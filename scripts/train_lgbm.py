"""Train LightGBM using the shared feature builder."""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import time
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc

import sys
sys.path.insert(0, ".")
from src.sentinel.features import build_features, features_to_array, FEATURE_COLUMNS
from src.sentinel.cost import load_costs, make_decision

# --- Load data ---
print("Loading data...")
df = pd.read_parquet("data/processed/train.parquet")
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["hour"] = df["trans_date_trans_time"].dt.hour
df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
df = df.sort_values("trans_date_trans_time").reset_index(drop=True)

# --- Temporal split ---
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx].copy()
val = df.iloc[split_idx:].copy()
print(f"Train: {len(train):,} rows | Val: {len(val):,} rows")

# --- Precompute velocity features per card (point-in-time) ---
print("Computing velocity features (this may take a few minutes)...")
start_time = time.time()

def compute_velocity_for_split(data: pd.DataFrame) -> list[dict]:
    """Compute point-in-time velocity features for each transaction."""
    # Sort by time
    data = data.sort_values("trans_date_trans_time").reset_index(drop=True)
    
    histories = []
    # Track per-card running stats
    card_txns = {}  # card_id -> list of (unix_time, amt, merchant)
    
    for idx, row in data.iterrows():
        card = row["cc_num"]
        unix_t = row["unix_time"]
        amt = row["amt"]
        merchant = row["merchant"]
        
        if card not in card_txns:
            card_txns[card] = []
        
        # Get history BEFORE this transaction (strictly before)
        past = card_txns[card]
        
        if len(past) == 0:
            history = None
        else:
            past_amts = [p[1] for p in past]
            # Count in last 1h (3600 seconds)
            count_1h = sum(1 for p in past if unix_t - p[0] <= 3600)
            # Count in last 24h
            count_24h = sum(1 for p in past if unix_t - p[0] <= 86400)
            # Sum in last 24h
            sum_24h = sum(p[1] for p in past if unix_t - p[0] <= 86400)
            # Distinct merchants in 24h
            merchants_24h = len(set(p[2] for p in past if unix_t - p[0] <= 86400))
            
            history = {
                "txn_count_1h": count_1h,
                "txn_count_24h": count_24h,
                "txn_sum_24h": sum_24h,
                "median_amt": float(np.median(past_amts)),
                "distinct_merchants_24h": merchants_24h,
            }
        
        histories.append(history)
        
        # Add this transaction to card history AFTER computing features
        card_txns[card].append((unix_t, amt, merchant))
        
        if idx % 200000 == 0 and idx > 0:
            print(f"  ...processed {idx:,} / {len(data):,} rows")
    
    return histories

train_histories = compute_velocity_for_split(train)
elapsed = time.time() - start_time
print(f"Velocity computed for train in {elapsed:.0f}s")

# --- Build feature matrix ---
print("Building feature matrix for train...")
train_features = []
for idx, (_, row) in enumerate(train.iterrows()):
    txn = row.to_dict()
    f = build_features(txn, history=train_histories[idx])
    train_features.append(features_to_array(f))

X_train = np.vstack(train_features)
y_train = train["is_fraud"].values
print(f"X_train shape: {X_train.shape}")

# --- Same for validation (fresh velocity tracking) ---
print("Computing velocity + features for validation...")
val_histories = compute_velocity_for_split(val)

val_features = []
for idx, (_, row) in enumerate(val.iterrows()):
    txn = row.to_dict()
    f = build_features(txn, history=val_histories[idx])
    val_features.append(features_to_array(f))

X_val = np.vstack(val_features)
y_val = val["is_fraud"].values
print(f"X_val shape: {X_val.shape}")

# --- Train LightGBM ---
print("\nTraining LightGBM...")
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLUMNS)
val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLUMNS, reference=train_data)

params = {
    "objective": "binary",
    "metric": "average_precision",
    "verbosity": -1,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "scale_pos_weight": neg / pos,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42,
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=30),
        lgb.log_evaluation(period=50),
    ],
)

# --- Evaluate ---
print("\nEvaluating...")
y_pred = model.predict(X_val)

precision, recall, _ = precision_recall_curve(y_val, y_pred)
pr_auc = auc(recall, precision)

# ₹ cost
costs = load_costs()
total_cost = 0.0
decisions = {"ALLOW": 0, "REVIEW": 0, "BLOCK": 0}

for i in range(len(y_val)):
    result = make_decision(p_fraud=float(y_pred[i]), amount=float(val.iloc[i]["amt"]), costs=costs)
    decisions[result["decision"]] += 1
    if result["decision"] == "ALLOW" and y_val[i] == 1:
        total_cost += float(val.iloc[i]["amt"]) + costs["chargeback_fee_inr"]
    elif result["decision"] == "BLOCK" and y_val[i] == 0:
        total_cost += costs["gross_margin"] * float(val.iloc[i]["amt"]) + costs["friction_cost_inr"] + costs["churn_probability"] * costs["customer_ltv_inr"]
    elif result["decision"] == "REVIEW":
        total_cost += costs["review_cost_inr"]
        if y_val[i] == 1 and np.random.random() > costs["analyst_catch_rate"]:
            total_cost += float(val.iloc[i]["amt"]) + costs["chargeback_fee_inr"]

print(f"\n{'='*50}")
print(f"  LightGBM Results")
print(f"{'='*50}")
print(f"  PR-AUC:     {pr_auc:.4f}")
print(f"  Total ₹:    {total_cost:,.0f}")
print(f"  Decisions:  {decisions}")
print(f"  Best iteration: {model.best_iteration}")

# --- Feature importance ---
importance = model.feature_importance(importance_type="gain")
feat_imp = sorted(zip(FEATURE_COLUMNS, importance), key=lambda x: x[1], reverse=True)
print(f"\n  Feature Importance (gain):")
for name, imp in feat_imp:
    print(f"    {name:>30s}: {imp:>10.1f}")

# --- Save artifacts ---
print("\nSaving artifacts...")
artifacts = Path("artifacts/sparkov")
artifacts.mkdir(parents=True, exist_ok=True)

# Native LightGBM booster (faster than sklearn wrapper)
model.save_model(str(artifacts / "model.lgb"))

# Feature schema
schema = {
    "feature_columns": FEATURE_COLUMNS,
    "num_features": len(FEATURE_COLUMNS),
    "version": "v2-lgbm",
}
json.dump(schema, open(artifacts / "feature_schema.json", "w"), indent=2)

# Column order
json.dump({"columns": FEATURE_COLUMNS}, open(artifacts / "column_order.json", "w"), indent=2)

# Metrics
metrics = {
    "pr_auc": round(pr_auc, 4),
    "total_cost_inr": round(total_cost, 2),
    "decisions": decisions,
    "best_iteration": model.best_iteration,
    "val_rows": len(val),
    "val_fraud_rate": round(float(val["is_fraud"].mean()), 4),
}
json.dump(metrics, open(artifacts / "metrics.json", "w"), indent=2)

print(f"\n✅ model.lgb saved")
print(f"✅ feature_schema.json saved")
print(f"✅ column_order.json saved")
print(f"✅ metrics.json saved")

# --- Load baseline metrics for comparison ---
try:
    baseline = json.load(open(artifacts / "baseline_metrics.json"))
    print(f"\n{'='*50}")
    print(f"  COMPARISON")
    print(f"{'='*50}")
    print(f"  Approve all:  ₹{baseline['approve_all_cost_inr']:>12,.0f}")
    print(f"  Rules:        ₹{baseline['rules_cost_inr']:>12,.0f}  (PR-AUC: {baseline['rules_pr_auc']:.4f})")
    print(f"  LR:           ₹{baseline['lr_cost_inr']:>12,.0f}  (PR-AUC: {baseline['lr_pr_auc']:.4f})")
    print(f"  LightGBM:     ₹{total_cost:>12,.0f}  (PR-AUC: {pr_auc:.4f})")
except FileNotFoundError:
    pass