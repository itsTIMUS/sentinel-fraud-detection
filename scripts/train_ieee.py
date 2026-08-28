"""Train LightGBM on IEEE-CIS dataset — proves architecture handles real-world data."""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
import time
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

import sys
sys.path.insert(0, ".")
from src.sentinel.features.ieee_builder import (
    build_ieee_features, ieee_features_to_array, IEEE_FEATURE_COLUMNS
)
from src.sentinel.model_wrapper import LGBMWrapper
from src.sentinel.cost import load_costs, make_decision

# --- Load data ---
print("Loading IEEE-CIS data...")
df = pd.read_csv("data/raw/ieee/train_transaction.csv")
print(f"Total rows: {len(df):,} | Columns: {len(df.columns)} | Fraud rate: {df['isFraud'].mean():.4f}")

# --- Temporal split (sort by TransactionDT) ---
df = df.sort_values("TransactionDT").reset_index(drop=True)
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx].copy()
val = df.iloc[split_idx:].copy()
print(f"Train: {len(train):,} rows | Val: {len(val):,} rows")
print(f"Train fraud rate: {train['isFraud'].mean():.4f} | Val fraud rate: {val['isFraud'].mean():.4f}")

# --- Build features ---
print("\nBuilding train features...")
start = time.time()
train_features = []
for idx, row in train.iterrows():
    f = build_ieee_features(row.to_dict())
    train_features.append(ieee_features_to_array(f))
    if idx % 100000 == 0 and idx > 0:
        print(f"  ...{idx:,} rows")

X_train = np.vstack(train_features)
y_train = train["isFraud"].values
print(f"Train features built in {time.time() - start:.0f}s | Shape: {X_train.shape}")

print("Building val features...")
start = time.time()
val_features = []
for idx, row in val.iterrows():
    f = build_ieee_features(row.to_dict())
    val_features.append(ieee_features_to_array(f))

X_val = np.vstack(val_features)
y_val = val["isFraud"].values
print(f"Val features built in {time.time() - start:.0f}s | Shape: {X_val.shape}")

# --- Train LightGBM ---
print("\nTraining LightGBM...")
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

train_data = lgb.Dataset(X_train, label=y_train, feature_name=IEEE_FEATURE_COLUMNS)
val_data = lgb.Dataset(X_val, label=y_val, feature_name=IEEE_FEATURE_COLUMNS, reference=train_data)

params = {
    "objective": "binary",
    "metric": "average_precision",
    "verbosity": -1,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "scale_pos_weight": neg / pos,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
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

# --- Evaluate raw ---
y_pred_raw = model.predict(X_val)
precision_arr, recall_arr, _ = precision_recall_curve(y_val, y_pred_raw)
pr_auc_raw = auc(recall_arr, precision_arr)
brier_raw = brier_score_loss(y_val, y_pred_raw)

print(f"\nRaw LightGBM:")
print(f"  PR-AUC: {pr_auc_raw:.4f}")
print(f"  Brier:  {brier_raw:.6f}")

# --- Calibrate ---
print("\nCalibrating...")
cal_split = int(len(X_val) * 0.7)
X_cal_train, X_cal_test = X_val[:cal_split], X_val[cal_split:]
y_cal_train, y_cal_test = y_val[:cal_split], y_val[cal_split:]

wrapper = LGBMWrapper(booster=model)
calibrator = CalibratedClassifierCV(wrapper, method="isotonic", cv="prefit")
calibrator.fit(X_cal_train, y_cal_train)

y_pred_cal = calibrator.predict_proba(X_cal_test)[:, 1]
brier_cal = brier_score_loss(y_cal_test, y_pred_cal)
print(f"  Brier before: {brier_raw:.6f}")
print(f"  Brier after:  {brier_cal:.6f}")
print(f"  Improvement:  {((brier_raw - brier_cal) / brier_raw) * 100:.1f}%")

# --- Cost evaluation ---
print("\nCost evaluation on validation set...")
costs = load_costs()
y_pred_full = calibrator.predict_proba(X_val)[:, 1]

total_cost_sentinel = 0.0
total_cost_approve_all = 0.0
decisions = {"ALLOW": 0, "REVIEW": 0, "BLOCK": 0}
tp, fp, fn, tn = 0, 0, 0, 0

for i in range(len(y_val)):
    amt = float(val.iloc[i]["TransactionAmt"])
    true_label = y_val[i]
    p = float(y_pred_full[i])

    result = make_decision(p_fraud=p, amount=amt, costs=costs)
    decisions[result["decision"]] += 1

    if result["decision"] == "ALLOW" and true_label == 1:
        total_cost_sentinel += amt + costs["chargeback_fee_inr"]
        fn += 1
    elif result["decision"] == "BLOCK" and true_label == 0:
        total_cost_sentinel += costs["gross_margin"] * amt + costs["friction_cost_inr"] + costs["churn_probability"] * costs["customer_ltv_inr"]
        fp += 1
    elif result["decision"] == "BLOCK" and true_label == 1:
        tp += 1
    elif result["decision"] == "ALLOW" and true_label == 0:
        tn += 1
    elif result["decision"] == "REVIEW":
        total_cost_sentinel += costs["review_cost_inr"]
        if true_label == 1 and np.random.random() > costs["analyst_catch_rate"]:
            total_cost_sentinel += amt + costs["chargeback_fee_inr"]

    if true_label == 1:
        total_cost_approve_all += amt + costs["chargeback_fee_inr"]

savings_pct = (1 - total_cost_sentinel / total_cost_approve_all) * 100

print(f"\n{'='*50}")
print(f"  IEEE-CIS RESULTS")
print(f"{'='*50}")
print(f"  PR-AUC:     {pr_auc_raw:.4f}")
print(f"  Brier:      {brier_cal:.6f}")
print(f"  Decisions:  {decisions}")
print(f"  TP: {tp:,} | FP: {fp:,} | FN: {fn:,} | TN: {tn:,}")
if tp + fp > 0:
    print(f"  Precision:  {tp/(tp+fp):.4f}")
if tp + fn > 0:
    print(f"  Recall:     {tp/(tp+fn):.4f}")
print(f"\n  Approve all:      ₹{total_cost_approve_all:>12,.0f}")
print(f"  Sentinel:         ₹{total_cost_sentinel:>12,.0f}")
print(f"  Savings:          {savings_pct:.1f}%")

# Feature importance
importance = model.feature_importance(importance_type="gain")
feat_imp = sorted(zip(IEEE_FEATURE_COLUMNS, importance), key=lambda x: x[1], reverse=True)
print(f"\n  Top 10 Features (gain):")
for name, imp in feat_imp[:10]:
    print(f"    {name:>25s}: {imp:>12.1f}")

# --- Save artifacts ---
print("\nSaving artifacts...")
ieee_artifacts = Path("artifacts/ieee")
ieee_artifacts.mkdir(parents=True, exist_ok=True)

model.save_model(str(ieee_artifacts / "model.lgb"))
joblib.dump(calibrator, ieee_artifacts / "calibrator.joblib")

json.dump({
    "feature_columns": IEEE_FEATURE_COLUMNS,
    "num_features": len(IEEE_FEATURE_COLUMNS),
    "version": "v1-ieee",
}, open(ieee_artifacts / "feature_schema.json", "w"), indent=2)

json.dump({
    "pr_auc": round(pr_auc_raw, 4),
    "brier_before": round(brier_raw, 6),
    "brier_after": round(brier_cal, 6),
    "total_cost_sentinel_inr": round(total_cost_sentinel, 2),
    "total_cost_approve_all_inr": round(total_cost_approve_all, 2),
    "savings_pct": round(savings_pct, 1),
    "decisions": decisions,
    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    "precision": round(tp/(tp+fp), 4) if tp+fp > 0 else 0,
    "recall": round(tp/(tp+fn), 4) if tp+fn > 0 else 0,
    "val_rows": len(val),
    "val_fraud_rate": round(float(val["isFraud"].mean()), 4),
}, open(ieee_artifacts / "metrics.json", "w"), indent=2)

print(f"\n✅ IEEE-CIS model saved to artifacts/ieee/")
print(f"✅ metrics.json saved")

# --- Comparison with Sparkov ---
try:
    sparkov_metrics = json.load(open("artifacts/sparkov/metrics.json"))
    print(f"\n{'='*50}")
    print(f"  SPARKOV vs IEEE-CIS COMPARISON")
    print(f"{'='*50}")
    print(f"  {'Metric':<25s} {'Sparkov':>12s} {'IEEE-CIS':>12s}")
    print(f"  {'PR-AUC':<25s} {sparkov_metrics.get('pr_auc',0):>12.4f} {pr_auc_raw:>12.4f}")
    print(f"  {'Savings vs approve-all':<25s} {sparkov_metrics.get('savings_vs_approve_all_pct',0):>11.1f}% {savings_pct:>11.1f}%")
    print(f"  {'Precision':<25s} {sparkov_metrics.get('precision_at_op',0):>12.4f} {tp/(tp+fp) if tp+fp>0 else 0:>12.4f}")
    print(f"  {'Recall':<25s} {sparkov_metrics.get('recall_at_op',0):>12.4f} {tp/(tp+fn) if tp+fn>0 else 0:>12.4f}")
except FileNotFoundError:
    pass