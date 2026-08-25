"""Rules baseline + Logistic Regression baseline, scored on PR-AUC and ₹ cost."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
import json
import sys
sys.path.insert(0, ".")
from src.sentinel.cost import load_costs, make_decision

# --- Load data ---
print("Loading data...")
df = pd.read_parquet("data/processed/train.parquet")
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df = df.sort_values("trans_date_trans_time").reset_index(drop=True)

# --- Temporal split: last 20% by time = validation ---
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx].copy()
val = df.iloc[split_idx:].copy()
print(f"Train: {len(train):,} rows ({train['is_fraud'].mean():.4f} fraud rate)")
print(f"Val:   {len(val):,} rows ({val['is_fraud'].mean():.4f} fraud rate)")
print(f"Val period: {val['trans_date_trans_time'].min()} to {val['trans_date_trans_time'].max()}")

# --- Helper: compute total ₹ cost on a set of predictions ---
costs = load_costs()

def compute_total_cost(y_true, y_scores, threshold=0.5):
    """Compute total ₹ cost using cost model at a given threshold."""
    total = 0.0
    decisions = {"ALLOW": 0, "REVIEW": 0, "BLOCK": 0}
    for true_label, score, amt in zip(y_true, y_scores, val["amt"]):
        result = make_decision(p_fraud=float(score), amount=float(amt), costs=costs)
        decisions[result["decision"]] += 1
        # Actual cost depends on true label + decision
        if result["decision"] == "ALLOW" and true_label == 1:
            total += amt + costs["chargeback_fee_inr"]  # fraud got through
        elif result["decision"] == "BLOCK" and true_label == 0:
            total += costs["gross_margin"] * amt + costs["friction_cost_inr"] + costs["churn_probability"] * costs["customer_ltv_inr"]
        elif result["decision"] == "REVIEW":
            total += costs["review_cost_inr"]
            if true_label == 1 and np.random.random() > costs["analyst_catch_rate"]:
                total += amt + costs["chargeback_fee_inr"]  # analyst missed it
    return total, decisions

def evaluate(name, y_true, y_scores):
    """Compute PR-AUC and ₹ cost."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    total_cost, decisions = compute_total_cost(y_true, y_scores)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  PR-AUC:     {pr_auc:.4f}")
    print(f"  Total ₹:    {total_cost:,.0f}")
    print(f"  Decisions:  {decisions}")
    return pr_auc, total_cost

# ============================================================
# BASELINE 1: Rules
# ============================================================
print("\n--- Building Rules Baseline ---")
val["hour"] = val["trans_date_trans_time"].dt.hour

def rules_score(row):
    """Hand-written rules returning a pseudo-probability."""
    score = 0.0
    # Night transaction (22:00-04:00)
    if row["hour"] >= 22 or row["hour"] <= 4:
        score += 0.3
    # High-risk category
    if row["category"] in ("shopping_net", "misc_net", "grocery_pos"):
        score += 0.2
    # High amount (above ₹500)
    if row["amt"] > 500:
        score += 0.3
    # Very high amount
    if row["amt"] > 1000:
        score += 0.2
    return min(score, 1.0)

rules_scores = val.apply(rules_score, axis=1).values
rules_prauc, rules_cost = evaluate("Rules Baseline", val["is_fraud"].values, rules_scores)

# ============================================================
# BASELINE 2: Logistic Regression (8 simple features)
# ============================================================
print("\n--- Building LR Baseline ---")

# Simple features for both train and val
def make_simple_features(data):
    feats = pd.DataFrame()
    feats["log_amt"] = np.log1p(data["amt"])
    feats["hour"] = data["trans_date_trans_time"].dt.hour
    feats["day_of_week"] = data["trans_date_trans_time"].dt.dayofweek
    feats["is_night"] = ((feats["hour"] >= 22) | (feats["hour"] <= 5)).astype(float)
    feats["city_pop_log"] = np.log1p(data["city_pop"])
    # Category encoding
    le = LabelEncoder()
    feats["category_enc"] = le.fit_transform(data["category"])
    feats["lat"] = data["lat"]
    feats["long"] = data["long"]
    return feats, le

X_train_lr, le = make_simple_features(train)
y_train = train["is_fraud"].values

# Reuse the same label encoder for val
X_val_lr = pd.DataFrame()
X_val_lr["log_amt"] = np.log1p(val["amt"])
X_val_lr["hour"] = val["trans_date_trans_time"].dt.hour
X_val_lr["day_of_week"] = val["trans_date_trans_time"].dt.dayofweek
X_val_lr["is_night"] = ((X_val_lr["hour"] >= 22) | (X_val_lr["hour"] <= 5)).astype(float)
X_val_lr["city_pop_log"] = np.log1p(val["city_pop"])
X_val_lr["category_enc"] = val["category"].map(dict(zip(le.classes_, le.transform(le.classes_)))).fillna(-1)
X_val_lr["lat"] = val["lat"]
X_val_lr["long"] = val["long"]

y_val = val["is_fraud"].values

# Train LR with class weight
lr = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    solver="lbfgs",
    random_state=42
)
lr.fit(X_train_lr, y_train)

lr_scores = lr.predict_proba(X_val_lr)[:, 1]
lr_prauc, lr_cost = evaluate("Logistic Regression", y_val, lr_scores)

# ============================================================
# BASELINE 0: Approve Everything (floor)
# ============================================================
approve_all_cost = val[val["is_fraud"] == 1]["amt"].sum() + val["is_fraud"].sum() * costs["chargeback_fee_inr"]
print(f"\n{'='*50}")
print(f"  Approve Everything (floor)")
print(f"{'='*50}")
print(f"  Total ₹:    {approve_all_cost:,.0f}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*50}")
print(f"  SUMMARY")
print(f"{'='*50}")
print(f"  Approve all:  ₹{approve_all_cost:>12,.0f}")
print(f"  Rules:        ₹{rules_cost:>12,.0f}  (PR-AUC: {rules_prauc:.4f})")
print(f"  LR:           ₹{lr_cost:>12,.0f}  (PR-AUC: {lr_prauc:.4f})")

# --- Save LR model for API ---
Path("artifacts/sparkov").mkdir(parents=True, exist_ok=True)
joblib.dump(lr, "artifacts/sparkov/model_lr.joblib")
joblib.dump(le, "artifacts/sparkov/label_encoder.joblib")

# Save comparison metrics
metrics = {
    "approve_all_cost_inr": round(approve_all_cost, 2),
    "rules_pr_auc": round(rules_prauc, 4),
    "rules_cost_inr": round(rules_cost, 2),
    "lr_pr_auc": round(lr_prauc, 4),
    "lr_cost_inr": round(lr_cost, 2),
    "val_rows": len(val),
    "val_fraud_rate": round(val["is_fraud"].mean(), 4),
}
json.dump(metrics, open("artifacts/sparkov/baseline_metrics.json", "w"), indent=2)

print(f"\n✅ LR model saved to artifacts/sparkov/model_lr.joblib")
print(f"✅ Metrics saved to artifacts/sparkov/baseline_metrics.json")