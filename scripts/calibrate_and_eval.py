"""Day 3: Calibration + break-even curve + held-out evaluation."""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
import time
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    precision_recall_curve, auc, confusion_matrix,
    classification_report, brier_score_loss, precision_score, recall_score, f1_score
)
from sklearn.base import BaseEstimator, ClassifierMixin

import sys
sys.path.insert(0, ".")
from src.sentinel.features import build_features, features_to_array, FEATURE_COLUMNS
from src.sentinel.cost import load_costs, make_decision

ARTIFACTS = Path("artifacts/sparkov")
PLOTS = Path("reports/plots")
costs = load_costs()

# ============================================================
# PART 1: Load model + rebuild validation data
# ============================================================
print("Loading model and data...")
booster = lgb.Booster(model_file=str(ARTIFACTS / "model.lgb"))

df = pd.read_parquet("data/processed/train.parquet")
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["hour"] = df["trans_date_trans_time"].dt.hour
df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
df = df.sort_values("trans_date_trans_time").reset_index(drop=True)

# Same temporal split as training
split_idx = int(len(df) * 0.8)
val = df.iloc[split_idx:].copy()
print(f"Validation: {len(val):,} rows ({val['is_fraud'].mean():.4f} fraud rate)")

# --- Rebuild val features + velocity ---
print("Rebuilding validation features (this takes a few minutes)...")
start_time = time.time()

val_sorted = val.sort_values("trans_date_trans_time").reset_index(drop=True)
card_txns = {}
val_features = []
val_histories = []

for idx, row in val_sorted.iterrows():
    card = row["cc_num"]
    unix_t = row["unix_time"]
    amt = row["amt"]
    merchant = row["merchant"]

    if card not in card_txns:
        card_txns[card] = []

    past = card_txns[card]
    if len(past) == 0:
        history = None
    else:
        past_amts = [p[1] for p in past]
        count_1h = sum(1 for p in past if unix_t - p[0] <= 3600)
        count_24h = sum(1 for p in past if unix_t - p[0] <= 86400)
        sum_24h = sum(p[1] for p in past if unix_t - p[0] <= 86400)
        merchants_24h = len(set(p[2] for p in past if unix_t - p[0] <= 86400))
        history = {
            "txn_count_1h": count_1h,
            "txn_count_24h": count_24h,
            "txn_sum_24h": sum_24h,
            "median_amt": float(np.median(past_amts)),
            "distinct_merchants_24h": merchants_24h,
        }

    txn_dict = row.to_dict()
    f = build_features(txn_dict, history=history)
    val_features.append(features_to_array(f))
    card_txns[card].append((unix_t, amt, merchant))

    if idx % 100000 == 0 and idx > 0:
        print(f"  ...{idx:,} rows")

X_val = np.vstack(val_features)
y_val = val_sorted["is_fraud"].values
print(f"Validation features built in {time.time() - start_time:.0f}s")

# Raw scores
y_raw = booster.predict(X_val)

# ============================================================
# PART 2: Calibration
# ============================================================
print("\n--- Calibrating ---")

# Wrap the booster for sklearn compatibility
class LGBMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, booster):
        self.booster = booster
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        raw = self.booster.predict(X)
        return np.column_stack([1 - raw, raw])

    def predict(self, X):
        return (self.booster.predict(X) > 0.5).astype(int)

wrapper = LGBMWrapper(booster)

# Use last 30% of validation for calibration, first 70% for calibration training
cal_split = int(len(X_val) * 0.7)
X_cal_train, X_cal_test = X_val[:cal_split], X_val[cal_split:]
y_cal_train, y_cal_test = y_val[:cal_split], y_val[cal_split:]

calibrator = CalibratedClassifierCV(wrapper, method="isotonic", cv="prefit")
calibrator.fit(X_cal_train, y_cal_train)

# Calibrated scores on the calibration test set
y_cal_raw = booster.predict(X_cal_test)
y_cal_calibrated = calibrator.predict_proba(X_cal_test)[:, 1]

# Brier scores
brier_before = brier_score_loss(y_cal_test, y_cal_raw)
brier_after = brier_score_loss(y_cal_test, y_cal_calibrated)
print(f"Brier score BEFORE calibration: {brier_before:.6f}")
print(f"Brier score AFTER calibration:  {brier_after:.6f}")
print(f"Improvement: {((brier_before - brier_after) / brier_before) * 100:.1f}%")

# ============================================================
# PART 3: Reliability Diagram (before/after)
# ============================================================
print("\nPlotting reliability diagram...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before
prob_true_b, prob_pred_b = calibration_curve(y_cal_test, y_cal_raw, n_bins=10, strategy="uniform")
axes[0].plot(prob_pred_b, prob_true_b, "s-", color="#e74c3c", label="LightGBM raw")
axes[0].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
axes[0].set_xlabel("Predicted probability")
axes[0].set_ylabel("True frequency")
axes[0].set_title(f"BEFORE Calibration (Brier: {brier_before:.6f})")
axes[0].legend()

# After
prob_true_a, prob_pred_a = calibration_curve(y_cal_test, y_cal_calibrated, n_bins=10, strategy="uniform")
axes[1].plot(prob_pred_a, prob_true_a, "s-", color="#2ecc71", label="Calibrated (isotonic)")
axes[1].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
axes[1].set_xlabel("Predicted probability")
axes[1].set_ylabel("True frequency")
axes[1].set_title(f"AFTER Calibration (Brier: {brier_after:.6f})")
axes[1].legend()

plt.tight_layout()
plt.savefig(PLOTS / "calibration_curve.png", dpi=150)
plt.close()
print("✅ Calibration curve saved")

# ============================================================
# PART 4: Break-Even Curve (THE PITCH CENTREPIECE)
# ============================================================
print("\nPlotting break-even curve...")

amounts = np.linspace(10, 30000, 500)
thresholds = []

for amt in amounts:
    # Solve: cost_allow(p, amt) = cost_block(p, amt)
    # p * (amt * (1 - recovery) + chargeback) = (1-p) * (margin*amt + friction + churn*ltv)
    # p * A = (1-p) * B
    # p * A = B - p * B
    # p * (A + B) = B
    # p = B / (A + B)
    A = amt * (1 - costs["goods_recovery_rate"]) + costs["chargeback_fee_inr"]
    B = costs["gross_margin"] * amt + costs["friction_cost_inr"] + costs["churn_probability"] * costs["customer_ltv_inr"]
    p_breakeven = B / (A + B)
    thresholds.append(p_breakeven)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(amounts, thresholds, color="#3498db", linewidth=2.5)
ax.fill_between(amounts, thresholds, 1.0, alpha=0.15, color="#e74c3c", label="BLOCK zone")
ax.fill_between(amounts, 0, thresholds, alpha=0.15, color="#2ecc71", label="ALLOW zone")
ax.set_xlabel("Transaction Amount (₹)", fontsize=12)
ax.set_ylabel("Break-Even Fraud Probability", fontsize=12)
ax.set_title("Sentinel Break-Even Curve: Threshold as f(Amount)\nSmall transactions → relaxed | Large transactions → paranoid", fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(0, 30000)
ax.set_ylim(0, 1.0)

# Annotate key points
for amt_point in [500, 5000, 20000]:
    A = amt_point * (1 - costs["goods_recovery_rate"]) + costs["chargeback_fee_inr"]
    B = costs["gross_margin"] * amt_point + costs["friction_cost_inr"] + costs["churn_probability"] * costs["customer_ltv_inr"]
    p = B / (A + B)
    ax.annotate(f"₹{amt_point:,}: p={p:.3f}",
                xy=(amt_point, p), xytext=(amt_point + 1500, p + 0.08),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(PLOTS / "breakeven_curve.png", dpi=150)
plt.close()
print("✅ Break-even curve saved")

# ============================================================
# PART 5: Held-Out Test Set Evaluation (ONE TIME ONLY)
# ============================================================
print("\n" + "=" * 60)
print("  HELD-OUT TEST SET EVALUATION — RUN ONCE, NO TUNING AFTER")
print("=" * 60)

test_df = pd.read_parquet("data/processed/test.parquet")
test_df["trans_date_trans_time"] = pd.to_datetime(test_df["trans_date_trans_time"])
test_df["hour"] = test_df["trans_date_trans_time"].dt.hour
test_df["day_of_week"] = test_df["trans_date_trans_time"].dt.dayofweek
test_df = test_df.sort_values("trans_date_trans_time").reset_index(drop=True)
print(f"Test set: {len(test_df):,} rows ({test_df['is_fraud'].mean():.4f} fraud rate)")

# Build test features with velocity
print("Building test features...")
start_time = time.time()
card_txns_test = {}
test_features = []

for idx, row in test_df.iterrows():
    card = row["cc_num"]
    unix_t = row["unix_time"]
    amt = row["amt"]
    merchant = row["merchant"]

    if card not in card_txns_test:
        card_txns_test[card] = []

    past = card_txns_test[card]
    if len(past) == 0:
        history = None
    else:
        past_amts = [p[1] for p in past]
        count_1h = sum(1 for p in past if unix_t - p[0] <= 3600)
        count_24h = sum(1 for p in past if unix_t - p[0] <= 86400)
        sum_24h = sum(p[1] for p in past if unix_t - p[0] <= 86400)
        merchants_24h = len(set(p[2] for p in past if unix_t - p[0] <= 86400))
        history = {
            "txn_count_1h": count_1h,
            "txn_count_24h": count_24h,
            "txn_sum_24h": sum_24h,
            "median_amt": float(np.median(past_amts)),
            "distinct_merchants_24h": merchants_24h,
        }

    txn_dict = row.to_dict()
    f = build_features(txn_dict, history=history)
    test_features.append(features_to_array(f))
    card_txns_test[card].append((unix_t, amt, merchant))

    if idx % 100000 == 0 and idx > 0:
        print(f"  ...{idx:,} rows")

X_test = np.vstack(test_features)
y_test = test_df["is_fraud"].values
print(f"Test features built in {time.time() - start_time:.0f}s")

# Calibrated predictions on test
y_test_calibrated = calibrator.predict_proba(X_test)[:, 1]
y_test_raw = booster.predict(X_test)

# PR-AUC
precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_test_calibrated)
test_pr_auc = auc(recall_arr, precision_arr)

# Brier
test_brier = brier_score_loss(y_test, y_test_calibrated)

# ₹ Cost evaluation with cost-aware decisions
total_cost_sentinel = 0.0
total_cost_approve_all = 0.0
total_cost_naive_05 = 0.0
decisions = {"ALLOW": 0, "REVIEW": 0, "BLOCK": 0}
tp, fp, fn, tn = 0, 0, 0, 0
cost_tp, cost_fp, cost_fn, cost_tn = 0.0, 0.0, 0.0, 0.0

for i in range(len(y_test)):
    amt = float(test_df.iloc[i]["amt"])
    true_label = y_test[i]
    p = float(y_test_calibrated[i])

    # Sentinel cost-aware decision
    result = make_decision(p_fraud=p, amount=amt, costs=costs)
    decisions[result["decision"]] += 1

    if result["decision"] == "ALLOW" and true_label == 1:
        total_cost_sentinel += amt + costs["chargeback_fee_inr"]
        fn += 1
        cost_fn += amt + costs["chargeback_fee_inr"]
    elif result["decision"] == "BLOCK" and true_label == 0:
        total_cost_sentinel += costs["gross_margin"] * amt + costs["friction_cost_inr"] + costs["churn_probability"] * costs["customer_ltv_inr"]
        fp += 1
        cost_fp += costs["gross_margin"] * amt + costs["friction_cost_inr"] + costs["churn_probability"] * costs["customer_ltv_inr"]
    elif result["decision"] == "BLOCK" and true_label == 1:
        tp += 1  # correctly blocked fraud, no cost
    elif result["decision"] == "ALLOW" and true_label == 0:
        tn += 1  # correctly allowed legit, no cost
    elif result["decision"] == "REVIEW":
        total_cost_sentinel += costs["review_cost_inr"]
        if true_label == 1 and np.random.random() > costs["analyst_catch_rate"]:
            total_cost_sentinel += amt + costs["chargeback_fee_inr"]

    # Approve-all baseline
    if true_label == 1:
        total_cost_approve_all += amt + costs["chargeback_fee_inr"]

    # Naive 0.5 threshold
    if p >= 0.5 and true_label == 0:
        total_cost_naive_05 += costs["gross_margin"] * amt + costs["friction_cost_inr"] + costs["churn_probability"] * costs["customer_ltv_inr"]
    elif p < 0.5 and true_label == 1:
        total_cost_naive_05 += amt + costs["chargeback_fee_inr"]

print(f"\n  PR-AUC:          {test_pr_auc:.4f}")
print(f"  Brier score:     {test_brier:.6f}")
print(f"  Decisions:       {decisions}")
print(f"\n  Confusion (BLOCK = positive prediction):")
print(f"    TP (blocked fraud):     {tp:>6,}")
print(f"    FP (blocked legit):     {fp:>6,}")
print(f"    FN (allowed fraud):     {fn:>6,}")
print(f"    TN (allowed legit):     {tn:>6,}")

# Precision/Recall at operating point (BLOCK decision)
if tp + fp > 0:
    op_precision = tp / (tp + fp)
else:
    op_precision = 0
if tp + fn > 0:
    op_recall = tp / (tp + fn)
else:
    op_recall = 0
print(f"\n  At operating point (BLOCK decisions):")
print(f"    Precision: {op_precision:.4f}")
print(f"    Recall:    {op_recall:.4f}")

print(f"\n  ₹ Cost Comparison (held-out test):")
print(f"    Approve everything: ₹{total_cost_approve_all:>12,.0f}")
print(f"    Naive 0.5 threshold: ₹{total_cost_naive_05:>12,.0f}")
print(f"    Sentinel (cost-aware): ₹{total_cost_sentinel:>12,.0f}")
savings_pct = (1 - total_cost_sentinel / total_cost_approve_all) * 100
print(f"    Savings vs approve-all: {savings_pct:.1f}%")

# ============================================================
# PART 6: Confusion Matrix Plot (in ₹)
# ============================================================
print("\nPlotting confusion matrix...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count-based
cm_counts = np.array([[tn, fp], [fn, tp]])
im = axes[0].imshow(cm_counts, cmap="Blues")
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(["ALLOW", "BLOCK"])
axes[0].set_yticklabels(["Legit", "Fraud"])
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_title("Confusion Matrix (Counts)")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, f"{cm_counts[i, j]:,}", ha="center", va="center",
                     color="white" if cm_counts[i, j] > cm_counts.max() / 2 else "black", fontsize=14)

# ₹-based
cm_cost = np.array([[0, cost_fp], [cost_fn, 0]])
im2 = axes[1].imshow(cm_cost, cmap="Reds")
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(["ALLOW", "BLOCK"])
axes[1].set_yticklabels(["Legit", "Fraud"])
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].set_title("Confusion Matrix (₹ Cost)")
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f"₹{cm_cost[i, j]:,.0f}", ha="center", va="center",
                     color="white" if cm_cost[i, j] > cm_cost.max() / 2 else "black", fontsize=13)

plt.tight_layout()
plt.savefig(PLOTS / "confusion_matrix.png", dpi=150)
plt.close()
print("✅ Confusion matrix saved")

# ============================================================
# PART 7: PR Curve
# ============================================================
print("Plotting PR curve...")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recall_arr, precision_arr, color="#3498db", linewidth=2, label=f"LightGBM calibrated (PR-AUC={test_pr_auc:.4f})")
ax.axhline(y=y_test.mean(), color="gray", linestyle="--", label=f"Baseline (fraud rate={y_test.mean():.4f})")
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve (Held-Out Test Set)", fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(PLOTS / "pr_curve.png", dpi=150)
plt.close()
print("✅ PR curve saved")

# ============================================================
# PART 8: Save final artifacts
# ============================================================
print("\nSaving final artifacts...")

# Save calibrator
joblib.dump(calibrator, ARTIFACTS / "calibrator.joblib")

# Save thresholds
thresholds_data = {
    "policy": "expected_cost_v2",
    "calibration": "isotonic",
    "note": "Thresholds are computed per-transaction based on amount. No single fixed threshold.",
    "example_breakevens": {
        "500_inr": round(float(thresholds[np.argmin(np.abs(amounts - 500))]), 4),
        "5000_inr": round(float(thresholds[np.argmin(np.abs(amounts - 5000))]), 4),
        "20000_inr": round(float(thresholds[np.argmin(np.abs(amounts - 20000))]), 4),
    }
}
json.dump(thresholds_data, open(ARTIFACTS / "thresholds.json", "w"), indent=2)

# Update metrics
final_metrics = {
    "pr_auc": round(test_pr_auc, 4),
    "brier_score_calibrated": round(test_brier, 6),
    "brier_before_calibration": round(brier_before, 6),
    "total_cost_sentinel_inr": round(total_cost_sentinel, 2),
    "total_cost_approve_all_inr": round(total_cost_approve_all, 2),
    "total_cost_naive_05_inr": round(total_cost_naive_05, 2),
    "savings_vs_approve_all_pct": round(savings_pct, 1),
    "precision_at_op": round(op_precision, 4),
    "recall_at_op": round(op_recall, 4),
    "decisions": decisions,
    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    "test_rows": len(test_df),
    "test_fraud_rate": round(float(test_df["is_fraud"].mean()), 4),
    "evaluated_once": True,
}
json.dump(final_metrics, open(ARTIFACTS / "metrics.json", "w"), indent=2)

print(f"\n✅ calibrator.joblib saved")
print(f"✅ thresholds.json saved")
print(f"✅ metrics.json updated with test results")
print(f"\n🏁 MODEL IS FROZEN. No more tuning.")