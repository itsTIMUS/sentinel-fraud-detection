"""Oracle (perfect info) + best fixed threshold baselines.
Answers: 'How much of the achievable savings does Sentinel actually capture?'"""

import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import time
import matplotlib.pyplot as plt
from pathlib import Path
from src.sentinel.features import build_features, features_to_array, FEATURE_COLUMNS
from src.sentinel.model_wrapper import LGBMWrapper
from src.sentinel.cost import load_costs, make_decision

costs = load_costs()
PLOTS = Path("reports/plots")

# --- Load predictions (reuse from tornado) ---
print("Loading Sparkov test data...")
booster = lgb.Booster(model_file="artifacts/sparkov/model.lgb")
calibrator = joblib.load("artifacts/sparkov/calibrator.joblib")

test_df = pd.read_parquet("data/processed/test.parquet")
test_df["trans_date_trans_time"] = pd.to_datetime(test_df["trans_date_trans_time"])
test_df["hour"] = test_df["trans_date_trans_time"].dt.hour
test_df["day_of_week"] = test_df["trans_date_trans_time"].dt.dayofweek
test_df = test_df.sort_values("trans_date_trans_time").reset_index(drop=True)

print("Building features...")
start = time.time()
card_txns = {}
test_features = []
for idx, row in test_df.iterrows():
    card, unix_t, amt, merchant = row["cc_num"], row["unix_time"], row["amt"], row["merchant"]
    if card not in card_txns:
        card_txns[card] = []
    past = card_txns[card]
    if len(past) == 0:
        history = None
    else:
        past_amts = [p[1] for p in past]
        history = {
            "txn_count_1h": sum(1 for p in past if unix_t - p[0] <= 3600),
            "txn_count_24h": sum(1 for p in past if unix_t - p[0] <= 86400),
            "txn_sum_24h": sum(p[1] for p in past if unix_t - p[0] <= 86400),
            "median_amt": float(np.median(past_amts)),
            "distinct_merchants_24h": len(set(p[2] for p in past if unix_t - p[0] <= 86400)),
        }
    test_features.append(features_to_array(build_features(row.to_dict(), history=history)))
    card_txns[card].append((unix_t, amt, merchant))
    if idx % 200000 == 0 and idx > 0:
        print(f"  ...{idx:,} rows")

X_test = np.vstack(test_features)
y_test = test_df["is_fraud"].values
y_pred = calibrator.predict_proba(X_test)[:, 1]
amounts = test_df["amt"].values
print(f"Ready in {time.time() - start:.0f}s")

# --- Helper: compute total cost ---
def compute_cost(y_true, y_scores, amts, costs_dict):
    total = 0.0
    decisions = {"ALLOW": 0, "CHALLENGE": 0, "REVIEW": 0, "BLOCK": 0}
    for i in range(len(y_true)):
        result = make_decision(p_fraud=float(y_scores[i]), amount=float(amts[i]), costs=costs_dict)
        decisions[result["decision"]] += 1
        if result["decision"] == "ALLOW" and y_true[i] == 1:
            total += float(amts[i]) + costs_dict["chargeback_fee_inr"]
        elif result["decision"] == "BLOCK" and y_true[i] == 0:
            total += costs_dict["gross_margin"] * float(amts[i]) + costs_dict["friction_cost_inr"] + costs_dict["churn_probability"] * costs_dict["customer_ltv_inr"]
        elif result["decision"] == "CHALLENGE":
            if y_true[i] == 0:
                total += costs_dict.get("challenge_friction_inr", 15)
            else:
                if np.random.random() > costs_dict.get("fraudster_3ds_dropout", 0.95):
                    total += float(amts[i]) + costs_dict["chargeback_fee_inr"]
        elif result["decision"] == "REVIEW":
            total += costs_dict["review_cost_inr"] + costs_dict.get("review_delay_churn_inr", 30)
            if y_true[i] == 1 and np.random.random() > costs_dict["analyst_catch_rate"]:
                total += float(amts[i]) + costs_dict["chargeback_fee_inr"]
    return total, decisions

# ============================================================
# 1. APPROVE ALL
# ============================================================
cost_approve_all = sum(amounts[y_test == 1]) + y_test.sum() * costs["chargeback_fee_inr"]
print(f"\n1. Approve all:          ₹{cost_approve_all:>12,.0f}")

# ============================================================
# 2. ORACLE (perfect information — knows every fraud)
# ============================================================
# Oracle blocks all fraud, allows all legit. Cost = 0 for fraud (blocked correctly)
# But blocking has a cost: for very small frauds, allowing might be cheaper
np.random.seed(42)
oracle_scores = y_test.astype(float)  # p=1.0 for fraud, p=0.0 for legit
cost_oracle, dec_oracle = compute_cost(y_test, oracle_scores, amounts, costs)
print(f"2. Oracle (perfect):     ₹{cost_oracle:>12,.0f}  decisions: {dec_oracle}")

# ============================================================
# 3. BEST FIXED THRESHOLD (tuned on validation, not test)
# ============================================================
print("\n3. Finding best fixed threshold...")
# Use validation set for tuning
val_df = pd.read_parquet("data/processed/train.parquet")
val_df["trans_date_trans_time"] = pd.to_datetime(val_df["trans_date_trans_time"])
val_df["hour"] = val_df["trans_date_trans_time"].dt.hour
val_df["day_of_week"] = val_df["trans_date_trans_time"].dt.dayofweek
val_df = val_df.sort_values("trans_date_trans_time").reset_index(drop=True)
split_idx = int(len(val_df) * 0.8)
val_df = val_df.iloc[split_idx:].copy()

# Build val features
card_txns_v = {}
val_features = []
for idx, row in val_df.iterrows():
    card, unix_t, amt, merchant = row["cc_num"], row["unix_time"], row["amt"], row["merchant"]
    if card not in card_txns_v:
        card_txns_v[card] = []
    past = card_txns_v[card]
    if len(past) == 0:
        history = None
    else:
        past_amts = [p[1] for p in past]
        history = {
            "txn_count_1h": sum(1 for p in past if unix_t - p[0] <= 3600),
            "txn_count_24h": sum(1 for p in past if unix_t - p[0] <= 86400),
            "txn_sum_24h": sum(p[1] for p in past if unix_t - p[0] <= 86400),
            "median_amt": float(np.median(past_amts)),
            "distinct_merchants_24h": len(set(p[2] for p in past if unix_t - p[0] <= 86400)),
        }
    val_features.append(features_to_array(build_features(row.to_dict(), history=history)))
    card_txns_v[card].append((unix_t, amt, merchant))

X_val = np.vstack(val_features)
y_val = val_df["is_fraud"].values
y_pred_val = calibrator.predict_proba(X_val)[:, 1]
amounts_val = val_df["amt"].values

# Search for best threshold on validation
best_t = 0.5
best_cost = float("inf")
threshold_costs = []

for t in np.arange(0.01, 0.99, 0.01):
    cost_t = 0.0
    for i in range(len(y_val)):
        if y_pred_val[i] >= t:
            # Block
            if y_val[i] == 0:
                cost_t += costs["gross_margin"] * float(amounts_val[i]) + costs["friction_cost_inr"] + costs["churn_probability"] * costs["customer_ltv_inr"]
        else:
            # Allow
            if y_val[i] == 1:
                cost_t += float(amounts_val[i]) + costs["chargeback_fee_inr"]
    threshold_costs.append((t, cost_t))
    if cost_t < best_cost:
        best_cost = cost_t
        best_t = t

print(f"   Best threshold on validation: {best_t:.2f}")

# Apply best threshold to TEST set
cost_fixed = 0.0
dec_fixed = {"ALLOW": 0, "BLOCK": 0}
for i in range(len(y_test)):
    if y_pred[i] >= best_t:
        dec_fixed["BLOCK"] += 1
        if y_test[i] == 0:
            cost_fixed += costs["gross_margin"] * float(amounts[i]) + costs["friction_cost_inr"] + costs["churn_probability"] * costs["customer_ltv_inr"]
    else:
        dec_fixed["ALLOW"] += 1
        if y_test[i] == 1:
            cost_fixed += float(amounts[i]) + costs["chargeback_fee_inr"]

print(f"   Fixed threshold (t={best_t:.2f}): ₹{cost_fixed:>12,.0f}  decisions: {dec_fixed}")

# ============================================================
# 4. SENTINEL (cost-aware, 4 actions)
# ============================================================
np.random.seed(42)
cost_sentinel, dec_sentinel = compute_cost(y_test, y_pred, amounts, costs)
print(f"\n4. Sentinel (cost-aware): ₹{cost_sentinel:>12,.0f}  decisions: {dec_sentinel}")

# ============================================================
# SUMMARY
# ============================================================
achievable = cost_approve_all - cost_oracle
captured = cost_approve_all - cost_sentinel
pct_captured = (captured / achievable) * 100 if achievable > 0 else 0

print(f"\n{'='*60}")
print(f"  COMPLETE BASELINE COMPARISON")
print(f"{'='*60}")
print(f"  {'Strategy':<30s} {'Total ₹':>12s} {'Savings':>10s}")
print(f"  {'Approve everything':<30s} {'₹'+f'{cost_approve_all:,.0f}':>12s} {'—':>10s}")
print(f"  {'Best fixed threshold':<30s} {'₹'+f'{cost_fixed:,.0f}':>12s} {f'{(1-cost_fixed/cost_approve_all)*100:.1f}%':>10s}")
print(f"  {'Sentinel (4-action)':<30s} {'₹'+f'{cost_sentinel:,.0f}':>12s} {f'{(1-cost_sentinel/cost_approve_all)*100:.1f}%':>10s}")
print(f"  {'Oracle (perfect info)':<30s} {'₹'+f'{cost_oracle:,.0f}':>12s} {f'{(1-cost_oracle/cost_approve_all)*100:.1f}%':>10s}")
print(f"")
print(f"  Achievable savings (approve-all → oracle): ₹{achievable:,.0f}")
print(f"  Sentinel captures: ₹{captured:,.0f} = {pct_captured:.1f}% of achievable")
print(f"  Remaining gap to oracle: ₹{cost_sentinel - cost_oracle:,.0f}")

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
strategies = ["Approve All", f"Fixed t={best_t:.2f}", "Sentinel", "Oracle"]
costs_list = [cost_approve_all, cost_fixed, cost_sentinel, cost_oracle]
colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]

bars = ax.bar(strategies, costs_list, color=colors, width=0.6)
ax.set_ylabel("Total ₹ Cost", fontsize=12)
ax.set_title("Baseline Comparison: How Much of Achievable Savings Does Sentinel Capture?", fontsize=13)

for bar, cost in zip(bars, costs_list):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
            f"₹{cost:,.0f}", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(PLOTS / "oracle_comparison.png", dpi=150)
plt.close()
print("\n✅ Oracle comparison chart saved")