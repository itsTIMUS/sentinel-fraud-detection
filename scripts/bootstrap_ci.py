"""Bootstrap confidence intervals on total ₹ cost."""

import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import time
from src.sentinel.features import build_features, features_to_array, FEATURE_COLUMNS
from src.sentinel.model_wrapper import LGBMWrapper
from src.sentinel.cost import load_costs, make_decision

costs = load_costs()

# --- Load test predictions ---
print("Loading Sparkov test data...")
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

# --- Precompute per-transaction costs ---
print("\nPrecomputing per-transaction costs...")
per_txn_costs = np.zeros(len(y_test))

for i in range(len(y_test)):
    amt = float(amounts[i])
    true_label = y_test[i]
    p = float(y_pred[i])
    result = make_decision(p_fraud=p, amount=amt, costs=costs)
    d = result["decision"]

    if d == "ALLOW" and true_label == 1:
        per_txn_costs[i] = amt + costs["chargeback_fee_inr"]
    elif d == "BLOCK" and true_label == 0:
        per_txn_costs[i] = costs["gross_margin"] * amt + costs["friction_cost_inr"] + costs["churn_probability"] * costs["customer_ltv_inr"]
    elif d == "CHALLENGE":
        if true_label == 0:
            per_txn_costs[i] = costs.get("challenge_friction_inr", 15)
        else:
            per_txn_costs[i] = (1 - costs.get("fraudster_3ds_dropout", 0.95)) * (amt + costs["chargeback_fee_inr"])
    elif d == "REVIEW":
        per_txn_costs[i] = costs["review_cost_inr"] + costs.get("review_delay_churn_inr", 30)
        if true_label == 1:
            per_txn_costs[i] += (1 - costs["analyst_catch_rate"]) * (amt + costs["chargeback_fee_inr"])

base_cost = per_txn_costs.sum()
print(f"Base total cost: ₹{base_cost:,.0f}")

# --- Bootstrap ---
N_BOOTSTRAP = 1000
print(f"\nRunning {N_BOOTSTRAP} bootstrap resamples...")

bootstrap_costs = []
n = len(per_txn_costs)

for b in range(N_BOOTSTRAP):
    indices = np.random.randint(0, n, size=n)
    sample_cost = per_txn_costs[indices].sum()
    bootstrap_costs.append(sample_cost)
    if (b + 1) % 200 == 0:
        print(f"  ...{b+1}/{N_BOOTSTRAP}")

bootstrap_costs = np.array(bootstrap_costs)

# --- Results ---
ci_lower = np.percentile(bootstrap_costs, 2.5)
ci_upper = np.percentile(bootstrap_costs, 97.5)
mean_cost = bootstrap_costs.mean()
std_cost = bootstrap_costs.std()

print(f"\n{'='*60}")
print(f"  BOOTSTRAP CONFIDENCE INTERVALS ({N_BOOTSTRAP} resamples)")
print(f"{'='*60}")
print(f"  Point estimate:  ₹{base_cost:>12,.0f}")
print(f"  Bootstrap mean:  ₹{mean_cost:>12,.0f}")
print(f"  Bootstrap std:   ₹{std_cost:>12,.0f}")
print(f"  95% CI:          ₹{ci_lower:>12,.0f}  to  ₹{ci_upper:>12,.0f}")
print(f"  Width:           ₹{ci_upper - ci_lower:>12,.0f}")
print(f"")

# Savings CI
approve_all = sum(amounts[y_test == 1]) + y_test.sum() * costs["chargeback_fee_inr"]
savings_lower = (1 - ci_upper / approve_all) * 100
savings_upper = (1 - ci_lower / approve_all) * 100

print(f"  Approve-all cost: ₹{approve_all:>12,.0f}")
print(f"  Savings range:   {savings_lower:.1f}%  to  {savings_upper:.1f}%")
print(f"")
print(f"  ► Sentinel saves ₹{base_cost:,.0f} ± ₹{std_cost:,.0f} (95% CI: ₹{ci_lower:,.0f} – ₹{ci_upper:,.0f})")
print(f"  ► Savings: {(1-base_cost/approve_all)*100:.1f}% (95% CI: {savings_lower:.1f}% – {savings_upper:.1f}%)")