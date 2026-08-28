"""Re-evaluate both datasets with the updated cost policy (CHALLENGE + review fix).
No retraining — same models, new decision logic only."""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, ".")
from src.sentinel.features import build_features, features_to_array, FEATURE_COLUMNS
from src.sentinel.features.ieee_builder import build_ieee_features, ieee_features_to_array, IEEE_FEATURE_COLUMNS
from src.sentinel.model_wrapper import LGBMWrapper
from src.sentinel.cost import load_costs, make_decision

costs = load_costs()

# ============================================================
# SPARKOV — re-evaluate held-out test with new policy
# ============================================================
print("=" * 60)
print("  SPARKOV — Re-evaluation with CHALLENGE + review fix")
print("=" * 60)

sparkov_booster = lgb.Booster(model_file="artifacts/sparkov/model.lgb")
sparkov_calibrator = joblib.load("artifacts/sparkov/calibrator.joblib")

test_df = pd.read_parquet("data/processed/test.parquet")
test_df["trans_date_trans_time"] = pd.to_datetime(test_df["trans_date_trans_time"])
test_df["hour"] = test_df["trans_date_trans_time"].dt.hour
test_df["day_of_week"] = test_df["trans_date_trans_time"].dt.dayofweek
test_df = test_df.sort_values("trans_date_trans_time").reset_index(drop=True)

print(f"Test set: {len(test_df):,} rows")
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
y_pred = sparkov_calibrator.predict_proba(X_test)[:, 1]
print(f"Features built in {time.time() - start:.0f}s")

# Evaluate with new policy
total_cost_new = 0.0
total_cost_approve_all = 0.0
decisions_new = {"ALLOW": 0, "CHALLENGE": 0, "REVIEW": 0, "BLOCK": 0}
tp, fp, fn, tn = 0, 0, 0, 0

for i in range(len(y_test)):
    amt = float(test_df.iloc[i]["amt"])
    true_label = y_test[i]
    p = float(y_pred[i])
    result = make_decision(p_fraud=p, amount=amt, costs=costs)
    decisions_new[result["decision"]] += 1

    if result["decision"] == "ALLOW" and true_label == 1:
        total_cost_new += amt + costs["chargeback_fee_inr"]
        fn += 1
    elif result["decision"] == "BLOCK" and true_label == 0:
        total_cost_new += costs["gross_margin"] * amt + costs["friction_cost_inr"] + costs["churn_probability"] * costs["customer_ltv_inr"]
        fp += 1
    elif result["decision"] == "BLOCK" and true_label == 1:
        tp += 1
    elif result["decision"] == "ALLOW" and true_label == 0:
        tn += 1
    elif result["decision"] == "CHALLENGE":
        # Legit customer: small friction
        if true_label == 0:
            total_cost_new += costs["challenge_friction_inr"]
        # Fraud: most drop off, few get through
        else:
            if np.random.random() > costs["fraudster_3ds_dropout"]:
                total_cost_new += amt + costs["chargeback_fee_inr"]
    elif result["decision"] == "REVIEW":
        total_cost_new += costs["review_cost_inr"] + costs.get("review_delay_churn_inr", 80)
        if true_label == 1 and np.random.random() > costs["analyst_catch_rate"]:
            total_cost_new += amt + costs["chargeback_fee_inr"]

    if true_label == 1:
        total_cost_approve_all += amt + costs["chargeback_fee_inr"]

sparkov_savings = (1 - total_cost_new / total_cost_approve_all) * 100

print(f"\n  Decisions: {decisions_new}")
print(f"  TP: {tp:,} | FP: {fp:,} | FN: {fn:,} | TN: {tn:,}")
print(f"  Approve all:      ₹{total_cost_approve_all:>12,.0f}")
print(f"  OLD policy:       ₹     309,487  (92.9%)")
print(f"  NEW policy:       ₹{total_cost_new:>12,.0f}  ({sparkov_savings:.1f}%)")

# ============================================================
# IEEE-CIS — re-evaluate with new policy
# ============================================================
print("\n" + "=" * 60)
print("  IEEE-CIS — Re-evaluation with CHALLENGE + review fix")
print("=" * 60)

ieee_booster = lgb.Booster(model_file="artifacts/ieee/model.lgb")
ieee_calibrator = joblib.load("artifacts/ieee/calibrator.joblib")

ieee_df = pd.read_csv("data/raw/ieee/train_transaction.csv")
ieee_df = ieee_df.sort_values("TransactionDT").reset_index(drop=True)
split_idx = int(len(ieee_df) * 0.8)
ieee_val = ieee_df.iloc[split_idx:].copy()

print(f"Val set: {len(ieee_val):,} rows")
print("Building features...")
start = time.time()

ieee_features = []
for idx, row in ieee_val.iterrows():
    f = build_ieee_features(row.to_dict())
    ieee_features.append(ieee_features_to_array(f))

X_ieee = np.vstack(ieee_features)
y_ieee = ieee_val["isFraud"].values
y_pred_ieee = ieee_calibrator.predict_proba(X_ieee)[:, 1]
print(f"Features built in {time.time() - start:.0f}s")

# Evaluate
total_cost_ieee_new = 0.0
total_cost_ieee_approve = 0.0
decisions_ieee = {"ALLOW": 0, "CHALLENGE": 0, "REVIEW": 0, "BLOCK": 0}
tp_i, fp_i, fn_i, tn_i = 0, 0, 0, 0

for i in range(len(y_ieee)):
    amt = float(ieee_val.iloc[i]["TransactionAmt"])
    true_label = y_ieee[i]
    p = float(y_pred_ieee[i])
    result = make_decision(p_fraud=p, amount=amt, costs=costs)
    decisions_ieee[result["decision"]] += 1

    if result["decision"] == "ALLOW" and true_label == 1:
        total_cost_ieee_new += amt + costs["chargeback_fee_inr"]
        fn_i += 1
    elif result["decision"] == "BLOCK" and true_label == 0:
        total_cost_ieee_new += costs["gross_margin"] * amt + costs["friction_cost_inr"] + costs["churn_probability"] * costs["customer_ltv_inr"]
        fp_i += 1
    elif result["decision"] == "BLOCK" and true_label == 1:
        tp_i += 1
    elif result["decision"] == "ALLOW" and true_label == 0:
        tn_i += 1
    elif result["decision"] == "CHALLENGE":
        if true_label == 0:
            total_cost_ieee_new += costs["challenge_friction_inr"]
        else:
            if np.random.random() > costs["fraudster_3ds_dropout"]:
                total_cost_ieee_new += amt + costs["chargeback_fee_inr"]
    elif result["decision"] == "REVIEW":
        total_cost_ieee_new += costs["review_cost_inr"] + costs.get("review_delay_churn_inr", 80)
        if true_label == 1 and np.random.random() > costs["analyst_catch_rate"]:
            total_cost_ieee_new += amt + costs["chargeback_fee_inr"]

    if true_label == 1:
        total_cost_ieee_approve += amt + costs["chargeback_fee_inr"]

ieee_savings = (1 - total_cost_ieee_new / total_cost_ieee_approve) * 100

print(f"\n  Decisions: {decisions_ieee}")
print(f"  TP: {tp_i:,} | FP: {fp_i:,} | FN: {fn_i:,} | TN: {tn_i:,}")
print(f"  Approve all:      ₹{total_cost_ieee_approve:>12,.0f}")
print(f"  OLD policy:       ₹   2,539,429  (62.1%)")
print(f"  NEW policy:       ₹{total_cost_ieee_new:>12,.0f}  ({ieee_savings:.1f}%)")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  FINAL COMPARISON: OLD vs NEW POLICY")
print("=" * 60)
print(f"  {'Dataset':<15s} {'Old ₹':>15s} {'New ₹':>15s} {'Old Savings':>12s} {'New Savings':>12s}")
print(f"  {'Sparkov':<15s} {'₹309,487':>15s} {'₹'+f'{total_cost_new:,.0f}':>15s} {'92.9%':>12s} {f'{sparkov_savings:.1f}%':>12s}")
print(f"  {'IEEE-CIS':<15s} {'₹2,539,429':>15s} {'₹'+f'{total_cost_ieee_new:,.0f}':>15s} {'62.1%':>12s} {f'{ieee_savings:.1f}%':>12s}")