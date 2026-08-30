"""₹ decomposition — where does the remaining cost actually live?"""

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

# --- Load test predictions ---
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

# --- Decompose costs ---
print("\nDecomposing costs...")
np.random.seed(42)

cost_fn_fraud = 0.0       # Allowed fraud: lost goods + chargeback
cost_fn_chargeback = 0.0  # Chargeback portion of FN
cost_fp_margin = 0.0      # Blocked legit: lost margin
cost_fp_friction = 0.0    # Blocked legit: friction
cost_fp_churn = 0.0       # Blocked legit: churn × LTV
cost_challenge_friction = 0.0  # Challenge friction on legit
cost_challenge_fraud = 0.0     # Fraud that got through challenge
cost_review_analyst = 0.0      # Review analyst time + delay
cost_review_missed = 0.0       # Fraud missed by analyst

counts = {"ALLOW_legit": 0, "ALLOW_fraud": 0, "CHALLENGE_legit": 0, "CHALLENGE_fraud": 0,
          "REVIEW_legit": 0, "REVIEW_fraud": 0, "BLOCK_legit": 0, "BLOCK_fraud": 0}

for i in range(len(y_test)):
    amt = float(amounts[i])
    true_label = y_test[i]
    p = float(y_pred[i])
    result = make_decision(p_fraud=p, amount=amt, costs=costs)
    decision = result["decision"]

    if decision == "ALLOW":
        if true_label == 1:
            cost_fn_fraud += amt
            cost_fn_chargeback += costs["chargeback_fee_inr"]
            counts["ALLOW_fraud"] += 1
        else:
            counts["ALLOW_legit"] += 1

    elif decision == "BLOCK":
        if true_label == 0:
            cost_fp_margin += costs["gross_margin"] * amt
            cost_fp_friction += costs["friction_cost_inr"]
            cost_fp_churn += costs["churn_probability"] * costs["customer_ltv_inr"]
            counts["BLOCK_legit"] += 1
        else:
            counts["BLOCK_fraud"] += 1

    elif decision == "CHALLENGE":
        if true_label == 0:
            cost_challenge_friction += costs.get("challenge_friction_inr", 15)
            counts["CHALLENGE_legit"] += 1
        else:
            if np.random.random() > costs.get("fraudster_3ds_dropout", 0.95):
                cost_challenge_fraud += amt + costs["chargeback_fee_inr"]
            counts["CHALLENGE_fraud"] += 1

    elif decision == "REVIEW":
        cost_review_analyst += costs["review_cost_inr"] + costs.get("review_delay_churn_inr", 30)
        if true_label == 1:
            if np.random.random() > costs["analyst_catch_rate"]:
                cost_review_missed += amt + costs["chargeback_fee_inr"]
            counts["REVIEW_fraud"] += 1
        else:
            counts["REVIEW_legit"] += 1

total = (cost_fn_fraud + cost_fn_chargeback + cost_fp_margin + cost_fp_friction +
         cost_fp_churn + cost_challenge_friction + cost_challenge_fraud +
         cost_review_analyst + cost_review_missed)

# --- Print breakdown ---
print(f"\n{'='*60}")
print(f"  ₹ COST DECOMPOSITION (Total: ₹{total:,.0f})")
print(f"{'='*60}")
print(f"\n  MISSED FRAUD (allowed fraud that cost us):")
print(f"    Fraud goods lost:       ₹{cost_fn_fraud:>10,.0f}  ({cost_fn_fraud/total*100:>5.1f}%)")
print(f"    Chargeback fees:        ₹{cost_fn_chargeback:>10,.0f}  ({cost_fn_chargeback/total*100:>5.1f}%)")
print(f"    Subtotal:               ₹{cost_fn_fraud+cost_fn_chargeback:>10,.0f}  ({(cost_fn_fraud+cost_fn_chargeback)/total*100:>5.1f}%)")

print(f"\n  FALSE BLOCKS (blocked legit customers):")
print(f"    Lost margin:            ₹{cost_fp_margin:>10,.0f}  ({cost_fp_margin/total*100:>5.1f}%)")
print(f"    Friction cost:          ₹{cost_fp_friction:>10,.0f}  ({cost_fp_friction/total*100:>5.1f}%)")
print(f"    Churn × LTV:            ₹{cost_fp_churn:>10,.0f}  ({cost_fp_churn/total*100:>5.1f}%)")
print(f"    Subtotal:               ₹{cost_fp_margin+cost_fp_friction+cost_fp_churn:>10,.0f}  ({(cost_fp_margin+cost_fp_friction+cost_fp_churn)/total*100:>5.1f}%)")

print(f"\n  CHALLENGE COSTS:")
print(f"    Friction on legit:      ₹{cost_challenge_friction:>10,.0f}  ({cost_challenge_friction/total*100:>5.1f}%)")
print(f"    Fraud through 3DS:      ₹{cost_challenge_fraud:>10,.0f}  ({cost_challenge_fraud/total*100:>5.1f}%)")
print(f"    Subtotal:               ₹{cost_challenge_friction+cost_challenge_fraud:>10,.0f}  ({(cost_challenge_friction+cost_challenge_fraud)/total*100:>5.1f}%)")

print(f"\n  REVIEW COSTS:")
print(f"    Analyst time + delay:   ₹{cost_review_analyst:>10,.0f}  ({cost_review_analyst/total*100:>5.1f}%)")
print(f"    Missed by analyst:      ₹{cost_review_missed:>10,.0f}  ({cost_review_missed/total*100:>5.1f}%)")
print(f"    Subtotal:               ₹{cost_review_analyst+cost_review_missed:>10,.0f}  ({(cost_review_analyst+cost_review_missed)/total*100:>5.1f}%)")

print(f"\n  DECISION COUNTS:")
for k, v in sorted(counts.items()):
    print(f"    {k:<25s}: {v:>8,}")

# --- Waterfall chart ---
print("\nPlotting waterfall chart...")

categories = [
    "Missed Fraud\n(goods)",
    "Missed Fraud\n(chargeback)",
    "False Block\n(margin)",
    "False Block\n(friction)",
    "False Block\n(churn)",
    "Challenge\n(friction)",
    "Challenge\n(fraud through)",
    "Review\n(analyst)",
    "Review\n(missed)",
]
values = [
    cost_fn_fraud, cost_fn_chargeback,
    cost_fp_margin, cost_fp_friction, cost_fp_churn,
    cost_challenge_friction, cost_challenge_fraud,
    cost_review_analyst, cost_review_missed,
]

# Color by type
colors = [
    "#e74c3c", "#e74c3c",           # missed fraud
    "#f39c12", "#f39c12", "#f39c12", # false blocks
    "#3498db", "#3498db",            # challenge
    "#9b59b6", "#9b59b6",            # review
]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(categories, values, color=colors, width=0.7)

# Add value labels
for bar, val in zip(bars, values):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f"₹{val:,.0f}", ha="center", fontsize=9, fontweight="bold")

ax.set_ylabel("₹ Cost", fontsize=12)
ax.set_title(f"Where Does ₹{total:,.0f} Actually Come From?\nCost Decomposition by Source", fontsize=13)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#e74c3c", label="Missed Fraud"),
    Patch(facecolor="#f39c12", label="False Blocks"),
    Patch(facecolor="#3498db", label="Challenge Costs"),
    Patch(facecolor="#9b59b6", label="Review Costs"),
]
ax.legend(handles=legend_elements, fontsize=10)

plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(PLOTS / "cost_waterfall.png", dpi=150)
plt.close()
print("✅ Waterfall chart saved")



# ============================================================
# IEEE-CIS WATERFALL
# ============================================================
print("\n\n" + "="*60)
print("  IEEE-CIS COST DECOMPOSITION")
print("="*60)

ieee_booster = lgb.Booster(model_file="artifacts/ieee/model.lgb")
ieee_calibrator = joblib.load("artifacts/ieee/calibrator.joblib")

from src.sentinel.features.ieee_builder import build_ieee_features, ieee_features_to_array

ieee_df = pd.read_csv("data/raw/ieee/train_transaction.csv")
ieee_df = ieee_df.sort_values("TransactionDT").reset_index(drop=True)
split_idx = int(len(ieee_df) * 0.8)
ieee_val = ieee_df.iloc[split_idx:].copy()

print(f"IEEE val set: {len(ieee_val):,} rows")
print("Building features...")
ieee_features = []
for idx, row in ieee_val.iterrows():
    ieee_features.append(ieee_features_to_array(build_ieee_features(row.to_dict())))

X_ieee = np.vstack(ieee_features)
y_ieee = ieee_val["isFraud"].values
y_pred_ieee = ieee_calibrator.predict_proba(X_ieee)[:, 1]
amounts_ieee = ieee_val["TransactionAmt"].values

np.random.seed(42)
i_fn_fraud = 0.0; i_fn_cb = 0.0
i_fp_margin = 0.0; i_fp_friction = 0.0; i_fp_churn = 0.0
i_ch_friction = 0.0; i_ch_fraud = 0.0
i_rv_analyst = 0.0; i_rv_missed = 0.0
i_counts = {"ALLOW_legit": 0, "ALLOW_fraud": 0, "CHALLENGE_legit": 0, "CHALLENGE_fraud": 0,
            "REVIEW_legit": 0, "REVIEW_fraud": 0, "BLOCK_legit": 0, "BLOCK_fraud": 0}

for i in range(len(y_ieee)):
    amt = float(amounts_ieee[i])
    true_label = y_ieee[i]
    p = float(y_pred_ieee[i])
    result = make_decision(p_fraud=p, amount=amt, costs=costs)
    d = result["decision"]

    if d == "ALLOW":
        if true_label == 1:
            i_fn_fraud += amt; i_fn_cb += costs["chargeback_fee_inr"]
            i_counts["ALLOW_fraud"] += 1
        else:
            i_counts["ALLOW_legit"] += 1
    elif d == "BLOCK":
        if true_label == 0:
            i_fp_margin += costs["gross_margin"] * amt
            i_fp_friction += costs["friction_cost_inr"]
            i_fp_churn += costs["churn_probability"] * costs["customer_ltv_inr"]
            i_counts["BLOCK_legit"] += 1
        else:
            i_counts["BLOCK_fraud"] += 1
    elif d == "CHALLENGE":
        if true_label == 0:
            i_ch_friction += costs.get("challenge_friction_inr", 15)
            i_counts["CHALLENGE_legit"] += 1
        else:
            if np.random.random() > costs.get("fraudster_3ds_dropout", 0.95):
                i_ch_fraud += amt + costs["chargeback_fee_inr"]
            i_counts["CHALLENGE_fraud"] += 1
    elif d == "REVIEW":
        i_rv_analyst += costs["review_cost_inr"] + costs.get("review_delay_churn_inr", 30)
        if true_label == 1 and np.random.random() > costs["analyst_catch_rate"]:
            i_rv_missed += amt + costs["chargeback_fee_inr"]
        if true_label == 1:
            i_counts["REVIEW_fraud"] += 1
        else:
            i_counts["REVIEW_legit"] += 1

i_total = i_fn_fraud + i_fn_cb + i_fp_margin + i_fp_friction + i_fp_churn + i_ch_friction + i_ch_fraud + i_rv_analyst + i_rv_missed

print(f"\n  MISSED FRAUD:")
print(f"    Fraud goods lost:       ₹{i_fn_fraud:>10,.0f}  ({i_fn_fraud/i_total*100:>5.1f}%)")
print(f"    Chargeback fees:        ₹{i_fn_cb:>10,.0f}  ({i_fn_cb/i_total*100:>5.1f}%)")
print(f"    Subtotal:               ₹{i_fn_fraud+i_fn_cb:>10,.0f}  ({(i_fn_fraud+i_fn_cb)/i_total*100:>5.1f}%)")

print(f"\n  FALSE BLOCKS:")
print(f"    Lost margin:            ₹{i_fp_margin:>10,.0f}  ({i_fp_margin/i_total*100:>5.1f}%)")
print(f"    Friction cost:          ₹{i_fp_friction:>10,.0f}  ({i_fp_friction/i_total*100:>5.1f}%)")
print(f"    Churn × LTV:            ₹{i_fp_churn:>10,.0f}  ({i_fp_churn/i_total*100:>5.1f}%)")
print(f"    Subtotal:               ₹{i_fp_margin+i_fp_friction+i_fp_churn:>10,.0f}  ({(i_fp_margin+i_fp_friction+i_fp_churn)/i_total*100:>5.1f}%)")

print(f"\n  CHALLENGE COSTS:")
print(f"    Friction on legit:      ₹{i_ch_friction:>10,.0f}  ({i_ch_friction/i_total*100:>5.1f}%)")
print(f"    Fraud through 3DS:      ₹{i_ch_fraud:>10,.0f}  ({i_ch_fraud/i_total*100:>5.1f}%)")
print(f"    Subtotal:               ₹{i_ch_friction+i_ch_fraud:>10,.0f}  ({(i_ch_friction+i_ch_fraud)/i_total*100:>5.1f}%)")

print(f"\n  REVIEW COSTS:")
print(f"    Analyst time + delay:   ₹{i_rv_analyst:>10,.0f}  ({i_rv_analyst/i_total*100:>5.1f}%)")
print(f"    Missed by analyst:      ₹{i_rv_missed:>10,.0f}  ({i_rv_missed/i_total*100:>5.1f}%)")
print(f"    Subtotal:               ₹{i_rv_analyst+i_rv_missed:>10,.0f}  ({(i_rv_analyst+i_rv_missed)/i_total*100:>5.1f}%)")

print(f"\n  Total: ₹{i_total:,.0f}")

print(f"\n  DECISION COUNTS:")
for k, v in sorted(i_counts.items()):
    print(f"    {k:<25s}: {v:>8,}")

# --- Side-by-side comparison ---
print(f"\n{'='*60}")
print(f"  SPARKOV vs IEEE-CIS COST STRUCTURE")
print(f"{'='*60}")
print(f"  {'Source':<25s} {'Sparkov':>15s} {'IEEE-CIS':>15s}")
print(f"  {'Missed fraud':<25s} {'₹'+f'{cost_fn_fraud+cost_fn_chargeback:,.0f}':>15s} ({'₹'+f'{i_fn_fraud+i_fn_cb:,.0f}':>15s})")
print(f"  {'False blocks':<25s} {'₹'+f'{cost_fp_margin+cost_fp_friction+cost_fp_churn:,.0f}':>15s} ({'₹'+f'{i_fp_margin+i_fp_friction+i_fp_churn:,.0f}':>15s})")
print(f"  {'Challenge costs':<25s} {'₹'+f'{cost_challenge_friction+cost_challenge_fraud:,.0f}':>15s} ({'₹'+f'{i_ch_friction+i_ch_fraud:,.0f}':>15s})")
print(f"  {'Review costs':<25s} {'₹'+f'{cost_review_analyst+cost_review_missed:,.0f}':>15s} ({'₹'+f'{i_rv_analyst+i_rv_missed:,.0f}':>15s})")
print(f"  {'TOTAL':<25s} {'₹'+f'{total:,.0f}':>15s} ({'₹'+f'{i_total:,.0f}':>15s})")

# --- IEEE waterfall plot ---
i_categories = [
    "Missed Fraud\n(goods)", "Missed Fraud\n(chargeback)",
    "False Block\n(margin)", "False Block\n(friction)", "False Block\n(churn)",
    "Challenge\n(friction)", "Challenge\n(fraud through)",
    "Review\n(analyst)", "Review\n(missed)",
]
i_values = [i_fn_fraud, i_fn_cb, i_fp_margin, i_fp_friction, i_fp_churn,
            i_ch_friction, i_ch_fraud, i_rv_analyst, i_rv_missed]
i_colors = ["#e74c3c", "#e74c3c", "#f39c12", "#f39c12", "#f39c12",
            "#3498db", "#3498db", "#9b59b6", "#9b59b6"]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(i_categories, i_values, color=i_colors, width=0.7)
for bar, val in zip(bars, i_values):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
                f"₹{val:,.0f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("₹ Cost", fontsize=12)
ax.set_title(f"IEEE-CIS: Where Does ₹{i_total:,.0f} Actually Come From?", fontsize=13)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#e74c3c", label="Missed Fraud"),
    Patch(facecolor="#f39c12", label="False Blocks"),
    Patch(facecolor="#3498db", label="Challenge Costs"),
    Patch(facecolor="#9b59b6", label="Review Costs"),
]
ax.legend(handles=legend_elements, fontsize=10)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(PLOTS / "cost_waterfall_ieee.png", dpi=150)
plt.close()
print("\n✅ IEEE waterfall chart saved")