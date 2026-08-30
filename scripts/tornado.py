"""Sensitivity analysis — tornado chart showing which cost parameters matter most."""

import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path
from src.sentinel.features import build_features, features_to_array, FEATURE_COLUMNS
from src.sentinel.model_wrapper import LGBMWrapper
from src.sentinel.cost import load_costs, make_decision
import time

PLOTS = Path("reports/plots")
costs_base = load_costs()

# --- Load Sparkov test predictions ---
print("Loading Sparkov test data and building predictions...")
booster = lgb.Booster(model_file="artifacts/sparkov/model.lgb")
calibrator = joblib.load("artifacts/sparkov/calibrator.joblib")

test_df = pd.read_parquet("data/processed/test.parquet")
test_df["trans_date_trans_time"] = pd.to_datetime(test_df["trans_date_trans_time"])
test_df["hour"] = test_df["trans_date_trans_time"].dt.hour
test_df["day_of_week"] = test_df["trans_date_trans_time"].dt.dayofweek
test_df = test_df.sort_values("trans_date_trans_time").reset_index(drop=True)

# Build features + predictions
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
print(f"Predictions ready in {time.time() - start:.0f}s")

# --- Function to compute total cost with given parameters ---
def compute_total_cost(costs, y_true, y_scores, amts):
    total = 0.0
    decisions = {"ALLOW": 0, "CHALLENGE": 0, "REVIEW": 0, "BLOCK": 0}
    for i in range(len(y_true)):
        result = make_decision(p_fraud=float(y_scores[i]), amount=float(amts[i]), costs=costs)
        decisions[result["decision"]] += 1
        if result["decision"] == "ALLOW" and y_true[i] == 1:
            total += float(amts[i]) + costs["chargeback_fee_inr"]
        elif result["decision"] == "BLOCK" and y_true[i] == 0:
            total += costs["gross_margin"] * float(amts[i]) + costs["friction_cost_inr"] + costs["churn_probability"] * costs["customer_ltv_inr"]
        elif result["decision"] == "CHALLENGE":
            if y_true[i] == 0:
                total += costs.get("challenge_friction_inr", 15)
            else:
                if np.random.random() > costs.get("fraudster_3ds_dropout", 0.95):
                    total += float(amts[i]) + costs["chargeback_fee_inr"]
        elif result["decision"] == "REVIEW":
            total += costs["review_cost_inr"] + costs.get("review_delay_churn_inr", 30)
            if y_true[i] == 1 and np.random.random() > costs["analyst_catch_rate"]:
                total += float(amts[i]) + costs["chargeback_fee_inr"]
    return total, decisions

# --- Baseline cost ---
np.random.seed(42)
base_cost, base_decisions = compute_total_cost(costs_base, y_test, y_pred, amounts)
print(f"\nBaseline total cost: ₹{base_cost:,.0f}")
print(f"Baseline decisions: {base_decisions}")

# --- Vary each parameter ±50% ---
params_to_vary = [
    ("chargeback_fee_inr", "Chargeback Fee"),
    ("gross_margin", "Gross Margin"),
    ("friction_cost_inr", "Friction Cost"),
    ("churn_probability", "Churn Probability"),
    ("customer_ltv_inr", "Customer LTV"),
    ("review_cost_inr", "Review Cost"),
    ("analyst_catch_rate", "Analyst Catch Rate"),
    ("review_delay_churn_inr", "Review Delay Cost"),
    ("challenge_friction_inr", "Challenge Friction"),
    ("challenge_success_rate", "Challenge Success"),
    ("fraudster_3ds_dropout", "Fraudster 3DS Dropout"),
]

print("\nRunning sensitivity analysis (this takes a few minutes)...")
results = []

for param_key, param_label in params_to_vary:
    base_val = costs_base[param_key]
    
    # -50%
    costs_low = costs_base.copy()
    costs_low[param_key] = base_val * 0.5
    # Clamp rates to [0, 1]
    if param_key in ("gross_margin", "churn_probability", "analyst_catch_rate",
                     "challenge_success_rate", "fraudster_3ds_dropout", "goods_recovery_rate"):
        costs_low[param_key] = max(0.0, min(1.0, costs_low[param_key]))
    
    np.random.seed(42)
    cost_low, _ = compute_total_cost(costs_low, y_test, y_pred, amounts)
    
    # +50%
    costs_high = costs_base.copy()
    costs_high[param_key] = base_val * 1.5
    if param_key in ("gross_margin", "churn_probability", "analyst_catch_rate",
                     "challenge_success_rate", "fraudster_3ds_dropout", "goods_recovery_rate"):
        costs_high[param_key] = max(0.0, min(1.0, costs_high[param_key]))
    
    np.random.seed(42)
    cost_high, _ = compute_total_cost(costs_high, y_test, y_pred, amounts)
    
    delta_low = cost_low - base_cost
    delta_high = cost_high - base_cost
    swing = abs(cost_high - cost_low)
    
    results.append({
        "param": param_label,
        "base": base_val,
        "cost_low": cost_low,
        "cost_high": cost_high,
        "delta_low": delta_low,
        "delta_high": delta_high,
        "swing": swing,
    })
    
    print(f"  {param_label:<25s}: -50% → ₹{cost_low:>10,.0f} ({delta_low:>+10,.0f}) | +50% → ₹{cost_high:>10,.0f} ({delta_high:>+10,.0f}) | swing: ₹{swing:>10,.0f}")

# Sort by swing
results.sort(key=lambda x: x["swing"], reverse=True)

# --- Plot tornado chart ---
print("\nPlotting tornado chart...")

fig, ax = plt.subplots(figsize=(12, 8))
y_positions = range(len(results))
labels = [r["param"] for r in results]

# Plot bars
for i, r in enumerate(results):
    # Low bar (from baseline)
    color_low = "#2ecc71" if r["delta_low"] < 0 else "#e74c3c"
    ax.barh(i, r["delta_low"], left=0, height=0.6, color=color_low, alpha=0.8)
    
    # High bar (from baseline)
    color_high = "#e74c3c" if r["delta_high"] > 0 else "#2ecc71"
    ax.barh(i, r["delta_high"], left=0, height=0.6, color=color_high, alpha=0.8)

ax.set_yticks(y_positions)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("Change in Total ₹ Cost (vs baseline)", fontsize=12)
ax.set_title(f"Sensitivity Analysis: Which Cost Parameters Matter Most?\nBaseline: ₹{base_cost:,.0f} | Each parameter varied ±50%", fontsize=13)
ax.axvline(x=0, color="black", linewidth=1)
ax.invert_yaxis()

# Add swing values on the right
for i, r in enumerate(results):
    ax.text(max(abs(r["delta_low"]), abs(r["delta_high"])) + 1000, i,
            f"±₹{r['swing']:,.0f}", va="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(PLOTS / "tornado_sensitivity.png", dpi=150)
plt.close()
print("✅ Tornado chart saved to reports/plots/tornado_sensitivity.png")

# --- Summary table ---
print(f"\n{'='*70}")
print(f"  SENSITIVITY RANKING (sorted by ₹ swing)")
print(f"{'='*70}")
print(f"  {'Parameter':<25s} {'Base Value':>12s} {'₹ Swing':>12s} {'Rank':>6s}")
for i, r in enumerate(results):
    swing_str = f"₹{r['swing']:,.0f}"
    rank_str = f"#{i+1}"
    print(f"  {r['param']:<25s} {str(r['base']):>12s} {swing_str:>12s} {rank_str:>6s}")