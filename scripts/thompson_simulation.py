"""Thompson Sampling simulation — system learns cost parameters from outcomes."""

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
from src.sentinel.cost import load_costs

PLOTS = Path("reports/plots")
true_costs = load_costs()

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


# ============================================================
# THOMPSON SAMPLING SIMULATION
# ============================================================
print("\n" + "="*60)
print("  THOMPSON SAMPLING SIMULATION")
print("="*60)

# Parameters we can learn (observable outcomes)
# - chargeback_fee: observed when chargeback arrives
# - challenge_success_rate: observed from OTP completions
# - fraudster_3ds_dropout: observed from challenge outcomes

# Parameters we CANNOT learn (no observable feedback)
# - churn_probability: takes months to observe
# - customer_ltv: takes months/years
# - friction_cost: customer frustration isn't measurable

# --- Prior distributions (start with wide uncertainty) ---
# Using Normal distributions with wide std for learnable params
class ThompsonParameter:
    """A cost parameter with a learnable distribution."""
    def __init__(self, name, prior_mean, prior_std, true_value, learnable=True):
        self.name = name
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.true_value = true_value
        self.learnable = learnable
        
        # Running estimates (online Bayesian update)
        self.mean = prior_mean
        self.std = prior_std
        self.observations = []
        self.mean_history = [prior_mean]
        self.std_history = [prior_std]
    
    def sample(self):
        """Sample from current posterior."""
        sampled = np.random.normal(self.mean, self.std)
        # Clamp to reasonable range
        if "rate" in self.name or "dropout" in self.name or "success" in self.name:
            return max(0.01, min(0.99, sampled))
        else:
            return max(1.0, sampled)
    
    def update(self, observed_value):
        """Bayesian update with new observation."""
        if not self.learnable:
            return
        self.observations.append(observed_value)
        n = len(self.observations)
        # Online mean + shrinking variance
        self.mean = (self.prior_mean / self.prior_std**2 + sum(self.observations) / (self.prior_std**2 / n)) / \
                    (1 / self.prior_std**2 + n / (self.prior_std**2 / n))
        # Simplified: running average with decaying std
        self.mean = (self.prior_mean + sum(self.observations)) / (1 + n)
        self.std = self.prior_std / np.sqrt(1 + n)
        self.mean_history.append(self.mean)
        self.std_history.append(self.std)


# Initialize parameters with WRONG priors (to show learning)
params = {
    "chargeback_fee_inr": ThompsonParameter(
        "chargeback_fee_inr",
        prior_mean=2000,       # start with WRONG guess (true=1500)
        prior_std=500,
        true_value=1500,
        learnable=True
    ),
    "challenge_success_rate": ThompsonParameter(
        "challenge_success_rate",
        prior_mean=0.70,       # start with WRONG guess (true=0.85)
        prior_std=0.15,
        true_value=0.85,
        learnable=True
    ),
    "fraudster_3ds_dropout": ThompsonParameter(
        "fraudster_3ds_dropout",
        prior_mean=0.80,       # start with WRONG guess (true=0.95)
        prior_std=0.10,
        true_value=0.95,
        learnable=True
    ),
}

# Fixed parameters (not learnable — acknowledged)
fixed_params = {
    "goods_recovery_rate": true_costs["goods_recovery_rate"],
    "gross_margin": true_costs["gross_margin"],
    "friction_cost_inr": true_costs["friction_cost_inr"],
    "churn_probability": true_costs["churn_probability"],
    "customer_ltv_inr": true_costs["customer_ltv_inr"],
    "review_cost_inr": true_costs["review_cost_inr"],
    "analyst_catch_rate": true_costs["analyst_catch_rate"],
    "review_delay_churn_inr": true_costs["review_delay_churn_inr"],
    "challenge_friction_inr": true_costs["challenge_friction_inr"],
    "retry_recovery_rate": true_costs["retry_recovery_rate"],
}


def make_thompson_decision(p_fraud, amount, sampled_costs):
    """Same as make_decision but with sampled parameters."""
    p_fraud = max(0.001, min(p_fraud, 0.999))
    
    rec = sampled_costs["goods_recovery_rate"]
    cb = sampled_costs["chargeback_fee_inr"]
    margin = sampled_costs["gross_margin"]
    friction = sampled_costs["friction_cost_inr"]
    churn = sampled_costs["churn_probability"]
    ltv = sampled_costs["customer_ltv_inr"]
    rho = sampled_costs.get("retry_recovery_rate", 0.0)
    
    cost_allow = p_fraud * (amount * (1 - rec) + cb)
    cost_block = (1 - p_fraud) * ((1 - rho) * margin * amount + friction + churn * ltv)
    
    review_delay = sampled_costs.get("review_delay_churn_inr", 30)
    cost_review = (sampled_costs["review_cost_inr"] + review_delay +
                   p_fraud * (1 - sampled_costs["analyst_catch_rate"]) * (amount * (1 - rec) + cb))
    
    ch_friction = sampled_costs.get("challenge_friction_inr", 15)
    ch_success = sampled_costs.get("challenge_success_rate", 0.85)
    ch_dropout = sampled_costs.get("fraudster_3ds_dropout", 0.95)
    
    cost_challenge = (ch_friction +
                      (1 - ch_success) * (1 - p_fraud) * ((1 - rho) * margin * amount + friction + churn * ltv) +
                      p_fraud * (1 - ch_dropout) * (amount * (1 - rec) + cb))
    
    options = {"ALLOW": cost_allow, "CHALLENGE": cost_challenge, "REVIEW": cost_review, "BLOCK": cost_block}
    return min(options, key=options.get), options


# --- Run simulation ---
print("\nRunning Thompson Sampling over test transactions...")
np.random.seed(42)

total_cost_thompson = 0.0
total_cost_fixed = 0.0
thompson_costs_over_time = []
fixed_costs_over_time = []
exploration_count = 0

# Use first 50,000 transactions for simulation (faster)
N_SIM = min(50000, len(y_test))

for i in range(N_SIM):
    p = float(y_pred[i])
    amt = float(amounts[i])
    true_label = y_test[i]
    
    # --- Thompson decision (sampled parameters) ---
    sampled_costs = fixed_params.copy()
    for key, param in params.items():
        sampled_costs[key] = param.sample()
    
    thompson_decision, _ = make_thompson_decision(p, amt, sampled_costs)
    
    # --- Fixed decision (true parameters) ---
    fixed_costs_full = {**fixed_params, **{k: v.true_value for k, v in params.items()}}
    fixed_decision, _ = make_thompson_decision(p, amt, fixed_costs_full)
    
    if thompson_decision != fixed_decision:
        exploration_count += 1
    
    # --- Compute actual costs based on TRUE outcomes ---
    def actual_cost(decision, true_label, amt):
        if decision == "ALLOW" and true_label == 1:
            return amt + 1500  # true chargeback
        elif decision == "BLOCK" and true_label == 0:
            return (1 - 0.5) * 0.18 * amt + 250 + 0.04 * 6000
        elif decision == "CHALLENGE":
            if true_label == 0:
                return 15  # friction
            else:
                # True dropout rate is 0.95
                if np.random.random() > 0.95:
                    return amt + 1500
                return 0
        elif decision == "REVIEW":
            cost = 45 + 30
            if true_label == 1 and np.random.random() > 0.92:
                cost += amt + 1500
            return cost
        return 0
    
    t_cost = actual_cost(thompson_decision, true_label, amt)
    f_cost = actual_cost(fixed_decision, true_label, amt)
    
    total_cost_thompson += t_cost
    total_cost_fixed += f_cost
    
    # --- Observe outcomes and update parameters ---
    # Chargeback fee: if fraud was allowed, observe the actual fee
    if (thompson_decision == "ALLOW" and true_label == 1):
        params["chargeback_fee_inr"].update(1500)  # true fee
    
    # Challenge success: if challenged a legit customer, observe completion
    if thompson_decision == "CHALLENGE" and true_label == 0:
        completed = np.random.random() < 0.85  # true success rate
        params["challenge_success_rate"].update(1.0 if completed else 0.0)
    
    # Fraudster dropout: if challenged a fraudster, observe dropout
    if thompson_decision == "CHALLENGE" and true_label == 1:
        dropped = np.random.random() < 0.95  # true dropout rate
        params["fraudster_3ds_dropout"].update(1.0 if dropped else 0.0)
    
    # Track costs over time
    if (i + 1) % 1000 == 0:
        thompson_costs_over_time.append(total_cost_thompson)
        fixed_costs_over_time.append(total_cost_fixed)
    
    if (i + 1) % 10000 == 0:
        print(f"  ...{i+1:,}/{N_SIM:,} | Thompson: ₹{total_cost_thompson:,.0f} | Fixed: ₹{total_cost_fixed:,.0f} | Explored: {exploration_count}")

# --- Results ---
print(f"\n{'='*60}")
print(f"  THOMPSON SAMPLING RESULTS ({N_SIM:,} transactions)")
print(f"{'='*60}")
print(f"  Fixed policy cost:    ₹{total_cost_fixed:>12,.0f}")
print(f"  Thompson policy cost: ₹{total_cost_thompson:>12,.0f}")
print(f"  Exploration decisions: {exploration_count:,} ({exploration_count/N_SIM*100:.1f}%)")
print(f"")
print(f"  Parameter convergence:")
for key, param in params.items():
    print(f"    {key:<30s}: prior={param.prior_mean:.4f} → learned={param.mean:.4f} (true={param.true_value:.4f})")

# --- Plot 1: Parameter convergence ---
print("\nPlotting convergence charts...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (key, param) in zip(axes, params.items()):
    steps = range(len(param.mean_history))
    means = param.mean_history
    stds = param.std_history
    
    ax.plot(steps, means, color="#3498db", linewidth=2, label="Learned estimate")
    ax.fill_between(steps,
                     [m - 2*s for m, s in zip(means, stds)],
                     [m + 2*s for m, s in zip(means, stds)],
                     alpha=0.2, color="#3498db", label="95% CI")
    ax.axhline(y=param.true_value, color="#2ecc71", linewidth=2, linestyle="--", label=f"True value ({param.true_value})")
    ax.axhline(y=param.prior_mean, color="#e74c3c", linewidth=1, linestyle=":", label=f"Prior ({param.prior_mean})")
    
    ax.set_xlabel("Observations", fontsize=11)
    ax.set_ylabel("Parameter value", fontsize=11)
    ax.set_title(key.replace("_", " ").title(), fontsize=12)
    ax.legend(fontsize=9)

plt.suptitle("Thompson Sampling: Parameters Converge to True Values\nStarting from wrong priors, learning from observed outcomes",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS / "thompson_convergence.png", dpi=150)
plt.close()
print("✅ Convergence chart saved")

# --- Plot 2: Cumulative cost comparison ---
fig, ax = plt.subplots(figsize=(10, 6))
x = range(1000, N_SIM + 1, 1000)
ax.plot(x, thompson_costs_over_time, color="#3498db", linewidth=2, label="Thompson Sampling (learns)")
ax.plot(x, fixed_costs_over_time, color="#e74c3c", linewidth=2, label="Fixed parameters (optimal)")
ax.set_xlabel("Transactions processed", fontsize=12)
ax.set_ylabel("Cumulative ₹ cost", fontsize=12)
ax.set_title("Thompson Sampling: Exploration Cost Converges to Optimal\nEarly: slightly worse (exploring) → Later: matches optimal", fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(PLOTS / "thompson_cumulative_cost.png", dpi=150)
plt.close()
print("✅ Cumulative cost chart saved")

# --- Identifiability caveat ---
print(f"\n{'='*60}")
print(f"  IDENTIFIABILITY CAVEAT (say this in the video)")
print(f"{'='*60}")
print(f"  LEARNABLE (observable feedback):")
print(f"    chargeback_fee_inr      → arrives on bank statement")
print(f"    challenge_success_rate  → OTP completion observed")
print(f"    fraudster_3ds_dropout   → challenge outcome observed")
print(f"")
print(f"  NOT LEARNABLE (no passive feedback):")
print(f"    churn_probability       → takes months to observe")
print(f"    customer_ltv_inr        → takes months/years")
print(f"    friction_cost_inr       → customer frustration unmeasurable")
print(f"")
print(f"  This distinction is the signal. Saying it out loud is")
print(f"  stronger than implementing the bandit.")