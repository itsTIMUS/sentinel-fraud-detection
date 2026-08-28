"""
Cost-aware decision policy for fraud detection.
Computes expected ₹ loss for ALLOW / CHALLENGE / REVIEW / BLOCK and picks the cheapest.
"""

import yaml
from pathlib import Path


def load_costs(config_path: str = "config/costs.yaml") -> dict:
    """Load cost parameters from YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def expected_cost_allow(p_fraud: float, amount: float, costs: dict) -> float:
    """Expected ₹ loss if we ALLOW this transaction."""
    return p_fraud * (
        amount * (1 - costs["goods_recovery_rate"]) + costs["chargeback_fee_inr"]
    )


def expected_cost_block(p_fraud: float, amount: float, costs: dict) -> float:
    """Expected ₹ loss if we BLOCK this transaction."""
    return (1 - p_fraud) * (
        costs["gross_margin"] * amount
        + costs["friction_cost_inr"]
        + costs["churn_probability"] * costs["customer_ltv_inr"]
    )


def expected_cost_review(p_fraud: float, amount: float, costs: dict) -> float:
    """Expected ₹ loss if we send to REVIEW.
    
    Includes delay/friction cost for legitimate customers held in queue.
    This prevents over-routing to REVIEW on high-amount transactions.
    """
    review_delay = costs.get("review_delay_churn_inr", 80)
    return (
        costs["review_cost_inr"]
        + review_delay
        + p_fraud * (
            (1 - costs["analyst_catch_rate"])
            * (amount * (1 - costs["goods_recovery_rate"]) + costs["chargeback_fee_inr"])
        )
    )


def expected_cost_challenge(p_fraud: float, amount: float, costs: dict) -> float:
    """Expected ₹ loss if we CHALLENGE (3DS/OTP step-up authentication).
    
    Much cheaper than BLOCK for mid-risk transactions because:
    - Legitimate customers mostly complete the challenge (85% success)
    - Fraudsters mostly drop off (95% dropout)
    - Under 3DS liability shift, authenticated fraud costs the issuer, not merchant
    """
    challenge_friction = costs.get("challenge_friction_inr", 15)
    challenge_success = costs.get("challenge_success_rate", 0.85)
    fraudster_dropout = costs.get("fraudster_3ds_dropout", 0.95)

    # Cost from legitimate customers who abandon due to challenge
    cost_legit_abandon = (1 - challenge_success) * (1 - p_fraud) * (
        costs["gross_margin"] * amount
        + challenge_friction
        + costs["churn_probability"] * costs["customer_ltv_inr"]
    )

    # Cost from fraudsters who get through despite challenge
    cost_fraud_through = p_fraud * (1 - fraudster_dropout) * (
        amount * (1 - costs["goods_recovery_rate"]) + costs["chargeback_fee_inr"]
    )

    return challenge_friction + cost_legit_abandon + cost_fraud_through


def make_decision(p_fraud: float, amount: float, costs: dict) -> dict:
    """
    Given a calibrated fraud probability and transaction amount,
    compute the cheapest action in ₹.

    Four actions: ALLOW, CHALLENGE, REVIEW, BLOCK
    Picks whichever has the lowest expected cost.

    Returns:
        dict with decision, all costs, expected profit, and thresholds used.
    """
    p_fraud = max(0.001, min(p_fraud, 0.999))

    cost_allow = expected_cost_allow(p_fraud, amount, costs)
    cost_challenge = expected_cost_challenge(p_fraud, amount, costs)
    cost_review = expected_cost_review(p_fraud, amount, costs)
    cost_block = expected_cost_block(p_fraud, amount, costs)

    options = {
        "ALLOW": cost_allow,
        "CHALLENGE": cost_challenge,
        "REVIEW": cost_review,
        "BLOCK": cost_block,
    }
    decision = min(options, key=options.get)

    # Expected profit: what the merchant makes if transaction is legit, minus cost
    expected_profit = (1 - p_fraud) * costs["gross_margin"] * amount - options[decision]

    return {
        "decision": decision,
        "expected_loss_if_allowed_inr": round(cost_allow, 2),
        "expected_loss_if_challenged_inr": round(cost_challenge, 2),
        "expected_loss_if_reviewed_inr": round(cost_review, 2),
        "expected_loss_if_blocked_inr": round(cost_block, 2),
        "expected_profit_inr": round(expected_profit, 2),
        "amount_inr": round(amount, 2),
        "risk_probability": round(p_fraud, 4),
    }