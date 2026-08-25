"""
Cost-aware decision policy for fraud detection.
Computes expected ₹ loss for ALLOW / REVIEW / BLOCK and picks the cheapest.
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
    """Expected ₹ loss if we send to REVIEW."""
    return costs["review_cost_inr"] + p_fraud * (
        (1 - costs["analyst_catch_rate"])
        * (amount * (1 - costs["goods_recovery_rate"]) + costs["chargeback_fee_inr"])
    )


def make_decision(p_fraud: float, amount: float, costs: dict) -> dict:
    """
    Given a calibrated fraud probability and transaction amount,
    compute the cheapest action in ₹.

    Returns:
        dict with decision, costs breakdown, and thresholds used.
    """
    p_fraud = max(0.001, min(p_fraud, 0.999))

    cost_allow = expected_cost_allow(p_fraud, amount, costs)
    cost_block = expected_cost_block(p_fraud, amount, costs)
    cost_review = expected_cost_review(p_fraud, amount, costs)

    options = {
        "ALLOW": cost_allow,
        "REVIEW": cost_review,
        "BLOCK": cost_block,
    }
    decision = min(options, key=options.get)

    return {
        "decision": decision,
        "expected_loss_if_allowed_inr": round(cost_allow, 2),
        "expected_loss_if_blocked_inr": round(cost_block, 2),
        "expected_loss_if_reviewed_inr": round(cost_review, 2),
        "amount_inr": round(amount, 2),
        "risk_probability": round(p_fraud, 4),
    }