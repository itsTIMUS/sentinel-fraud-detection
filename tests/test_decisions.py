"""Tests for cost-aware decision policy."""

from src.sentinel.cost import load_costs, make_decision


def get_costs():
    return load_costs("config/costs.yaml")


def test_low_risk_small_amount_allows():
    """Low fraud probability + small amount → ALLOW (blocking costs more than fraud)."""
    costs = get_costs()
    result = make_decision(p_fraud=0.01, amount=500, costs=costs)
    assert result["decision"] == "ALLOW", f"Expected ALLOW, got {result['decision']}"


def test_high_risk_large_amount_blocks():
    """High fraud probability + large amount → BLOCK (fraud costs more than blocking)."""
    costs = get_costs()
    result = make_decision(p_fraud=0.90, amount=50000, costs=costs)
    assert result["decision"] == "BLOCK", f"Expected BLOCK, got {result['decision']}"


def test_medium_risk_reviews():
    """Medium probability → REVIEW (cheaper than both ALLOW and BLOCK)."""
    costs = get_costs()
    result = make_decision(p_fraud=0.05, amount=5000, costs=costs)
    assert result["decision"] == "REVIEW", f"Expected REVIEW, got {result['decision']}"


def test_decision_has_required_fields():
    """Every decision must include all cost fields."""
    costs = get_costs()
    result = make_decision(p_fraud=0.10, amount=1000, costs=costs)
    required = [
        "decision",
        "expected_loss_if_allowed_inr",
        "expected_loss_if_blocked_inr",
        "expected_loss_if_reviewed_inr",
        "amount_inr",
        "risk_probability",
    ]
    for field in required:
        assert field in result, f"Missing field: {field}"


def test_probability_is_clipped():
    """Extreme probabilities should be clipped to [0.001, 0.999]."""
    costs = get_costs()
    result = make_decision(p_fraud=0.0, amount=1000, costs=costs)
    assert result["risk_probability"] == 0.001

    result = make_decision(p_fraud=1.0, amount=1000, costs=costs)
    assert result["risk_probability"] == 0.999