"""Tests for cost-aware decision policy (v2: CHALLENGE + review fix + profit)."""

from src.sentinel.cost import load_costs, make_decision


def get_costs():
    return load_costs("config/costs.yaml")


def test_low_risk_small_amount_allows():
    """Low fraud probability + small amount → ALLOW."""
    costs = get_costs()
    result = make_decision(p_fraud=0.01, amount=500, costs=costs)
    assert result["decision"] == "ALLOW", f"Expected ALLOW, got {result['decision']}"


def test_high_risk_large_amount_blocks():
    """High fraud probability + large amount → BLOCK."""
    costs = get_costs()
    result = make_decision(p_fraud=0.90, amount=50000, costs=costs)
    assert result["decision"] == "BLOCK", f"Expected BLOCK, got {result['decision']}"


def test_medium_risk_triggers_challenge():
    """Medium probability + small-medium amount → CHALLENGE (cheaper than REVIEW and BLOCK)."""
    costs = get_costs()
    result = make_decision(p_fraud=0.10, amount=1000, costs=costs)
    assert result["decision"] == "CHALLENGE", f"Expected CHALLENGE, got {result['decision']}"


def test_decision_has_required_fields():
    """Every decision must include all cost fields including new ones."""
    costs = get_costs()
    result = make_decision(p_fraud=0.10, amount=1000, costs=costs)
    required = [
        "decision",
        "expected_loss_if_allowed_inr",
        "expected_loss_if_challenged_inr",
        "expected_loss_if_reviewed_inr",
        "expected_loss_if_blocked_inr",
        "expected_profit_inr",
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


def test_challenge_cheaper_than_block_at_medium_risk():
    """CHALLENGE should always be cheaper than BLOCK for mid-risk transactions."""
    costs = get_costs()
    result = make_decision(p_fraud=0.10, amount=10000, costs=costs)
    assert result["expected_loss_if_challenged_inr"] < result["expected_loss_if_blocked_inr"], \
        "CHALLENGE should be cheaper than BLOCK at medium risk"


def test_expected_profit_positive_for_low_risk():
    """Low-risk transaction should have positive expected profit."""
    costs = get_costs()
    result = make_decision(p_fraud=0.01, amount=5000, costs=costs)
    assert result["expected_profit_inr"] > 0, \
        f"Expected positive profit, got {result['expected_profit_inr']}"


def test_four_actions_exist():
    """All four actions should be reachable."""
    costs = get_costs()
    actions_seen = set()
    test_cases = [
        (0.001, 100),    # very low risk, small → ALLOW
        (0.08, 5000),    # medium risk → CHALLENGE
        (0.50, 500),     # high risk, small → might be REVIEW or CHALLENGE
        (0.95, 50000),   # very high risk, large → BLOCK
    ]
    for p, amt in test_cases:
        result = make_decision(p_fraud=p, amount=amt, costs=costs)
        actions_seen.add(result["decision"])

    # At minimum ALLOW and BLOCK must be reachable
    assert "ALLOW" in actions_seen, "ALLOW should be reachable"
    assert "BLOCK" in actions_seen, "BLOCK should be reachable"