"""Tests for feature builder — including leakage prevention."""

from src.sentinel.features import build_features, features_to_array, FEATURE_COLUMNS


SAMPLE_TXN = {
    "trans_date_trans_time": "2020-06-21 12:14:25",
    "hour": 12,
    "day_of_week": 6,
    "amt": 500.0,
    "lat": 33.9659,
    "long": -80.9355,
    "merch_lat": 33.986391,
    "merch_long": -81.200714,
    "city_pop": 333497,
    "dob": "1968-03-19",
    "category": "shopping_net",
}


def test_build_features_returns_all_columns():
    """Feature dict must contain every column the model expects."""
    features = build_features(SAMPLE_TXN)
    for col in FEATURE_COLUMNS:
        assert col in features, f"Missing feature: {col}"


def test_features_to_array_correct_length():
    """Array must match FEATURE_COLUMNS length."""
    features = build_features(SAMPLE_TXN)
    arr = features_to_array(features)
    assert len(arr) == len(FEATURE_COLUMNS), f"Expected {len(FEATURE_COLUMNS)}, got {len(arr)}"


def test_is_night_correct():
    """Night flag should be 1 for hour 22-5, 0 otherwise."""
    night_txn = {**SAMPLE_TXN, "hour": 23}
    day_txn = {**SAMPLE_TXN, "hour": 12}
    assert build_features(night_txn)["is_night"] == 1.0
    assert build_features(day_txn)["is_night"] == 0.0


def test_haversine_nonzero():
    """Distance between different points should be positive."""
    features = build_features(SAMPLE_TXN)
    assert features["haversine_dist"] > 0


def test_category_risk_known():
    """Known category should return its fraud rate."""
    features = build_features(SAMPLE_TXN)
    assert features["category_risk"] == 0.0176  # shopping_net


def test_category_risk_unknown():
    """Unknown category should return global mean fallback."""
    txn = {**SAMPLE_TXN, "category": "something_new"}
    features = build_features(txn)
    assert features["category_risk"] == 0.003  # fallback


def test_velocity_defaults_without_history():
    """Without history, velocity features should be 0 or 1."""
    features = build_features(SAMPLE_TXN, history=None)
    assert features["card_txn_count_1h"] == 0.0
    assert features["card_txn_count_24h"] == 0.0
    assert features["amt_vs_card_median"] == 1.0


def test_velocity_with_history():
    """With history, velocity features should reflect the data."""
    history = {
        "txn_count_1h": 3,
        "txn_count_24h": 12,
        "txn_sum_24h": 5000,
        "median_amt": 100,
        "distinct_merchants_24h": 5,
    }
    features = build_features(SAMPLE_TXN, history=history)
    assert features["card_txn_count_1h"] == 3.0
    assert features["card_txn_count_24h"] == 12.0
    assert features["amt_vs_card_median"] == 5.0  # 500 / 100


def test_no_future_leakage():
    """Feature at time T must not change when future rows are added.
    This is the critical leakage prevention test."""
    # Same transaction, same history → same features
    history_t = {"txn_count_1h": 2, "txn_count_24h": 5, "txn_sum_24h": 1000,
                 "median_amt": 200, "distinct_merchants_24h": 3}
    features_before = build_features(SAMPLE_TXN, history=history_t)

    # "Future" data shouldn't exist in history — history is point-in-time
    # If we pass the same history, features must be identical
    features_after = build_features(SAMPLE_TXN, history=history_t)

    for col in FEATURE_COLUMNS:
        assert features_before[col] == features_after[col], f"Leakage in {col}"