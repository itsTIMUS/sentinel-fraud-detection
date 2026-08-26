"""
Feature builder for Sentinel.
SINGLE SOURCE OF TRUTH — used by both training (batch) and serving (real-time).
This guarantees zero training-serving skew.
"""

import numpy as np
from math import radians, cos, sin, asin, sqrt


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two lat/long points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


def build_features(txn: dict, history: dict | None = None) -> dict[str, float]:
    """
    Convert a raw transaction dict into model-ready features.

    Args:
        txn: raw transaction fields (from CSV row or API request)
        history: optional velocity/behavioral data for this card
                 (from velocity store in API, or precomputed in training)

    Returns:
        dict of feature_name → float, ready for model input
    """
    f = {}

    # --- Temporal ---
    hour = int(txn.get("hour", 0))
    f["hour"] = hour
    f["day_of_week"] = int(txn.get("day_of_week", 0))
    f["is_night"] = 1.0 if (hour >= 22 or hour <= 5) else 0.0
    f["is_weekend"] = 1.0 if f["day_of_week"] >= 5 else 0.0

    # --- Amount ---
    amt = float(txn.get("amt", 0))
    f["log_amt"] = float(np.log1p(amt))
    f["amt"] = amt

    # --- Geography ---
    try:
        f["haversine_dist"] = haversine(
            float(txn.get("lat", 0)),
            float(txn.get("long", 0)),
            float(txn.get("merch_lat", 0)),
            float(txn.get("merch_long", 0)),
        )
    except (ValueError, TypeError):
        f["haversine_dist"] = 0.0

    # --- Demographics ---
    f["city_pop_log"] = float(np.log1p(float(txn.get("city_pop", 0))))

    # Age from DOB
    try:
        import datetime
        dob = txn.get("dob", "")
        if dob:
            # Approximate age using transaction year
            birth_year = int(str(dob)[:4])
            txn_year = int(str(txn.get("trans_date_trans_time", "2020"))[:4])
            f["age"] = float(txn_year - birth_year)
        else:
            f["age"] = 40.0  # median fallback
    except (ValueError, TypeError):
        f["age"] = 40.0

    # --- Category encoding (will be mapped to risk score) ---
    # Category fraud rates from training data (point-in-time, smoothed)
    CATEGORY_RISK = {
        "shopping_net": 0.0176, "misc_net": 0.0145, "grocery_pos": 0.0143,
        "shopping_pos": 0.0073, "gas_transport": 0.0048, "misc_pos": 0.0033,
        "grocery_net": 0.0030, "travel": 0.0030, "entertainment": 0.0026,
        "personal_care": 0.0025, "kids_pets": 0.0020, "food_dining": 0.0015,
        "home": 0.0015, "health_fitness": 0.0015,
    }
    category = str(txn.get("category", "")).lower().strip()
    f["category_risk"] = CATEGORY_RISK.get(category, 0.003)  # global mean fallback

    # --- Velocity (from entity history) ---
    if history:
        f["card_txn_count_1h"] = float(history.get("txn_count_1h", 0))
        f["card_txn_count_24h"] = float(history.get("txn_count_24h", 0))
        f["card_txn_sum_24h"] = float(history.get("txn_sum_24h", 0))
        f["amt_vs_card_median"] = amt / max(float(history.get("median_amt", 1)), 1)
        f["card_distinct_merchants_24h"] = float(history.get("distinct_merchants_24h", 0))
    else:
        f["card_txn_count_1h"] = 0.0
        f["card_txn_count_24h"] = 0.0
        f["card_txn_sum_24h"] = 0.0
        f["amt_vs_card_median"] = 1.0
        f["card_distinct_merchants_24h"] = 0.0

    return f


# Feature order for model input — MUST match training column order
FEATURE_COLUMNS = [
    "hour", "day_of_week", "is_night", "is_weekend",
    "log_amt", "amt",
    "haversine_dist",
    "city_pop_log", "age",
    "category_risk",
    "card_txn_count_1h", "card_txn_count_24h", "card_txn_sum_24h",
    "amt_vs_card_median", "card_distinct_merchants_24h",
]


def features_to_array(features: dict) -> np.ndarray:
    """Convert feature dict to numpy array in correct column order."""
    return np.array([features[col] for col in FEATURE_COLUMNS], dtype=np.float64)