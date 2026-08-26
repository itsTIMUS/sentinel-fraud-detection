"""Reason codes from LightGBM pred_contrib → human-readable explanations."""

import numpy as np
from src.sentinel.features import FEATURE_COLUMNS

# Human-readable reason code dictionary
REASON_CODES = {
    "hour": {
        "code": "TIME_OF_DAY",
        "template": "transaction at hour {value:.0f}",
    },
    "day_of_week": {
        "code": "DAY_OF_WEEK",
        "template": "unusual day-of-week pattern",
    },
    "is_night": {
        "code": "NIGHT_TXN",
        "template": "nighttime transaction (22:00–05:00)",
    },
    "is_weekend": {
        "code": "WEEKEND_TXN",
        "template": "weekend transaction",
    },
    "log_amt": {
        "code": "AMT_UNUSUAL",
        "template": "unusual transaction amount",
    },
    "amt": {
        "code": "AMT_RAW",
        "template": "transaction amount ₹{value:.2f}",
    },
    "haversine_dist": {
        "code": "GEO_DISTANCE",
        "template": "{value:.0f} km from merchant location",
    },
    "city_pop_log": {
        "code": "CITY_POP",
        "template": "unusual city population pattern",
    },
    "age": {
        "code": "AGE_PATTERN",
        "template": "age-based risk pattern",
    },
    "category_risk": {
        "code": "HIGH_RISK_CATEGORY",
        "template": "high-risk merchant category",
    },
    "card_txn_count_1h": {
        "code": "VELOCITY_1H",
        "template": "{value:.0f} transactions on this card in last hour",
    },
    "card_txn_count_24h": {
        "code": "VELOCITY_24H",
        "template": "{value:.0f} transactions on this card in 24 hours",
    },
    "card_txn_sum_24h": {
        "code": "SPEND_24H",
        "template": "₹{value:.0f} total spend on this card in 24 hours",
    },
    "amt_vs_card_median": {
        "code": "AMT_VS_MEDIAN",
        "template": "amount is {value:.1f}x this card's median spend",
    },
    "card_distinct_merchants_24h": {
        "code": "MERCHANT_DIVERSITY",
        "template": "{value:.0f} distinct merchants in 24 hours",
    },
}


def get_reason_codes(
    booster,
    feature_array: np.ndarray,
    feature_values: dict,
    top_k: int = 3,
) -> list[dict]:
    """
    Extract top-k reason codes from LightGBM pred_contrib.

    Args:
        booster: LightGBM Booster (native, not sklearn wrapper)
        feature_array: 1D or 2D numpy array of features
        feature_values: dict of feature_name → actual value (for human-readable detail)
        top_k: number of top contributors to return

    Returns:
        list of dicts with code, contribution, detail
    """
    # pred_contrib returns [feature_contribs..., bias] per row
    contribs = booster.predict(
        feature_array.reshape(1, -1), pred_contrib=True
    )[0]

    # Exclude bias (last element)
    feature_contribs = contribs[:-1]

    # Pair with feature names and sort by absolute contribution
    paired = list(zip(FEATURE_COLUMNS, feature_contribs))
    paired.sort(key=lambda x: abs(x[1]), reverse=True)

    reasons = []
    for name, contrib in paired[:top_k]:
        info = REASON_CODES.get(name, {"code": name.upper(), "template": f"{name} signal"})
        value = feature_values.get(name, 0)

        try:
            detail = info["template"].format(value=value)
        except (KeyError, ValueError):
            detail = info["template"]

        reasons.append({
            "code": info["code"],
            "contribution": round(float(contrib), 4),
            "detail": detail,
        })

    return reasons