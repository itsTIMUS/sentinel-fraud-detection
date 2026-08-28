"""
IEEE-CIS specific feature builder.
Maps IEEE columns to features compatible with the Sentinel architecture.
"""

import numpy as np


# IEEE-CIS has different columns than Sparkov, so we build a separate feature set
# but output the SAME format: dict[str, float]

IEEE_FEATURE_COLUMNS = [
    "log_amt", "amt",
    "product_cd_enc",
    "card1", "card2", "card3", "card5",
    "addr1", "addr2",
    "dist1",
    "P_emaildomain_enc", "R_emaildomain_enc",
    "C1", "C2", "C5", "C6", "C13", "C14",
    "D1", "D4", "D10", "D15",
    "V12", "V13", "V54", "V56", "V75", "V78", "V83", "V87",
]


# Top email domains by frequency (from IEEE-CIS EDA)
EMAIL_ENCODING = {
    "gmail.com": 1, "yahoo.com": 2, "outlook.com": 3, "hotmail.com": 4,
    "anonymous.com": 5, "aol.com": 6, "comcast.net": 7, "icloud.com": 8,
    "yahoo.com.mx": 9, "msn.com": 10, "live.com": 11, "att.net": 12,
}

PRODUCT_ENCODING = {"W": 0, "C": 1, "R": 2, "H": 3, "S": 4}


def build_ieee_features(txn: dict) -> dict[str, float]:
    """Convert an IEEE-CIS transaction row to model-ready features."""
    f = {}

    # Amount
    amt = float(txn.get("TransactionAmt", 0))
    f["log_amt"] = float(np.log1p(amt))
    f["amt"] = amt

    # Product code
    f["product_cd_enc"] = float(PRODUCT_ENCODING.get(str(txn.get("ProductCD", "")), -1))

    # Card features (use as-is, they're already numeric)
    for col in ["card1", "card2", "card3", "card5"]:
        val = txn.get(col, 0)
        f[col] = float(val) if val == val else 0.0  # NaN check

    # Address
    for col in ["addr1", "addr2"]:
        val = txn.get(col, 0)
        f[col] = float(val) if val == val else 0.0

    # Distance
    val = txn.get("dist1", 0)
    f["dist1"] = float(val) if val == val else 0.0

    # Email domains
    p_email = str(txn.get("P_emaildomain", "")).lower()
    r_email = str(txn.get("R_emaildomain", "")).lower()
    f["P_emaildomain_enc"] = float(EMAIL_ENCODING.get(p_email, 0))
    f["R_emaildomain_enc"] = float(EMAIL_ENCODING.get(r_email, 0))

    # Counting features (C1-C14, use top ones by importance)
    for col in ["C1", "C2", "C5", "C6", "C13", "C14"]:
        val = txn.get(col, 0)
        f[col] = float(val) if val == val else 0.0

    # Time delta features (D1-D15, use top ones)
    for col in ["D1", "D4", "D10", "D15"]:
        val = txn.get(col, 0)
        f[col] = float(val) if val == val else 0.0

    # Vesta features (V1-V339, use top ones by known importance)
    for col in ["V12", "V13", "V54", "V56", "V75", "V78", "V83", "V87"]:
        val = txn.get(col, 0)
        f[col] = float(val) if val == val else 0.0

    return f


def ieee_features_to_array(features: dict) -> np.ndarray:
    """Convert feature dict to numpy array in correct column order."""
    return np.array([features[col] for col in IEEE_FEATURE_COLUMNS], dtype=np.float64)