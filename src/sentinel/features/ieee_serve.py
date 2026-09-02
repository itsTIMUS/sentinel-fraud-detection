"""IEEE-CIS feature builder for serve time — uses saved lookup tables."""

import numpy as np
import pickle
from pathlib import Path


class IEEEFeatureBuilder:
    """Builds IEEE-CIS features using pre-computed lookup tables."""

    def __init__(self, lookups_path: str = "artifacts/ieee/lookups.pkl"):
        with open(lookups_path, "rb") as f:
            self.lookups = pickle.load(f)

        # Load feature schema for column order
        import json
        schema = json.load(open(Path(lookups_path).parent / "feature_schema.json"))
        self.feature_columns = schema["feature_columns"]

        self.email_map = self.lookups["email_map"]

    def build_features(self, txn: dict) -> dict:
        """Build features from an IEEE-CIS shaped transaction."""
        f = {}

        # Amount
        amt = float(txn.get("TransactionAmt", 0) or 0)
        f["log_amt"] = float(np.log1p(amt))
        f["amt"] = amt
        f["amt_decimal"] = round(amt - int(amt), 2)
        f["amt_is_round"] = 1.0 if f["amt_decimal"] == 0 else 0.0

        # Product code
        f["ProductCD"] = float({"W": 0, "C": 1, "R": 2, "H": 3, "S": 4}.get(
            str(txn.get("ProductCD", "")), -1))

        # Card features
        for col in ["card1", "card2", "card3", "card5"]:
            val = txn.get(col, -1)
            f[col] = float(val) if val == val and val is not None else -1.0

        f["card4"] = float({"visa": 0, "mastercard": 1, "american express": 2,
                            "discover": 3}.get(str(txn.get("card4", "")).lower(), -1))
        f["card6"] = float({"debit": 0, "credit": 1, "charge card": 2,
                            "debit or credit": 3}.get(str(txn.get("card6", "")).lower(), -1))

        # Address
        for col in ["addr1", "addr2"]:
            val = txn.get(col, -1)
            f[col] = float(val) if val == val and val is not None else -1.0

        f["dist1"] = float(txn.get("dist1", -1) or -1)

        # Email
        p_email = str(txn.get("P_emaildomain", "")).lower()
        r_email = str(txn.get("R_emaildomain", "")).lower()
        f["P_email"] = float(self.email_map.get(p_email, 0))
        f["R_email"] = float(self.email_map.get(r_email, 0))
        f["email_match"] = 1.0 if p_email == r_email and p_email != "nan" else 0.0
        f["email_missing"] = 1.0 if p_email in ("", "nan") else 0.0

        # C features
        for col in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
                     "C9", "C10", "C11", "C12", "C13", "C14"]:
            val = txn.get(col, 0)
            f[col] = float(val) if val == val and val is not None else 0.0

        # D features
        for col in ["D1", "D2", "D3", "D4", "D5", "D10", "D11", "D15"]:
            val = txn.get(col, -1)
            f[col] = float(val) if val == val and val is not None else -1.0

        # V features
        top_v = ["V12", "V13", "V14", "V17", "V20", "V23", "V26", "V29", "V30",
                 "V35", "V36", "V37", "V38", "V40", "V44", "V45", "V47", "V48",
                 "V54", "V56", "V62", "V69", "V75", "V76", "V78", "V82", "V83",
                 "V86", "V87", "V127", "V130", "V131", "V139", "V140",
                 "V147", "V149", "V152", "V160",
                 "V201", "V202", "V203", "V204", "V205", "V206", "V207", "V208",
                 "V209", "V210", "V212", "V213",
                 "V246", "V258", "V263", "V264", "V265",
                 "V271", "V274", "V277", "V279", "V280", "V282", "V283", "V285",
                 "V288", "V289", "V294", "V306", "V307", "V308", "V310", "V312",
                 "V313", "V314", "V315", "V317", "V318", "V320", "V321"]
        for col in top_v:
            val = txn.get(col, -999)
            f[col] = float(val) if val == val and val is not None else -999.0

        # Identity
        f["DeviceType"] = float({"mobile": 0, "desktop": 1}.get(
            str(txn.get("DeviceType", "")).lower(), -1))
        f["has_identity"] = 0.0 if str(txn.get("DeviceType", "")) in ("", "nan", "None") else 1.0

        browser = str(txn.get("id_31", "")).lower()
        f["browser_chrome"] = 1.0 if "chrome" in browser else 0.0
        f["browser_safari"] = 1.0 if "safari" in browser else 0.0
        f["browser_firefox"] = 1.0 if "firefox" in browser else 0.0
        f["browser_ie"] = 1.0 if any(x in browser for x in ["ie", "edge", "trident"]) else 0.0

        os_info = str(txn.get("id_30", "")).lower()
        f["os_windows"] = 1.0 if "windows" in os_info else 0.0
        f["os_ios"] = 1.0 if "ios" in os_info else 0.0
        f["os_android"] = 1.0 if "android" in os_info else 0.0
        f["os_mac"] = 1.0 if "mac" in os_info else 0.0

        f["has_screen"] = 0.0 if str(txn.get("id_33", "")) in ("", "nan", "None") else 1.0

        # Frequency from lookup tables
        card1_val = txn.get("card1", -1)
        f["card1_freq"] = self.lookups["card1_freq"].get(card1_val, 0.0)
        card2_val = txn.get("card2", -1)
        f["card2_freq"] = self.lookups["card2_freq"].get(card2_val, 0.0)
        addr1_val = txn.get("addr1", -1)
        f["addr1_freq"] = self.lookups["addr1_freq"].get(addr1_val, 0.0)
        p_email_domain = txn.get("P_emaildomain", "")
        f["P_emaildomain_freq"] = self.lookups.get("P_emaildomain_freq", {}).get(p_email_domain, 0.0)

        # UID-based lookups
        uid = f"{int(card1_val)}_{int(addr1_val)}_0"
        uid_s = self.lookups["uid_stats"]
        f["uid_count"] = uid_s["uid_count"].get(uid, 0.0)
        f["uid_mean_amt"] = uid_s["uid_mean_amt"].get(uid, 0.0)
        f["uid_std_amt"] = uid_s["uid_std_amt"].get(uid, 0.0)
        f["uid_max_amt"] = uid_s["uid_max_amt"].get(uid, 0.0)
        f["amt_vs_uid_mean"] = amt / max(f["uid_mean_amt"], 1.0)
        f["amt_vs_uid_max"] = amt / max(f["uid_max_amt"], 1.0)

        # Card1 lookups
        c1_s = self.lookups["card1_stats"]
        f["card1_count"] = c1_s["card1_count"].get(card1_val, 0.0)
        f["card1_mean_amt"] = c1_s["card1_mean_amt"].get(card1_val, 0.0)
        f["card1_std_amt"] = c1_s["card1_std_amt"].get(card1_val, 0.0)
        f["amt_vs_card1_mean"] = amt / max(f["card1_mean_amt"], 1.0)

        # Addr1 lookups
        a1_s = self.lookups["addr1_stats"]
        f["addr1_count"] = a1_s["addr1_count"].get(addr1_val, 0.0)
        f["addr1_mean_amt"] = a1_s["addr1_mean_amt"].get(addr1_val, 0.0)

        # Email lookups
        e_s = self.lookups["email_stats"]
        f["email_count"] = e_s["email_count"].get(p_email_domain, 0.0)

        # UID velocity (use defaults at serve time — no historical window available)
        f["uid_daily_mean"] = f["uid_count"] / 30.0  # rough estimate
        f["uid_daily_max"] = f["uid_daily_mean"] * 2.0

        # Time features
        txn_dt = float(txn.get("TransactionDT", 0) or 0)
        f["hour"] = int((txn_dt % 86400) / 3600)
        f["is_night"] = 1.0 if (f["hour"] >= 22 or f["hour"] <= 5) else 0.0

        return f

    def to_array(self, features: dict) -> np.ndarray:
        """Convert to numpy array in correct column order."""
        result = []
        for col in self.feature_columns:
            result.append(features.get(col, 0.0))
        return np.array(result, dtype=np.float64)