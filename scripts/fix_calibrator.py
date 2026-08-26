"""Re-save calibrator with importable LGBMWrapper."""

import sys
sys.path.insert(0, ".")
import numpy as np
import lightgbm as lgb
import joblib
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from src.sentinel.model_wrapper import LGBMWrapper
from src.sentinel.features import build_features, features_to_array

ARTIFACTS = Path("artifacts/sparkov")
booster = lgb.Booster(model_file=str(ARTIFACTS / "model.lgb"))

print("Rebuilding val features...")
df = pd.read_parquet("data/processed/train.parquet")
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["hour"] = df["trans_date_trans_time"].dt.hour
df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
val = df.iloc[int(len(df) * 0.8):].copy().sort_values("trans_date_trans_time").reset_index(drop=True)

card_txns = {}
val_features = []
for idx, row in val.iterrows():
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
    val_features.append(features_to_array(build_features(row.to_dict(), history=history)))
    card_txns[card].append((unix_t, amt, merchant))
    if idx % 100000 == 0 and idx > 0:
        print(f"  ...{idx:,}")

X_val = np.vstack(val_features)
y_val = val["is_fraud"].values
cal_split = int(len(X_val) * 0.7)

wrapper = LGBMWrapper(booster=booster)
calibrator = CalibratedClassifierCV(wrapper, method="isotonic", cv="prefit")
calibrator.fit(X_val[:cal_split], y_val[:cal_split])
joblib.dump(calibrator, ARTIFACTS / "calibrator.joblib")
print("✅ Calibrator saved with src.sentinel.model_wrapper.LGBMWrapper")