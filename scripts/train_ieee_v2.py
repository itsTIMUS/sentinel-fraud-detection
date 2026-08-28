"""IEEE-CIS v2: merge identity features (device, browser, OS) + retrain."""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
import time
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

import sys
sys.path.insert(0, ".")
from src.sentinel.model_wrapper import LGBMWrapper
from src.sentinel.cost import load_costs, make_decision

costs = load_costs()

# --- Load and merge ---
print("Loading IEEE-CIS transaction + identity data...")
txn = pd.read_csv("data/raw/ieee/train_transaction.csv")
identity = pd.read_csv("data/raw/ieee/train_identity.csv")

print(f"Transactions: {len(txn):,} rows, {len(txn.columns)} cols")
print(f"Identity: {len(identity):,} rows, {len(identity.columns)} cols")

# Merge on TransactionID
df = txn.merge(identity, on="TransactionID", how="left")
print(f"After merge: {len(df):,} rows, {len(df.columns)} cols")
print(f"Identity match rate: {identity['TransactionID'].isin(txn['TransactionID']).mean():.1%}")

# --- Feature engineering ---
print("\nBuilding features...")

# Email encodings
EMAIL_ENC = {
    "gmail.com": 1, "yahoo.com": 2, "outlook.com": 3, "hotmail.com": 4,
    "anonymous.com": 5, "aol.com": 6, "comcast.net": 7, "icloud.com": 8,
    "yahoo.com.mx": 9, "msn.com": 10, "live.com": 11, "att.net": 12,
}
PRODUCT_ENC = {"W": 0, "C": 1, "R": 2, "H": 3, "S": 4}

def build_features_v2(row):
    """Enhanced IEEE features with identity data."""
    f = {}

    # Amount
    amt = float(row.get("TransactionAmt", 0) or 0)
    f["log_amt"] = float(np.log1p(amt))
    f["amt"] = amt

    # Product
    f["product_cd"] = float(PRODUCT_ENC.get(str(row.get("ProductCD", "")), -1))

    # Card features
    for col in ["card1", "card2", "card3", "card5"]:
        val = row.get(col, 0)
        f[col] = float(val) if val == val else 0.0

    # Address
    for col in ["addr1", "addr2"]:
        val = row.get(col, 0)
        f[col] = float(val) if val == val else 0.0

    # Distance
    val = row.get("dist1", 0)
    f["dist1"] = float(val) if val == val else 0.0

    # Email
    p_email = str(row.get("P_emaildomain", "")).lower()
    r_email = str(row.get("R_emaildomain", "")).lower()
    f["P_email_enc"] = float(EMAIL_ENC.get(p_email, 0))
    f["R_email_enc"] = float(EMAIL_ENC.get(r_email, 0))
    f["email_match"] = 1.0 if p_email == r_email and p_email != "nan" else 0.0

    # Counting features
    for col in ["C1", "C2", "C5", "C6", "C13", "C14"]:
        val = row.get(col, 0)
        f[col] = float(val) if val == val else 0.0

    # Time delta features
    for col in ["D1", "D4", "D10", "D15"]:
        val = row.get(col, 0)
        f[col] = float(val) if val == val else 0.0

    # Vesta features (top by known importance)
    for col in ["V12", "V13", "V54", "V56", "V75", "V78", "V83", "V87",
                "V258", "V201", "V246", "V315", "V294"]:
        val = row.get(col, 0)
        f[col] = float(val) if val == val else 0.0

    # --- NEW: Identity features ---
    # Device type (mobile vs desktop)
    device_type = str(row.get("DeviceType", "")).lower()
    f["is_mobile"] = 1.0 if device_type == "mobile" else 0.0
    f["has_device_info"] = 0.0 if str(row.get("DeviceType", "nan")) == "nan" else 1.0

    # Browser (id_31)
    browser = str(row.get("id_31", "")).lower()
    f["is_chrome"] = 1.0 if "chrome" in browser else 0.0
    f["is_safari"] = 1.0 if "safari" in browser else 0.0
    f["is_firefox"] = 1.0 if "firefox" in browser else 0.0
    f["is_edge"] = 1.0 if "edge" in browser else 0.0

    # OS (id_30)
    os_info = str(row.get("id_30", "")).lower()
    f["is_windows"] = 1.0 if "windows" in os_info else 0.0
    f["is_ios"] = 1.0 if "ios" in os_info else 0.0
    f["is_android"] = 1.0 if "android" in os_info else 0.0
    f["is_mac"] = 1.0 if "mac" in os_info else 0.0

    # Screen resolution (id_33) — presence indicates more info
    f["has_screen_res"] = 0.0 if str(row.get("id_33", "nan")) == "nan" else 1.0

    # UID construction (pseudo-account key)
    # Helps group transactions by likely same person
    card1 = str(row.get("card1", ""))
    addr1 = str(row.get("addr1", ""))
    f["uid_hash"] = float(hash(card1 + "_" + addr1) % 100000)

    return f

# Build feature matrix
FEATURE_COLS_V2 = None
features_list = []
start = time.time()

for idx, row in df.iterrows():
    f = build_features_v2(row)
    if FEATURE_COLS_V2 is None:
        FEATURE_COLS_V2 = sorted(f.keys())
    features_list.append([f[col] for col in FEATURE_COLS_V2])
    if idx % 100000 == 0 and idx > 0:
        print(f"  ...{idx:,} rows")

X = np.array(features_list, dtype=np.float64)
y = df["isFraud"].values
print(f"Features built in {time.time() - start:.0f}s | Shape: {X.shape}")
print(f"Feature count: {len(FEATURE_COLS_V2)}")

# --- Temporal split ---
df_sorted_idx = df["TransactionDT"].argsort().values
X = X[df_sorted_idx]
y = y[df_sorted_idx]

split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]
print(f"\nTrain: {len(X_train):,} | Val: {len(X_val):,}")
print(f"Train fraud: {y_train.mean():.4f} | Val fraud: {y_val.mean():.4f}")

# --- Train ---
print("\nTraining LightGBM v2...")
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS_V2)
val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS_V2, reference=train_data)

params = {
    "objective": "binary",
    "metric": "average_precision",
    "verbosity": -1,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "scale_pos_weight": neg / pos,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "seed": 42,
}

model = lgb.train(
    params, train_data,
    num_boost_round=500,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
)

# --- Evaluate raw ---
y_pred_raw = model.predict(X_val)
prec, rec, _ = precision_recall_curve(y_val, y_pred_raw)
pr_auc = auc(rec, prec)
brier_raw = brier_score_loss(y_val, y_pred_raw)

print(f"\nRaw: PR-AUC={pr_auc:.4f}, Brier={brier_raw:.6f}")

# --- Calibrate ---
print("Calibrating...")
cal_split = int(len(X_val) * 0.7)
wrapper = LGBMWrapper(booster=model)
calibrator = CalibratedClassifierCV(wrapper, method="isotonic", cv="prefit")
calibrator.fit(X_val[:cal_split], y_val[:cal_split])

y_pred_cal = calibrator.predict_proba(X_val)[:, 1]
brier_cal = brier_score_loss(y_val, y_pred_cal)
print(f"Calibrated: Brier={brier_cal:.6f} (improvement: {((brier_raw-brier_cal)/brier_raw)*100:.1f}%)")

# --- Cost evaluation with new policy ---
print("\nCost evaluation...")
total_cost = 0.0
total_cost_approve = 0.0
decisions = {"ALLOW": 0, "CHALLENGE": 0, "REVIEW": 0, "BLOCK": 0}
tp, fp, fn, tn = 0, 0, 0, 0

# Need amounts aligned with sorted order
amounts_sorted = df["TransactionAmt"].values[df_sorted_idx][split_idx:]

for i in range(len(y_val)):
    amt = float(amounts_sorted[i])
    true_label = y_val[i]
    p = float(y_pred_cal[i])
    result = make_decision(p_fraud=p, amount=amt, costs=costs)
    decisions[result["decision"]] += 1

    if result["decision"] == "ALLOW" and true_label == 1:
        total_cost += amt + costs["chargeback_fee_inr"]
        fn += 1
    elif result["decision"] == "BLOCK" and true_label == 0:
        total_cost += costs["gross_margin"] * amt + costs["friction_cost_inr"] + costs["churn_probability"] * costs["customer_ltv_inr"]
        fp += 1
    elif result["decision"] == "BLOCK" and true_label == 1:
        tp += 1
    elif result["decision"] == "ALLOW" and true_label == 0:
        tn += 1
    elif result["decision"] == "CHALLENGE":
        if true_label == 0:
            total_cost += costs["challenge_friction_inr"]
        else:
            if np.random.random() > costs["fraudster_3ds_dropout"]:
                total_cost += amt + costs["chargeback_fee_inr"]
    elif result["decision"] == "REVIEW":
        total_cost += costs["review_cost_inr"] + costs.get("review_delay_churn_inr", 80)
        if true_label == 1 and np.random.random() > costs["analyst_catch_rate"]:
            total_cost += amt + costs["chargeback_fee_inr"]

    if true_label == 1:
        total_cost_approve += amt + costs["chargeback_fee_inr"]

savings = (1 - total_cost / total_cost_approve) * 100

print(f"\n{'='*60}")
print(f"  IEEE-CIS v2 RESULTS (with identity features)")
print(f"{'='*60}")
print(f"  PR-AUC:     {pr_auc:.4f}  (was 0.4969)")
print(f"  Brier:      {brier_cal:.6f}")
print(f"  Decisions:  {decisions}")
print(f"  TP: {tp:,} | FP: {fp:,} | FN: {fn:,} | TN: {tn:,}")
if tp + fp > 0:
    print(f"  Precision:  {tp/(tp+fp):.4f}  (was 0.8669)")
if tp + fn > 0:
    print(f"  Recall:     {tp/(tp+fn):.4f}  (was 0.5960)")
print(f"\n  Approve all:  ₹{total_cost_approve:>12,.0f}")
print(f"  v1 (old):     ₹   2,539,429  (62.1%)")
print(f"  v1 + policy:  ₹   1,687,561  (74.8%)")
print(f"  v2 (new):     ₹{total_cost:>12,.0f}  ({savings:.1f}%)")

# Feature importance
importance = model.feature_importance(importance_type="gain")
feat_imp = sorted(zip(FEATURE_COLS_V2, importance), key=lambda x: x[1], reverse=True)
print(f"\n  Top 15 Features:")
for name, imp in feat_imp[:15]:
    print(f"    {name:>25s}: {imp:>12.1f}")

# --- Save artifacts ---
print("\nSaving artifacts...")
artifacts = Path("artifacts/ieee")
model.save_model(str(artifacts / "model.lgb"))
joblib.dump(calibrator, artifacts / "calibrator.joblib")
json.dump({"feature_columns": FEATURE_COLS_V2, "version": "v2-identity"}, 
          open(artifacts / "feature_schema.json", "w"), indent=2)
json.dump({
    "pr_auc": round(pr_auc, 4),
    "brier_calibrated": round(brier_cal, 6),
    "total_cost_inr": round(total_cost, 2),
    "total_cost_approve_all_inr": round(total_cost_approve, 2),
    "savings_pct": round(savings, 1),
    "decisions": decisions,
    "precision": round(tp/(tp+fp), 4) if tp+fp > 0 else 0,
    "recall": round(tp/(tp+fn), 4) if tp+fn > 0 else 0,
    "val_rows": len(y_val),
    "version": "v2-identity",
}, open(artifacts / "metrics.json", "w"), indent=2)

print("✅ IEEE-CIS v2 artifacts saved")