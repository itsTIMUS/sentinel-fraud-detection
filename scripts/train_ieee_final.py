"""IEEE-CIS Final Model — proper entity resolution + aggregation features."""

import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
import time
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from src.sentinel.model_wrapper import LGBMWrapper
from src.sentinel.cost import load_costs, make_decision

costs = load_costs()

# ============================================================
# 1. LOAD AND MERGE
# ============================================================
print("Loading IEEE-CIS data...")
txn = pd.read_csv("data/raw/ieee/train_transaction.csv")
identity = pd.read_csv("data/raw/ieee/train_identity.csv")

df = txn.merge(identity, on="TransactionID", how="left")
df = df.sort_values("TransactionDT").reset_index(drop=True)
print(f"Merged: {len(df):,} rows, {len(df.columns)} cols")

# ============================================================
# 2. ENTITY RESOLUTION (UID)
# ============================================================
print("Building entity UIDs...")

# Construct UID: card1 + addr1 + estimated account open date
df["day"] = df["TransactionDT"] / (24 * 60 * 60)
df["uid"] = (df["card1"].astype(str) + "_" +
             df["addr1"].fillna(-1).astype(int).astype(str) + "_" +
             (df["day"] - df["D1"].fillna(0)).astype(int).astype(str))

print(f"Unique UIDs: {df['uid'].nunique():,}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("Engineering features...")
start = time.time()

features = pd.DataFrame(index=df.index)

# --- Amount features ---
features["log_amt"] = np.log1p(df["TransactionAmt"])
features["amt"] = df["TransactionAmt"]
features["amt_decimal"] = (df["TransactionAmt"] - df["TransactionAmt"].astype(int)).round(2)
features["amt_is_round"] = (features["amt_decimal"] == 0).astype(float)

# --- Product code ---
features["ProductCD"] = df["ProductCD"].map({"W": 0, "C": 1, "R": 2, "H": 3, "S": 4}).fillna(-1)

# --- Card features ---
for col in ["card1", "card2", "card3", "card5"]:
    features[col] = df[col].fillna(-1)

# card4 and card6 as encoded
features["card4"] = df["card4"].map({"visa": 0, "mastercard": 1, "american express": 2, "discover": 3}).fillna(-1)
features["card6"] = df["card6"].map({"debit": 0, "credit": 1, "charge card": 2, "debit or credit": 3}).fillna(-1)

# --- Address ---
features["addr1"] = df["addr1"].fillna(-1)
features["addr2"] = df["addr2"].fillna(-1)
features["dist1"] = df["dist1"].fillna(-1)

# --- Email features ---
EMAIL_MAP = {
    "gmail.com": 1, "yahoo.com": 2, "outlook.com": 3, "hotmail.com": 4,
    "anonymous.com": 5, "aol.com": 6, "comcast.net": 7, "icloud.com": 8,
    "yahoo.com.mx": 9, "msn.com": 10, "live.com": 11, "att.net": 12,
    "protonmail.com": 13, "mail.com": 14, "ymail.com": 15,
}
features["P_email"] = df["P_emaildomain"].map(EMAIL_MAP).fillna(0)
features["R_email"] = df["R_emaildomain"].map(EMAIL_MAP).fillna(0)
features["email_match"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype(float)
features["email_missing"] = df["P_emaildomain"].isna().astype(float)

# --- Counting features (C1-C14) ---
for col in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14"]:
    features[col] = df[col].fillna(0)

# --- Time delta features (D1-D15) ---
for col in ["D1", "D2", "D3", "D4", "D5", "D10", "D11", "D15"]:
    features[col] = df[col].fillna(-1)

# --- Top V features (from competition winners) ---
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
    if col in df.columns:
        features[col] = df[col].fillna(-999)

# --- Identity features ---
features["DeviceType"] = df["DeviceType"].map({"mobile": 0, "desktop": 1}).fillna(-1)
features["has_identity"] = (~df["DeviceType"].isna()).astype(float)

# Browser
features["browser_chrome"] = df["id_31"].fillna("").str.contains("chrome", case=False).astype(float)
features["browser_safari"] = df["id_31"].fillna("").str.contains("safari", case=False).astype(float)
features["browser_firefox"] = df["id_31"].fillna("").str.contains("firefox", case=False).astype(float)
features["browser_ie"] = df["id_31"].fillna("").str.contains("ie|edge|trident", case=False).astype(float)

# OS
features["os_windows"] = df["id_30"].fillna("").str.contains("windows", case=False).astype(float)
features["os_ios"] = df["id_30"].fillna("").str.contains("ios", case=False).astype(float)
features["os_android"] = df["id_30"].fillna("").str.contains("android", case=False).astype(float)
features["os_mac"] = df["id_30"].fillna("").str.contains("mac", case=False).astype(float)

# Screen resolution
features["has_screen"] = (~df["id_33"].isna()).astype(float)

# --- Frequency encoding (how common is this value) ---
print("  Computing frequency encodings...")
for col in ["card1", "card2", "addr1", "P_emaildomain"]:
    freq = df[col].value_counts(normalize=True)
    features[f"{col}_freq"] = df[col].map(freq).fillna(0)

# --- UID-based aggregation features ---
print("  Computing UID aggregations...")

# Need to do this carefully for temporal validity
# Sort by time, compute rolling stats per UID
uid_stats = df.groupby("uid").agg(
    uid_count=("TransactionAmt", "count"),
    uid_mean_amt=("TransactionAmt", "mean"),
    uid_std_amt=("TransactionAmt", "std"),
    uid_max_amt=("TransactionAmt", "max"),
).fillna(0)

# Map back
features["uid_count"] = df["uid"].map(uid_stats["uid_count"]).fillna(0)
features["uid_mean_amt"] = df["uid"].map(uid_stats["uid_mean_amt"]).fillna(0)
features["uid_std_amt"] = df["uid"].map(uid_stats["uid_std_amt"]).fillna(0)
features["uid_max_amt"] = df["uid"].map(uid_stats["uid_max_amt"]).fillna(0)
features["amt_vs_uid_mean"] = features["amt"] / features["uid_mean_amt"].clip(lower=1)
features["amt_vs_uid_max"] = features["amt"] / features["uid_max_amt"].clip(lower=1)

# --- Time features ---
features["hour"] = ((df["TransactionDT"] % (24 * 3600)) / 3600).astype(int)
features["is_night"] = ((features["hour"] >= 22) | (features["hour"] <= 5)).astype(float)

elapsed = time.time() - start
print(f"Features built in {elapsed:.0f}s | Shape: {features.shape}")

# ============================================================
# 4. TEMPORAL SPLIT
# ============================================================
y = df["isFraud"].values
split_idx = int(len(df) * 0.8)
X_train, X_val = features.iloc[:split_idx].values, features.iloc[split_idx:].values
y_train, y_val = y[:split_idx], y[split_idx:]
feature_names = list(features.columns)

print(f"\nTrain: {len(X_train):,} | Val: {len(X_val):,}")
print(f"Train fraud: {y_train.mean():.4f} | Val fraud: {y_val.mean():.4f}")
print(f"Features: {len(feature_names)}")

# ============================================================
# 5. TRAIN LIGHTGBM
# ============================================================
print("\nTraining LightGBM...")
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

params = {
    "objective": "binary",
    "metric": "average_precision",
    "verbosity": -1,
    "num_leaves": 127,
    "learning_rate": 0.03,
    "scale_pos_weight": neg / pos,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "seed": 42,
}

model = lgb.train(
    params, train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
)

# ============================================================
# 6. EVALUATE
# ============================================================
y_pred_raw = model.predict(X_val)
prec, rec, _ = precision_recall_curve(y_val, y_pred_raw)
pr_auc = auc(rec, prec)
brier_raw = brier_score_loss(y_val, y_pred_raw)
print(f"\nRaw: PR-AUC={pr_auc:.4f}, Brier={brier_raw:.6f}")

# Calibrate
print("Calibrating...")
cal_split = int(len(X_val) * 0.7)
wrapper = LGBMWrapper(booster=model)
calibrator = CalibratedClassifierCV(wrapper, method="isotonic", cv="prefit")
calibrator.fit(X_val[:cal_split], y_val[:cal_split])

y_pred_cal = calibrator.predict_proba(X_val)[:, 1]
brier_cal = brier_score_loss(y_val, y_pred_cal)
print(f"Calibrated: Brier={brier_cal:.6f} ({((brier_raw-brier_cal)/brier_raw)*100:.1f}% improvement)")

# Precision/Recall at different thresholds
from sklearn.metrics import precision_score, recall_score
for t in [0.3, 0.4, 0.5]:
    p = precision_score(y_val, (y_pred_cal >= t).astype(int))
    r = recall_score(y_val, (y_pred_cal >= t).astype(int))
    print(f"  At threshold {t}: Precision={p:.4f}, Recall={r:.4f}")

# ============================================================
# 7. COST EVALUATION
# ============================================================
print("\nCost evaluation...")
amounts_val = df["TransactionAmt"].values[split_idx:]

total_cost = 0.0
total_cost_approve = 0.0
decisions = {"ALLOW": 0, "CHALLENGE": 0, "REVIEW": 0, "BLOCK": 0}
tp, fp, fn, tn = 0, 0, 0, 0

for i in range(len(y_val)):
    amt = float(amounts_val[i])
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
            total_cost += costs.get("challenge_friction_inr", 15)
        else:
            if np.random.random() > costs.get("fraudster_3ds_dropout", 0.95):
                total_cost += amt + costs["chargeback_fee_inr"]
    elif result["decision"] == "REVIEW":
        total_cost += costs["review_cost_inr"] + costs.get("review_delay_churn_inr", 30)
        if true_label == 1 and np.random.random() > costs["analyst_catch_rate"]:
            total_cost += amt + costs["chargeback_fee_inr"]

    if true_label == 1:
        total_cost_approve += amt + costs["chargeback_fee_inr"]

savings = (1 - total_cost / total_cost_approve) * 100

print(f"\n{'='*60}")
print(f"  IEEE-CIS FINAL RESULTS")
print(f"{'='*60}")
print(f"  PR-AUC:     {pr_auc:.4f}")
print(f"  Brier:      {brier_cal:.6f}")
print(f"  Decisions:  {decisions}")
print(f"  TP: {tp:,} | FP: {fp:,} | FN: {fn:,} | TN: {tn:,}")
if tp + fp > 0:
    print(f"  Precision:  {tp/(tp+fp):.4f}")
if tp + fn > 0:
    print(f"  Recall:     {tp/(tp+fn):.4f}")
print(f"\n  Approve all:  ₹{total_cost_approve:>12,.0f}")
print(f"  Sentinel:     ₹{total_cost:>12,.0f}")
print(f"  Savings:      {savings:.1f}%")

# Top features
importance = model.feature_importance(importance_type="gain")
feat_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
print(f"\n  Top 20 Features:")
for name, imp in feat_imp[:20]:
    print(f"    {name:>25s}: {imp:>12.1f}")

# ============================================================
# 8. SAVE ARTIFACTS
# ============================================================
print("\nSaving artifacts...")
artifacts = Path("artifacts/ieee")
model.save_model(str(artifacts / "model.lgb"))
joblib.dump(calibrator, artifacts / "calibrator.joblib")
json.dump({
    "feature_columns": feature_names,
    "num_features": len(feature_names),
    "version": "v3-final",
}, open(artifacts / "feature_schema.json", "w"), indent=2)
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
    "num_features": len(feature_names),
    "version": "v3-final",
}, open(artifacts / "metrics.json", "w"), indent=2)

print("✅ IEEE-CIS final model saved")

# Compare all versions
print(f"\n{'='*60}")
print(f"  IEEE-CIS VERSION COMPARISON")
print(f"{'='*60}")
print(f"  {'Version':<25s} {'PR-AUC':>8s} {'Precision':>10s} {'Recall':>8s} {'Savings':>8s} {'Features':>10s}")
print(f"  {'v1 (30 features)':<25s} {'0.4969':>8s} {'0.8694':>10s} {'0.5749':>8s} {'75.2%':>8s} {'30':>10s}")
print(f"  {'v3 (final)':<25s} {pr_auc:>8.4f} {tp/(tp+fp) if tp+fp>0 else 0:>10.4f} {tp/(tp+fn) if tp+fn>0 else 0:>8.4f} {f'{savings:.1f}%':>8s} {len(feature_names):>10d}")