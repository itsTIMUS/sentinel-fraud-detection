"""Quick EDA — 4 plots, 45 minutes, stop."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Setup
Path("reports/plots").mkdir(parents=True, exist_ok=True)
df = pd.read_csv("data/raw/sparkov/fraudTrain.csv")

# Parse datetime
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["hour"] = df["trans_date_trans_time"].dt.hour

print(f"Shape: {df.shape}")
print(f"Date range: {df['trans_date_trans_time'].min()} to {df['trans_date_trans_time'].max()}")
print(f"Fraud count: {df['is_fraud'].sum()} / {len(df)} ({df['is_fraud'].mean():.4f})")

# --- Plot 1: Class Balance ---
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["is_fraud"].value_counts()
ax.bar(["Legitimate", "Fraud"], [counts[0], counts[1]], color=["#2ecc71", "#e74c3c"])
ax.set_title("Class Balance")
ax.set_ylabel("Count")
for i, v in enumerate([counts[0], counts[1]]):
    ax.text(i, v + 5000, f"{v:,}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("reports/plots/01_class_balance.png", dpi=150)
plt.close()
print("✅ Plot 1: Class balance saved")

# --- Plot 2: Amount Distribution by Class (log scale) ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df[df["is_fraud"] == 0]["amt"], bins=100, alpha=0.6, label="Legit", color="#2ecc71", density=True)
ax.hist(df[df["is_fraud"] == 1]["amt"], bins=100, alpha=0.6, label="Fraud", color="#e74c3c", density=True)
ax.set_xscale("log")
ax.set_xlabel("Amount (log scale)")
ax.set_ylabel("Density")
ax.set_title("Amount Distribution: Legit vs Fraud")
ax.legend()
plt.tight_layout()
plt.savefig("reports/plots/02_amount_distribution.png", dpi=150)
plt.close()
print("✅ Plot 2: Amount distribution saved")

# --- Plot 3: Fraud Rate by Hour ---
fig, ax = plt.subplots(figsize=(8, 4))
fraud_by_hour = df.groupby("hour")["is_fraud"].mean()
ax.bar(fraud_by_hour.index, fraud_by_hour.values, color="#3498db")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Fraud Rate")
ax.set_title("Fraud Rate by Hour of Day")
ax.set_xticks(range(0, 24))
plt.tight_layout()
plt.savefig("reports/plots/03_fraud_by_hour.png", dpi=150)
plt.close()
print("✅ Plot 3: Fraud by hour saved")

# --- Plot 4: Fraud Rate by Category (top 10) ---
fig, ax = plt.subplots(figsize=(10, 5))
fraud_by_cat = df.groupby("category")["is_fraud"].mean().sort_values(ascending=True)
fraud_by_cat.plot(kind="barh", ax=ax, color="#9b59b6")
ax.set_xlabel("Fraud Rate")
ax.set_title("Fraud Rate by Category")
plt.tight_layout()
plt.savefig("reports/plots/04_fraud_by_category.png", dpi=150)
plt.close()
print("✅ Plot 4: Fraud by category saved")

# --- Summary stats for DECISIONS.md ---
print("\n--- Key Stats for DECISIONS.md ---")
print(f"Date range: {df['trans_date_trans_time'].min()} to {df['trans_date_trans_time'].max()}")
print(f"Total rows: {len(df):,}")
print(f"Fraud rate: {df['is_fraud'].mean():.4f} ({df['is_fraud'].sum():,} frauds)")
print(f"Amount range: ₹{df['amt'].min():.2f} to ₹{df['amt'].max():.2f}")
print(f"Median amount: ₹{df['amt'].median():.2f}")
print(f"Unique cards: {df['cc_num'].nunique():,}")
print(f"Unique merchants: {df['merchant'].nunique():,}")
print(f"Categories: {df['category'].nunique()}")
print(f"Highest fraud category: {fraud_by_cat.idxmax()} ({fraud_by_cat.max():.4f})")
print(f"Peak fraud hour: {fraud_by_hour.idxmax()} ({fraud_by_hour.max():.4f})")