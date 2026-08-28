# 🛡️ Sentinel — Cost-Aware Fraud Detection for Payment Gateways

> Most fraud systems optimise a metric. Sentinel optimises rupees — it computes, per transaction, whether blocking costs more than the fraud would, and it can show you the audit trail for every decision it has ever made.

## The Problem

A merchant runs an online store. Some payments are stolen cards. When fraud goes through, the cardholder complains, the bank claws money back, and the merchant loses the goods, the money, and pays a chargeback penalty fee.

The obvious fix — block anything suspicious — is worse than the disease. Block a real customer at checkout and you lose the sale, possibly the customer forever.

**Sentinel is the dial between "lose money to thieves" and "lose money by insulting customers" — set automatically, priced in ₹.**

## How It Works

For every payment, in under 50 milliseconds, Sentinel:

1. Examines context — not just "₹40,000 is big" but "₹40,000 from a card that never spent over ₹3,000, at 3am, 700km from usual location, 4th attempt in 10 minutes"
2. Estimates a calibrated probability of fraud
3. Computes expected ₹ loss if allowed vs blocked — approves, challenges, or blocks whichever is cheapest
4. Records why — permanently, in language a support agent can read

## Defense-Only Compliance

This system is strictly defense-only. No fraud-generation tooling, no evasion testing, no adversarial attack simulation. Robustness is evaluated through distribution shift and concept drift analysis only.

## Status

🚧 Under construction — Razorpay Buildathon Track 02 submission.

## 2026-08-28 — IEEE-CIS: Dual-Dataset Validation

### Why IEEE-CIS
Sparkov is simulated data — we cannot depend only on synthetic patterns. IEEE-CIS is real anonymized e-commerce fraud data from Vesta Corporation. Training on both proves the architecture is dataset-agnostic.

### IEEE-CIS dataset characteristics
- 590,540 transactions, 394 columns (vs Sparkov's 23)
- 3.5% fraud rate (vs Sparkov's 0.58%)
- Many anonymized features (V1-V339, C1-C14, D1-D15)
- No geographic or merchant name data — reason codes are weaker

### Feature engineering for IEEE-CIS
Used 30 features: transaction amount, product code, card attributes (card1-5), address, distance, email domains, counting features (C1-C14), time deltas (D1-D15), and top Vesta features (V12-V87). Feature selection based on known importance from competition leaderboard analysis.

### IEEE-CIS results
- PR-AUC: 0.4969 (vs Sparkov 0.9513)
- Brier score after calibration: 0.024268 (63.1% improvement from raw)
- Precision: 86.7% | Recall: 59.6%
- Decisions: 93,828 ALLOW / 23,191 REVIEW / 1,089 BLOCK
- Cost savings: 62.1% vs approve-all (₹2,539,429 vs ₹6,705,934)

### Why IEEE-CIS performance is lower — and why that's the point
Sparkov is simulated with clean, predictable fraud patterns. Real-world fraud (IEEE-CIS) is much harder — anonymized features, messier data, adversarial fraudsters. Lower PR-AUC is expected and honest. The key insight: even at 0.50 PR-AUC, the cost-aware policy still saves 62% vs approve-all. The architecture works regardless of model accuracy.

### What stayed the same (proving modularity)
- cost.py — identical ₹ logic, no changes
- make_decision() — same ALLOW/REVIEW/BLOCK policy
- Cost parameters (costs.yaml) — same ₹ assumptions
- API contract — same JSON in/out
- Audit ledger schema — unchanged
- Dashboard — same 4 tabs

### What changed (only dataset-specific files)
- NEW: src/sentinel/features/ieee_builder.py (30 IEEE-specific features)
- NEW: scripts/train_ieee.py (training script)
- NEW: artifacts/ieee/ (separate model + calibrator + metrics)

### Top features (IEEE-CIS, by gain)
1. R_emaildomain_enc — receiver email domain (615K)
2. C1 — counting feature (571K)
3. C13 — counting feature (533K)
4. card1 — card attribute (454K)
5. card2 — card attribute (441K)

Compared to Sparkov where log_amt dominated, IEEE-CIS relies heavily on email domain and counting features — different fraud signals for different datasets, same decision framework.

### Kaggle API key issue
kaggle.json was missing from .kaggle folder. Recreated via Kaggle Settings → Create Legacy API Key. The key is only used for downloading data — no impact on existing model or artifacts.