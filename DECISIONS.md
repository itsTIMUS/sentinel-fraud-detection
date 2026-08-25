# 📓 DECISIONS.md — Engineering Decision Log

## 2026-08-25 — Day 0: Project Setup

### Why Track 02 (AI Risk Manager)
Card-not-present fraud is a real business loss priced in ₹, not a toy classification problem. The rubric rewards problem taste — picking something that actually matters.

### Why Sparkov dataset first
Human-readable columns (card, merchant, category, geo, time) enable real reason codes like "unusual amount for this card" and "712km from usual location." Explainability is the demo story. IEEE-CIS planned as Phase 2 to prove robustness on real anonymized data.

### What's out of scope
Chargeback dispute automation, return abuse, account takeover, training pipeline product, feature store product, model retraining automation, multi-tenant auth.

### Key design decision: shared features.py
One feature function used by both Kaggle training and FastAPI serving. VS Code is the source of truth. This eliminates training-serving skew by construction.

### Why not SMOTE
SMOTE inflates offline metrics and destroys probability calibration. Using `scale_pos_weight` instead. Calibrated probabilities are essential because every downstream ₹ decision depends on p being a real probability.

### Why LightGBM over LLM for scoring
1000x faster, costs nothing to run, produces calibrated probabilities, fully auditable via pred_contrib. LLM will only narrate (off hot path) with deterministic template fallback.


--- Key Stats for DECISIONS.md ---
Date range: 2019-01-01 00:00:18 to 2020-06-21 12:13:37
Total rows: 1,296,675
Fraud rate: 0.0058 (7,506 frauds)
Amount range: ₹1.00 to ₹28948.90
Median amount: ₹47.52
Unique cards: 983
Unique merchants: 693
Categories: 14
Highest fraud category: shopping_net (0.0176)
Peak fraud hour: 22 (0.0288)
## 2026-08-25 — Day 1: Cost Model + Walking Skeleton

### EDA observations
- 1,296,675 rows, 0.58% fraud rate (7,506 frauds)
- Date range: 2019-01-01 to 2020-06-21
- Peak fraud hours: 22:00-03:00 (~3% fraud rate vs ~0.1% daytime)
- Highest fraud category: shopping_net (1.76%)
- 983 unique cards, 693 merchants, 14 categories
- Amount range: ₹1.00 to ₹28,948.90, median ₹47.52

### Baseline results (temporal validation split, last 20%)
- Approve everything: ₹3,123,266
- Rules baseline: ₹5,290,381 (PR-AUC: 0.3489)
- Logistic Regression: ₹18,289,845 (PR-AUC: 0.3186)

### Why baselines cost MORE than approve-all
LR sends 210K/259K transactions to REVIEW at ₹45 each = ~₹9.5M in review costs alone. The raw probabilities are uncalibrated and too aggressive. This proves calibration (Day 3) is essential — uncalibrated scores make the cost model useless.

### Walking skeleton achieved
curl → FastAPI → LR model → cost decision → SQLite audit ledger. Full loop working. Model will be swapped to LightGBM on Day 2 without changing the API contract.