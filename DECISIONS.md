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