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

## 2026-08-26 — Day 2: Features + LightGBM

### Decision: Train locally instead of Kaggle
Original plan used Kaggle for training. Since our machine handles 1.3M rows fine (696s for velocity computation, LightGBM trains in seconds), we trained locally. This eliminates handoff friction, copy-paste errors, and version mismatch risk between environments.

### features.py — the keystone file
Wrote `src/sentinel/features/builder.py` with `build_features()` function used by BOTH training and API serving. 15 features across 5 families:
- **Temporal:** hour, day_of_week, is_night, is_weekend
- **Amount:** log_amt, raw amt
- **Geography:** haversine distance (customer ↔ merchant)
- **Demographics:** city_pop_log, age from DOB
- **Velocity:** card_txn_count_1h, card_txn_count_24h, card_txn_sum_24h, amt_vs_card_median, card_distinct_merchants_24h
- **Category:** category_risk (smoothed fraud rate from training data)

### Velocity computation: point-in-time only
Every velocity aggregate uses ONLY transactions strictly before the current one. Sorted by time first. No future leakage. Took ~696s for 1M rows — acceptable for offline training.

### Category risk encoding
Used hardcoded fraud rates from EDA (e.g., shopping_net: 0.0176) instead of runtime target encoding. Avoids leakage, avoids needing the full dataset at serve time. Unknown categories fall back to global mean (0.003).

### File structure fix
`features.py` couldn't live at `src/sentinel/features.py` because `src/sentinel/features/` was already a package directory. Moved to `src/sentinel/features/builder.py` with re-export in `__init__.py`.

### LightGBM results
- PR-AUC: 0.9692 (vs Rules 0.3489, LR 0.3186)
- Total ₹ cost: 338,557 (vs approve-all 3,123,266 = 89% reduction)
- Decisions: 254,777 ALLOW / 2,778 REVIEW / 1,780 BLOCK
- Model used 500 boosting rounds, no early stopping triggered
- scale_pos_weight = neg/pos for class imbalance (NOT SMOTE)

### Top features by importance (gain)
1. log_amt (45.2M) — how big is the purchase
2. card_txn_sum_24h (15.0M) — spending velocity
3. is_night (2.3M) — fraud spikes 22:00–03:00
4. hour (1.1M) — time of day
5. card_distinct_merchants_24h (884K) — card used at many places

### API swap: LR → LightGBM
Swapped model in API without changing the contract. Latency dropped from 753ms to 5ms. Same curl, same JSON, same audit ledger — just better decisions behind the scenes.

### Why raw LightGBM scores still need calibration
LightGBM outputs raw scores, not true probabilities. With scale_pos_weight, these are further distorted. The cost model depends on p being a real probability — if p=0.05 doesn't mean "5% chance of fraud," every ₹ calculation downstream is fiction. Calibration is Day 3's top priority.


## 2026-08-27 — Day 3: Calibration + Evaluation (Model Frozen)

### Calibration results
- Brier score BEFORE calibration: 0.001731
- Brier score AFTER calibration: 0.000655
- Improvement: 62.1%
- Method: Isotonic regression (CalibratedClassifierCV with cv="prefit")
- Used 70% of validation for calibration fitting, 30% for verification

### Why calibration matters
Raw LightGBM scores with scale_pos_weight are NOT real probabilities. Before calibration, when the model said "60% fraud chance," the true frequency was ~12%. Every ₹ calculation downstream depends on p being real — without calibration, the entire cost model is fiction. The reliability diagram proves this visually.

### Break-even curve (THE pitch centrepiece)
The break-even probability is a function of transaction amount:
- ₹500 transaction: threshold = 0.225 (relaxed — blocking costs more than the fraud)
- ₹5,000 transaction: threshold = 0.176
- ₹20,000 transaction: threshold = 0.160 (paranoid — fraud cost dominates)
This means Sentinel is deliberately lenient on small transactions and deliberately strict on large ones. No single fixed threshold — computed per transaction.

### Held-out test evaluation (RUN ONCE, NO TUNING AFTER)
Test set: 555,719 rows (0.39% fraud rate)

**Model performance:**
- PR-AUC: 0.9513
- Brier score: 0.000583
- Precision at operating point: 94.8%
- Recall at operating point: 95.6%

**Confusion matrix:**
- TP (blocked fraud): 1,823
- FP (blocked legit): 100
- FN (allowed fraud): 85
- TN (allowed legit): 551,897

**Decisions distribution:** 551,982 ALLOW / 1,814 REVIEW / 1,923 BLOCK

### ₹ Cost comparison (held-out test) — THE money shot
| Strategy | Total ₹ Cost | Savings vs Approve-All |
|---|---|---|
| Approve everything | ₹4,350,825 | — |
| Naive 0.5 threshold | ₹527,133 | 87.9% |
| **Sentinel (cost-aware)** | **₹309,487** | **92.9%** |

### Full model progression (₹ cost on respective evaluation sets)
| Model | PR-AUC | Total ₹ Cost | Notes |
|---|---|---|---|
| Approve everything | — | ₹3,123,266 (val) / ₹4,350,825 (test) | Floor baseline |
| Rules baseline | 0.3489 | ₹5,290,381 | Worse than approve-all due to excessive reviews |
| Logistic Regression | 0.3186 | ₹18,289,845 | Terrible — sent 210K/259K to REVIEW at ₹45 each |
| LightGBM (raw) | 0.9692 | ₹338,557 | Massive jump, but uncalibrated |
| LightGBM (calibrated) | 0.9513 | ₹309,487 | Final model, honest probabilities |

### Problem encountered: calibrator pickle error
**What broke:** The calibrator was saved with `LGBMWrapper` defined in `__main__` (the training script). When uvicorn loads the API, `__main__` is uvicorn itself — Python couldn't find the class and crashed with `AttributeError: Can't get attribute 'LGBMWrapper'`.

**Root cause:** Python's pickle stores the full module path of classes. If a class is defined in a script run directly (`__main__`), the path becomes `__main__.ClassName`. Any other process loading that pickle needs `__main__` to contain that class — which it won't.

**How we fixed it:**
1. Created `src/sentinel/model_wrapper.py` with `LGBMWrapper` class
2. Re-ran calibration importing from that module (`from src.sentinel.model_wrapper import LGBMWrapper`)
3. Now pickle stores `src.sentinel.model_wrapper.LGBMWrapper` — findable from any process
4. API imports `model_wrapper` so the class is available during deserialization

**Lesson:** Always define serializable classes in importable modules, never in `__main__`. This is a classic Python gotcha with joblib/pickle.

### Model is FROZEN
No more tuning, no more touching the test set. From here, it's all product engineering (Day 4) and documentation (Day 5).

## 2026-08-27 — Day 4: Product Engineering

### Reason codes via pred_contrib
LightGBM's `pred_contrib=True` gives exact per-feature SHAP contributions in microseconds — no expensive TreeExplainer needed. Top-3 reasons returned with every decision in human-readable format: "unusual transaction amount", "712 km from merchant location", "4 transactions on this card in last hour."

### Velocity store (SQLite)
Real-time per-card transaction tracking. Records every scored transaction and provides velocity features (txn_count_1h, txn_count_24h, txn_sum_24h, distinct_merchants_24h) for future lookups. Tested by sending same card twice — second request showed velocity data from first.

### Degraded mode
When velocity store is unavailable, API does NOT crash. It scores with stateless features, sets `degraded: true`, and biases probability upward (1.5x) to push borderline cases toward REVIEW. DegradedVelocityStore fallback class returns None for all lookups.

### Separate /health and /ready endpoints
- `/health` — liveness: is the process alive?
- `/ready` — readiness: is the model loaded, calibrator loaded, velocity store reachable, ledger available?

### Streamlit dashboard (4 tabs)
1. **Live Score** — input transaction, see decision + reason codes + latency + cost breakdown
2. **Review Queue** — REVIEW decisions from audit ledger, most recent first
3. **Cost Dashboard** — interactive break-even curve with sliders for chargeback fee, margin, friction, review cost
4. **Model Card** — PR-AUC, precision, recall, Brier, cost savings, intended use, limitations

### Load test results
- p50: 12.4ms, p95: 14.7ms, p99: 33.0ms
- 200 requests, 0 errors
- Target was p99 < 50ms — achieved

### Problem: localhost resolving to IPv6 on Windows
Load test showed 2,060ms per request. Root cause: Windows tries IPv6 (::1) first for `localhost`, waits ~2 seconds before falling back to IPv4. Fix: use `127.0.0.1` instead of `localhost`. Actual latency was 12ms all along.

### SQLite optimizations
Added `PRAGMA synchronous=NORMAL` (skip disk fsync per commit) and `PRAGMA busy_timeout=5000` (wait for write locks instead of crashing) to both velocity store and audit ledger.

### Decision: deterministic templates over LLM narration
The plan included optional LLM narration (Groq/Gemini). We chose deterministic reason code templates instead. Reasons: faster, free, deterministic, auditable. LLM adds latency, cost, and non-determinism to a system where auditability is a selling point. This IS the "AI judgment" rubric answer — knowing where NOT to use AI.