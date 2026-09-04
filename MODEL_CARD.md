# Model Card — Sentinel Fraud Detection

## Model Details

| Field | Value |
|---|---|
| **Model name** | Sentinel Cost-Aware Fraud Scorer |
| **Version** | v0.4 (Sparkov) / v3-final (IEEE-CIS) |
| **Type** | LightGBM gradient boosted trees |
| **Calibration** | Isotonic regression (sklearn CalibratedClassifierCV) |
| **Decision policy** | Expected cost minimisation (4 actions: ALLOW/CHALLENGE/REVIEW/BLOCK) |
| **Training data** | Sparkov: 1,037,340 rows (80% temporal split) / IEEE-CIS: 472,432 rows |
| **Framework** | LightGBM 4.3.0, scikit-learn 1.4.2 |
| **License** | MIT |
| **Developer** | itsTIMUS (Heritage Institute of Technology) |
| **Date** | August–September 2026 |

---

## Intended Use

### Primary use
Real-time fraud scoring for card-not-present payment transactions. The system receives a transaction, estimates fraud probability, and recommends the cheapest action in ₹.

### Intended users
- Payment gateway risk teams
- Merchant fraud analysts
- E-commerce risk operations

### Out of scope
- Credit scoring or creditworthiness assessment
- Identity verification or KYC
- Law enforcement or criminal investigation
- Account takeover detection
- Return/refund abuse detection

---

## Training Data

### Sparkov (Primary — Demo Model)
- **Source:** Kaggle (kartik2112/fraud-detection), CC0-1.0 license
- **Type:** Simulated credit card transactions
- **Size:** 1,296,675 transactions (Jan 2019 – Jun 2020)
- **Fraud rate:** 0.58% (7,506 frauds)
- **Split:** Temporal — first 80% train, last 20% validation
- **Features:** 15 (temporal, amount, geography, velocity, category risk)

### IEEE-CIS (Validation — Real-World Proof)
- **Source:** Kaggle IEEE-CIS Fraud Detection competition (Vesta Corporation)
- **Type:** Real anonymized e-commerce transactions
- **Size:** 590,540 transactions
- **Fraud rate:** 3.5% (20,663 frauds)
- **Split:** Temporal — first 80% train, last 20% validation
- **Features:** 150 (card attributes, entity resolution, V-features, identity, aggregations)

---

## Performance

### Sparkov (Held-Out Test — Evaluated Once)

| Metric | Value |
|---|---|
| PR-AUC | 0.9513 |
| Precision (at operating point) | 94.8% |
| Recall (at operating point) | 95.6% |
| Brier score (calibrated) | 0.000583 |
| p99 latency | 33ms |

### IEEE-CIS (Validation Set)

| Metric | Value |
|---|---|
| PR-AUC | 0.6240 |
| Precision | 92.2% |
| Recall | 62.9% |
| Brier score (calibrated) | 0.018703 |
| p99 latency | ~8ms |

### Cost Impact

| Strategy | Sparkov ₹ | Sparkov Savings | IEEE-CIS ₹ | IEEE-CIS Savings |
|---|---|---|---|---|
| Approve everything | ₹4,350,825 | — | ₹6,705,934 | — |
| Best fixed threshold | ₹450,242 | 89.7% | — | — |
| **Sentinel (4-action)** | **₹229,434** | **94.7%** | **₹1,651,632** | **75.4%** |
| Oracle (perfect info) | ₹0 | 100% | — | — |

Bootstrap 95% CI (Sparkov): ₹196,957 – ₹262,502 (savings: 94.0% – 95.5%)

---

## Features

### Sparkov (15 features)

| Family | Features | Signal |
|---|---|---|
| Temporal | hour, day_of_week, is_night, is_weekend | Fraud spikes at night (22:00–03:00) |
| Amount | log_amt, amt | Fraud amounts are higher than legitimate |
| Geography | haversine_dist | Distance between customer and merchant |
| Demographics | city_pop_log, age | Urban/rural patterns, age targeting |
| Category | category_risk | Historical fraud rate per merchant category |
| Velocity | card_txn_count_1h/24h, card_txn_sum_24h, amt_vs_card_median, card_distinct_merchants_24h | Card testing and rapid spending patterns |

### IEEE-CIS (150 features)

| Family | Count | Key Features |
|---|---|---|
| Entity (UID) | 8 | uid_count, uid_mean_amt, uid_std_amt, uid_daily_mean |
| Card aggregation | 4 | card1_count, card1_mean_amt, card1_std_amt, amt_vs_card1_mean |
| Address aggregation | 2 | addr1_count, addr1_mean_amt |
| Frequency encoding | 4 | card1_freq, card2_freq, addr1_freq, P_emaildomain_freq |
| Card attributes | 6 | card1–card6 |
| Counting (C) | 14 | C1–C14 |
| Time delta (D) | 8 | D1, D2, D3, D4, D5, D10, D11, D15 |
| Vesta (V) | 78 | Selected from V1–V339 based on competition analysis |
| Identity | 10 | DeviceType, browser, OS flags |
| Email | 4 | P_email, R_email, email_match, email_missing |
| Amount | 4 | log_amt, amt, amt_decimal, amt_is_round |
| Temporal | 2 | hour, is_night |

---

## Decision Policy

The model outputs a calibrated fraud probability. The cost engine then computes expected ₹ loss for each action and picks the cheapest:

| Action | Cost Formula | When It Wins |
|---|---|---|
| **ALLOW** | p × (amount + chargeback_fee) | Low risk, any amount |
| **CHALLENGE** | friction + (1-success)(1-p)(margin loss) + p(1-dropout)(fraud loss) | Mid risk, low-mid amount |
| **REVIEW** | analyst_cost + delay + p(1-catch_rate)(fraud loss) | Mid-high risk, high amount |
| **BLOCK** | (1-p)(1-ρ)(margin × amount) + friction + churn × LTV | High risk |

There is no fixed threshold. The break-even point shifts with transaction amount:
- ₹500: threshold = 0.225 (relaxed)
- ₹5,000: threshold = 0.176
- ₹20,000: threshold = 0.160 (paranoid)

---

## Limitations

### Data limitations
- **Sparkov is simulated.** Clean patterns, predictable fraud. Real-world performance will be lower (IEEE-CIS PR-AUC is 0.62, not 0.95).
- **IEEE-CIS is anonymized.** V1–V339 features have unknown meaning. Reason codes are less interpretable than Sparkov's.
- **No real Razorpay data.** Both datasets are proxies. Actual performance depends on Razorpay's transaction distribution.

### Model limitations
- **Velocity features require history.** New cards/entities score with defaults — no velocity signal. First-transaction detection is weaker.
- **No concept drift detection.** The model doesn't know when it's going stale. Production deployment needs rolling PR-AUC monitoring.
- **Category risk is static.** Hardcoded from training data EDA. A new fraud pattern targeting a previously safe category won't be reflected until retrain.

### Cost parameter limitations
- **Parameters are estimates, not ground truth.** Chargeback fee ₹1,500, margin 18%, churn 4% are industry benchmarks. Each merchant would need to tune these.
- **Sensitivity matters.** Tornado analysis shows fraudster_3ds_dropout (₹171K swing) and chargeback_fee (₹162K swing) are the top two sensitivities. If these are significantly wrong, total cost changes meaningfully.
- **CHALLENGE rate of ~13% on IEEE-CIS may be commercially high.** Real merchants target single-digit 3DS rates. A capacity constraint would be needed in production.

### Ethical considerations
- **No demographic features used for scoring.** Gender, race, ethnicity, religion are not inputs.
- **False positives disproportionately affect edge cases.** New customers, unusual purchase patterns, international transactions may receive more CHALLENGEs.
- **Defense-only system.** No fraud generation, no evasion testing, no adversarial simulation.

---

## Evaluation Methodology

### Temporal split (no random splitting)
All splits are by time — train on past, evaluate on future. This prevents information leakage that random splits would introduce.

### PR-AUC over accuracy
With 0.58% fraud rate, a model predicting "never fraud" gets 99.4% accuracy. PR-AUC measures precision-recall tradeoff and is meaningful for imbalanced classes.

### ₹ cost as primary metric
Traditional ML metrics (F1, accuracy) don't reflect business impact. A false positive on a ₹50 transaction costs ₹580. A false negative on a ₹40,000 transaction costs ₹41,500. We optimise total ₹ cost, not a metric.

### Bootstrap confidence intervals
1,000 resamples give 95% CI: savings of 94.0%–95.5%. This confirms the result is stable, not dependent on a particular test set composition.

### Oracle baseline
Sentinel captures 94.7% of achievable savings (approve-all → oracle). This is a more honest metric than "94.7% vs approve-all" alone.

---

## Sensitivity Analysis

Tornado chart results (each parameter varied ±50%):

| Parameter | ₹ Swing | Learnable via Thompson Sampling? |
|---|---|---|
| Fraudster 3DS dropout | ₹171,248 | ✅ Yes (challenge outcome) |
| Chargeback fee | ₹162,028 | ✅ Yes (bank statement) |
| Challenge success rate | ₹120,001 | ✅ Yes (OTP completion) |
| Challenge friction | ₹32,063 | ❌ No |
| Churn probability | ₹27,314 | ❌ No |
| Customer LTV | ₹27,314 | ❌ No |

The top three sensitivities are all learnable from observed outcomes. The unlearnable parameters have limited impact (₹27–32K swing).

---

## Deployment

### API Endpoints
| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check |
| `/ready` | GET | Readiness check (model, calibrator, store, ledger) |
| `/v1/score` | POST | Sparkov model scoring (15 features) |
| `/v1/score/ieee` | POST | IEEE-CIS model scoring (150 features) |

### Infrastructure
- **API:** FastAPI + Uvicorn (single worker)
- **Velocity store:** SQLite (WAL mode, PRAGMA synchronous=NORMAL)
- **Audit ledger:** SQLite (append-only, every decision recorded)
- **Dashboard:** Streamlit (5 tabs)
- **Containerisation:** Dockerfile + docker-compose.yml provided

### Resilience
- **Degraded mode:** Velocity store failure → score with stateless features, bias toward REVIEW
- **Reject inference:** 1% of BLOCKs silently allowed for unbiased retraining labels
- **Retry recovery:** 50% of blocked legit customers retry successfully

---

## Citation
Sparkov dataset: https://www.kaggle.com/datasets/kartik2112/fraud-detection (CC0-1.0)
IEEE-CIS dataset: https://www.kaggle.com/c/ieee-fraud-detection (Vesta Corporation)
LightGBM: Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (NeurIPS 2017)

---

Built by [itsTIMUS](https://github.com/itsTIMUS) · Heritage Institute of Technology · Razorpay Buildathon 2026 · Track 02