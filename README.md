# 🛡️ Sentinel - Cost-Aware Fraud Detection

**Razorpay Buildathon 2026 · Track 02: AI Risk Manager**

> Most fraud systems optimise F1 score. Sentinel optimises rupees.
>
> For every transaction, it computes whether blocking costs more than the fraud would - and picks the cheapest action. Not the safest. Not the most conservative. The cheapest, in ₹.

---

## The Problem, In Money

A merchant runs an online store. Someone pays ₹40,000 with a stolen card. The real cardholder disputes it. The bank claws back ₹40,000 **and** charges ₹1,500 penalty. The merchant loses the goods, the money, and the fee.

The obvious fix - block anything suspicious - is worse. Block a real customer buying a ₹200 grocery order and you lose ₹580 (margin + friction + lifetime value risk). The fraud would've cost you ₹17. You just lost 34× more by "being safe."

**There's a dial between "lose money to thieves" and "lose money by insulting customers." Sentinel sets it automatically, per transaction, priced in ₹.**

---

## The Result

| Strategy | Total ₹ Cost | Savings |
|---|---|---|
| Approve everything | ₹4,350,825 | - |
| Best fixed threshold (t=0.21) | ₹450,242 | 89.7% |
| **Sentinel (4-action, cost-aware)** | **₹229,434** | **94.7%** |
| Oracle (perfect information) | ₹0 | 100% |

**Sentinel captures 94.7% of achievable savings** (bootstrap 95% CI: 94.0%–95.5%).

Validated on two datasets: Sparkov (simulated, 1.3M transactions) and IEEE-CIS (real anonymized e-commerce, 590K transactions).

| Metric | Sparkov | IEEE-CIS |
|---|---|---|
| PR-AUC | 0.9513 | 0.6240 |
| Precision | 94.8% | 92.2% |
| Recall | 95.6% | 62.9% |
| ₹ Savings | 94.7% | 75.4% |
| Features | 15 | 150 |
| p99 Latency | 33ms | 8ms |

IEEE-CIS is lower because real fraud is harder than simulated. That's the point - the cost-aware policy still saves 75% on genuinely hard data.

---

## How It Works

```
Transaction arrives (₹5,000, shopping_net, 2am, card used 4× today)
    │
    ├── Velocity store: "this card spent ₹12,000 in the last hour"
    │
    ├── Feature builder: 15 signals (amount, time, distance, velocity, category risk)
    │   Same function for training AND serving - zero skew by construction
    │
    ├── LightGBM → calibrated probability: p = 0.12
    │   Isotonic calibration (Brier improved 62%) - when it says 12%, it means 12%
    │
    ├── Cost engine computes 4 options:
    │   ALLOW:     ₹780   (12% chance of losing ₹5,000 + ₹1,500 penalty)
    │   CHALLENGE: ₹83    (send OTP - 95% of fraudsters drop off)
    │   REVIEW:    ₹127   (analyst reviews - ₹45 time + ₹30 delay)
    │   BLOCK:     ₹1,210 (lose margin + friction + churn risk)
    │
    ├── Decision: CHALLENGE (₹83 is cheapest)
    │
    ├── Reason codes: "unusual amount (+6.5), nighttime (+2.3), 4th txn in 1h (+1.8)"
    │
    └── Audit ledger: decision permanently recorded with full ₹ breakdown
```

### The Break-Even Curve

There is no single threshold. The threshold is a function of amount:

- **₹500 transaction:** threshold = 0.225 (relaxed - blocking costs more than fraud)
- **₹5,000 transaction:** threshold = 0.176
- **₹20,000 transaction:** threshold = 0.160 (paranoid - fraud cost dominates)

Small transactions get slack. Large transactions get scrutinised. This is computed per transaction, not hardcoded.

### Four Actions, Not Two

Most systems have ALLOW and BLOCK. Sentinel has four:

| Action | When It Wins | Typical Cost |
|---|---|---|
| **ALLOW** | Low risk, any amount | ₹0 if legit |
| **CHALLENGE** | Mid risk, low-mid amount | ₹15–283 (OTP/3DS) |
| **REVIEW** | Mid-high risk, high amount | ₹75 (analyst time) |
| **BLOCK** | High risk, any amount | ₹0 if fraud |

CHALLENGE (3DS/OTP step-up) is the biggest profit lever. At ₹20,000 with p=0.16: ALLOW costs ₹3,440, BLOCK costs ₹3,436, **CHALLENGE costs ₹283** - 12× cheaper than both. Because 85% of legit customers complete the OTP, but 95% of fraudsters drop off.

---

## Architecture

```
razorpay/
├── src/sentinel/
│   ├── api.py              FastAPI: /v1/score, /v1/score/ieee, /health, /ready
│   ├── cost.py             ₹ decision engine (ALLOW/CHALLENGE/REVIEW/BLOCK)
│   ├── features/
│   │   ├── builder.py      Sparkov features (15) - shared train/serve
│   │   └── ieee_serve.py   IEEE-CIS features (150) - entity resolution
│   ├── explain.py          pred_contrib → human-readable reason codes
│   ├── store.py            SQLite velocity store + degraded fallback
│   ├── ledger.py           Append-only audit ledger
│   └── model_wrapper.py    LGBMWrapper for calibrator compatibility
├── app/
│   └── console.py          Streamlit dashboard (5 tabs)
├── config/
│   └── costs.yaml          All ₹ parameters (change one number, thresholds auto-update)
├── artifacts/
│   ├── sparkov/            Frozen model + calibrator + metrics
│   └── ieee/               IEEE model + calibrator + lookup tables
├── scripts/                Training, evaluation, analysis scripts
├── tests/                  14 tests (cost model + feature builder)
├── reports/plots/          All generated charts
├── Dockerfile              Containerised deployment
├── docker-compose.yml      API + dashboard in one command
└── DECISIONS.md            Dated engineering decision log
```

### Dual-Model API

Same API, same cost engine, two different models:

| Endpoint | Dataset | Features | Use Case |
|---|---|---|---|
| `POST /v1/score` | Sparkov | 15 (explainable) | Demo with human-readable reason codes |
| `POST /v1/score/ieee` | IEEE-CIS | 150 (entity resolution) | Real-world data validation |

Both share `cost.py`, `make_decision()`, the audit ledger, and the dashboard. This proves the architecture is dataset-agnostic.

---

## Resilience

### Degraded Mode

Kill the velocity store. The API doesn't crash - it scores with stateless features, sets `degraded: true`, and biases probability 1.5× upward to push borderline cases toward REVIEW. It degrades, it doesn't fail.

### Reject Inference

1% of would-be BLOCK decisions are silently allowed through (flagged as `holdout_allowed` in the audit ledger). This maintains an unbiased label stream for future retraining. Without it, the model only sees outcomes for transactions it approved - creating a feedback loop that makes it progressively blind.

### Retry Recovery

When a legitimate customer is blocked, ~50% try again (different card, next session). The cost model accounts for this: `cost_block` is reduced by the retry recovery rate (ρ=0.5), so the system doesn't overvalue blocking.

---

## Sensitivity Analysis

Every ₹ parameter is an estimate. The tornado chart shows which ones matter:

| Rank | Parameter | ₹ Swing (±50%) | Learnable? |
|---|---|---|---|
| #1 | Fraudster 3DS dropout | ₹171,248 | ✅ Yes (challenge outcome) |
| #2 | Chargeback fee | ₹162,028 | ✅ Yes (bank statement) |
| #3 | Challenge success rate | ₹120,001 | ✅ Yes (OTP completion) |
| #4 | Challenge friction | ₹32,063 | ❌ No |
| #5 | Churn probability | ₹27,314 | ❌ No |

The two biggest sensitivities are both observable in production. The parameters that matter most are the ones Thompson Sampling can learn.

---

## Thompson Sampling - Self-Learning Parameters

In production, Sentinel wouldn't use fixed ₹ estimates forever. We simulated Thompson Sampling over 50,000 transactions, starting with deliberately wrong parameter guesses:

| Parameter | Started At (wrong) | Learned | True Value |
|---|---|---|---|
| Chargeback fee | ₹2,000 | ₹1,531 | ₹1,500 ✅ |
| Fraudster 3DS dropout | 0.80 | 0.975 | 0.95 ✅ |
| Challenge success rate | 0.70 | 0.71 | 0.85 ⚠️ |

Exploration cost: 0.2% of decisions (114/50,000). The system self-corrects from observed outcomes.

**The identifiability caveat:** Not all parameters are learnable from passive observation. Chargeback fee and 3DS dropout are observable (bank statements, OTP outcomes). But churn probability and customer LTV take months/years to observe - those need deliberate A/B tests, not passive learning. Knowing what you CAN'T learn automatically is as important as what you can.

---

## Cost Decomposition

Where does the remaining ₹229,434 actually come from?

| Source | ₹ | % |
|---|---|---|
| Chargeback fees (₹1,500 × 85 missed frauds) | ₹127,500 | 55.8% |
| Fraud through challenge (5% get past OTP) | ₹30,995 | 13.6% |
| Challenge friction on legit customers | ₹22,590 | 9.9% |
| Fraud goods lost | ₹12,900 | 5.6% |
| False block friction + churn | ₹22,540 | 9.9% |
| False block margin loss | ₹3,451 | 1.5% |
| Review analyst time | ₹8,475 | 3.7% |

56% of remaining cost is the fixed ₹1,500 chargeback penalty - not goods lost. Reducing chargebacks by even a few would save more than any model improvement.

---

## Model Progression

| Model | PR-AUC | ₹ Cost | What It Proved |
|---|---|---|---|
| Approve everything | - | ₹4,350,825 | The floor |
| Hand-written rules | 0.35 | ₹5,290,381 | Rules alone aren't enough |
| Logistic Regression | 0.32 | ₹18,289,845 | Uncalibrated scores destroy cost models |
| LightGBM (raw) | 0.97 | ₹338,557 | Trees dominate tabular data |
| LightGBM (calibrated) | 0.95 | ₹309,487 | Calibration is mandatory |
| **Sentinel v2 (4-action)** | **0.95** | **₹229,434** | **Policy change > model change** |

The single largest improvement came from adding CHALLENGE - a formula change, no retraining. The decision policy matters as much as the model.

---

## Where I Chose Not to Use an LLM

The scorer is not an LLM. LightGBM is:
- **1000× faster** (5ms vs seconds)
- **Free** (no API costs at any scale)
- **Calibratable** (probabilities you can trust for ₹ math)
- **Auditable** (pred_contrib gives exact feature contributions)
- **Deterministic** (same input → same output, every time)

An LLM cannot produce calibrated probabilities. If the fraud probability isn't a real probability, every downstream ₹ calculation is fiction. The model's job is to output a number; the cost engine's job is to decide. Mixing them would be worse engineering, not better.

Reason codes use deterministic templates, not LLM generation. "712 km from merchant location" is more auditable than whatever GPT-4 might hallucinate.

---

## Defense-Only Compliance

This system is strictly defense-only:
- No fraud generation tooling
- No evasion testing
- No adversarial attack simulation
- No offensive capability of any kind

Robustness is evaluated through distribution shift (Sparkov vs IEEE-CIS) and sensitivity analysis (tornado chart), not through adversarial probing.

---

## How To Run

### Without Docker (recommended for judging)

```bash
# Clone
git clone https://github.com/itsTIMUS/sentinel-fraud-detection.git
cd sentinel-fraud-detection

# Setup
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt

# Start API
python -m uvicorn src.sentinel.api:app --host 0.0.0.0 --port 8000

# Start dashboard (new terminal)
streamlit run app/console.py

# Test
curl -X POST http://127.0.0.1:8000/v1/score -H "Content-Type: application/json" -d '{"trans_date_trans_time":"2020-06-21 12:14:25","cc_num":2703186189652095,"merchant":"fraud_Kirlin and Sons","category":"personal_care","amt":2.86,"first":"Jeff","last":"Elliott","gender":"M","street":"351 Darlene Green","city":"Columbia","state":"SC","zip":29209,"lat":33.9659,"long":-80.9355,"city_pop":333497,"job":"Mechanical engineer","dob":"1968-03-19","trans_num":"test123","unix_time":1371816865,"merch_lat":33.986391,"merch_long":-81.200714}'

# Run tests
python -m pytest tests/ -v
```

### With Docker

```bash
docker compose up
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

---

## Honest Limitations

- **Sparkov is simulated.** Clean patterns, predictable fraud. Real-world performance (IEEE-CIS) is lower - PR-AUC 0.62 vs 0.95.
- **Cost parameters are estimates.** Chargeback fee ₹1,500, margin 18%, churn 4% - these are industry benchmarks, not contractual values. The tornado chart shows which ones matter if they're wrong.
- **CHALLENGE (3DS) rate of 19% on IEEE-CIS is commercially high.** Real merchants fight for single-digit 3DS rates. A capacity-constrained λ solver would limit challenge volume in production.
- **No concept drift detection.** The model doesn't know when it's going stale. Production deployment would need rolling PR-AUC monitoring and automated retraining triggers.
- **Velocity features need history.** A brand-new card scores with defaults - no velocity signal. The system is weaker on first transactions.
- **I used an AI assistant (Claude) for code generation.** The architecture decisions, bug fixes, and analysis are documented in DECISIONS.md with honest timestamps. The initial LR baseline was poorly tuned, the calibrator had a pickle deserialization bug, and the first cost model underpriced REVIEW - I caught and fixed each of these.

---

## Production Roadmap

What Sentinel would need for real deployment:

1. **Thompson Sampling (live)** - replace fixed cost params with posterior sampling. Learn chargeback fees and 3DS dropout rates from real outcomes. The simulation proves convergence; production needs a feedback loop.

2. **Dynamic chargeback fee** - Visa VAMP and Mastercard ECP impose cascading penalties. The chargeback that pushes the 30-day ratio over 1% costs 10–50× a normal one. Making the fee parameter dynamic and convex would auto-tighten near the cliff.

3. **Concept drift detection** - rolling weekly PR-AUC + Population Stability Index on feature distributions. Alert when the model's view of fraud diverges from reality.

4. **Redis for velocity** - SQLite handles the demo, but concurrent writes under load need Redis. The `store.py` interface is already abstracted for this swap.

5. **Reject inference at scale** - the 1% holdout mechanism exists. Production needs a 30–120 day label maturity window before retraining on those outcomes.

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| Model | LightGBM 4.3.0 | Best on tabular data, pred_contrib for free |
| Calibration | Isotonic (sklearn) | 62% Brier improvement, makes ₹ math trustworthy |
| API | FastAPI + Uvicorn | Async, auto-docs, Pydantic validation |
| Velocity | SQLite (WAL mode) | Good enough for demo, Redis-swappable |
| Dashboard | Streamlit | 5 tabs, interactive cost sliders |
| Explainability | pred_contrib | Exact tree SHAP in microseconds |

---

## Engineering Log

See [DECISIONS.md](DECISIONS.md) for the complete dated log of every architectural decision, bug encountered, and lesson learned - from Day 0 setup through dual-model API integration.

---

Built by [itsTIMUS](https://github.com/itsTIMUS) · Heritage Institute of Technology · Razorpay Buildathon 2026