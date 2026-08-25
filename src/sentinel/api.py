"""FastAPI scoring service for Sentinel."""

import time
import uuid
import numpy as np
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

import sys
sys.path.insert(0, ".")
from src.sentinel.cost import load_costs, make_decision
from src.sentinel.ledger import AuditLedger

# --- App ---
app = FastAPI(
    title="Sentinel — Cost-Aware Fraud Detection",
    version="0.1.0",
    description="Scores transactions and decides ALLOW / REVIEW / BLOCK based on expected ₹ cost.",
)

# --- Load artifacts ONCE at startup ---
ARTIFACTS = Path("artifacts/sparkov")
costs = load_costs()

# Load LR model (will be swapped to LightGBM on Day 2)
model = joblib.load(ARTIFACTS / "model_lr.joblib")
label_encoder = joblib.load(ARTIFACTS / "label_encoder.joblib")

print(f"✅ Model loaded from {ARTIFACTS / 'model_lr.joblib'}")
print(f"✅ Cost config loaded")
ledger = AuditLedger("data/audit.db")
print(f"✅ Audit ledger initialized")


# --- Request / Response schemas ---
class TransactionRequest(BaseModel):
    """Incoming transaction to score."""
    trans_date_trans_time: str = Field(..., example="2020-06-21 12:14:25")
    cc_num: int = Field(..., example=2703186189652095)
    merchant: str = Field(..., example="fraud_Kirlin and Sons")
    category: str = Field(..., example="personal_care")
    amt: float = Field(..., gt=0, example=2.86)
    first: str = Field(..., example="Jeff")
    last: str = Field(..., example="Elliott")
    gender: str = Field(..., example="M")
    street: str = Field(..., example="351 Darlene Green")
    city: str = Field(..., example="Columbia")
    state: str = Field(..., example="SC")
    zip: int = Field(..., example=29209)
    lat: float = Field(..., example=33.9659)
    long: float = Field(..., example=-80.9355)
    city_pop: int = Field(..., example=333497)
    job: str = Field(..., example="Mechanical engineer")
    dob: str = Field(..., example="1968-03-19")
    trans_num: str = Field(..., example="2da90c7d74bd46a0caf3777415b3ebd3")
    unix_time: int = Field(..., example=1371816865)
    merch_lat: float = Field(..., example=33.986391)
    merch_long: float = Field(..., example=-81.200714)


class DecisionResponse(BaseModel):
    """Scoring decision returned to caller."""
    decision_id: str
    decision: str
    risk_probability: float
    expected_loss_if_allowed_inr: float
    expected_loss_if_blocked_inr: float
    expected_loss_if_reviewed_inr: float
    amount_inr: float
    model_version: str
    latency_ms: float
    degraded: bool


# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "healthy", "model": "lr-v0.1"}


@app.post("/v1/score", response_model=DecisionResponse)
def score_transaction(txn: TransactionRequest):
    """Score a single transaction and return ALLOW / REVIEW / BLOCK."""
    start = time.perf_counter()

    try:
        # Build simple features (same as baselines.py)
        import pandas as pd
        hour = pd.to_datetime(txn.trans_date_trans_time).hour
        day_of_week = pd.to_datetime(txn.trans_date_trans_time).dayofweek

        # Encode category
        cat_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        cat_enc = cat_map.get(txn.category, -1)

        features = np.array([[
            np.log1p(txn.amt),
            hour,
            day_of_week,
            1.0 if (hour >= 22 or hour <= 5) else 0.0,
            np.log1p(txn.city_pop),
            cat_enc,
            txn.lat,
            txn.long,
        ]])

        # Score
        p_fraud = float(model.predict_proba(features)[:, 1][0])

        # Decide
        result = make_decision(p_fraud=p_fraud, amount=txn.amt, costs=costs)

        latency = (time.perf_counter() - start) * 1000
        dec_id = f"dec_{uuid.uuid4().hex[:12]}"
        # Log to audit ledger
        ledger.log({
            "decision_id": dec_id,
            "trans_num": txn.trans_num,
            "amount_inr": txn.amt,
            "risk_probability": result["risk_probability"],
            "decision": result["decision"],
            "expected_loss_if_allowed_inr": result["expected_loss_if_allowed_inr"],
            "expected_loss_if_blocked_inr": result["expected_loss_if_blocked_inr"],
            "expected_loss_if_reviewed_inr": result["expected_loss_if_reviewed_inr"],
            "model_version": "lr-v0.1",
            "latency_ms": round(latency, 2),
            "degraded": False,
        })

        return DecisionResponse(
            decision_id=dec_id,
            decision=result["decision"],
            risk_probability=result["risk_probability"],
            expected_loss_if_allowed_inr=result["expected_loss_if_allowed_inr"],
            expected_loss_if_blocked_inr=result["expected_loss_if_blocked_inr"],
            expected_loss_if_reviewed_inr=result["expected_loss_if_reviewed_inr"],
            amount_inr=result["amount_inr"],
            model_version="lr-v0.1",
            latency_ms=round(latency, 2),
            degraded=False,
        )

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))