"""FastAPI scoring service for Sentinel."""

import time
import uuid
import numpy as np
import lightgbm as lgb
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

import sys
sys.path.insert(0, ".")
from src.sentinel.cost import load_costs, make_decision
from src.sentinel.ledger import AuditLedger
from src.sentinel.features import build_features, features_to_array

# --- App ---
app = FastAPI(
    title="Sentinel — Cost-Aware Fraud Detection",
    version="0.2.0",
    description="Scores transactions and decides ALLOW / REVIEW / BLOCK based on expected ₹ cost.",
)

# --- Load artifacts ONCE at startup ---
ARTIFACTS = Path("artifacts/sparkov")
costs = load_costs()
booster = lgb.Booster(model_file=str(ARTIFACTS / "model.lgb"))
ledger = AuditLedger("data/audit.db")

# Warm the model (first predict is slow due to JIT)
dummy = np.zeros((1, 15))
booster.predict(dummy)

print("✅ LightGBM model loaded and warmed")
print("✅ Cost config loaded")
print("✅ Audit ledger initialized")


# --- Request / Response schemas ---
class TransactionRequest(BaseModel):
    """Incoming transaction to score."""
    model_config = {"protected_namespaces": ()}

    trans_date_trans_time: str = Field(..., examples=["2020-06-21 12:14:25"])
    cc_num: int = Field(..., examples=[2703186189652095])
    merchant: str = Field(..., examples=["fraud_Kirlin and Sons"])
    category: str = Field(..., examples=["personal_care"])
    amt: float = Field(..., gt=0, examples=[2.86])
    first: str = Field(..., examples=["Jeff"])
    last: str = Field(..., examples=["Elliott"])
    gender: str = Field(..., examples=["M"])
    street: str = Field(..., examples=["351 Darlene Green"])
    city: str = Field(..., examples=["Columbia"])
    state: str = Field(..., examples=["SC"])
    zip: int = Field(..., examples=[29209])
    lat: float = Field(..., examples=[33.9659])
    long: float = Field(..., examples=[-80.9355])
    city_pop: int = Field(..., examples=[333497])
    job: str = Field(..., examples=["Mechanical engineer"])
    dob: str = Field(..., examples=["1968-03-19"])
    trans_num: str = Field(..., examples=["2da90c7d74bd46a0caf3777415b3ebd3"])
    unix_time: int = Field(..., examples=[1371816865])
    merch_lat: float = Field(..., examples=[33.986391])
    merch_long: float = Field(..., examples=[-81.200714])


class DecisionResponse(BaseModel):
    """Scoring decision returned to caller."""
    model_config = {"protected_namespaces": ()}

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
    return {"status": "healthy", "model": "lgbm-v0.2"}


@app.post("/v1/score", response_model=DecisionResponse)
def score_transaction(txn: TransactionRequest):
    """Score a single transaction and return ALLOW / REVIEW / BLOCK."""
    start = time.perf_counter()

    try:
        # Parse datetime for features
        dt = pd.to_datetime(txn.trans_date_trans_time)

        # Build features using the SHARED feature builder
        txn_dict = {
            "hour": dt.hour,
            "day_of_week": dt.dayofweek,
            "amt": txn.amt,
            "lat": txn.lat,
            "long": txn.long,
            "merch_lat": txn.merch_lat,
            "merch_long": txn.merch_long,
            "city_pop": txn.city_pop,
            "dob": txn.dob,
            "trans_date_trans_time": txn.trans_date_trans_time,
            "category": txn.category,
        }

        features = build_features(txn_dict, history=None)
        feature_array = features_to_array(features).reshape(1, -1)

        # Score with native LightGBM booster
        p_fraud = float(booster.predict(feature_array)[0])

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
            "model_version": "lgbm-v0.2",
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
            model_version="lgbm-v0.2",
            latency_ms=round(latency, 2),
            degraded=False,
        )

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))