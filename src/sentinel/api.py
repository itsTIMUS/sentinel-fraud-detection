"""FastAPI scoring service for Sentinel."""

import time
import uuid
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

import sys
sys.path.insert(0, ".")
from src.sentinel.cost import load_costs, make_decision
from src.sentinel.ledger import AuditLedger
from src.sentinel.features import build_features, features_to_array
from src.sentinel.model_wrapper import LGBMWrapper
from src.sentinel.explain import get_reason_codes
from src.sentinel.store import VelocityStore, DegradedVelocityStore

# --- App ---
app = FastAPI(
    title="Sentinel — Cost-Aware Fraud Detection",
    version="0.4.0",
    description="Scores transactions and decides ALLOW / REVIEW / BLOCK based on expected ₹ cost.",
)

# --- Load artifacts ONCE at startup ---
ARTIFACTS = Path("artifacts/sparkov")
costs = load_costs()
booster = lgb.Booster(model_file=str(ARTIFACTS / "model.lgb"))
calibrator = joblib.load(ARTIFACTS / "calibrator.joblib")
ledger = AuditLedger("data/audit.db")

# Velocity store with fallback
try:
    velocity_store = VelocityStore("data/velocity.db")
    print("✅ Velocity store initialized")
except Exception:
    velocity_store = DegradedVelocityStore()
    print("⚠️ Velocity store unavailable — running in degraded mode")

# Warm the model
dummy = np.zeros((1, 15))
booster.predict(dummy)

print("✅ LightGBM model loaded and warmed")
print("✅ Calibrator loaded")
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
    reason_codes: list = []


# --- Endpoints ---
@app.get("/health")
def health():
    """Liveness check — is the service running?"""
    return {"status": "healthy", "model": "lgbm-v0.3-calibrated"}


@app.get("/ready")
def ready():
    """Readiness check — is everything loaded and reachable?"""
    return {
        "ready": True,
        "model_loaded": booster is not None,
        "calibrator_loaded": calibrator is not None,
        "velocity_store": velocity_store.is_available,
        "ledger": True,
    }


@app.post("/v1/score", response_model=DecisionResponse)
def score_transaction(txn: TransactionRequest):
    """Score a single transaction and return ALLOW / REVIEW / BLOCK."""
    start = time.perf_counter()

    try:
        dt = pd.to_datetime(txn.trans_date_trans_time)

        # Get velocity history (returns None if store is unavailable)
        history = velocity_store.get_history(str(txn.cc_num), txn.unix_time)
        is_degraded = not velocity_store.is_available

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

        features = build_features(txn_dict, history=history)
        feature_array = features_to_array(features).reshape(1, -1)

        # Score with calibrated model
        p_fraud = float(calibrator.predict_proba(feature_array)[:, 1][0])

        # In degraded mode, bias toward REVIEW for safety
        if is_degraded and p_fraud > 0.01:
            p_fraud = min(p_fraud * 1.5, 0.999)

        # Decide
        result = make_decision(p_fraud=p_fraud, amount=txn.amt, costs=costs)

        # Get reason codes
        reasons = get_reason_codes(booster, feature_array, features, top_k=3)

        # Record transaction in velocity store for future lookups
        velocity_store.record(
            card_id=str(txn.cc_num),
            unix_time=txn.unix_time,
            amount=txn.amt,
            merchant=txn.merchant,
        )

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
            "model_version": "lgbm-v0.3-calibrated",
            "latency_ms": round(latency, 2),
            "degraded": is_degraded,
            "reason_codes": str(reasons),
        })

        return DecisionResponse(
            decision_id=dec_id,
            decision=result["decision"],
            risk_probability=result["risk_probability"],
            expected_loss_if_allowed_inr=result["expected_loss_if_allowed_inr"],
            expected_loss_if_blocked_inr=result["expected_loss_if_blocked_inr"],
            expected_loss_if_reviewed_inr=result["expected_loss_if_reviewed_inr"],
            amount_inr=result["amount_inr"],
            model_version="lgbm-v0.3-calibrated",
            latency_ms=round(latency, 2),
            degraded=is_degraded,
            reason_codes=reasons,
        )

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))