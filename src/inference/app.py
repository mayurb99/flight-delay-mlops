"""
src/inference/app.py
════════════════════════════════════════════════════════
Flight Delay Prediction — FastAPI Inference Server

Project Lecture 2: Approval + Blue/Green Deploy
Served inside Docker container on SageMaker endpoint.

Endpoints:
  GET  /health         — health check for load balancer + smoke test
  POST /predict        — single flight prediction
  POST /predict/batch  — up to 500 flights in one request

CRITICAL: engineer_features() and FEATURE_COLS are imported
from features.py — the same file used in training.
This is the guarantee of zero training-serving skew.
If features.py changes, both training AND serving change together.

SageMaker calls /invocations for predictions.
The /ping endpoint is used for health checks by SageMaker.
We map both to match SageMaker's expected interface.
════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import pickle
import logging
import tarfile
from typing import List, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# ── Find features.py ──────────────────────────────────
# In Docker container: features.py is copied alongside app.py
# This sys.path ensures the import works in both local dev and container
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from features import engineer_features, FEATURE_COLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── App startup time (for uptime calculation) ─────────
_START_TIME = time.time()
_PREDICTION_COUNT = 0
_LAST_PREDICTION_TIME = None

# ── Model paths to try ────────────────────────────────
# SageMaker extracts model.tar.gz to /opt/ml/model/
MODEL_PATHS = [
    "/opt/ml/model/model.pkl",
    os.path.join(os.path.dirname(__file__), "../../model.pkl"),
    "model.pkl",
]


# ════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ════════════════════════════════════════════════════════

class FlightRequest(BaseModel):
    """
    Single flight prediction request.
    Mirrors the raw BTS dataset columns available 2 hours before departure.
    At serving time we do NOT have ARR_DELAY — that is what we are predicting.
    """
    FL_DATE:           str   = Field(..., description="Flight date YYYY-MM-DD")
    OP_CARRIER:        str   = Field(..., description="Airline carrier code e.g. AA, UA, DL")
    ORIGIN:            str   = Field(..., description="Origin airport IATA code e.g. JFK")
    DEST:              str   = Field(..., description="Destination airport IATA code e.g. LAX")
    CRS_DEP_TIME:      int   = Field(..., ge=0,   le=2359, description="Scheduled departure time HHMM")
    CRS_ARR_TIME:      int   = Field(..., ge=0,   le=2359, description="Scheduled arrival time HHMM")
    CRS_ELAPSED_TIME:  float = Field(..., gt=0,   description="Scheduled elapsed time in minutes")
    DISTANCE:          float = Field(..., gt=0,   description="Flight distance in miles")

    @field_validator("OP_CARRIER")
    @classmethod
    def carrier_not_empty(cls, v):
        if not v.strip():
            raise ValueError("OP_CARRIER cannot be empty")
        return v.strip().upper()

    @field_validator("ORIGIN", "DEST")
    @classmethod
    def airport_code_format(cls, v):
        v = v.strip().upper()
        if not (2 <= len(v) <= 4):
            raise ValueError(f"Airport code must be 2-4 characters: {v}")
        return v


class PredictionResponse(BaseModel):
    delayed:           int   # 0 = on time, 1 = delayed
    delay_probability: float # probability of 15+ min delay
    risk_level:        str   # LOW / MEDIUM / HIGH
    confidence:        str   # HIGH / MEDIUM / LOW confidence in prediction
    flight:            str   # "{ORIGIN}-{DEST} {OP_CARRIER}"


class BatchRequest(BaseModel):
    flights: List[FlightRequest] = Field(..., max_length=500)


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    total:       int
    delayed_count: int
    processing_ms: float


class HealthResponse(BaseModel):
    status:          str   # healthy / degraded
    model_loaded:    bool
    uptime_seconds:  float
    predictions_served: int
    last_prediction: Optional[str]
    model_path:      Optional[str]


# ════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════

_model = None
_model_path = None


def load_model():
    """
    Load model.pkl from known paths.
    Handles model.tar.gz extraction if needed.
    Cached after first load.
    """
    global _model, _model_path

    if _model is not None:
        return _model

    # Try direct pkl paths first
    for path in MODEL_PATHS:
        if os.path.exists(path):
            with open(path, "rb") as f:
                _model = pickle.load(f)
            _model_path = path
            logger.info(f"Model loaded from: {path}")
            return _model

    # Try extracting from tar.gz
    tar_paths = [
        "/opt/ml/model/model.tar.gz",
        "model.tar.gz",
    ]
    for tar_path in tar_paths:
        if os.path.exists(tar_path):
            extract_dir = os.path.dirname(tar_path)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(extract_dir)
            pkl_path = os.path.join(extract_dir, "model.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    _model = pickle.load(f)
                _model_path = pkl_path
                logger.info(f"Model extracted and loaded from: {tar_path}")
                return _model

    raise FileNotFoundError(
        f"No model.pkl found. Tried: {MODEL_PATHS}"
    )


def classify_risk(probability: float) -> str:
    """Map probability to risk tier."""
    if probability < 0.25:  return "LOW"
    if probability < 0.55:  return "MEDIUM"
    return "HIGH"


def classify_confidence(probability: float) -> str:
    """Higher confidence when probability is far from 0.5."""
    distance = abs(probability - 0.5)
    if distance >= 0.35:  return "HIGH"
    if distance >= 0.15:  return "MEDIUM"
    return "LOW"


def predict_single(flight: FlightRequest) -> PredictionResponse:
    """Run one prediction. Used by both /predict and /predict/batch."""
    global _PREDICTION_COUNT, _LAST_PREDICTION_TIME

    model = load_model()

    # Build raw DataFrame — same structure as training data
    raw = pd.DataFrame([{
        "FL_DATE":          flight.FL_DATE,
        "OP_CARRIER":       flight.OP_CARRIER,
        "ORIGIN":           flight.ORIGIN,
        "DEST":             flight.DEST,
        "CRS_DEP_TIME":     flight.CRS_DEP_TIME,
        "CRS_ARR_TIME":     flight.CRS_ARR_TIME,
        "CRS_ELAPSED_TIME": flight.CRS_ELAPSED_TIME,
        "DISTANCE":         flight.DISTANCE,
        "DEP_DELAY":        0.0,  # not known at prediction time — placeholder
        "ARR_DELAY":        0.0,  # target we are predicting
    }])

    # Apply same feature engineering as training
    features = engineer_features(raw)[FEATURE_COLS]

    # Predict
    prediction   = int(model.predict(features)[0])
    probability  = round(float(model.predict_proba(features)[0, 1]), 4)

    _PREDICTION_COUNT += 1
    _LAST_PREDICTION_TIME = datetime.utcnow().isoformat()

    return PredictionResponse(
        delayed           = prediction,
        delay_probability = probability,
        risk_level        = classify_risk(probability),
        confidence        = classify_confidence(probability),
        flight            = f"{flight.ORIGIN}-{flight.DEST} {flight.OP_CARRIER}",
    )


# ════════════════════════════════════════════════════════
# APP + ENDPOINTS
# ════════════════════════════════════════════════════════

app = FastAPI(
    title="Flight Delay Prediction API",
    description="Predicts whether a US domestic flight will be delayed 15+ minutes",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """Pre-load model at startup so first request is not slow."""
    try:
        load_model()
        logger.info("✓ Model pre-loaded at startup")
    except FileNotFoundError as e:
        logger.warning(f"Model not found at startup: {e}")


@app.get("/health", response_model=HealthResponse)
@app.get("/ping", response_model=HealthResponse)  # SageMaker health check endpoint
def health():
    """
    Health check endpoint.
    Used by:
      - SageMaker load balancer (/ping)
      - deploy.yml smoke test (/health)
      - Monitoring dashboards
    Returns HTTP 200 when model is loaded.
    Returns HTTP 503 when model is not available.
    """
    try:
        load_model()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False

    response = HealthResponse(
        status           = "healthy" if model_loaded else "degraded",
        model_loaded     = model_loaded,
        uptime_seconds   = round(time.time() - _START_TIME, 1),
        predictions_served = _PREDICTION_COUNT,
        last_prediction  = _LAST_PREDICTION_TIME,
        model_path       = _model_path,
    )

    if not model_loaded:
        raise HTTPException(status_code=503, detail=response.dict())

    return response


@app.post("/predict", response_model=PredictionResponse)
@app.post("/invocations", response_model=PredictionResponse)  # SageMaker invocation endpoint
def predict(flight: FlightRequest):
    """
    Predict delay probability for a single flight.

    Input: flight details known 2 hours before departure
    Output: delay prediction, probability, risk level

    SageMaker calls /invocations — we map both paths here.
    """
    try:
        return predict_single(flight)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    """
    Predict delay probability for up to 500 flights.

    Returns predictions in same order as input flights.
    Reports total delayed count for quick overview.
    """
    start = time.perf_counter()

    try:
        predictions = [predict_single(f) for f in batch.flights]
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")
    except Exception as e:
        logger.exception(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return BatchResponse(
        predictions   = predictions,
        total         = len(predictions),
        delayed_count = sum(p.delayed for p in predictions),
        processing_ms = elapsed_ms,
    )
