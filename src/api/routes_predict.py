"""Real-time delay prediction endpoints + health/readiness."""
import time

from fastapi import APIRouter, HTTPException, Request

from src.api.metrics import (
    ERROR_COUNT,
    MODEL_VERSION_TRAFFIC,
    PREDICTION_DISTRIBUTION,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    RISK_LEVEL_COUNT,
)
from src.api.schemas import DelayPredictRequest, DelayPredictResponse
from src.ml.predictor import predictor

router = APIRouter()


@router.get("/health")
def health():
    """Liveness: the process is up. Never checks dependencies."""
    return {"status": "ok"}


@router.get("/ready")
def ready():
    """Readiness: can we actually serve? Fails if no production model loaded."""
    if not predictor.ready():
        raise HTTPException(status_code=503, detail="production model not loaded")
    return {"status": "ready"}


@router.post("/predict/delay", response_model=DelayPredictResponse)
def predict_delay(body: DelayPredictRequest, request: Request):
    endpoint = "/predict/delay"
    start = time.perf_counter()
    try:
        prediction = predictor.predict(body.model_dump())
    except RuntimeError as exc:
        ERROR_COUNT.labels(endpoint=endpoint, error_type="model_unavailable").inc()
        REQUEST_COUNT.labels(endpoint=endpoint, status="503").inc()
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        ERROR_COUNT.labels(endpoint=endpoint, error_type="internal").inc()
        REQUEST_COUNT.labels(endpoint=endpoint, status="500").inc()
        raise

    latency_s = time.perf_counter() - start
    request.state.model_version = prediction.model_version

    REQUEST_COUNT.labels(endpoint=endpoint, status="200").inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency_s)
    PREDICTION_DISTRIBUTION.observe(prediction.delay_probability)
    MODEL_VERSION_TRAFFIC.labels(model_version=prediction.model_version).inc()
    RISK_LEVEL_COUNT.labels(risk_level=prediction.risk_level).inc()

    return DelayPredictResponse(
        shipment_id=prediction.shipment_id,
        delay_probability=prediction.delay_probability,
        risk_level=prediction.risk_level,
        model_name=prediction.model_name,
        model_version=prediction.model_version,
        latency_ms=round(latency_s * 1000, 2),
        request_id=request.state.request_id,
    )
