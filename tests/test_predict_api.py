"""Phase 1 tests: prediction API, validation, canary, batch, drift."""
import numpy as np
import pandas as pd

from src.ml.canary import route_to_canary
from src.ml.drift import detect_drift, psi


def test_predict_happy_path(client, sample_shipment):
    resp = client.post("/predict/delay", json=sample_shipment)
    assert resp.status_code == 200
    body = resp.json()
    assert body["shipment_id"] == "SHP_001"
    assert 0.0 <= body["delay_probability"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["model_version"] in {"v1", "v2"}
    assert body["request_id"].startswith("req_")
    assert body["latency_ms"] > 0


def test_predict_validation_rejects_bad_input(client, sample_shipment):
    bad = {**sample_shipment, "port_congestion_score": 3.5}  # out of [0,1]
    resp = client.post("/predict/delay", json=bad)
    assert resp.status_code == 422

    missing = {k: v for k, v in sample_shipment.items() if k != "shipment_id"}
    assert client.post("/predict/delay", json=missing).status_code == 422


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").status_code == 200


def test_canary_routing_is_sticky_and_near_10_percent():
    ids = [f"SHP_{i:05d}" for i in range(5000)]
    routed = [route_to_canary(s, canary_fraction=0.10) for s in ids]
    fraction = sum(routed) / len(routed)
    assert 0.07 < fraction < 0.13  # ~10% with hash-bucket tolerance
    # deterministic: same id always routes the same way
    assert routed == [route_to_canary(s, canary_fraction=0.10) for s in ids]


def test_batch_scoring_row_count(tmp_path):
    from scripts.run_batch_scoring import score
    from src.config import settings

    input_path = settings.data_dir / "shipments_batch.csv"
    output_path = score(input_path, tmp_path)
    n_in = len(pd.read_csv(input_path))
    out = pd.read_csv(output_path)
    assert len(out) == n_in
    assert set(out["model_version"]) == {"v1"}  # batch always uses production
    assert out["delay_probability"].between(0, 1).all()


def test_drift_detector():
    rng = np.random.default_rng(0)
    baseline = rng.beta(2, 3, 5000)
    same = rng.beta(2, 3, 1000)
    shifted = np.clip(rng.beta(2, 3, 1000) + 0.3, 0, 1)

    assert psi(baseline, same) < 0.10
    assert psi(baseline, shifted) > 0.25

    base_df = pd.DataFrame({"port_congestion_score": baseline,
                            "weather_risk_score": baseline,
                            "booking_lead_days": baseline})
    drift_df = pd.DataFrame({"port_congestion_score": shifted,
                             "weather_risk_score": same,
                             "booking_lead_days": same})
    results = {r.feature: r.status for r in detect_drift(base_df, drift_df)}
    assert results["port_congestion_score"] == "alert"
    assert results["weather_risk_score"] == "ok"
