import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Tests must be offline and deterministic — never hit a real LLM API,
# regardless of what the local .env says. Must be set before src imports.
os.environ["LLM_PROVIDER"] = "mock"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def client():
    from src.api.main import app
    return TestClient(app)


@pytest.fixture()
def sample_shipment():
    return {
        "shipment_id": "SHP_001",
        "customer_id": "CUS_A",
        "origin_port": "SGN",
        "destination_port": "SIN",
        "container_type": "40HC",
        "booking_lead_days": 5,
        "transshipment_count": 1,
        "port_congestion_score": 0.72,
        "weather_risk_score": 0.35,
        "historical_delay_rate": 0.18,
        "carrier_reliability_score": 0.81,
    }
