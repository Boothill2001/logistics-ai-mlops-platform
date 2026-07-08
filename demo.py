"""End-to-end demo script.

Runs against a live API (default http://localhost:8000):
  1. health/ready
  2. real-time prediction (+ canary distribution over 40 shipments)
  3. batch scoring + drift detection (normal vs post-holiday drifted data)
  4. copilot: all 5 intents, exercised as 3 different users
  5. human approval flow
  6. audit log tail

Start the API first:
  .venv/Scripts/python -m uvicorn src.api.main:app --port 8000
or: docker compose up
"""
import json
import sys
from collections import Counter
from pathlib import Path

import httpx

# Windows consoles often default to cp1252 — force UTF-8 so Vietnamese prints
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE = "http://localhost:8000"
client = httpx.Client(base_url=BASE, timeout=120)


def section(title: str):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def main():
    section("1. Health & readiness")
    print("GET /health ->", client.get("/health").json())
    print("GET /ready  ->", client.get("/ready").json())

    section("2. Real-time prediction + canary routing")
    shipment = {
        "shipment_id": "SHP_00001", "customer_id": "CUS_A",
        "origin_port": "SGN", "destination_port": "SIN", "container_type": "40HC",
        "booking_lead_days": 3, "transshipment_count": 2,
        "port_congestion_score": 0.85, "weather_risk_score": 0.60,
        "historical_delay_rate": 0.30, "carrier_reliability_score": 0.45,
    }
    resp = client.post("/predict/delay", json=shipment).json()
    print(json.dumps(resp, indent=2))

    versions = Counter()
    for i in range(40):
        r = client.post("/predict/delay", json={**shipment, "shipment_id": f"SHP_{i:05d}"}).json()
        versions[r["model_version"]] += 1
    print(f"Canary split over 40 shipments: {dict(versions)} (expect ~90/10 v1/v2)")

    section("3. Batch scoring + drift detection")
    from scripts.run_batch_scoring import score
    from src.config import settings
    score(settings.data_dir / "shipments_batch.csv", settings.data_dir / "batch_output")
    print("\n--- post-holiday drifted data (expect PSI alert on congestion) ---")
    score(settings.data_dir / "shipments_drifted.csv", settings.data_dir / "batch_output")

    section("4. Copilot — 5 intents, 3 users")
    conversations = [
        ("sales_001", "Summarize the contract of customer A."),
        ("intern_001", "Summarize the contract of customer A."),       # -> denied
        ("intern_001", "What is the delay compensation policy?"),      # public doc -> ok
        ("ops_001", "Is there any private note about customer A?"),    # -> no leak
        ("ops_001", "Does shipment SHP_00001 have delay risk?"),
        ("sales_001", "Top 5 shipments with highest delay risk today?"),
        ("ops_001", "Create a report for that customer."),             # -> clarify
        ("ops_001", "Send email to customer A about a 3-day shipment delay."),
    ]
    draft_id = None
    for user_id, message in conversations:
        r = client.post("/copilot/chat", headers={"X-User-Id": user_id},
                        json={"message": message}).json()
        print(f"\n[{user_id}] {message}")
        print(f"  intent={r['intent']} status={r['status']} sources={r['sources']}")
        print(f"  {r['response'][:300]}")
        if r.get("draft_id"):
            draft_id = r["draft_id"]

    section("5. Human-in-the-loop approval")
    denied = client.post("/copilot/approve", headers={"X-User-Id": "intern_001"},
                         json={"draft_id": draft_id, "approve": True})
    print(f"intern_001 tries to approve -> HTTP {denied.status_code} (expected 403)")
    approved = client.post("/copilot/approve", headers={"X-User-Id": "ops_001"},
                           json={"draft_id": draft_id, "approve": True}).json()
    print(f"ops_001 approves -> {approved['status']}: {approved['detail']}")

    section("6. Audit log (last 5 events)")
    from src.audit.log import read_audit
    for event in read_audit(limit=5):
        print(json.dumps(event, ensure_ascii=False)[:200])

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
