"""Phase 3+4 tests: copilot routing, human-in-the-loop, cross-customer RBAC."""
import pandas as pd
import pytest

from src.auth.rbac import DENIED_MESSAGE
from src.config import settings
from src.copilot.drafts import draft_store
from src.copilot.graph import run_copilot
from src.copilot.intent_rules import classify_intent_rules


@pytest.fixture(scope="module", autouse=True)
def ensure_batch_output():
    """batch_query needs last night's scored CSV."""
    path = settings.data_dir / "batch_output" / "predictions_latest.csv"
    if not path.exists():
        from scripts.run_batch_scoring import score
        score(settings.data_dir / "shipments_batch.csv", settings.data_dir / "batch_output")


def _shipment_id_of(customer: str) -> str:
    df = pd.read_csv(settings.data_dir / "shipments_batch.csv")
    return df[df["customer_id"] == customer].iloc[0]["shipment_id"]


def test_intent_classification_five_sample_questions():
    cases = {
        "Summarize the contract of customer A.": "knowledge_search",
        "Does shipment SHP_00001 have delay risk?": "ml_prediction",
        "Top 5 shipments with highest delay risk today?": "batch_query",
        "Send email to customer A about a 3-day shipment delay.": "dangerous_action",
        "Create a report for that customer.": "missing_info",
    }
    for message, expected in cases.items():
        assert classify_intent_rules(message) == expected, message


def test_ml_prediction_route_returns_real_model_output():
    shipment_id = _shipment_id_of("CUS_A")
    result = run_copilot("ops_001", f"Does shipment {shipment_id} have delay risk?", "req_t1")
    assert result["intent"] == "ml_prediction"
    assert result["status"] == "ok"
    assert result["prediction"]["model_version"] in {"v1", "v2"}
    assert shipment_id in result["response"]


def test_batch_query_is_row_level_filtered():
    result = run_copilot("sales_001", "Top 5 shipments with highest risk today?", "req_t2")
    assert result["intent"] == "batch_query"
    assert all(row["customer_id"] == "CUS_A" for row in result["batch_rows"])


def test_dangerous_action_creates_draft_never_sends():
    result = run_copilot("ops_001", "Send email to customer A about a 3-day shipment delay.", "req_t3")
    assert result["intent"] == "dangerous_action"
    assert result["status"] == "pending_approval"
    draft = draft_store.get(result["draft_id"])
    assert draft.status == "pending_approval"
    assert "NOT sent" in result["response"]


def test_approval_flow(client):
    chat = client.post("/copilot/chat", headers={"X-User-Id": "ops_001"},
                       json={"message": "Send email to customer A about the delay."})
    draft_id = chat.json()["draft_id"]
    assert chat.json()["status"] == "pending_approval"

    denied = client.post("/copilot/approve", headers={"X-User-Id": "intern_001"},
                         json={"draft_id": draft_id, "approve": True})
    assert denied.status_code == 403

    approved = client.post("/copilot/approve", headers={"X-User-Id": "ops_001"},
                           json={"draft_id": draft_id, "approve": True})
    assert approved.json()["status"] == "approved"


def test_missing_info_asks_clarifying_question():
    result = run_copilot("ops_001", "Create a report for that customer.", "req_t4")
    assert result["status"] == "needs_clarification"
    assert "customer" in result["response"].lower()


def test_cross_customer_rbac_denied():
    result = run_copilot("sales_001", "Summarize contract of customer B.", "req_t5")
    assert result["status"] == "denied"
    assert result["response"] == DENIED_MESSAGE
    ok = run_copilot("ops_001", "Summarize contract of customer B.", "req_t6")
    assert ok["status"] != "denied"


def test_unknown_user_rejected(client):
    resp = client.post("/copilot/chat", headers={"X-User-Id": "hacker_999"},
                       json={"message": "hello"})
    assert resp.status_code == 401
