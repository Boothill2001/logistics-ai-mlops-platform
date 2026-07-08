"""The 14 copilot graph nodes.

Design boundary enforced throughout: the LLM classifies, plans, and drafts
text. It never executes — predictions come from the ML predictor, documents
from permission-filtered retrieval, batch numbers from the scored CSV. If the
LLM hallucinates, it can only hallucinate *words*, never actions or numbers.
"""
import logging
import re

import pandas as pd

from src.audit.log import audit
from src.auth.rbac import DENIED_MESSAGE, can_access_customer
from src.auth.users import get_user
from src.config import settings
from src.copilot.drafts import draft_store
from src.copilot.llm import get_llm
from src.copilot.state import CopilotState
from src.ml.predictor import predictor
from src.rag.answerer import answer_question
from src.rag.store import DocStore

logger = logging.getLogger("copilot.nodes")

from src.copilot.intent_rules import (  # noqa: E402
    CUSTOMER_RE as _CUSTOMER_RE,
    INTENTS,
    SHIPMENT_RE as _SHIPMENT_RE,
    classify_intent_rules,
    extract_customer as _extract_customer,
)

INTENT_PROMPT = """Classify the user intent into exactly one label:
- knowledge_search: question about contracts, invoices, policies, procedures, documents
- ml_prediction: asks about delay risk of ONE specific shipment (has a shipment id)
- batch_query: analytics over many shipments (top N, highest risk, this week)
- dangerous_action: asks to send/draft an email or notify a customer
- missing_info: the target is truly unspecified (no customer name, no shipment id, no period). If a customer or shipment IS named, it is NOT missing_info.

Examples:
"Summarize the contract of customer A." -> knowledge_search
"Does shipment SHP_00123 have delay risk?" -> ml_prediction
"Top 5 shipments with highest delay risk today?" -> batch_query
"Send email to customer A about a 3-day shipment delay." -> dangerous_action
"Create a report for that customer." -> missing_info

Message: {message}

Reply with only the label."""

# --- lazy singletons --------------------------------------------------------
_doc_store: DocStore | None = None


def _store() -> DocStore:
    global _doc_store
    if _doc_store is None:
        _doc_store = DocStore()
    return _doc_store


# --- nodes ------------------------------------------------------------------

def input_validation_node(state: CopilotState) -> CopilotState:
    message = (state.get("message") or "").strip()
    if not message or len(message) > 2000:
        return {"status": "error", "error": "Message must be 1-2000 characters."}
    if get_user(state["user_id"]) is None:
        return {"status": "error", "error": "Unknown user."}
    return {"status": "ok", "message": message}


def intent_classification_node(state: CopilotState) -> CopilotState:
    llm = get_llm()
    label = llm.complete(INTENT_PROMPT.format(message=state["message"])).strip().lower()
    if label not in INTENTS:
        label = classify_intent_rules(state["message"])
    updates: CopilotState = {"intent": label}
    if sid := _SHIPMENT_RE.search(state["message"]):
        updates["shipment_id"] = sid.group(1).upper()
    if cid := _extract_customer(state["message"]):
        updates["customer_id"] = cid
    return updates


def permission_policy_node(state: CopilotState) -> CopilotState:
    """Early customer-level check. The RAG metadata filter remains the
    backstop — defense in depth, not a single gate."""
    user = get_user(state["user_id"])
    customer_id = state.get("customer_id")
    if customer_id and not can_access_customer(user, customer_id):
        audit("permission_denied", user.user_id,
              customer_id=customer_id, message=state["message"])
        return {"status": "denied", "response": DENIED_MESSAGE}
    return {}


def planner_node(state: CopilotState) -> CopilotState:
    plans = {
        "knowledge_search": ["retrieve permitted documents", "answer with citations"],
        "ml_prediction": ["look up shipment features", "run production/canary model", "explain risk"],
        "batch_query": ["read last night's scored batch", "filter to user's customers", "rank by risk"],
        "dangerous_action": ["draft email with LLM", "hold for human approval", "never auto-send"],
        "missing_info": ["ask a clarifying question"],
    }
    return {"plan": plans[state["intent"]]}


def router_decision_node(state: CopilotState) -> CopilotState:
    routes = {
        "knowledge_search": "rag_retrieval",
        "ml_prediction": "ml_inference",
        "batch_query": "batch_query",
        "dangerous_action": "email_draft",
        "missing_info": "final_response",
    }
    return {"route": routes[state["intent"]]}


def rag_retrieval_node(state: CopilotState) -> CopilotState:
    user = get_user(state["user_id"])
    result = answer_question(state["message"], user, _store(), get_llm())
    return {"rag_text": result.text, "rag_sources": result.sources}


def ml_inference_node(state: CopilotState) -> CopilotState:
    shipment_id = state.get("shipment_id")
    batch = pd.read_csv(settings.data_dir / "shipments_batch.csv")
    row = batch[batch["shipment_id"].str.upper() == (shipment_id or "").upper()]
    if row.empty:
        return {"error": f"Shipment {shipment_id} not found in current data."}
    prediction = predictor.predict(row.iloc[0].to_dict())
    return {"prediction": {
        "shipment_id": prediction.shipment_id,
        "delay_probability": prediction.delay_probability,
        "risk_level": prediction.risk_level,
        "model_version": prediction.model_version,
        "drivers": _explain_drivers(row.iloc[0].to_dict()),
    }}


def _explain_drivers(record: dict) -> list[str]:
    """Simple, honest feature-based explanation (no SHAP in the demo)."""
    drivers = []
    if record["port_congestion_score"] > 0.6:
        drivers.append(f"high port congestion ({record['port_congestion_score']:.2f})")
    if record["weather_risk_score"] > 0.5:
        drivers.append(f"elevated weather risk ({record['weather_risk_score']:.2f})")
    if record["booking_lead_days"] < 5:
        drivers.append(f"short booking lead time ({record['booking_lead_days']} days)")
    if record["carrier_reliability_score"] < 0.5:
        drivers.append(f"low carrier reliability ({record['carrier_reliability_score']:.2f})")
    if record["historical_delay_rate"] > 0.3:
        drivers.append(f"poor historical performance ({record['historical_delay_rate']:.2f})")
    return drivers or ["no single dominant risk factor"]


def batch_query_node(state: CopilotState) -> CopilotState:
    path = settings.data_dir / "batch_output" / "predictions_latest.csv"
    if not path.exists():
        return {"error": "No batch predictions available — nightly job has not run."}
    user = get_user(state["user_id"])
    df = pd.read_csv(path)
    # Row-level security: users only see shipments of customers they may access
    df = df[df["customer_id"].isin(user.customer_access)]
    top = df.nlargest(5, "delay_probability")
    return {"batch_rows": top[["shipment_id", "customer_id", "delay_probability",
                               "risk_level"]].to_dict("records")}


def email_draft_node(state: CopilotState) -> CopilotState:
    llm = get_llm()
    draft_text = llm.complete(
        "Write a professional delay notification email to the customer.\n"
        f"Context from user request: {state['message']}\n"
        "Do not invent specific dates or compensation amounts."
    )
    draft = draft_store.create(state["user_id"], draft_text)
    audit("email_draft_created", state["user_id"],
          draft_id=draft.draft_id, request=state["message"])
    return {"email_draft": draft_text, "draft_id": draft.draft_id}


def human_approval_node(state: CopilotState) -> CopilotState:
    """Interrupt point: the graph parks the action as pending. Nothing is
    sent — a human resolves it later via POST /copilot/approve."""
    return {"status": "pending_approval"}


def observation_check_node(state: CopilotState) -> CopilotState:
    """Did the chosen executor actually produce a result?"""
    route = state.get("route")
    produced = {
        "rag_retrieval": bool(state.get("rag_text")),
        "ml_inference": bool(state.get("prediction")) or bool(state.get("error")),
        "batch_query": state.get("batch_rows") is not None or bool(state.get("error")),
        "email_draft": bool(state.get("draft_id")),
        "final_response": True,
    }.get(route, True)
    if not produced and not state.get("replanned"):
        return {"status": "replan_needed"}
    return {}


def replan_node(state: CopilotState) -> CopilotState:
    """One retry, then degrade honestly. Fallback route is RAG (safest —
    permission-filtered, grounded), never a guess."""
    logger.info("replanning for request %s", state.get("request_id"))
    return {"replanned": True, "route": "rag_retrieval", "status": "ok"}


def final_response_node(state: CopilotState) -> CopilotState:
    if state.get("status") == "denied":
        return {}
    intent = state.get("intent")

    if intent == "missing_info":
        return {"status": "needs_clarification",
                "response": "Which customer and what time period would you like the report for?"}

    if state.get("error") and not state.get("rag_text"):
        return {"status": "error", "response": state["error"]}

    if intent == "ml_prediction" and state.get("prediction"):
        p = state["prediction"]
        return {"status": "ok", "response": (
            f"Shipment {p['shipment_id']}: delay probability {p['delay_probability']:.0%} "
            f"({p['risk_level']} risk, model {p['model_version']}). "
            f"Main drivers: {', '.join(p['drivers'])}."
        )}

    if intent == "batch_query" and state.get("batch_rows") is not None:
        if not state["batch_rows"]:
            return {"status": "ok", "response": "No shipments visible to you in the latest batch."}
        lines = [f"{r['shipment_id']} ({r['customer_id']}): "
                 f"{r['delay_probability']:.0%} [{r['risk_level']}]"
                 for r in state["batch_rows"]]
        return {"status": "ok", "response": "Top delay risks in latest batch:\n" + "\n".join(lines)}

    if intent == "dangerous_action" and state.get("draft_id"):
        return {"response": (
            "Email draft created (NOT sent — requires human approval, "
            f"draft_id={state['draft_id']}):\n\n{state['email_draft']}"
        )}

    # knowledge_search / fallback
    return {"status": state.get("status", "ok"),
            "response": state.get("rag_text", "Insufficient data to answer.")}


def logging_audit_node(state: CopilotState) -> CopilotState:
    audit("copilot_interaction", state["user_id"],
          request_id=state.get("request_id"),
          intent=state.get("intent"),
          route=state.get("route"),
          status=state.get("status"),
          sources=state.get("rag_sources", []),
          draft_id=state.get("draft_id"))
    return {}
