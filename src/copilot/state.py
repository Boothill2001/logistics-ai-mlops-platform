"""Copilot graph state.

One flat TypedDict flowing through every node. Nodes only fill the fields
they own; routing decisions read `status`, `intent` and `route`.
"""
from typing import Any, TypedDict


class CopilotState(TypedDict, total=False):
    # request context
    user_id: str
    message: str
    request_id: str

    # control flow
    status: str          # "ok" | "denied" | "needs_clarification" | "pending_approval" | "error"
    intent: str          # knowledge_search | ml_prediction | batch_query | dangerous_action | missing_info
    plan: list[str]      # human-readable plan steps (LLM plans, backend executes)
    route: str           # executor node chosen by RouterDecisionNode
    replanned: bool
    error: str

    # extracted entities
    shipment_id: str
    customer_id: str

    # execution results
    rag_text: str
    rag_sources: list[str]
    prediction: dict[str, Any]
    batch_rows: list[dict[str, Any]]
    email_draft: str
    draft_id: str

    # final
    response: str
