"""LangGraph wiring for the copilot.

Flow:
  input_validation -> intent_classification -> permission_policy -> planner
    -> router_decision -> [rag_retrieval | ml_inference | batch_query
                           | email_draft -> human_approval | final_response]
    -> observation_check -> (replan -> router loop, once) -> final_response
    -> logging_audit -> END

Early exits: validation error or permission denial skip straight to the
final response (still passing through audit — denials are logged too).
"""
from functools import lru_cache

from langgraph.graph import END, StateGraph

from src.copilot import nodes
from src.copilot.state import CopilotState


def _after_validation(state: CopilotState) -> str:
    return "final_response" if state.get("status") == "error" else "intent_classification"


def _after_policy(state: CopilotState) -> str:
    return "final_response" if state.get("status") == "denied" else "planner"


def _route_executor(state: CopilotState) -> str:
    return state["route"]


def _after_observation(state: CopilotState) -> str:
    return "replan" if state.get("status") == "replan_needed" else "final_response"


def build_graph():
    g = StateGraph(CopilotState)

    g.add_node("input_validation", nodes.input_validation_node)
    g.add_node("intent_classification", nodes.intent_classification_node)
    g.add_node("permission_policy", nodes.permission_policy_node)
    g.add_node("planner", nodes.planner_node)
    g.add_node("router_decision", nodes.router_decision_node)
    g.add_node("rag_retrieval", nodes.rag_retrieval_node)
    g.add_node("ml_inference", nodes.ml_inference_node)
    g.add_node("batch_query", nodes.batch_query_node)
    g.add_node("email_draft", nodes.email_draft_node)
    g.add_node("human_approval", nodes.human_approval_node)
    g.add_node("observation_check", nodes.observation_check_node)
    g.add_node("replan", nodes.replan_node)
    g.add_node("final_response", nodes.final_response_node)
    g.add_node("logging_audit", nodes.logging_audit_node)

    g.set_entry_point("input_validation")
    g.add_conditional_edges("input_validation", _after_validation,
                            {"intent_classification": "intent_classification",
                             "final_response": "final_response"})
    g.add_edge("intent_classification", "permission_policy")
    g.add_conditional_edges("permission_policy", _after_policy,
                            {"planner": "planner", "final_response": "final_response"})
    g.add_edge("planner", "router_decision")
    g.add_conditional_edges("router_decision", _route_executor,
                            {"rag_retrieval": "rag_retrieval",
                             "ml_inference": "ml_inference",
                             "batch_query": "batch_query",
                             "email_draft": "email_draft",
                             "final_response": "final_response"})
    g.add_edge("rag_retrieval", "observation_check")
    g.add_edge("ml_inference", "observation_check")
    g.add_edge("batch_query", "observation_check")
    g.add_edge("email_draft", "human_approval")
    g.add_edge("human_approval", "observation_check")
    g.add_conditional_edges("observation_check", _after_observation,
                            {"replan": "replan", "final_response": "final_response"})
    g.add_edge("replan", "router_decision")
    g.add_edge("final_response", "logging_audit")
    g.add_edge("logging_audit", END)

    return g.compile()


@lru_cache(maxsize=1)
def get_copilot():
    return build_graph()


def run_copilot(user_id: str, message: str, request_id: str) -> CopilotState:
    graph = get_copilot()
    result = graph.invoke({"user_id": user_id, "message": message,
                           "request_id": request_id, "status": "ok"})
    result.setdefault("status", "ok")
    result.setdefault("response", "")
    return result
