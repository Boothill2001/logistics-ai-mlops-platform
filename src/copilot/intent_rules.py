"""Deterministic intent rules.

Used by MockLLM (so tests/demos classify without a network call) and as the
safety net when the real LLM returns a label outside the allowed set — an
LLM classifier must never be able to invent a new intent.
"""
import re

INTENTS = ["knowledge_search", "ml_prediction", "batch_query", "dangerous_action", "missing_info"]

SHIPMENT_RE = re.compile(r"\b(SHP_\w+)\b", re.IGNORECASE)
CUSTOMER_RE = re.compile(r"\b(CUS_[A-Z])\b|customer\s+([A-D])\b", re.IGNORECASE)


def classify_intent_rules(message: str) -> str:
    m = message.lower()
    if re.search(r"\b(email|notify|send|g[ửu]i|inform)\b", m):
        return "dangerous_action"
    if re.search(r"that customer|for that client|khách hàng đó|customer đó|báo cáo cho khách", m):
        return "missing_info"
    if SHIPMENT_RE.search(message):
        return "ml_prediction"
    if re.search(r"\btop\b|\bhighest\b|cao nhất|tuần này|this week|hôm nay|today|shipment nào", m):
        return "batch_query"
    if re.search(r"\b(report|báo cáo|summary)\b", m) and not CUSTOMER_RE.search(message):
        return "missing_info"
    return "knowledge_search"


def extract_customer(message: str) -> str | None:
    match = CUSTOMER_RE.search(message)
    if not match:
        return None
    return (match.group(1) or f"CUS_{match.group(2)}").upper()
