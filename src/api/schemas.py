"""Pydantic request/response contracts for the API."""
from pydantic import BaseModel, Field


class DelayPredictRequest(BaseModel):
    shipment_id: str = Field(min_length=1, max_length=64)
    customer_id: str = Field(min_length=1, max_length=64)
    origin_port: str = Field(min_length=2, max_length=8)
    destination_port: str = Field(min_length=2, max_length=8)
    container_type: str = Field(min_length=2, max_length=8)
    booking_lead_days: int = Field(ge=0, le=365)
    transshipment_count: int = Field(ge=0, le=10)
    port_congestion_score: float = Field(ge=0.0, le=1.0)
    weather_risk_score: float = Field(ge=0.0, le=1.0)
    historical_delay_rate: float = Field(ge=0.0, le=1.0)
    carrier_reliability_score: float = Field(ge=0.0, le=1.0)


class DelayPredictResponse(BaseModel):
    shipment_id: str
    delay_probability: float
    risk_level: str
    model_name: str
    model_version: str
    latency_ms: float
    request_id: str


class CopilotChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class CopilotChatResponse(BaseModel):
    response: str
    intent: str
    status: str  # "ok" | "needs_clarification" | "pending_approval" | "denied" | "error"
    sources: list[str] = []
    draft_id: str | None = None
    request_id: str


class ApprovalRequest(BaseModel):
    draft_id: str
    approve: bool


class ApprovalResponse(BaseModel):
    draft_id: str
    status: str  # "approved" | "rejected"
    detail: str
