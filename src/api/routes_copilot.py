"""Copilot endpoints.

Auth is a trusted X-User-Id header — a demo stand-in for a validated JWT.
"""
from fastapi import APIRouter, Header, HTTPException, Request

from src.api.schemas import (
    ApprovalRequest,
    ApprovalResponse,
    CopilotChatRequest,
    CopilotChatResponse,
)
from src.audit.log import audit
from src.auth.users import get_user
from src.copilot.drafts import draft_store
from src.copilot.graph import run_copilot

router = APIRouter(prefix="/copilot")


def _resolve_user(x_user_id: str | None):
    if not x_user_id or get_user(x_user_id) is None:
        raise HTTPException(status_code=401, detail="Unknown or missing X-User-Id")
    return get_user(x_user_id)


@router.post("/chat", response_model=CopilotChatResponse)
def chat(body: CopilotChatRequest, request: Request,
         x_user_id: str | None = Header(default=None)):
    user = _resolve_user(x_user_id)
    result = run_copilot(user.user_id, body.message, request.state.request_id)
    return CopilotChatResponse(
        response=result["response"],
        intent=result.get("intent", "unknown"),
        status=result["status"],
        sources=result.get("rag_sources", []),
        draft_id=result.get("draft_id"),
        request_id=request.state.request_id,
    )


@router.post("/approve", response_model=ApprovalResponse)
def approve(body: ApprovalRequest, x_user_id: str | None = Header(default=None)):
    user = _resolve_user(x_user_id)
    if user.role not in {"ops", "admin"}:
        raise HTTPException(status_code=403, detail="Only ops or admin can approve drafts")
    draft = draft_store.resolve(body.draft_id, body.approve)
    if draft is None:
        raise HTTPException(status_code=404, detail="Draft not found")
    audit("draft_resolved", user.user_id, draft_id=draft.draft_id, status=draft.status)
    detail = ("Draft approved. (Demo: dispatch would be handled by a separate "
              "backend job — the copilot itself never sends email.)"
              if draft.status == "approved" else "Draft rejected.")
    return ApprovalResponse(draft_id=draft.draft_id, status=draft.status, detail=detail)
