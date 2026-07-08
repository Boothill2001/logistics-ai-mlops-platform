"""In-memory store for email drafts awaiting human approval.

Human-in-the-loop contract: the graph NEVER sends anything. It produces a
draft with status pending_approval; a human calls POST /copilot/approve.
Even after approval the demo only marks the draft approved — actual dispatch
would be a separate, non-LLM backend job reading approved drafts.
"""
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Draft:
    draft_id: str
    user_id: str
    content: str
    status: str = "pending_approval"  # -> "approved" | "rejected"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DraftStore:
    def __init__(self):
        self._drafts: dict[str, Draft] = {}
        self._lock = threading.Lock()

    def create(self, user_id: str, content: str) -> Draft:
        with self._lock:
            draft = Draft(draft_id=f"draft_{uuid.uuid4().hex[:10]}", user_id=user_id, content=content)
            self._drafts[draft.draft_id] = draft
            return draft

    def get(self, draft_id: str) -> Draft | None:
        return self._drafts.get(draft_id)

    def resolve(self, draft_id: str, approve: bool) -> Draft | None:
        with self._lock:
            draft = self._drafts.get(draft_id)
            if draft and draft.status == "pending_approval":
                draft.status = "approved" if approve else "rejected"
            return draft


draft_store = DraftStore()
