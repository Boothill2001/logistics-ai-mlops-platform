"""Append-only audit log (JSONL).

Every security-relevant event is recorded: retrievals (which documents reached
which user's context), predictions made via the copilot, email drafts, and
approval decisions. This is the data lineage story: for any answer the system
gave, you can reconstruct who asked, what was retrieved, and what was done.

Demo note: production would ship these to an immutable store (e.g. S3 with
object lock, or a SIEM), not a local file.
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.config import settings

_lock = threading.Lock()


def _audit_path() -> Path:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    return settings.logs_dir / "audit.jsonl"


def audit(event_type: str, user_id: str, **fields) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        "user_id": user_id,
        **fields,
    }
    with _lock:
        with _audit_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_audit(limit: int = 50) -> list[dict]:
    path = _audit_path()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines[-limit:]]
