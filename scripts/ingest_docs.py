"""Ingest data/docs into Chroma with permission metadata.

Permission metadata is assigned HERE, at ingest time, from a manifest — not
inferred by the LLM or guessed at query time. Access control facts must come
from a governed source (in production: the DMS/ECM system of record).
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings  # noqa: E402
from src.rag.chunker import chunk_text  # noqa: E402
from src.rag.store import DocStore  # noqa: E402

# Manifest: filename -> (doc_type, customer_id, permission_group)
DOC_MANIFEST = {
    "contract_customer_A.txt": ("contract", "CUS_A", "customer_A"),
    "invoice_customer_A.txt": ("invoice", "CUS_A", "customer_A"),
    "shipping_policy.txt": ("policy", "ALL", "public"),
    "delay_sop.txt": ("sop", "ALL", "public"),
    "dangerous_private_note.txt": ("private_note", "CUS_A", "admin_only"),
}


def main() -> None:
    store = DocStore()
    now = datetime.now(timezone.utc).isoformat()

    for filename, (doc_type, customer_id, permission_group) in DOC_MANIFEST.items():
        path = settings.data_dir / "docs" / filename
        document_id = path.stem
        chunks = chunk_text(path.read_text(encoding="utf-8"))
        store.add_chunks(
            ids=[f"{document_id}::chunk_{c.chunk_index}" for c in chunks],
            texts=[c.text for c in chunks],
            metadatas=[{
                "document_id": document_id,
                "customer_id": customer_id,
                "doc_type": doc_type,
                "permission_group": permission_group,
                "created_at": now,
            } for c in chunks],
        )
        print(f"ingested {filename}: {len(chunks)} chunks [{permission_group}]")

    print(f"total chunks in collection: {store.count()}")


if __name__ == "__main__":
    main()
