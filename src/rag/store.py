"""Chroma vector store wrapper with permission-aware retrieval.

THE key security property lives here: the permission filter is applied
*inside the vector search* (Chroma `where` clause), BEFORE any document text
is returned. Documents a user cannot access are never candidates — they can
never reach the LLM context, so the model cannot leak what it never saw.

This is deliberately NOT done by retrieving first and filtering after:
post-filtering means restricted text exists in application memory per-request
and one missed code path leaks it. Pre-filtering makes leakage structurally
impossible at this layer.
"""
from dataclasses import dataclass

import chromadb

from src.audit.log import audit
from src.auth.rbac import allowed_permission_groups
from src.auth.users import User
from src.config import settings
from src.rag.embeddings import LocalHashEmbedding

COLLECTION = "logistics_docs"

# Lexical embeddings put absolute distances on a coarse scale; anything worse
# than this is treated as "no relevant context" rather than shown to the LLM.
MAX_DISTANCE = 0.95


@dataclass
class RetrievedChunk:
    text: str
    document_id: str
    doc_type: str
    customer_id: str
    distance: float


class DocStore:
    def __init__(self, persist_dir: str | None = None):
        self._client = chromadb.PersistentClient(path=str(persist_dir or settings.chroma_dir))
        self._collection = self._client.get_or_create_collection(
            COLLECTION,
            embedding_function=LocalHashEmbedding(),
            # cosine distance: vectors are l2-normalized, so distance is in
            # [0, 2] and MAX_DISTANCE below has a stable meaning
            metadata={"hnsw:space": "cosine"},
        )

    # -- ingest ----------------------------------------------------------
    def add_chunks(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        required = {"document_id", "customer_id", "doc_type", "permission_group", "created_at"}
        for meta in metadatas:
            missing = required - set(meta)
            if missing:
                raise ValueError(f"chunk metadata missing {missing}")
        self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    def count(self) -> int:
        return self._collection.count()

    # -- retrieval ---------------------------------------------------------
    def retrieve(self, query: str, user: User, top_k: int = 4) -> list[RetrievedChunk]:
        groups = allowed_permission_groups(user)
        result = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            # Permission filter INSIDE the search — non-negotiable ordering.
            where={"permission_group": {"$in": groups}},
        )
        chunks = []
        for text, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            if dist <= MAX_DISTANCE:
                chunks.append(RetrievedChunk(
                    text=text,
                    document_id=meta["document_id"],
                    doc_type=meta["doc_type"],
                    customer_id=meta["customer_id"],
                    distance=round(float(dist), 4),
                ))
        # Data lineage: record exactly which documents reached this user's context
        audit(
            "rag_retrieval", user.user_id, query=query,
            allowed_groups=groups,
            retrieved=[{"document_id": c.document_id, "distance": c.distance} for c in chunks],
        )
        return chunks
