"""Grounded answering over retrieved chunks.

Contract: the answer is built ONLY from retrieved context. If retrieval
returns nothing usable, we say so ("Không đủ dữ liệu") instead of letting the
LLM improvise — a wrong-but-confident answer about a contract clause is worse
than no answer. Every answer carries citations to its source documents.
"""
from dataclasses import dataclass, field

from src.auth.users import User
from src.rag.store import DocStore, RetrievedChunk

NO_DATA_MESSAGE = (
    "Insufficient data in the documents you have access to. Cannot answer this question."
)

ANSWER_PROMPT = """You are a logistics operations assistant. Answer the user's question using ONLY the context below. If the context does not contain the answer, reply exactly: "NO_SUFFICIENT_DATA".
Cite sources inline as [document_id]. Do not invent facts, numbers, or clauses.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RagAnswer:
    text: str
    sources: list[str] = field(default_factory=list)
    grounded: bool = True


def build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.document_id}]\n{c.text}" for c in chunks)


def answer_question(question: str, user: User, store: DocStore, llm) -> RagAnswer:
    chunks = store.retrieve(question, user)
    if not chunks:
        return RagAnswer(text=NO_DATA_MESSAGE, sources=[], grounded=False)

    prompt = ANSWER_PROMPT.format(context=build_context(chunks), question=question)
    raw = llm.complete(prompt)
    if "NO_SUFFICIENT_DATA" in raw or "KHONG_DU_DU_LIEU" in raw:
        return RagAnswer(text=NO_DATA_MESSAGE, sources=[], grounded=False)

    sources = sorted({c.document_id for c in chunks})
    return RagAnswer(text=raw.strip(), sources=sources, grounded=True)
