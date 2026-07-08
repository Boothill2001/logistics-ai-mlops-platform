"""LLM provider abstraction.

Two implementations behind one interface:
- DeepSeekLLM — real API (OpenAI-compatible endpoint).
- MockLLM    — deterministic, offline, zero-cost. Used in tests and demos.

Why both: tests must not depend on network/credits/nondeterminism, and the
system must degrade gracefully if the LLM vendor is down (MockLLM is also the
fallback path). Switch via LLM_PROVIDER env var.
"""
import logging
import re
from abc import ABC, abstractmethod

from src.config import settings

logger = logging.getLogger("copilot.llm")


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class DeepSeekLLM(LLMProvider):
    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )

    @property
    def name(self) -> str:
        return "deepseek"

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
        )
        return response.choices[0].message.content or ""


class MockLLM(LLMProvider):
    """Deterministic rule-based stand-in.

    For RAG answering it extractively summarizes the provided context (so
    grounding still holds); for classification-style prompts it keys off
    prompt markers. Good enough to exercise every code path offline.
    """

    @property
    def name(self) -> str:
        return "mock"

    def complete(self, prompt: str) -> str:
        if "Context:" in prompt and "Question:" in prompt:
            return self._answer_from_context(prompt)
        if "Write a professional delay notification email" in prompt:
            return self._email_draft(prompt)
        # Generic fallback
        return "OK."

    def _answer_from_context(self, prompt: str) -> str:
        context = prompt.split("Context:", 1)[1].split("Question:", 1)[0].strip()
        question = prompt.split("Question:", 1)[1].split("Answer:", 1)[0].strip()
        if not context:
            return "KHONG_DU_DU_LIEU"
        # Extractive: keep sentences sharing terms with the question
        doc_ids = re.findall(r"\[([\w\-]+)\]", context)
        q_terms = {w.lower() for w in re.findall(r"\w{4,}", question)}
        sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\[[\w\-]+\]", "", context))
        hits = [s.strip() for s in sentences
                if q_terms & {w.lower() for w in re.findall(r"\w{4,}", s)}]
        if not hits:
            return "KHONG_DU_DU_LIEU"
        citation = f" [{doc_ids[0]}]" if doc_ids else ""
        return " ".join(hits[:4]) + citation

    def _email_draft(self, prompt: str) -> str:
        return (
            "Subject: Update on your shipment — revised ETA\n\n"
            "Dear Customer,\n\n"
            "We are writing to inform you that your shipment is expected to be "
            "delayed. Our operations team has verified the cause and is working "
            "on mitigation per our delay handling procedure. We will share the "
            "revised ETA and next steps shortly.\n\n"
            "We apologize for the inconvenience.\n\n"
            "Best regards,\nOperations Team, Pacific Line Logistics"
        )


def get_llm() -> LLMProvider:
    if settings.llm_provider == "deepseek" and settings.deepseek_api_key:
        try:
            return DeepSeekLLM()
        except Exception as exc:  # missing dep / bad config -> degrade, don't crash
            logger.warning("DeepSeek unavailable (%s), falling back to MockLLM", exc)
    return MockLLM()
