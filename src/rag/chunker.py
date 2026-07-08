"""Paragraph-aware chunking.

Splits on blank lines first (paragraphs are the natural semantic unit in
contracts/SOPs), then packs paragraphs into chunks up to max_chars with a
one-paragraph overlap so a clause split across chunks is still retrievable.
"""
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    chunk_index: int


def chunk_text(text: str, max_chars: int = 500) -> list[Chunk]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    current: list[str] = []
    size = 0

    def flush():
        nonlocal current, size
        if current:
            chunks.append(Chunk(text="\n\n".join(current), chunk_index=len(chunks)))
            # keep last paragraph as overlap into the next chunk
            current = current[-1:]
            size = len(current[0]) if current else 0

    for para in paragraphs:
        if size + len(para) > max_chars and current:
            flush()
        current.append(para)
        size += len(para)

    if current:
        chunks.append(Chunk(text="\n\n".join(current), chunk_index=len(chunks)))
    return chunks
