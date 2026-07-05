from __future__ import annotations
import re
from typing import List, Dict


def _split_into_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter (no heavy NLP dependency required)."""
    text = re.sub(r"\s+", " ", text).strip()
    # split on sentence-ending punctuation followed by a space + capital letter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 75,
) -> List[str]:
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > chunk_size:
            chunks.append(current.strip())
            overlap_text = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            current = (overlap_text + " " + sentence).strip()
        else:
            current = (current + " " + sentence).strip() if current else sentence

    if current:
        chunks.append(current.strip())

    return chunks


def chunk_documents(
    documents: List[Dict[str, str]],
    chunk_size: int = 500,
    chunk_overlap: int = 75,
) -> List[Dict[str, str]]:
    all_chunks: List[Dict[str, str]] = []
    for doc in documents:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append({"source": doc["source"], "chunk_id": i, "text": piece})
    return all_chunks
