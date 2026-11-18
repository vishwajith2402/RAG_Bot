from __future__ import annotations

import re

from .schemas import Chunk, Page


SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def chunk_pages(pages: list[Page], chunk_size: int = 900, overlap: int = 160) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        page_chunks = _split_text(page.text, chunk_size=chunk_size, overlap=overlap)
        for index, text in enumerate(page_chunks):
            chunks.append(
                Chunk(
                    chunk_id=f"{page.document_id}:{page.page_number}:{index}",
                    document_id=page.document_id,
                    document_name=page.document_name,
                    page_number=page.page_number,
                    text=text,
                )
            )
    return chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    sentences = SENTENCE_PATTERN.split(normalized)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > chunk_size:
            chunks.extend(_split_long_sentence(sentence, chunk_size, overlap))
            continue
        if current and current_len + len(sentence) + 1 > chunk_size:
            chunks.append(" ".join(current).strip())
            current = _overlap_tail(current, overlap)
            current_len = sum(len(item) + 1 for item in current)
        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _split_long_sentence(sentence: str, chunk_size: int, overlap: int) -> list[str]:
    parts = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(sentence):
        parts.append(sentence[start : start + chunk_size].strip())
        start += step
    return [part for part in parts if part]


def _overlap_tail(sentences: list[str], overlap: int) -> list[str]:
    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        if total + len(sentence) > overlap:
            break
        tail.insert(0, sentence)
        total += len(sentence)
    return tail

