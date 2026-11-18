from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Page:
    document_id: str
    document_name: str
    page_number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    semantic_score: float
    keyword_score: float


@dataclass(frozen=True)
class ChatTurn:
    role: str
    content: str


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[RetrievedChunk]
    used_provider: str
    confidence: float

