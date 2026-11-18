from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

from .analysis import DocumentProfile, build_document_profiles
from .answer import generate_answer
from .chunking import chunk_pages
from .config import AppConfig
from .document_loader import load_document
from .embeddings import EmbeddingModel
from .retrieval import HybridRetriever
from .schemas import Answer, ChatTurn, Chunk, Page


@dataclass
class RagEngine:
    config: AppConfig
    embedder: EmbeddingModel
    pages: list[Page]
    chunks: list[Chunk]
    retriever: HybridRetriever
    profiles: list[DocumentProfile]
    vectors: np.ndarray

    @classmethod
    def from_uploads(cls, uploads: list[tuple[str, bytes]], config: AppConfig) -> "RagEngine":
        pages: list[Page] = []
        for file_name, content in uploads:
            pages.extend(load_document(file_name, content))
        if not pages:
            raise ValueError("No readable text was found in the uploaded documents.")

        chunks = chunk_pages(pages, chunk_size=config.chunk_size, overlap=config.chunk_overlap)
        if not chunks:
            raise ValueError("The uploaded documents did not produce searchable chunks.")

        embedder = EmbeddingModel(config.embedding_model, config.fallback_embedding_model)
        vectors = embedder.encode([chunk.text for chunk in chunks])
        retriever = HybridRetriever(chunks, vectors)
        profiles = build_document_profiles(pages, chunks)
        return cls(
            config=config,
            embedder=embedder,
            pages=pages,
            chunks=chunks,
            retriever=retriever,
            profiles=profiles,
            vectors=vectors,
        )

    @classmethod
    def load_index(cls, index_path: Path, config: AppConfig) -> "RagEngine":
        payload = pickle.loads(index_path.read_bytes())
        embedder = EmbeddingModel(payload.get("embedding_model", config.embedding_model), config.fallback_embedding_model)
        vectors = payload["vectors"]
        chunks = payload["chunks"]
        pages = payload["pages"]
        profiles = payload.get("profiles") or build_document_profiles(pages, chunks)
        retriever = HybridRetriever(chunks, vectors)
        return cls(config=config, embedder=embedder, pages=pages, chunks=chunks, retriever=retriever, profiles=profiles, vectors=vectors)

    def save_index(self, index_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedding_model": self.embedder.model_name,
            "pages": self.pages,
            "chunks": self.chunks,
            "profiles": self.profiles,
            "vectors": self.vectors,
        }
        index_path.write_bytes(pickle.dumps(payload))

    @property
    def document_names(self) -> list[str]:
        return sorted({page.document_name for page in self.pages})

    def ask(
        self,
        question: str,
        history: list[ChatTurn],
        provider: str,
        document_names: list[str] | None = None,
        rerank: bool = False,
    ) -> Answer:
        search_query = _contextualize_question(question, history)
        query_vector = self.embedder.encode([search_query])[0]
        search_k = max(self.config.top_k, 30 if document_names or rerank else self.config.top_k)
        retrieved = self.retriever.search(search_query, query_vector, top_k=search_k)
        if document_names:
            allowed = set(document_names)
            retrieved = [item for item in retrieved if item.chunk.document_name in allowed]
        if rerank:
            retrieved = _rerank(search_query, retrieved)
        retrieved = retrieved[: self.config.top_k]
        return generate_answer(question, retrieved, history, provider, self.config)

    def semantic_search(self, query: str, document_names: list[str] | None = None, top_k: int = 12) -> list:
        query_vector = self.embedder.encode([query])[0]
        results = self.retriever.search(query, query_vector, top_k=max(top_k, 30 if document_names else top_k))
        if document_names:
            allowed = set(document_names)
            results = [item for item in results if item.chunk.document_name in allowed]
        return results[:top_k]


def _rerank(query: str, retrieved: list) -> list:
    try:
        from sentence_transformers import CrossEncoder

        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        model = CrossEncoder(model_name)
        pairs = [(query, item.chunk.text) for item in retrieved]
        scores = model.predict(pairs)
        return [item for _, item in sorted(zip(scores, retrieved), key=lambda pair: pair[0], reverse=True)]
    except Exception:
        query_terms = {token.lower() for token in query.split() if len(token) > 3}
        return sorted(
            retrieved,
            key=lambda item: (
                len(query_terms & {token.strip(".,:;()[]").lower() for token in item.chunk.text.split()}),
                item.score,
            ),
            reverse=True,
        )


def _contextualize_question(question: str, history: list[ChatTurn]) -> str:
    pronounish = {"that", "this", "it", "they", "those", "there", "above", "previous", "same"}
    question_terms = set(question.lower().replace("?", "").split())
    if question_terms & pronounish:
        recent = " ".join(turn.content for turn in history[-6:] if turn.role in {"user", "assistant"})
        return f"Conversation context: {recent}\nStandalone question: {question}".strip()
    recent_users = " ".join(turn.content for turn in history[-3:] if turn.role == "user")
    return f"{recent_users} {question}".strip()
