from __future__ import annotations

import math
import re

import numpy as np

from .schemas import Chunk, RetrievedChunk


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            raise ValueError("Cannot build a retriever without chunks.")
        self.chunks = chunks
        self.vectors = vectors.astype("float32")
        self._build_semantic_index()
        self._build_keyword_index()

    def search(self, query: str, query_vector: np.ndarray, top_k: int = 6) -> list[RetrievedChunk]:
        semantic_scores = self._semantic_scores(query_vector)
        keyword_scores = self._keyword_scores(query)

        combined = 0.68 * semantic_scores + 0.32 * keyword_scores
        indices = np.argsort(-combined)[: max(top_k, 1)]

        results = []
        for index in indices:
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[int(index)],
                    score=float(combined[index]),
                    semantic_score=float(semantic_scores[index]),
                    keyword_score=float(keyword_scores[index]),
                )
            )
        return results

    def _build_semantic_index(self) -> None:
        self.faiss_index = None
        try:
            import faiss

            index = faiss.IndexFlatIP(self.vectors.shape[1])
            index.add(self.vectors)
            self.faiss_index = index
        except Exception:
            self.faiss_index = None

    def _semantic_scores(self, query_vector: np.ndarray) -> np.ndarray:
        vector = query_vector.reshape(1, -1).astype("float32")
        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(vector, len(self.chunks))
            semantic = np.zeros(len(self.chunks), dtype="float32")
            for score, index in zip(scores[0], indices[0]):
                if index >= 0:
                    semantic[index] = max(0.0, float(score))
            return semantic
        scores = np.dot(self.vectors, vector[0])
        return np.clip(scores, 0.0, 1.0)

    def _build_keyword_index(self) -> None:
        self.tokenized_chunks = [_tokenize(chunk.text) for chunk in self.chunks]
        self.bm25 = None
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.tokenized_chunks)
        except Exception:
            self._build_simple_bm25()

    def _keyword_scores(self, query: str) -> np.ndarray:
        tokens = _tokenize(query)
        if not tokens:
            return np.zeros(len(self.chunks), dtype="float32")
        if self.bm25 is not None:
            raw = np.asarray(self.bm25.get_scores(tokens), dtype="float32")
        else:
            raw = self._simple_bm25_scores(tokens)
        if raw.max() <= 0:
            return np.zeros_like(raw, dtype="float32")
        return raw / raw.max()

    def _build_simple_bm25(self) -> None:
        self.doc_freq: dict[str, int] = {}
        self.term_freqs: list[dict[str, int]] = []
        lengths = []
        for tokens in self.tokenized_chunks:
            freqs: dict[str, int] = {}
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1
            self.term_freqs.append(freqs)
            lengths.append(len(tokens))
            for token in freqs:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
        self.avg_doc_len = float(np.mean(lengths)) if lengths else 1.0

    def _simple_bm25_scores(self, query_tokens: list[str]) -> np.ndarray:
        k1 = 1.5
        b = 0.75
        total_docs = len(self.chunks)
        scores = np.zeros(total_docs, dtype="float32")
        for i, freqs in enumerate(self.term_freqs):
            doc_len = sum(freqs.values()) or 1
            score = 0.0
            for token in query_tokens:
                if token not in freqs:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                tf = freqs[token]
                denom = tf + k1 * (1 - b + b * doc_len / self.avg_doc_len)
                score += idf * (tf * (k1 + 1) / denom)
            scores[i] = score
        return scores


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]

