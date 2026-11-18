from __future__ import annotations

import numpy as np


class EmbeddingModel:
    def __init__(self, model_name: str, fallback_model_name: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.mode = "sentence-transformers"
        except Exception:
            try:
                from sentence_transformers import SentenceTransformer

                if not fallback_model_name:
                    raise
                self.model = SentenceTransformer(fallback_model_name)
                self.model_name = fallback_model_name
                self.mode = "sentence-transformers"
            except Exception:
                from sklearn.feature_extraction.text import HashingVectorizer

                self.model = HashingVectorizer(
                    n_features=2048,
                    alternate_sign=False,
                    norm="l2",
                    stop_words="english",
                )
                self.model_name = "sklearn-hashing-vectorizer"
                self.mode = "hashing-vectorizer"

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.mode == "hashing-vectorizer":
            vectors = self.model.transform(texts).toarray()
            return np.asarray(vectors, dtype="float32")
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
        return np.asarray(vectors, dtype="float32")
