import numpy as np

from src.retrieval import HybridRetriever
from src.schemas import Chunk


def test_hybrid_retriever_prefers_keyword_match():
    chunks = [
        Chunk("1", "d", "policy.pdf", 1, "Vacation requests require manager approval."),
        Chunk("2", "d", "policy.pdf", 2, "The office kitchen has a coffee machine."),
    ]
    vectors = np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype="float32")
    retriever = HybridRetriever(chunks, vectors)

    results = retriever.search("manager approval vacation", np.asarray([0.7, 0.3], dtype="float32"), top_k=1)

    assert results[0].chunk.chunk_id == "1"

