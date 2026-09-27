import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.document_store import build_chunk_index
from app.retriever import Retriever


def test_retriever_returns_relevant_chunk():
    chunks = build_chunk_index("data/sample_docs")
    retriever = Retriever(chunks)

    results = retriever.retrieve("How many days of work from home are allowed?", top_k=2)

    assert len(results) > 0
    top_chunk, score = results[0]
    assert "work from home" in top_chunk.text.lower() or "8 days" in top_chunk.text.lower()
    assert score > 0


def test_retriever_handles_unrelated_query_gracefully():
    chunks = build_chunk_index("data/sample_docs")
    retriever = Retriever(chunks)

    results = retriever.retrieve("zzzz unrelated nonsense query zzzz", top_k=3)
    # Should not raise, may legitimately return an empty list if nothing matches
    assert isinstance(results, list)
