"""
Retrieval component of the RAG pipeline.

Uses TF-IDF + cosine similarity so the retriever runs fully offline with no
external API calls or model downloads. It is a drop-in interface: the
`embedding_backend` could be swapped for OpenAI/HuggingFace embeddings in a
production deployment without changing the rest of the pipeline.
"""

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.document_store import Chunk


class Retriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]
