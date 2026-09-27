"""
Loads plain-text documents and splits them into overlapping chunks that are
small enough to fit into an LLM prompt as retrieved context.
"""

import os
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


def load_documents(folder: str) -> dict:
    """Loads every .txt file in a folder into {filename: content}."""
    documents = {}
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                documents[filename] = f.read()
    return documents


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """Splits text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def build_chunk_index(folder: str, chunk_size: int = 400, overlap: int = 80) -> List[Chunk]:
    documents = load_documents(folder)
    all_chunks: List[Chunk] = []
    for doc_id, text in documents.items():
        for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            all_chunks.append(Chunk(doc_id=doc_id, chunk_id=i, text=chunk))
    return all_chunks
