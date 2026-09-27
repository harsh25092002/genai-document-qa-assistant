"""
Generation component of the RAG pipeline.

Builds a prompt from the retrieved chunks and calls an LLM (OpenAI's Chat
Completions API) to produce a grounded answer. If no API key is configured,
falls back to a simple extractive answer built directly from the highest
scoring chunk, so the project still runs end-to-end without any credentials.
"""

import os
from typing import List, Tuple

from app.document_store import Chunk

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context. If the answer is not contained in the context, say "
    "you don't have enough information."
)


def _build_context(results: List[Tuple[Chunk, float]]) -> str:
    parts = []
    for chunk, score in results:
        parts.append(f"[Source: {chunk.doc_id}, chunk {chunk.chunk_id}]\n{chunk.text}")
    return "\n\n".join(parts)


def answer_question(question: str, results: List[Tuple[Chunk, float]]) -> str:
    if not results:
        return "I couldn't find anything relevant in the indexed documents to answer that."

    context = _build_context(results)
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        return _generate_with_llm(question, context, api_key)
    return _fallback_extractive_answer(question, results)


def _generate_with_llm(question: str, context: str, api_key: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def _fallback_extractive_answer(question: str, results: List[Tuple[Chunk, float]]) -> str:
    """
    Used when OPENAI_API_KEY is not set. Returns the most relevant chunk
    directly, so the demo works fully offline without any paid API.
    """
    best_chunk, score = results[0]
    return (
        f"(No OPENAI_API_KEY set — showing the most relevant passage instead of "
        f"an LLM-generated answer)\n\nFrom '{best_chunk.doc_id}':\n{best_chunk.text}"
    )
