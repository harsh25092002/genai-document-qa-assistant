"""
FastAPI service exposing the GenAI Document Q&A Assistant.

Endpoints:
  GET  /health          -> liveness check
  POST /ask             -> {"question": "..."} -> {"answer": "...", "sources": [...]}
"""

from fastapi import FastAPI
from pydantic import BaseModel

from app.document_store import build_chunk_index
from app.qa_engine import answer_question
from app.retriever import Retriever

app = FastAPI(title="GenAI Document Q&A Assistant", version="1.0.0")

DOCS_FOLDER = "data/sample_docs"
_chunks = build_chunk_index(DOCS_FOLDER)
_retriever = Retriever(_chunks)


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": len(_chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    results = _retriever.retrieve(request.question, top_k=request.top_k)
    answer = answer_question(request.question, results)
    sources = [f"{chunk.doc_id}#chunk{chunk.chunk_id}" for chunk, _ in results]
    return AskResponse(answer=answer, sources=sources)
