# GenAI Document Q&A Assistant

A Retrieval-Augmented Generation (RAG) service that answers natural-language questions grounded in a set of documents, built with **Python, FastAPI, and OpenAI's Chat Completions API**, and containerized with **Docker** for cloud deployment.

## How it works
1. **Ingestion** — plain-text documents are split into overlapping chunks (`app/document_store.py`)
2. **Retrieval** — a TF-IDF + cosine-similarity retriever finds the most relevant chunks for a question (`app/retriever.py`) — fully offline, no external API needed for this step
3. **Generation** — the retrieved chunks are passed as context to an LLM (`gpt-4o-mini` via the OpenAI API) which produces a grounded answer (`app/qa_engine.py`)
4. If no `OPENAI_API_KEY` is set, the service gracefully falls back to returning the most relevant passage directly, so the whole pipeline still runs end-to-end with zero paid API calls — useful for demos and grading.

## Tech Stack
Python 3.11, FastAPI, scikit-learn (TF-IDF retrieval), OpenAI API, Docker

## Project Structure
```
app/
├── document_store.py   # Load + chunk documents
├── retriever.py         # TF-IDF retrieval
├── qa_engine.py          # LLM-based (or fallback) answer generation
└── main.py               # FastAPI app (/health, /ask)
data/sample_docs/         # Sample HR policy documents used for the demo
tests/                     # pytest tests for the retriever
Dockerfile
requirements.txt
```

## Running Locally
```bash
pip install -r requirements.txt
cp .env.example .env          # optionally add your OPENAI_API_KEY
uvicorn app.main:app --reload
```
Then:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many work-from-home days are allowed per month?"}'
```

## Running Tests
```bash
pytest tests/ -v
```

## Running with Docker
```bash
docker build -t genai-doc-qa .
docker run -p 8000:8000 --env-file .env genai-doc-qa
```

## Deploying to the Cloud (free-tier friendly)

**Option A — Render.com**
1. Push this repo to GitHub.
2. On Render, create a new **Web Service**, connect the repo, choose "Docker" as the environment.
3. Add `OPENAI_API_KEY` under Environment Variables (optional).
4. Render builds the Dockerfile and gives you a public HTTPS URL.

**Option B — AWS (EC2 Free Tier)**
1. Launch a `t2.micro` EC2 instance (Ubuntu, free tier eligible).
2. Install Docker, `git clone` this repo.
3. `docker build -t genai-doc-qa .` then `docker run -d -p 80:8000 --env-file .env genai-doc-qa`.
4. Open port 80 in the instance's security group to access it publicly.

**Option C — Azure App Service (container deployment)**
1. Push the built image to Azure Container Registry (`az acr build`).
2. Create an App Service (Linux, container) pointing at that image.
3. Set `OPENAI_API_KEY` in App Service > Configuration > Application settings.

## Notes
- Swap the TF-IDF retriever for OpenAI/HuggingFace embeddings + a vector database (e.g. FAISS, Pinecone) for larger document sets — the `Retriever` interface is designed to be a drop-in replacement.
- Add authentication (API key header) before exposing `/ask` publicly in production.
