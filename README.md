# Career Intelligence Assistant

A RAG system that analyzes a resume against one or more job descriptions and
answers questions about fit, skill gaps, experience alignment, and interview
prep. Built as a take-home for a Forward Deployed Engineer role — see
[`PRD.md`](./PRD.md) for the full scope, architecture decisions, and
hour-by-hour build plan this repo follows.

## Quick start

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY

docker-compose up --build
```

- Frontend: http://localhost:3000
- Backend API + docs: http://localhost:8000/docs

The first `docker-compose up --build` bakes the local embedding and
re-ranking models into the backend image at build time (see
`backend/Dockerfile`), so it takes a few minutes once; subsequent starts are
fast and need no network at request time for embeddings/reranking. Only the
Claude API call needs network + your API key at runtime.

### Running without Docker (faster iteration)

```bash
# backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # set ANTHROPIC_API_KEY
uvicorn app.main:app --reload

# frontend, in another terminal
cd frontend
npm install
npm run dev
```

`scripts/smoke_test_pipeline.py` in `backend/` runs parsing → chunking →
hybrid retrieval → reranking end to end on sample text, without needing a
Claude API key — useful for sanity-checking the RAG pipeline in isolation.

## Architecture

```
backend/app/
  api/        — FastAPI routes (documents, chat) + DI wiring — no business logic
  services/   — orchestration: ingestion_service, chat_service, document_registry
  rag/        — parsers, chunking, embeddings, vector_store, retrieval, reranker
  llm/        — Claude client: system prompt, context assembly, the API call
  core/       — config (env-driven settings), logging (structlog + timing), exceptions, rate limiting
  models/     — Pydantic schemas shared across every layer

frontend/src/
  api/        — typed fetch client + types mirroring the backend schemas exactly
  components/ — UploadPanel, ChatPanel, SourceChip + small hand-rolled UI primitives
```

`EmbeddingProvider` (`app/rag/embeddings.py`) and `VectorStore`
(`app/rag/vector_store.py`) are the two interfaces in this codebase built
around dependency injection on purpose — they're the seams most likely to
change (swapping in a hosted embedding API, or FAISS/Qdrant later). Nothing
else is injected "for its own sake."

## Key decisions (see PRD.md §3 and §5 for the full reasoning)

- **Embeddings & reranking run locally** (`sentence-transformers`, no API
  key) — the biggest reliability win for anyone running this cold.
- **ChromaDB**, chosen over FAISS specifically for native metadata
  filtering, which the document-type query filter depends on.
- **Hybrid retrieval**: semantic (Chroma) + keyword (BM25) merged via
  reciprocal rank fusion, then **re-ranked by a local cross-encoder** before
  context assembly — see `backend/tests/test_retrieval.py` for a test that
  demonstrates BM25 recovering a chunk semantic-only search missed entirely.
- **Grounding guardrail**: the system prompt instructs Claude to say when
  context is insufficient rather than reach for outside knowledge; low
  retrieval/rerank scores additionally flag a response as `grounded: false`
  so the UI can surface a low-confidence warning.
- **Rate limiting** is an in-process sliding-window counter — enough to
  demonstrate the guardrail for a single-instance take-home; a Redis-backed
  limiter is the named next step for a real deployment.

## What's cut, on purpose

See PRD.md §2 for the full list (multi-user auth, cloud deployment, OCR,
production rate limiting). Nothing here is a forgotten feature — it's a
named tradeoff given the 2-day budget.

## Testing

```bash
# backend
cd backend && source .venv/bin/activate && pytest -q

# frontend
cd frontend && npm run build && npx vitest run
```

Backend: 11 tests covering chunking metadata/overlap, the document-type
query classifier, BM25+RRF fusion recovering a keyword-strong chunk a
semantic-only search missed, and cross-encoder rerank ordering — all run
without hitting the network or needing an API key (the cross-encoder tests
stub the model itself, since they're testing the sort/threshold logic, not
the model). Frontend: a build/type-check pass plus a ChatPanel test that
mocks the API and asserts sources (with both scores) render correctly, and
that the input is disabled until both a resume and a JD are uploaded.

## What I'd do with more time

- A learned/trained query classifier instead of the current keyword
  heuristic for document-type filtering
- Multi-turn conversation memory (currently each question is independent)
- Streaming responses in the chat UI
- A Redis-backed rate limiter and a real job queue for ingestion, for
  multi-instance deployment
- Broader edge-case test coverage (malformed uploads, concurrent ingestion)
