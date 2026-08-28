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
  api/        — FastAPI routes (documents, chat, chat/stream) + DI wiring — no business logic
  services/   — orchestration: ingestion_service, chat_service, document_registry, conversation_store
  rag/        — parsers, chunking, embeddings, vector_store, retrieval, reranker
  llm/        — Claude client: system prompt, context assembly, the API call (plain + streaming)
  core/       — config (env-driven settings), logging (structlog + timing), exceptions, rate limiting
  models/     — Pydantic schemas shared across every layer

frontend/src/
  api/        — typed fetch client (incl. SSE stream reader) + types mirroring the backend schemas exactly
  components/ — UploadPanel, ChatPanel (streams responses token-by-token), SourceChip + hand-rolled UI primitives
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
- **Conversation memory** is session-scoped and in-memory only (`ConversationStore`,
  keyed by the frontend's per-tab `session_id`) — prior turns' plain Q&A text
  is replayed to Claude on each new turn, but old _retrieved context_ is not
  re-sent, so tokens don't grow unboundedly across a long conversation. Resets
  on backend restart; a real deployment would move this to Redis.
- **Streaming** (`POST /chat/stream`, SSE) runs the same guardrail →
  retrieve → rerank pipeline as the plain endpoint, then streams Claude's
  answer token-by-token. Sources arrive in a `context` event _before_ the
  text starts streaming, so the UI can show what's grounding the answer
  while it's still being generated — one retry policy applies to the plain
  endpoint's single call, not the stream (retrying after partial output was
  already sent to the client would duplicate it; a stream failure instead
  surfaces as an `error` SSE event).

## What's cut, on purpose

See PRD.md §2 for the full list (multi-user auth, cloud deployment, OCR,
production rate limiting). Nothing here is a forgotten feature — it's a
named tradeoff given the 2-day budget.

## AI tools in your development process

I started by writing a detailed project instruction that codified my architecture decisions, coding standards, and constraints before writing any application code. This instruction acted as a living spec — Claude operated within it, not outside it.

My workflow was: I decide the what and why, Claude drafts the how, I review and refine. Every architectural choice (no LangChain, FastAPI over Flask, chunking strategy, prompt design) was mine. Claude accelerated implementation.

My rules for AI-assisted development:

Never accept generated code without verifying
Never let it pick architecture — that's my job
Use it for boilerplate and repetitive patterns where the spec is clear
Don't use it for README/documentation — those must be my actual thoughts
When it produces something I don't understand, that's a red flag, not a feature

What this changes about engineering: the skill shifts from "can you write code" to "can you specify precisely what you want, evaluate what you get, and maintain coherence across a codebase." The project instruction I wrote is arguably the most important artifact in this rep

## Testing

```bash
# backend
cd backend && source .venv/bin/activate && pytest -q

# frontend
cd frontend && npm run build && npx vitest run
```

Backend: 37 tests covering chunking metadata/overlap, the document-type
query classifier, BM25+RRF fusion recovering a keyword-strong chunk a
semantic-only search missed, cross-encoder rerank ordering, conversation
history (ordering, per-session isolation, eviction once the turn cap is
hit), guardrails (no-documents, rate-limit-trip, oversized upload,
empty/unsupported file), and route-level integration tests against the real
FastAPI app (`tests/test_api_routes.py`, via `httpx.ASGITransport` +
`app.dependency_overrides`) — all run without hitting the network or
needing an API key.

That last file exists because of a real bug it caught: `get_chat_service`
and `get_ingestion_service` originally took a plain `settings: Settings |
None = None` parameter for testing convenience. FastAPI's dependency
resolver doesn't know that's "just a default" — a Pydantic-model-typed
parameter with no `Depends`/`Query`/`Path` marker is exactly what it treats
as an _implicit second request-body field_. `POST /chat` and `POST
/documents` were silently expecting `{"request": {...}, "settings": {...}}`
instead of a flat body, and every unit test up to that point was calling the
service classes directly, so nothing exercised the actual HTTP request
parsing to notice. Fixed by resolving `settings` via `Depends(get_settings)`
like everything else in `deps.py` — see the comment in `app/api/deps.py` for
the full explanation. It's a good example of why route-level tests earn
their keep even when the unit tests below them are solid.

Frontend: a build/type-check pass plus ChatPanel tests that drive the same
event-callback shape `streamChatMessage` uses in production (context →
delta → done, and a guardrail `error` event), and that the input is
disabled until both a resume and a JD are uploaded.

## What I'd do with more time

- A learned/trained query classifier instead of the current keyword
  heuristic for document-type filtering
- Move conversation memory and rate limiting to Redis, for multi-instance
  deployment and so history survives a restart
- A real job queue for ingestion (currently synchronous per-request)
- Broader edge-case test coverage (malformed uploads, concurrent ingestion,
  a stream that disconnects mid-answer)
- Per-document delete/re-index without a full re-ingest
