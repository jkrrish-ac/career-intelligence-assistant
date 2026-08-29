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

`docker-compose up` also starts a `redis` service and a `worker` service
alongside `backend`/`frontend`. Both are opt-in at the application level —
the backend defaults to in-process rate limiting/conversation memory and
synchronous ingestion (see "Recent additions" below) — but the containers
themselves start either way, so flipping `REDIS_URL`/`INGESTION_MODE` in
`.env` doesn't require restructuring your compose setup, just restarting
the backend.

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
  rag/        — parsers, chunking, embeddings, vector_store, retrieval, reranker, query_classifier
  llm/        — Claude client: system prompt, context assembly, the API call (plain + streaming + classify)
  core/       — config (env-driven settings), logging (structlog + timing), exceptions, rate limiting
  models/     — Pydantic schemas shared across every layer
  worker.py   — arq worker entrypoint for async ingestion (INGESTION_MODE=async only)

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
- **Rate limiting and conversation memory** default to in-process
  implementations (fine for a single instance), and switch to Redis-backed
  ones automatically when `REDIS_URL` is set — see "Recent additions" below.
- **Conversation memory** is session-scoped, keyed by the frontend's
  per-tab `session_id` — prior turns' plain Q&A text is replayed to Claude
  on each new turn, but old *retrieved context* is not re-sent, so tokens
  don't grow unboundedly across a long conversation.
- **Streaming** (`POST /chat/stream`, SSE) runs the same guardrail →
  retrieve → rerank pipeline as the plain endpoint, then streams Claude's
  answer token-by-token. Sources arrive in a `context` event *before* the
  text starts streaming, so the UI can show what's grounding the answer
  while it's still being generated — one retry policy applies to the plain
  endpoint's single call, not the stream (retrying after partial output was
  already sent to the client would duplicate it; a stream failure instead
  surfaces as an `error` SSE event).

## Recent additions (post-P1)

Five follow-on engineering items from the original "what I'd do with more
time" list, now implemented — each one opt-in via a config flag, so the
synchronous/in-process path this take-home was built and tested against
keeps working unchanged by default:

- **LLM-based query classifier** (`QUERY_CLASSIFIER_MODE=llm`, the
  default): replaces the keyword heuristic in `classify_query_target()`
  with a small, schema-constrained Claude tool-call (`ClaudeClient.classify`,
  `app/rag/query_classifier.py`) that picks which uploaded document a
  question targets. The original heuristic was **not removed** — it's the
  automatic fallback on any failure or timeout
  (`QUERY_CLASSIFIER_TIMEOUT_SECONDS`, default 3s), and it's what
  `QUERY_CLASSIFIER_MODE=heuristic` uses directly. The tradeoff, named
  rather than hidden: one extra small Claude call (latency + cost) per
  question, in exchange for handling phrasing the regex never anticipated.
- **Redis for conversation memory and rate limiting** (`REDIS_URL`, unset
  by default): `RedisConversationStore` and `RedisRateLimiter`
  (`app/services/conversation_store.py`, `app/core/rate_limit.py`)
  implement the exact same interface as their in-process counterparts, so
  `ChatService` and every caller are unaware which is running. Rate
  limiting uses a Redis sorted set (`ZADD`/`ZREMRANGEBYSCORE`/`ZCARD`);
  conversation history uses a capped, TTL'd Redis `LIST`
  (`LPUSH`/`LTRIM`). `docker-compose.yml` runs a `redis` service and points
  the backend at it automatically; parity between both implementations is
  tested directly in `tests/test_redis_backed_stores.py` via `fakeredis`,
  not just asserted in a docstring.
- **A real job queue for ingestion** (`INGESTION_MODE=async`, default
  `sync`): `POST /documents` returns `202` immediately with a `pending`
  document instead of blocking the request on parse/chunk/embed. An `arq`
  (Redis-backed) worker (`app/worker.py`, the `worker` service in
  `docker-compose.yml`) picks up the job and runs the exact same pipeline
  out of band via `IngestionService.process_pending()`, flipping the
  document to `ready` (or `failed`, with `error_message` set — a
  worker never crashes or infinitely retries on one bad file) when it's
  done. `GET /documents/{document_id}` lets a client poll a single
  document's status. Requires Redis, so it's off by default; the frontend
  doesn't poll for this yet (see PRD.md §12).
- **Per-document delete/re-index without a full re-ingest**: `PUT
  /documents/{document_id}` (`IngestionService.reindex_document`) replaces
  a document's chunks in place — same document_id, same position in the
  list — instead of the old delete-then-re-upload dance. Not fully atomic
  (a query landing between the delete and the re-add would briefly see
  zero chunks for that document); named in the method's docstring as an
  acceptable tradeoff for this traffic pattern.
- **Broader edge-case test coverage** (`tests/test_edge_cases.py`):
  malformed uploads hitting the *real* parsers (a zero-byte file, garbage
  content named `.pdf`/`.docx`) to confirm they map to `422` rather than an
  unhandled `500`; concurrent ingestion via `asyncio.gather()` against one
  shared `DocumentRegistry`, exercising its `threading.Lock` for the first
  time; and a stream-disconnect test that drives `ChatService.stream_answer`
  by hand and calls `.aclose()` on it mid-stream — the direct way to test
  what a real client disconnect relies on, since `httpx`'s `ASGITransport`
  turns out to fully buffer the response before the client sees anything
  (see the test file's comments for why that rules out a transport-level
  test here).

## What's cut, on purpose

See PRD.md §2 for the full list (multi-user auth, cloud deployment, OCR).
Production-grade rate limiting is no longer on this list — see "Recent
additions" above. Nothing here is a forgotten feature — it's a named
tradeoff given the original 2-day budget.

## Testing

```bash
# backend
cd backend && source .venv/bin/activate && pytest -q

# frontend
cd frontend && npm run build && npx vitest run
```

Backend: 75 tests covering chunking metadata/overlap, the document-type
query classifier (both the keyword heuristic and the LLM-based classifier,
including its fallback-on-failure and fallback-on-timeout paths), BM25+RRF
fusion recovering a keyword-strong chunk a semantic-only search missed,
cross-encoder rerank ordering, conversation history (ordering, per-session
isolation, eviction once the turn cap is hit -- run against both the
in-process and Redis/`fakeredis`-backed implementations to prove parity),
the Redis-backed rate limiter (same parity approach), guardrails
(no-documents, rate-limit-trip, oversized upload, empty/unsupported file),
per-document re-indexing (`PUT /documents/{id}`, keeps its document_id, 404s
on an unknown one), async ingestion (the arq task function, the
pending->ready/failed status transitions, the `202` route branch, and that
the default `INGESTION_MODE=sync` behaves exactly as before), edge cases
(malformed uploads hitting the real parsers, concurrent ingestion, a
stream-consumer disconnecting mid-answer), and route-level integration
tests against the real FastAPI app (`tests/test_api_routes.py`, via
`httpx.ASGITransport` + `app.dependency_overrides`) — all run without
hitting the network or needing an API key or a real Redis/arq worker.

That last file exists because of a real bug it caught: `get_chat_service`
and `get_ingestion_service` originally took a plain `settings: Settings |
None = None` parameter for testing convenience. FastAPI's dependency
resolver doesn't know that's "just a default" — a Pydantic-model-typed
parameter with no `Depends`/`Query`/`Path` marker is exactly what it treats
as an *implicit second request-body field*. `POST /chat` and `POST
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

The five items that used to be listed here (a learned query classifier,
Redis-backed memory/rate limiting, a real ingestion job queue, broader
edge-case tests, per-document re-index) are now implemented — see "Recent
additions" above. What's next after that:

- A cross-process-safe document registry (a real database, or at least file
  locking) — `documents.json` is now written by two processes (the backend
  and the arq worker) with only in-process locking; named explicitly in
  `document_registry.py`'s docstring rather than silently assumed
- Frontend support for the post-P1 additions: a "re-index" action next to
  each document, and a "processing…" state driven by polling
  `GET /documents/{id}` when `INGESTION_MODE=async` — both backend
  endpoints exist and are tested, but nothing in the UI calls them yet
- A fully trained (not zero-shot) query classifier, if per-query LLM
  latency/cost ever becomes a problem at real traffic volume
- Voyage AI / OpenAI embedding swap-in (the `EmbeddingProvider` interface
  already supports it — just needs the provider class and an API key)
