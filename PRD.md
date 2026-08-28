# Career Intelligence Assistant — Product Requirements Document

**Author:** JK · **Purpose:** Forward Deployed Engineer take-home assignment
**Date:** 2026-08-28 · **Build window:** 2 calendar days, ~6-8 focused hours total
**Status:** Locked scope for build — do not re-open scope debates mid-build; log new ideas in the Parking Lot (§10) instead

---

## 1. What's actually being evaluated

The assignment brief is short — analyze a resume against job descriptions, answer fit/gap/interview-prep questions — but the evaluation is not about feature count. It's about:

- Engineering philosophy: can I make good tradeoffs under a real time constraint and explain them
- Clean architecture: layers, DI, types — not a script that works
- RAG approach: are chunking/retrieval/prompting decisions deliberate or default
- Observability: can the reviewer see *why* the system answered the way it did
- Use of AI tooling: shown through this PRD and commit history, not just the final code

Given only 6-8 hours, the single biggest risk is **an ambitious architecture that never reaches a working end-to-end demo**. This PRD is built to fail safe: every hour has a working, demoable system at the end of it, and everything past the P0 line is explicitly optional. A smaller thing that runs cleanly and is well-reasoned beats a larger thing that's half-wired.

---

## 2. Scope: P0 / P1 / Cut

### P0 — Must exist for the demo to make sense (this is the whole time budget)

1. Upload one resume (PDF or plain text) and one-or-more job descriptions (PDF or plain text)
2. Documents are chunked, tagged with metadata (`source_type`, `document_id`, `section` if detectable), embedded locally, and stored in a persistent vector store
3. A chat endpoint that answers grounded questions using retrieved chunks, with the three example query types working end-to-end:
   - "What skills am I missing for this role?" (gap analysis, resume vs. one JD)
   - "How does my experience align with Job #2?" (alignment, resume vs. a named JD)
   - One interview-prep style question (e.g. "What should I prepare to talk about for this role?")
4. Responses cite which document/section they drew from (retrieval transparency, not full citation UI)
5. Guardrail: if no resume or no JD is uploaded, or the question is unanswerable from the documents, the assistant says so instead of hallucinating
6. Structured logging + timing on retrieval and the Claude call, visible in the terminal/log file
7. A minimal but clean chat UI: upload panel + chat panel, obviously usable without instructions
8. `docker-compose up` brings up backend + frontend with zero manual steps beyond `.env` keys
9. README documenting architecture, decisions, and how to run it (written as we go, not backfilled)

### P1 — Do only if P0 is done with time to spare

10. DOCX upload support (PDF + txt cover the demo; DOCX parsing is mechanical, add it last)
11. Keyword (BM25) + semantic hybrid retrieval with simple score fusion, instead of semantic-only
12. Section-aware semantic chunking (split on resume headers / bullet groups) instead of pure recursive character splitting
13. A handful of pytest tests on the pipeline (chunking metadata, retrieval filtering) and one frontend test
14. Basic per-request token usage tally surfaced in the UI or logs

### Explicitly cut — named on purpose, not forgotten

- **Cross-encoder re-ranking model.** A real re-ranker needs a second model download and adds latency for marginal gains at this corpus size (a handful of short documents). Replaced with reciprocal-rank fusion of the semantic + keyword scores, which is a legitimate lightweight re-ranking strategy and costs near-zero extra time. Documented in the README as a conscious tradeoff, with the real re-ranker named as the first thing to add post-take-home.
- **Multi-user auth, persistence across sessions beyond local disk, cloud deployment.** Out of scope for a take-home; noted as "not built" rather than silently absent.
- **Rate limiting as a real token-bucket/middleware.** A simple in-process request counter with a 429 response is enough to demonstrate the guardrail exists; a production rate limiter (Redis-backed) is named as a next step.
- **OCR / scanned-PDF support.** Assume text-layer PDFs, which is a reasonable assumption to state explicitly rather than silently fail on.

---

## 3. Tech stack decisions

| Layer | Choice | Why (given the 6-8 hour budget) |
|---|---|---|
| Backend | Python + FastAPI | Given; async-friendly, typed via Pydantic, fast to scaffold |
| Frontend | React + TypeScript | Given; shadcn/ui for components so time goes into UX decisions, not CSS |
| LLM | Claude API, `claude-sonnet-4-6` | Given |
| Embeddings | **`sentence-transformers/all-MiniLM-L6-v2`, run locally** | Zero API key, zero marginal cost, no network dependency for the evaluator to configure — the single biggest reliability win for a take-home a stranger has to run cold. 384-dim, ~80MB, fast on CPU, more than adequate quality for a handful of short documents. The embedding call is wrapped behind an `EmbeddingProvider` interface (see §4) so swapping in Voyage AI or OpenAI later is a one-line config change, not a rewrite — this is the DI principle from the project instructions applied to a real decision, and worth calling out explicitly in the README as evidence of that. |
| Vector store | **ChromaDB (embedded, persistent-on-disk)** | No separate server process to stand up under time pressure, ships as a single `pip install`, supports metadata filtering natively (needed for the `source_type` filter — "my experience" → resume chunks only), and persists to a mounted volume so `docker-compose` restarts don't lose ingested documents. FAISS was considered but has no native metadata filtering, which the requirements explicitly need. |
| Keyword retrieval (P1) | `rank_bm25` | In-process, no extra service, trivial to fuse with Chroma's cosine scores via reciprocal rank fusion |
| Containerization | Docker + docker-compose | Given |
| Testing | pytest / vitest | Given; scoped narrowly (§7) rather than chasing coverage |
| Observability | `structlog` + a timing decorator on retrieval and LLM calls | Structured JSON logs, not print statements — matches the "observability built-in" principle directly |

---

## 4. Architecture

```
frontend/ (React + TS)
  ├─ UploadPanel        — resume + JD upload, shows parsed doc list
  ├─ ChatPanel          — question input, streamed/plain answer, source chips
  └─ api/client.ts       — typed fetch wrappers matching backend schemas

backend/ (FastAPI)
  ├─ api/                — routes only: request/response models, status codes, no business logic
  │    ├─ documents.py    — POST /documents (upload), GET /documents
  │    └─ chat.py         — POST /chat
  ├─ services/            — orchestration layer, injected into routes via FastAPI Depends
  │    ├─ ingestion_service.py   — parse → chunk → embed → store
  │    └─ chat_service.py        — retrieve → assemble context → call LLM → guardrail-check response
  ├─ rag/                 — the actual RAG mechanics, each piece independently testable
  │    ├─ parsers.py       — PDF/DOCX/text extraction, one function per format behind a common interface
  │    ├─ chunking.py      — semantic-first, recursive-split fallback, configurable size/overlap
  │    ├─ embeddings.py    — EmbeddingProvider interface; LocalEmbeddingProvider implementation
  │    ├─ vector_store.py  — VectorStore interface; ChromaVectorStore implementation
  │    └─ retrieval.py     — semantic search + (P1) BM25 fusion + document-type filtering
  ├─ llm/
  │    └─ claude_client.py — thin wrapper: system prompt construction, call, token/timing logging
  ├─ core/
  │    ├─ config.py        — Pydantic Settings, reads .env, sensible defaults for every tunable
  │    └─ logging.py        — structlog setup
  └─ models/               — Pydantic schemas shared across layers (Document, Chunk, ChatRequest, ChatResponse, SourceRef)
```

Every arrow in this diagram is an interface, not a concrete class, at the two points that most plausibly change after the take-home is submitted: `EmbeddingProvider` and `VectorStore`. That's the DI principle applied where it actually earns its keep, rather than sprinkled everywhere for its own sake — worth one sentence in the README explaining that choice, since "dependency injection everywhere" without a reason reads as cargo-culting.

---

## 5. RAG pipeline specifics

**Document processing.** PDF via `pypdf` (or `pdfplumber` if `pypdf` mangles layout — decide during build, log the choice), DOCX via `python-docx` (P1), plain text passthrough. Each parsed document is tagged with `document_id` (uuid), `source_type` (`resume` | `job_description`), and a human label (filename or "Job #2" style index for JDs, since the example queries reference JDs by number).

**Chunking.** Default: split resumes/JDs on structural cues first — headers (`EXPERIENCE`, `SKILLS`, `EDUCATION`, etc.), bullet groups — and fall back to `RecursiveCharacterTextSplitter`-style splitting (LangChain's splitter is fine to use as a utility even though the rest of the pipeline is hand-rolled) for anything unstructured. `CHUNK_SIZE` (default 500 tokens) and `CHUNK_OVERLAP` (default 50 tokens) are `.env`-configurable. Each chunk keeps `document_id`, `source_type`, and `section` metadata through to storage.

**Retrieval.** P0: top-k semantic search (k configurable, default 5) against Chroma, with a metadata filter applied when the query clearly targets one type — "my experience/background" → `source_type=resume`, "Job #2 / this role / the posting" → `source_type=job_description` matched to the right `document_id`. The filter is a small rule-based classifier on the query (keyword match), not an LLM call — cheap and fast, and honest about being heuristic in the README. P1: fuse in BM25 keyword scores via reciprocal rank fusion before taking top-k, which is the "re-ranking step" the requirements ask for, done at a cost the time budget can afford. Every retrieved chunk carries its similarity score into the response payload for observability.

**Prompt construction.** System prompt fixes the assistant's role ("career advisor analyzing only the specific documents provided — never speculate beyond them") and response format (short lead-in, then structured breakdown — never a wall of text). Context is assembled with explicit delimiters, e.g.:

```
<resume>
[chunk text] (source: resume.pdf, section: Experience)
</resume>

<job_description id="Job #2">
[chunk text] (source: job2.pdf, section: Requirements)
</job_description>
```

Gap-analysis queries get an explicit chain-of-thought instruction ("first list the role's key requirements, then check each against the resume context, then summarize gaps") rather than asking for the answer directly — this is the one place extra prompt effort clearly pays off in answer quality. The system prompt explicitly instructs: if the retrieved context doesn't contain the answer, say so rather than filling the gap from general knowledge.

**Guardrails.** Upload validation: file type allowlist (`.pdf`, `.txt`, and `.docx` if P1 lands), size cap (`MAX_FILE_SIZE_MB` env var, default 10MB). Chat validation: reject/short-circuit if zero documents uploaded, or if uploaded but the wrong type for the question (e.g. asked to compare against "Job #2" but only one JD exists) — clear error message, not a 500. Rate limiting: an in-process sliding-window counter per session on `/chat`, returning 429 past a configurable threshold — enough to demonstrate the principle within the time budget.

**Error handling.** A small set of typed exceptions (`DocumentParseError`, `EmptyContextError`, `RateLimitExceeded`, etc.) raised in `rag/`/`services/` and caught by a single FastAPI exception handler that maps each to a structured JSON error response (`{ error_code, message, detail }`) and a log line with context — never a bare `except Exception` swallowing the real cause, and never a raw 500 with a stack trace reaching the frontend.

---

## 6. API contract (P0 surface)

```
POST /documents
  multipart/form-data: file, source_type (resume|job_description), label?
  → 201 { document_id, source_type, label, chunk_count }

GET /documents
  → 200 [ { document_id, source_type, label, uploaded_at, chunk_count } ]

POST /chat
  { message: string, session_id?: string }
  → 200 {
      answer: string,
      sources: [ { document_id, label, section?, score } ],
      timing: { retrieval_ms, llm_ms },
      token_usage: { input_tokens, output_tokens }
    }
```

All models are Pydantic on the backend and mirrored as TypeScript interfaces on the frontend — no `any`, no hand-parsed JSON.

---

## 7. UI/UX (P0)

Two-panel layout, shadcn/ui components:

- **Left — Document panel:** drag-and-drop upload, list of uploaded documents with type badge (Resume / JD) and label, delete action, empty state that explains what to upload
- **Right — Chat panel:** message thread, each assistant answer followed by a small "Sources" row (chip per source doc/section, expandable to show the retrieved snippet and score) — this is the single UX detail that most directly demonstrates the RAG pipeline is real and not a canned response
- Loading state during retrieval+generation (skeleton or spinner, not a frozen screen)
- Disabled chat input with a one-line explanation until at least one resume and one JD are uploaded (surfaces the guardrail in the UI, not just as an API error)

Anything beyond this (multi-session history, theming, animations) is explicitly deferred — the UI needs to read as deliberate and uncluttered, not maximal.

---

## 8. Testing plan (P1, scoped narrowly)

Given the time budget, tests target the parts most likely to silently break and hardest to eyeball-verify, not coverage percentage:

- `test_chunking.py` — chunk metadata survives chunking (document_id/source_type/section attached correctly); overlap behaves as configured
- `test_retrieval.py` — metadata filter correctly restricts results when a query implies a document type
- One `vitest` test on the ChatPanel — renders sources correctly given a mocked API response

No attempt at end-to-end/integration test automation within this budget; the demo script (§9) is the manual integration test.

---

## 9. Definition of done / demo script

The submission is done when this script runs cleanly, cold, on a machine that has only Docker and a Claude API key:

1. `docker-compose up` — no other setup
2. Upload one resume, two job descriptions
3. Ask "What skills am I missing for this role?" targeting one JD → grounded, structured answer with source chips
4. Ask "How does my experience align with Job #2?" → answer correctly scoped to the second JD specifically
5. Ask an interview-prep question → structured, actionable answer
6. Ask something unanswerable from the documents (e.g. "What's the company's revenue?") → assistant says it can't answer from the provided documents, doesn't hallucinate
7. Terminal shows structured logs with timing for each of the above calls

If all seven pass, ship it — polishing beyond this point should go toward the README (documenting decisions, tradeoffs, and the "what I'd do with more time" list), since that document is doing as much evaluation work as the code.

---

## 10. Hour-by-hour plan (~7 hrs across 2 evenings)

**Evening 1 (~4 hrs) — pipeline that works end-to-end, even if ugly**

- 0:00–0:30 — repo scaffold: FastAPI + React skeletons, Docker/docker-compose, `.env.example`, config loading
- 0:30–1:15 — `rag/parsers.py` (PDF + txt) and `rag/chunking.py`, sanity-checked against a real resume/JD in a scratch script (not yet wired to API)
- 1:15–2:00 — `rag/embeddings.py` (local provider) + `rag/vector_store.py` (Chroma) behind interfaces; `POST /documents` wired end-to-end
- 2:00–2:45 — `rag/retrieval.py` (semantic search + type filter) + `llm/claude_client.py` + system prompt v1; `POST /chat` wired end-to-end, tested from curl/Postman, not yet the UI
- 2:45–3:15 — guardrails: upload validation, empty-context handling, basic rate limit
- 3:15–4:00 — structlog setup + timing decorators on retrieval/LLM calls; commit; write README architecture section while it's fresh

**Evening 2 (~3-4 hrs) — make it demoable and defensible**

- 0:00–1:00 — frontend UploadPanel + ChatPanel wired to the real API (skip mocking — the backend already works)
- 1:00–1:30 — sources UI (chips + expandable snippet/score) — the highest-leverage UX item
- 1:30–2:00 — run the demo script (§9) end-to-end, fix whatever breaks
- 2:00–2:30 — *if time remains:* P1 items in order of value — BM25 fusion first (visible in retrieval quality), then pytest tests, then DOCX support
- 2:30–3:00 — README pass: decisions log, tradeoffs, "what I cut and why," "what I'd do with more time"
- 3:00–3:30 — buffer / polish / re-record or re-run the demo script clean

If Evening 1 runs long, the P1 block is the thing that gets dropped, not the README — an unpolished but honest README beats an extra feature the evaluator has to discover on their own.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| PDF text extraction is messy (multi-column resumes, tables) | Test against the actual resume being used for the demo early (Evening 1, step 2); fall back to `pdfplumber` if `pypdf` output is garbled; document the limitation rather than over-engineering extraction |
| Local embedding model download is slow/blocked in Docker build | Bake the model into the Docker image at build time (not first-request), so `docker-compose up` doesn't hang on a cold download during the actual evaluation |
| Query-type classification (resume vs. JD filter) misfires | Keep it as an additive filter, not exclusionary — if classification is ambiguous, search across all documents rather than wrongly narrowing and returning nothing |
| Running out of time before frontend is wired | Backend is fully demoable via curl/Swagger (`/docs`) at the end of Evening 1 regardless — worst case, the demo falls back to that with an honest note in the README |
| Claude API latency/rate limits during the live demo | Log and surface `llm_ms` timing so slowness is visible and explained, not silently confusing |

---

## 12. Parking lot (ideas deliberately deferred, not lost)

- Real cross-encoder re-ranking
- Multi-turn conversation memory across chat sessions
- Voyage AI / OpenAI embedding swap-in (interface already supports it — just needs the provider class and an API key)
- Streaming responses in the chat UI
- Per-document delete/re-index without a full re-ingest
- Redis-backed rate limiting
