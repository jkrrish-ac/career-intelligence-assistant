# Career Intelligence Assistant — Product Requirements Document

**Author:** JK · **Purpose:** Forward Deployed Engineer take-home assignment
**Date:** 2026-08-28 · **Build window:** 2 calendar days, ~6-8 focused hours *per day* (~12-16 hours total)
**Status:** Locked scope for build — do not re-open scope debates mid-build; log new ideas in the Parking Lot (§10) instead

---

## 1. What's actually being evaluated

The assignment brief is short — analyze a resume against job descriptions, answer fit/gap/interview-prep questions — but the evaluation is not about feature count. It's about:

- Engineering philosophy: can I make good tradeoffs under a real time constraint and explain them
- Clean architecture: layers, DI, types — not a script that works
- RAG approach: are chunking/retrieval/prompting decisions deliberate or default
- Observability: can the reviewer see *why* the system answered the way it did
- Use of AI tooling: shown through this PRD and commit history, not just the final code

Even with ~12-16 hours, the single biggest risk is **an ambitious architecture that never reaches a working end-to-end demo**. This PRD is built to fail safe: every session has a working, demoable system at the end of it, and everything past the P0 line is explicitly optional. A smaller thing that runs cleanly and is well-reasoned beats a larger thing that's half-wired — the extra time (vs. a bare-minimum take-home) goes toward doing the RAG pipeline properly (real hybrid retrieval, real re-ranking) rather than toward extra surface-level features.

---

## 2. Scope: P0 / P1 / Cut

### P0 — Must exist for the demo to make sense (targets ~10-12 of the ~12-16 hours)

1. Upload a resume and multiple job descriptions, in PDF, DOCX, or plain text
2. Section-aware chunking (headers/bullet groups first, recursive-split fallback), each chunk tagged with `source_type`, `document_id`, `section`, embedded locally, stored in a persistent vector store
3. **Hybrid retrieval**: semantic search (Chroma) fused with keyword search (BM25) via reciprocal rank fusion, then **re-ranked by a local cross-encoder** before context assembly — this is the real thing the requirements ask for, not a placeholder, because the extra time budget makes it feasible
4. Document-type filtering by query intent ("my experience" → resume chunks; "Job #2" → that JD's chunks)
5. A chat endpoint that answers grounded questions, with the three example query types working end-to-end:
   - "What skills am I missing for this role?" (gap analysis, resume vs. one JD, chain-of-thought prompted)
   - "How does my experience align with Job #2?" (alignment, resume vs. a named JD)
   - One interview-prep style question (e.g. "What should I prepare to talk about for this role?")
6. Responses cite which document/section/score they drew from (retrieval transparency, surfaced in the UI, not just the API)
7. Guardrails: upload validation (type/size), empty-context handling, unanswerable-question handling, rate limiting, typed errors — see §5
8. Structured logging + timing (retrieval, rerank, LLM call) + token usage tracking
9. A polished two-panel UI (upload + chat) with source chips, loading states, and the "no documents yet" guardrail surfaced visually
10. `docker-compose up` brings up backend + frontend with zero manual steps beyond `.env` keys (embedding model baked into the image, not downloaded on first request)
11. Core test suite: chunking metadata, retrieval filtering, rerank ordering (pytest) + one frontend test (vitest)
12. README documenting architecture, decisions, and how to run it (written as we go, not backfilled)

### P1 — Do only if P0 is done with time to spare (the ~2-4 hour buffer)

13. Session-scoped multi-turn conversation memory (chat history passed back to Claude, not just single-shot Q&A)
14. Token usage displayed in the UI itself, not just logs
15. A small "why this answer" debug affordance in the UI showing retrieved-vs-reranked scores side by side
16. Streaming responses in the chat UI
17. Broader edge-case test coverage beyond the P0 core suite

### Explicitly cut — named on purpose, not forgotten

- **Multi-user auth, persistence beyond local disk, cloud deployment.** Out of scope for a take-home; noted as "not built" rather than silently absent.
- **Production-grade rate limiting.** An in-process sliding-window counter demonstrates the guardrail; a Redis-backed token-bucket limiter is named as the next step.
- **OCR / scanned-PDF support.** Assume text-layer PDFs, stated explicitly rather than silently failing on scans.
- **Fine-tuned or hosted embedding models.** Local MiniLM is the default per §3; swapping in a hosted model is a config change, not a build-time task, so it's not attempted here.

---

## 3. Tech stack decisions

| Layer | Choice | Why (given the ~12-16 hour budget) |
|---|---|---|
| Backend | Python + FastAPI | Given; async-friendly, typed via Pydantic, fast to scaffold |
| Frontend | React + TypeScript | Given; shadcn/ui for components so time goes into UX decisions, not CSS |
| LLM | Claude API, `claude-sonnet-4-6` | Given |
| Embeddings | **`sentence-transformers/all-MiniLM-L6-v2`, run locally** | Zero API key, zero marginal cost, no network dependency for the evaluator to configure — the single biggest reliability win for a take-home a stranger has to run cold. 384-dim, ~80MB, fast on CPU, more than adequate quality for a handful of short documents. The embedding call is wrapped behind an `EmbeddingProvider` interface (see §4) so swapping in Voyage AI or OpenAI later is a one-line config change, not a rewrite — this is the DI principle from the project instructions applied to a real decision, and worth calling out explicitly in the README as evidence of that. |
| Vector store | **ChromaDB (embedded, persistent-on-disk)** | No separate server process to stand up under time pressure, ships as a single `pip install`, supports metadata filtering natively (needed for the `source_type` filter — "my experience" → resume chunks only), and persists to a mounted volume so `docker-compose` restarts don't lose ingested documents. FAISS was considered but has no native metadata filtering, which the requirements explicitly need. |
| Keyword retrieval | `rank_bm25` | In-process, no extra service, fused with Chroma's cosine scores via reciprocal rank fusion — the extra budget makes real hybrid retrieval (not just semantic) achievable as P0 |
| Re-ranking | Local cross-encoder, `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers) | Same library family as the embedding model, so no new dependency; ~80MB, runs on CPU fast enough for a handful of fused candidates. With the larger time budget this replaces a heuristic fusion-only approach and directly satisfies the "re-ranking step before context assembly" requirement rather than approximating it |
| Containerization | Docker + docker-compose | Given |
| Testing | pytest / vitest | Given; scoped to the pipeline's highest-risk seams (§8) rather than chasing coverage |
| Observability | `structlog` + a timing decorator on retrieval, rerank, and LLM calls | Structured JSON logs, not print statements — matches the "observability built-in" principle directly |

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
  │    ├─ retrieval.py     — semantic search + BM25 fusion (RRF) + document-type filtering
  │    └─ reranker.py      — cross-encoder re-ranking of fused candidates before context assembly
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

**Document processing.** PDF via `pypdf` (or `pdfplumber` if `pypdf` mangles layout — decide during build, log the choice), DOCX via `python-docx`, plain text passthrough — all three are P0 given the larger budget. Each parsed document is tagged with `document_id` (uuid), `source_type` (`resume` | `job_description`), and a human label (filename or "Job #2" style index for JDs, since the example queries reference JDs by number).

**Chunking.** Default: split resumes/JDs on structural cues first — headers (`EXPERIENCE`, `SKILLS`, `EDUCATION`, etc.), bullet groups — and fall back to `RecursiveCharacterTextSplitter`-style splitting (LangChain's splitter is fine to use as a utility even though the rest of the pipeline is hand-rolled) for anything unstructured. `CHUNK_SIZE` (default 500 tokens) and `CHUNK_OVERLAP` (default 50 tokens) are `.env`-configurable. Each chunk keeps `document_id`, `source_type`, and `section` metadata through to storage.

**Retrieval.** Top-k semantic search (k configurable, default 8-10 to give the fusion/rerank stages enough candidates) against Chroma, combined with BM25 keyword search over the same chunk set, merged via reciprocal rank fusion. A metadata filter narrows candidates *before* fusion when the query clearly targets one document type — "my experience/background" → `source_type=resume`, "Job #2 / this role / the posting" → `source_type=job_description` matched to the right `document_id`. The filter is a small rule-based classifier on the query (keyword match), not an LLM call — cheap, fast, and honestly documented as heuristic rather than dressed up as more than it is. The fused candidate set is then passed through the cross-encoder re-ranker (§3) to produce the final top-k (default 5) used for context assembly. Every chunk carries both its fused-retrieval score and its re-rank score into the response payload for observability — this is the concrete detail that lets the UI (§7) show *why* an answer was grounded the way it was, not just that it was.

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

**Guardrails.** Upload validation: file type allowlist (`.pdf`, `.txt`, `.docx`), size cap (`MAX_FILE_SIZE_MB` env var, default 10MB). Chat validation: reject/short-circuit if zero documents uploaded, or if uploaded but the wrong type for the question (e.g. asked to compare against "Job #2" but only one JD exists) — clear error message, not a 500. Rate limiting: an in-process sliding-window counter per session on `/chat`, returning 429 past a configurable threshold — enough to demonstrate the principle within the time budget.

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
      sources: [ { document_id, label, section?, retrieval_score, rerank_score } ],
      timing: { retrieval_ms, llm_ms },
      token_usage: { input_tokens, output_tokens }
    }
```

All models are Pydantic on the backend and mirrored as TypeScript interfaces on the frontend — no `any`, no hand-parsed JSON.

---

## 7. UI/UX (P0)

Two-panel layout, shadcn/ui components:

- **Left — Document panel:** drag-and-drop upload, list of uploaded documents with type badge (Resume / JD) and label, delete action, empty state that explains what to upload
- **Right — Chat panel:** message thread, each assistant answer followed by a small "Sources" row (chip per source doc/section, expandable to show the retrieved snippet plus its retrieval and re-rank scores) — this is the single UX detail that most directly demonstrates the RAG pipeline is real and not a canned response
- Loading state during retrieval+generation (skeleton or spinner, not a frozen screen)
- Disabled chat input with a one-line explanation until at least one resume and one JD are uploaded (surfaces the guardrail in the UI, not just as an API error)

Anything beyond this (multi-session history, theming, animations) is explicitly deferred — the UI needs to read as deliberate and uncluttered, not maximal.

---

## 8. Testing plan

Tests target the parts most likely to silently break and hardest to eyeball-verify, not coverage percentage.

**P0 core suite:**
- `test_chunking.py` — chunk metadata survives chunking (document_id/source_type/section attached correctly); overlap behaves as configured
- `test_retrieval.py` — metadata filter correctly restricts candidates when a query implies a document type; RRF fusion produces a sane merged ordering on a known small case
- `test_reranker.py` — cross-encoder re-ranking actually changes/improves ordering on a constructed example where the "obviously more relevant" chunk starts lower in the fused list
- One `vitest` test on the ChatPanel — renders sources (with both scores) correctly given a mocked API response

**P1 (if time allows):** guardrail edge cases (empty upload, unanswerable question, rate-limit trip) as explicit backend tests rather than only manual demo-script checks.

No attempt at full end-to-end/integration test automation within this budget; the demo script (§9) is the manual integration test.

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

## 10. Hour-by-hour plan (~14 hrs across 2 days, 6-8 hrs/day)

**Day 1 (~7 hrs) — the full backend pipeline works end-to-end, even if the UI is still curl/Swagger**

- 0:00–0:30 — repo scaffold: FastAPI + React skeletons, Docker/docker-compose, `.env.example`, config loading
- 0:30–1:15 — `rag/parsers.py` (PDF + DOCX + txt) and `rag/chunking.py` (section-aware, recursive fallback), sanity-checked against a real resume/JD in a scratch script
- 1:15–2:00 — `rag/embeddings.py` (local provider) + `rag/vector_store.py` (Chroma) behind interfaces; `POST /documents` wired end-to-end
- 2:00–2:45 — `rag/retrieval.py`: BM25 index + reciprocal rank fusion with Chroma's semantic results; document-type query filter
- 2:45–3:30 — `rag/reranker.py`: cross-encoder re-ranking of the fused candidates; verify on a couple of hand-checked queries that ordering actually improves
- 3:30–4:15 — `llm/claude_client.py` + system prompt v1 + context assembly with delimiters and chain-of-thought instruction for gap-analysis queries; `POST /chat` wired end-to-end, tested via curl/Swagger
- 4:15–5:00 — guardrails: upload validation, empty-context/unanswerable handling, typed exceptions + FastAPI exception handler, rate limiting
- 5:00–5:45 — `structlog` setup + timing decorators (retrieval, rerank, LLM) + token usage tracking
- 5:45–7:00 — `test_chunking.py`, `test_retrieval.py`, `test_reranker.py`; commit; write the README architecture section while it's fresh

**Day 2 (~7 hrs) — frontend, polish, and the buffer items**

- 0:00–1:00 — UploadPanel wired to the real API (document list, type badges, delete)
- 1:00–2:00 — ChatPanel wired to the real API: message thread, loading states, disabled-input guardrail
- 2:00–2:45 — Sources UI: chips + expandable snippet showing both retrieval and re-rank scores — the highest-leverage UX item
- 2:45–3:30 — Run the demo script (§9) end-to-end on a clean `docker-compose up`; fix whatever breaks
- 3:30–4:00 — `vitest` test on ChatPanel + any remaining P0 backend edge-case tests
- 4:00–5:30 — P1 buffer, in priority order: token usage in UI → "why this answer" debug affordance → conversation memory → streaming (stop wherever the clock runs out; each is independently shippable)
- 5:30–6:30 — README pass: decisions log, tradeoffs, "what I cut and why," "what I'd do with more time"
- 6:30–7:00 — buffer / final polish / re-run the demo script clean one more time

If Day 1 runs long, the P1 buffer block on Day 2 is what shrinks, not the README or the demo-script pass — an unpolished-but-honest README beats an extra feature the evaluator has to discover on their own, and a clean demo beats a longer feature list.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| PDF text extraction is messy (multi-column resumes, tables) | Test against the actual resume being used for the demo early (Evening 1, step 2); fall back to `pdfplumber` if `pypdf` output is garbled; document the limitation rather than over-engineering extraction |
| Local embedding model download is slow/blocked in Docker build | Bake the model into the Docker image at build time (not first-request), so `docker-compose up` doesn't hang on a cold download during the actual evaluation |
| Query-type classification (resume vs. JD filter) misfires | Keep it as an additive filter, not exclusionary — if classification is ambiguous, search across all documents rather than wrongly narrowing and returning nothing |
| Running out of time before frontend is wired | Backend is fully demoable via curl/Swagger (`/docs`) at the end of Day 1 regardless — worst case, the demo falls back to that with an honest note in the README |
| Claude API latency/rate limits during the live demo | Log and surface `llm_ms` timing so slowness is visible and explained, not silently confusing |
| Cross-encoder rerank adds noticeable latency on top of retrieval + LLM call | Keep the fused candidate set small (≤10) going into the reranker so it stays a low double-digit-ms step; if it's still a problem, log `rerank_ms` separately so the cost is visible and explained rather than silently making the whole request feel slow |

---

## 12. Parking lot (ideas deliberately deferred, not lost)

- Multi-turn conversation memory across chat sessions (P1 if time allows, otherwise here)
- Voyage AI / OpenAI embedding swap-in (interface already supports it — just needs the provider class and an API key)
- Streaming responses in the chat UI
- Per-document delete/re-index without a full re-ingest
- Redis-backed rate limiting
- A learned/trained re-ranker or query classifier, replacing the keyword-heuristic document-type filter
