# Project Instructions — Career Intelligence Assistant

## Context

I'm building a **Career Intelligence Assistant** as a take-home assignment for a **Forward Deployed Engineer** role. This is a RAG-based system that analyzes resumes against job descriptions, answers questions about fit, skill gaps, experience alignment, and interview prep.

**The evaluators care as much about HOW I build it as WHAT I build.** They're looking at: engineering philosophy, clean architecture, RAG approach decisions, creativity in UI/UX, observability, and how I use AI tools.

---

## My Tech Stack & Preferences

- **Backend:** Python (FastAPI)
- **Frontend:** React with TypeScript
- **LLM:** Claude API (Anthropic) — use claude-sonnet-4-6 for the assistant's inference
- **Embedding Model:** [To be decided — suggest and justify options]
- **Vector Store:** [To be decided — suggest lightweight options suitable for a take-home]
- **Containerization:** Docker + docker-compose
- **Testing:** pytest (backend), vitest or jest (frontend)

---

## Architecture Principles

Follow these engineering standards in ALL code you write:

1. **Clean separation of concerns** — distinct layers for API routes, services, RAG pipeline, and LLM interaction. No god files.
2. **Dependency injection** — services should be injectable, not hardcoded. Makes testing trivial.
3. **Typed everything** — Pydantic models on backend, TypeScript interfaces on frontend. No `any`.
4. **Error handling with context** — structured errors, not bare exceptions. Log what matters.
5. **Configuration via environment** — all secrets, model names, chunk sizes, retrieval params in `.env` with sensible defaults.
6. **Small, focused functions** — each function does one thing. If it needs a comment explaining "what", it's too complex.
7. **Observability built-in** — structured logging (not print statements), timing on LLM calls and retrieval, token usage tracking.
8. **README-driven development** — document decisions as we make them, not after.

---

## RAG Pipeline Requirements

When building the RAG components, follow this approach:

### Document Processing

- Support PDF and DOCX resume uploads
- Support multiple job description uploads (PDF, DOCX, or plain text)
- Extract text cleanly — handle formatting artifacts
- Tag each chunk with metadata: `source_type` (resume | job_description), `document_id`, `section` (if detectable)

### Chunking Strategy

- Use semantic chunking where possible (by section headers, bullet groups)
- Fall back to recursive character splitting with overlap
- Chunk sizes and overlap should be configurable
- Preserve document structure metadata through chunking

### Retrieval

- Hybrid retrieval if feasible (semantic + keyword)
- Re-ranking step before context assembly
- Filter by document type when the query implies it (e.g., "my experience" → resume chunks, "Job #2 requirements" → job description chunks)
- Return relevance scores for observability

### Prompt Engineering

- System prompt should define the assistant's role clearly: career advisor analyzing specific uploaded documents
- Include structured context with clear delimiters between resume content and job description content
- Use chain-of-thought for gap analysis queries
- Guard against hallucination: if information isn't in the documents, say so
- Format responses with actionable structure (not walls of text)

### Guardrails

- Input validation on uploads (file type, size limits)
- Output guardrails: keep responses grounded in uploaded documents
- Handle edge cases: no resume uploaded, no job descriptions, irrelevant queries
- Rate limiting on LLM calls

---

## UI/UX Requirements

The frontend should feel like a **polished product**, not a hackathon demo:

- Clean, modern design — use a component library (shadcn/ui preferred)
- **Document management panel:** show uploaded resume and job descriptions with status indicators
- **Chat interface:** conversational Q&A with the assistant
- **Suggested queries:** show example questions based on what's uploaded
- **Visual gap analysis:** when comparing resume to a job, show a structured skill match visualization (not just text)
- **Responsive layout** — should work on desktop
- Loading states, error states, empty states — handle them all
- Dark mode support is a plus but not required

---

## Code Quality Standards

- Every file should have a clear, single responsibility
- Use conventional commits
- Include docstrings on all public functions (backend)
- Type hints everywhere (Python + TypeScript)
- No commented-out code
- No TODO comments in submitted code — track them in README if needed
- Linting: ruff (Python), eslint + prettier (frontend)
- Pre-commit hooks if time permits

---

## Testing Strategy

- **Unit tests:** RAG pipeline components (chunking, retrieval, prompt assembly)
- **Integration tests:** API endpoints with mock LLM responses
- **Frontend:** component tests for key interactions
- Don't aim for 100% coverage — test the important paths and edge cases

---

## What NOT To Do

- Don't over-engineer. This is a take-home, not a production system. But make it clear I KNOW what production would look like (document it in README).
- Don't use LangChain unless there's a specific justified reason. Prefer lighter abstractions or direct implementation — shows I understand the internals.
- Don't leave the default Create React App or Vite boilerplate branding.
- Don't hardcode prompts in route handlers. Keep them in a dedicated prompts module.
- Don't skip error handling to save time. Handle the obvious failure modes.

---

## README Structure

The README must cover these (per assignment requirements):

1. Quick setup instructions (docker-compose up should be the happy path)
2. Architecture overview with a diagram (Mermaid is fine)
3. Productionization plan (AWS/GCP deployment, scaling, managed services)
4. RAG/LLM decisions: what I chose, what I considered, why
5. Key technical decisions
6. Engineering standards followed (and skipped)
7. How I used AI tools (be honest and specific)
8. What I'd do with more time

**Important:** Write the README in MY voice — concise, opinionated, specific. Not generic LLM-sounding prose. Use first person. Be direct about tradeoffs.

---

## How I Want You To Help Me

1. **When I ask you to write code:** Write production-quality code following the standards above. No placeholder implementations unless I explicitly ask for a stub.
2. **When I ask about architecture decisions:** Give me 2-3 options with clear tradeoffs, then recommend one with justification. I'll make the final call.
3. **When I ask you to review:** Be critical. Point out issues, suggest improvements, flag anything that looks like it was auto-generated without thought.
4. **Don't:** Add unnecessary abstractions, over-comment obvious code, or write generic docstrings that don't add information.
5. **Do:** Keep track of decisions we make so we can reference them in the README.
6. **Formatting:** Use minimal formatting in conversation. Code should speak for itself.

---

## Session Tracking

As we work, maintain awareness of:

- Decisions made and their rationale (for README)
- Components completed vs remaining
- Any technical debt introduced intentionally
- Time-sensitive tradeoffs (what to skip, what to polish)
