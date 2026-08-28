"""Claude API wrapper: system prompt, context assembly, the call itself.

Kept thin and single-purpose — prompt *construction* logic lives here where
it's testable in isolation from the FastAPI/service plumbing; the actual
`anthropic` SDK call is the only side-effecting part.
"""

from __future__ import annotations

import re
import time

from anthropic import AsyncAnthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.exceptions import LLMProviderError
from app.core.logging import get_logger
from app.models.schemas import ConversationTurn, RetrievedChunk, SourceType

logger = get_logger(__name__)

SYSTEM_PROMPT = """\
You are a career advisor assistant analyzing a specific candidate's resume \
against specific job descriptions that have been uploaded to this session.

Rules you must follow:
- Answer ONLY using the resume and job description context provided below. \
Never use outside knowledge about companies, roles, or salaries.
- If the provided context does not contain enough information to answer, \
say so plainly and state what's missing — never guess or fill gaps from \
general knowledge.
- Structure your answers for scanability: a one-line summary first, then a \
short breakdown (bullet points or short sections). Avoid walls of text.
- When asked about skill gaps or role fit, reason step by step: first list \
the role's key requirements from the job description context, then check \
each one against the resume context, then summarize matches and gaps.
- When multiple job descriptions are in context, be explicit about which \
job (by its label, e.g. "Job #2") each point of your answer refers to.
"""

_GAP_ANALYSIS_HINTS = re.compile(
    r"\b(missing|gap|lack|qualif|fit|align|match)\w*\b", re.IGNORECASE
)


def build_user_message(query: str, context_chunks: list[RetrievedChunk]) -> str:
    """Assemble the retrieved context with explicit delimiters per source
    type, plus the user's question. Chain-of-thought nudge is added only for
    gap/fit-style questions, where the extra reasoning step measurably helps."""

    resume_blocks = []
    jd_blocks: dict[str, list[str]] = {}

    for rc in context_chunks:
        chunk = rc.chunk
        tag = f"(source: {chunk.label}" + (f", section: {chunk.section})" if chunk.section else ")")
        block = f"{chunk.text}\n{tag}"
        if chunk.source_type == SourceType.RESUME:
            resume_blocks.append(block)
        else:
            jd_blocks.setdefault(chunk.label, []).append(block)

    context_parts: list[str] = []
    if resume_blocks:
        context_parts.append("<resume>\n" + "\n\n".join(resume_blocks) + "\n</resume>")
    for label, blocks in jd_blocks.items():
        context_parts.append(
            f'<job_description label="{label}">\n' + "\n\n".join(blocks) + "\n</job_description>"
        )

    context_text = "\n\n".join(context_parts) if context_parts else "(no relevant context found)"

    cot_hint = (
        "\n\nFirst identify the role's key requirements from the job description context, "
        "then compare each against the resume context, then summarize."
        if _GAP_ANALYSIS_HINTS.search(query)
        else ""
    )

    return f"{context_text}\n\n---\n\nQuestion: {query}{cot_hint}"


class ClaudeClient:
    def __init__(self, api_key: str, model: str, max_tokens: int) -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    @staticmethod
    def _build_messages(
        query: str, context_chunks: list[RetrievedChunk], history: list[ConversationTurn] | None
    ) -> list[dict]:
        """Prior turns carry the plain question/answer text, not the
        context blob that was assembled for that turn — the thread of
        conversation matters for follow-ups, but re-sending old retrieved
        context on every turn would grow tokens unboundedly for no benefit.
        Only the *current* question gets fresh context."""
        messages = [{"role": t["role"], "content": t["content"]} for t in (history or [])]
        messages.append({"role": "user", "content": build_user_message(query, context_chunks)})
        return messages

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call(self, messages: list[dict]) -> tuple[str, int, int]:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        answer = "".join(block.text for block in response.content if block.type == "text")
        return answer, response.usage.input_tokens, response.usage.output_tokens

    async def answer(
        self,
        query: str,
        context_chunks: list[RetrievedChunk],
        history: list[ConversationTurn] | None = None,
    ) -> dict:
        messages = self._build_messages(query, context_chunks, history)
        start = time.perf_counter()
        try:
            answer_text, input_tokens, output_tokens = await self._call(messages)
        except Exception as exc:
            logger.error("claude_call_failed", error=str(exc))
            raise LLMProviderError("The Claude API call failed after retries.") from exc
        llm_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "claude_call_complete",
            llm_ms=llm_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return {
            "answer": answer_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "llm_ms": llm_ms,
        }

    async def stream_answer(
        self,
        query: str,
        context_chunks: list[RetrievedChunk],
        history: list[ConversationTurn] | None = None,
    ):
        """Async generator: yields text deltas as they arrive, then a final
        dict with the same shape `answer()` returns (used by the SSE route
        to emit a closing event with usage/timing once the stream ends).

        Deliberately not wrapped in the same retry as `_call` — once partial
        text has been sent to the client, silently retrying and re-streaming
        from the top would duplicate output. If the stream fails mid-way,
        the SSE route below emits an error event instead."""
        messages = self._build_messages(query, context_chunks, history)
        start = time.perf_counter()
        try:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield {"type": "delta", "text": text}
                final_message = await stream.get_final_message()
        except Exception as exc:
            logger.error("claude_stream_failed", error=str(exc))
            raise LLMProviderError("The Claude API streaming call failed.") from exc

        llm_ms = round((time.perf_counter() - start) * 1000, 2)
        answer_text = "".join(block.text for block in final_message.content if block.type == "text")
        logger.info(
            "claude_stream_complete",
            llm_ms=llm_ms,
            input_tokens=final_message.usage.input_tokens,
            output_tokens=final_message.usage.output_tokens,
        )
        yield {
            "type": "done",
            "answer": answer_text,
            "input_tokens": final_message.usage.input_tokens,
            "output_tokens": final_message.usage.output_tokens,
            "llm_ms": llm_ms,
        }
