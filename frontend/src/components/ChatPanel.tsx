import { useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, ChevronDown, Loader2, Send, Sparkles } from 'lucide-react'
import type { DocumentMetadata, SourceRef, SourceType, TimingInfo, TokenUsage } from '../api/types'
import { ApiError } from '../api/types'
import { streamChatMessage } from '../api/client'
import { SourceChip } from './SourceChip'
import { MarkdownAnswer } from './MarkdownAnswer'
import { Button } from './ui/Button'
import { cn } from '../lib/cn'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: SourceRef[]
  grounded?: boolean
  timing?: TimingInfo
  tokenUsage?: TokenUsage
  streaming?: boolean
  error?: string
}

const SUGGESTED_QUESTIONS = [
  'What skills am I missing for this role?',
  'How does my experience align with Job #2?',
  'What should I prepare to talk about in an interview for this role?',
]

export function ChatPanel({ documents }: { documents: DocumentMetadata[] }) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const sessionId = useRef(crypto.randomUUID())
  const abortRef = useRef<AbortController | null>(null)

  const docTypeById = useMemo(() => {
    const map = new Map<string, SourceType>()
    for (const doc of documents) map.set(doc.document_id, doc.source_type)
    return map
  }, [documents])

  useEffect(() => () => abortRef.current?.abort(), [])

  const hasResume = documents.some((d) => d.source_type === 'resume')
  const hasJobDescription = documents.some((d) => d.source_type === 'job_description')
  const canChat = hasResume && hasJobDescription

  function updateAssistantMessage(id: string, patch: Partial<ChatMessage>) {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)))
  }

  async function handleSend(text: string) {
    const trimmed = text.trim()
    if (!trimmed || loading) return

    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: trimmed }
    const assistantId = crypto.randomUUID()
    setMessages((prev) => [
      ...prev,
      userMessage,
      { id: assistantId, role: 'assistant', content: '', streaming: true },
    ])
    setInput('')
    setLoading(true)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      await streamChatMessage(
        trimmed,
        sessionId.current,
        (event) => {
          switch (event.type) {
            case 'context':
              updateAssistantMessage(assistantId, { sources: event.sources, grounded: event.grounded })
              break
            case 'delta':
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId ? { ...m, content: m.content + event.text } : m,
                ),
              )
              break
            case 'done':
              updateAssistantMessage(assistantId, {
                timing: event.timing,
                tokenUsage: event.token_usage,
                streaming: false,
              })
              break
            case 'error':
              updateAssistantMessage(assistantId, { error: event.message, streaming: false })
              break
          }
        },
        controller.signal,
      )
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : 'Something went wrong reaching the assistant.'
      updateAssistantMessage(assistantId, { error: message, streaming: false })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 space-y-4 overflow-y-auto p-4">
        {messages.length === 0 && <EmptyState canChat={canChat} onPick={handleSend} />}
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} docTypeById={docTypeById} />
        ))}
      </div>

      <form
        className="flex items-center gap-2 border-t border-[var(--color-border)] p-3"
        onSubmit={(e) => {
          e.preventDefault()
          handleSend(input)
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={!canChat || loading}
          placeholder={
            canChat
              ? 'Ask about fit, skill gaps, or interview prep…'
              : 'Upload a resume and a job description to start chatting'
          }
          className="h-10 flex-1 rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 text-sm outline-none focus:border-[var(--color-accent)] disabled:opacity-50"
        />
        <Button type="submit" aria-label="Send message" disabled={!canChat || loading || !input.trim()}>
          <Send className="h-4 w-4" />
        </Button>
      </form>
    </div>
  )
}

function EmptyState({ canChat, onPick }: { canChat: boolean; onPick: (text: string) => void }) {
  if (!canChat) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-[var(--color-text-secondary)]">
        <Sparkles className="h-6 w-6" />
        <p className="text-sm">Upload a resume and at least one job description to get started.</p>
      </div>
    )
  }
  return (
    <div className="space-y-2">
      <p className="text-xs text-[var(--color-text-secondary)]">Try asking:</p>
      {SUGGESTED_QUESTIONS.map((q) => (
        <button
          key={q}
          onClick={() => onPick(q)}
          className="block w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-left text-sm hover:bg-[var(--color-surface-hover)]"
        >
          {q}
        </button>
      ))}
    </div>
  )
}

/** "your resume, Job #1, and Job #2" — a single plain-English sentence
 * fragment summarizing which uploaded documents actually grounded this
 * answer, so a user gets the gist before they ever look at (or need to
 * look at) an individual source chip. */
function summarizeSources(sources: SourceRef[], docTypeById: Map<string, SourceType>): string {
  const seen = new Set<string>()
  const parts: string[] = []

  for (const source of sources) {
    if (seen.has(source.document_id)) continue
    seen.add(source.document_id)
    const type = docTypeById.get(source.document_id)
    parts.push(type === 'resume' ? 'your resume' : source.label)
  }

  if (parts.length === 0) return ''
  if (parts.length === 1) return parts[0]
  if (parts.length === 2) return `${parts[0]} and ${parts[1]}`
  return `${parts.slice(0, -1).join(', ')}, and ${parts[parts.length - 1]}`
}

function MessageBubble({
  message,
  docTypeById,
}: {
  message: ChatMessage
  docTypeById: Map<string, SourceType>
}) {
  const isUser = message.role === 'user'
  const isThinking = message.streaming && !message.content && !message.error
  const hasSources = !isUser && !message.error && !!message.sources && message.sources.length > 0

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[85%] space-y-2 ${isUser ? 'items-end' : 'items-start'}`}>
        <div
          className={`rounded-xl px-3 py-2 text-sm ${
            isUser
              ? 'bg-[var(--color-accent)] text-white whitespace-pre-wrap'
              : message.error
                ? 'bg-[var(--color-danger)]/10 text-[var(--color-danger)] whitespace-pre-wrap'
                : 'bg-[var(--color-surface)] text-[var(--color-text-primary)]'
          }`}
        >
          {message.error ? (
            <span className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              {message.error}
            </span>
          ) : isThinking ? (
            <span className="flex items-center gap-2 text-[var(--color-text-secondary)]">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Reading your documents and asking Claude…
            </span>
          ) : isUser ? (
            message.content
          ) : (
            <>
              <MarkdownAnswer content={message.content} />
              {message.streaming && <span className="animate-pulse">▍</span>}
            </>
          )}
        </div>

        {hasSources && (
          <div className="space-y-2">
            {message.grounded === false && (
              <div className="flex items-start gap-2 rounded-lg bg-[var(--color-warning)]/10 px-2.5 py-2 text-xs text-[var(--color-warning)]">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span>
                  This answer may not be well supported by your documents — the closest match was
                  weak, so double-check it against the source material yourself.
                </span>
              </div>
            )}

            <div className="space-y-1">
              <p className="text-[10px] text-[var(--color-text-secondary)]">
                Based on <span className="text-[var(--color-text-primary)]">
                  {summarizeSources(message.sources!, docTypeById)}
                </span>
              </p>
              <div className="flex flex-col gap-1">
                {message.sources!.map((source, i) => (
                  <SourceChip key={`${source.document_id}-${i}`} source={source} />
                ))}
              </div>
            </div>

            {message.timing && message.tokenUsage && (
              <AnswerDetails timing={message.timing} tokenUsage={message.tokenUsage} />
            )}
          </div>
        )}
      </div>
    </div>
  )
}

/** Retrieval/rerank/LLM timings and token counts are real engineering
 * signal (and worth keeping, per PRD §7's retrieval-transparency goal) but
 * they're not something a candidate asking about their own resume needs to
 * see by default — tucked behind one small toggle instead of a permanent
 * monospace line, in plain English rather than internal stage names. */
function AnswerDetails({ timing, tokenUsage }: { timing: TimingInfo; tokenUsage: TokenUsage }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div>
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-1 text-[10px] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
      >
        <ChevronDown className={cn('h-3 w-3 transition-transform', { 'rotate-180': expanded })} />
        {expanded ? 'Hide' : 'Show'} answer details
      </button>
      {expanded && (
        <dl className="mt-1.5 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[10px] text-[var(--color-text-secondary)]">
          <dt>Searching your documents</dt>
          <dd className="text-right font-mono">{timing.retrieval_ms}ms</dd>
          {timing.rerank_ms !== null && (
            <>
              <dt>Ranking the results</dt>
              <dd className="text-right font-mono">{timing.rerank_ms}ms</dd>
            </>
          )}
          <dt>Writing the answer</dt>
          <dd className="text-right font-mono">{timing.llm_ms}ms</dd>
          <dt>Tokens used</dt>
          <dd className="text-right font-mono">
            {tokenUsage.input_tokens} in / {tokenUsage.output_tokens} out
          </dd>
        </dl>
      )}
    </div>
  )
}
