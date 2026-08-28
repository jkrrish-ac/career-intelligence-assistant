import { useRef, useState } from 'react'
import { AlertTriangle, Loader2, Send, Sparkles } from 'lucide-react'
import type { ChatResponse, DocumentMetadata } from '../api/types'
import { ApiError } from '../api/types'
import { sendChatMessage } from '../api/client'
import { SourceChip } from './SourceChip'
import { Button } from './ui/Button'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  response?: ChatResponse
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

  const hasResume = documents.some((d) => d.source_type === 'resume')
  const hasJobDescription = documents.some((d) => d.source_type === 'job_description')
  const canChat = hasResume && hasJobDescription

  async function handleSend(text: string) {
    const trimmed = text.trim()
    if (!trimmed || loading) return

    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: trimmed }
    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const response = await sendChatMessage(trimmed, sessionId.current)
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: 'assistant', content: response.answer, response },
      ])
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : 'Something went wrong reaching the assistant.'
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: 'assistant', content: '', error: message },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 space-y-4 overflow-y-auto p-4">
        {messages.length === 0 && (
          <EmptyState canChat={canChat} onPick={handleSend} />
        )}
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        {loading && (
          <div className="flex items-center gap-2 text-xs text-[var(--color-text-secondary)]">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Retrieving context and asking Claude…
          </div>
        )}
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

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user'
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[85%] space-y-2 ${isUser ? 'items-end' : 'items-start'}`}>
        <div
          className={`rounded-xl px-3 py-2 text-sm whitespace-pre-wrap ${
            isUser
              ? 'bg-[var(--color-accent)] text-white'
              : message.error
                ? 'bg-[var(--color-danger)]/10 text-[var(--color-danger)]'
                : 'bg-[var(--color-surface)] text-[var(--color-text-primary)]'
          }`}
        >
          {message.error ? (
            <span className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              {message.error}
            </span>
          ) : (
            message.content
          )}
        </div>

        {message.response && (
          <div className="space-y-1.5">
            {!message.response.grounded && (
              <p className="text-[10px] text-[var(--color-text-secondary)]">
                ⚠ Low-confidence match — the documents may not fully cover this question.
              </p>
            )}
            {message.response.sources.length > 0 && (
              <div className="space-y-1">
                <p className="text-[10px] font-semibold uppercase tracking-wide text-[var(--color-text-secondary)]">
                  Sources
                </p>
                <div className="flex flex-col gap-1">
                  {message.response.sources.map((source, i) => (
                    <SourceChip key={`${source.document_id}-${i}`} source={source} />
                  ))}
                </div>
              </div>
            )}
            <p className="font-mono text-[10px] text-[var(--color-text-secondary)]">
              retrieval {message.response.timing.retrieval_ms}ms
              {message.response.timing.rerank_ms !== null &&
                ` · rerank ${message.response.timing.rerank_ms}ms`}{' '}
              · llm {message.response.timing.llm_ms}ms · {message.response.token_usage.input_tokens}+
              {message.response.token_usage.output_tokens} tokens
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
