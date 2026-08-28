import type {
  ApiErrorBody,
  ChatResponse,
  DocumentMetadata,
  SourceType,
  StreamEvent,
  UploadResponse,
} from './types'
import { ApiError } from './types'

// Vite dev server proxies /api -> the backend (see vite.config.ts); in
// production the two are served from the same origin behind one reverse
// proxy (see docker-compose.yml), so this stays a relative path either way.
const BASE_URL = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, init)

  if (!response.ok) {
    let body: ApiErrorBody | null = null
    try {
      body = (await response.json()) as ApiErrorBody
    } catch {
      // Non-JSON error body (e.g. a proxy-level 502) — fall through to a generic message.
    }
    throw new ApiError(
      body?.message ?? `Request failed with status ${response.status}`,
      body?.error_code ?? 'unknown_error',
      response.status,
    )
  }

  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}

export async function uploadDocument(
  file: File,
  sourceType: SourceType,
  label?: string,
): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('source_type', sourceType)
  if (label) formData.append('label', label)

  return request<UploadResponse>('/documents', {
    method: 'POST',
    body: formData,
  })
}

export async function listDocuments(): Promise<DocumentMetadata[]> {
  return request<DocumentMetadata[]>('/documents')
}

export async function deleteDocument(documentId: string): Promise<void> {
  return request<void>(`/documents/${documentId}`, { method: 'DELETE' })
}

export async function sendChatMessage(message: string, sessionId?: string): Promise<ChatResponse> {
  return request<ChatResponse>('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId }),
  })
}

/**
 * Streams /chat/stream (Server-Sent Events framed as `data: <json>\n\n`)
 * and invokes `onEvent` for each parsed event as it arrives. Throws ApiError
 * for a non-2xx HTTP response (e.g. rate limit before the stream even
 * starts); a guardrail that fires mid-stream instead arrives as a
 * `{type: 'error'}` event, which the caller handles via onEvent.
 */
export async function streamChatMessage(
  message: string,
  sessionId: string | undefined,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${BASE_URL}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId }),
    signal,
  })

  if (!response.ok || !response.body) {
    let body: ApiErrorBody | null = null
    try {
      body = (await response.json()) as ApiErrorBody
    } catch {
      // fall through to generic message below
    }
    throw new ApiError(
      body?.message ?? `Request failed with status ${response.status}`,
      body?.error_code ?? 'unknown_error',
      response.status,
    )
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''
    for (const frame of frames) {
      const line = frame.split('\n').find((l) => l.startsWith('data: '))
      if (!line) continue
      onEvent(JSON.parse(line.slice('data: '.length)) as StreamEvent)
    }
  }
}
