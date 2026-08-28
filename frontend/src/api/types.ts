// Mirrors backend/app/models/schemas.py exactly — one place to see the
// contract between frontend and backend. No `any` anywhere in this file.

export type SourceType = 'resume' | 'job_description'

export interface DocumentMetadata {
  document_id: string
  source_type: SourceType
  label: string
  filename: string
  uploaded_at: string
  chunk_count: number
}

export interface UploadResponse {
  document_id: string
  source_type: SourceType
  label: string
  chunk_count: number
}

export interface SourceRef {
  document_id: string
  label: string
  section: string | null
  retrieval_score: number
  rerank_score: number | null
  snippet: string
}

export interface TimingInfo {
  retrieval_ms: number
  rerank_ms: number | null
  llm_ms: number
}

export interface TokenUsage {
  input_tokens: number
  output_tokens: number
}

export interface ChatResponse {
  answer: string
  sources: SourceRef[]
  timing: TimingInfo
  token_usage: TokenUsage
  grounded: boolean
}

export interface ApiErrorBody {
  error_code: string
  message: string
  detail: Record<string, unknown>
}

export class ApiError extends Error {
  readonly errorCode: string
  readonly status: number

  constructor(message: string, errorCode: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.errorCode = errorCode
    this.status = status
  }
}
