import { describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ChatPanel } from './ChatPanel'
import type { DocumentMetadata, StreamEvent } from '../api/types'
import * as apiClient from '../api/client'

const RESUME: DocumentMetadata = {
  document_id: 'doc-resume',
  source_type: 'resume',
  label: 'jk_resume',
  filename: 'resume.pdf',
  uploaded_at: new Date().toISOString(),
  chunk_count: 4,
  status: 'ready',
  error_message: null,
}

const JOB_DESCRIPTION: DocumentMetadata = {
  document_id: 'doc-jd',
  source_type: 'job_description',
  label: 'Job #1',
  filename: 'jd.pdf',
  uploaded_at: new Date().toISOString(),
  chunk_count: 3,
  status: 'ready',
  error_message: null,
}

/** Emits a canned sequence of SSE events through the same callback shape
 * `streamChatMessage` uses, so ChatPanel is tested the way it's actually
 * driven in production (incrementally), not against a single Promise. */
function fakeStream(events: StreamEvent[]) {
  return vi
    .spyOn(apiClient, 'streamChatMessage')
    .mockImplementation(async (_message, _sessionId, onEvent) => {
      for (const event of events) {
        onEvent(event)
      }
    })
}

describe('ChatPanel', () => {
  it('streams the answer incrementally and renders sources (with both scores) once available', async () => {
    const streamSpy = fakeStream([
      {
        type: 'context',
        grounded: true,
        sources: [
          {
            document_id: 'doc-jd',
            label: 'Job #1',
            section: 'Requirements',
            retrieval_score: 0.87,
            rerank_score: 4.21,
            snippet: 'Kubernetes and container orchestration experience required.',
          },
        ],
      },
      { type: 'delta', text: "You're missing " },
      { type: 'delta', text: 'hands-on Kubernetes experience.' },
      {
        type: 'done',
        timing: { retrieval_ms: 12.3, rerank_ms: 5.1, llm_ms: 812.4 },
        token_usage: { input_tokens: 512, output_tokens: 96 },
      },
    ])

    render(<ChatPanel documents={[RESUME, JOB_DESCRIPTION]} />)

    const input = screen.getByPlaceholderText(/ask about fit/i)
    await userEvent.type(input, 'What skills am I missing for this role?')
    await userEvent.click(screen.getByRole('button', { name: /send message/i }))

    expect(streamSpy).toHaveBeenCalledWith(
      'What skills am I missing for this role?',
      expect.any(String),
      expect.any(Function),
      expect.anything(),
    )

    await waitFor(() =>
      expect(screen.getByText(/missing hands-on kubernetes/i)).toBeInTheDocument(),
    )

    expect(screen.getByText('Job #1')).toBeInTheDocument()
    expect(screen.getByText(/rerank 4\.21/)).toBeInTheDocument()
    expect(screen.getByText(/512\+96 tokens/)).toBeInTheDocument()
  })

  it('surfaces a guardrail error event without crashing', async () => {
    fakeStream([
      {
        type: 'error',
        error_code: 'no_documents_uploaded',
        message: 'Upload at least a resume and one job description before asking questions.',
      },
    ])

    render(<ChatPanel documents={[RESUME, JOB_DESCRIPTION]} />)
    const input = screen.getByPlaceholderText(/ask about fit/i)
    await userEvent.type(input, 'Anything?')
    await userEvent.click(screen.getByRole('button', { name: /send message/i }))

    await waitFor(() =>
      expect(screen.getByText(/upload at least a resume/i)).toBeInTheDocument(),
    )
  })

  it('disables the input until both a resume and a job description are uploaded', () => {
    render(<ChatPanel documents={[RESUME]} />)
    expect(screen.getByPlaceholderText(/upload a resume/i)).toBeDisabled()
  })
})
