import { describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ChatPanel } from './ChatPanel'
import type { DocumentMetadata } from '../api/types'
import * as apiClient from '../api/client'

const RESUME: DocumentMetadata = {
  document_id: 'doc-resume',
  source_type: 'resume',
  label: 'jk_resume',
  filename: 'resume.pdf',
  uploaded_at: new Date().toISOString(),
  chunk_count: 4,
}

const JOB_DESCRIPTION: DocumentMetadata = {
  document_id: 'doc-jd',
  source_type: 'job_description',
  label: 'Job #1',
  filename: 'jd.pdf',
  uploaded_at: new Date().toISOString(),
  chunk_count: 3,
}

describe('ChatPanel', () => {
  it('renders the answer and its sources (with both scores) from a mocked API response', async () => {
    const sendChatMessageSpy = vi.spyOn(apiClient, 'sendChatMessage').mockResolvedValue({
      answer: "You're missing hands-on Kubernetes experience for this role.",
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
      timing: { retrieval_ms: 12.3, rerank_ms: 5.1, llm_ms: 812.4 },
      token_usage: { input_tokens: 512, output_tokens: 96 },
      grounded: true,
    })

    render(<ChatPanel documents={[RESUME, JOB_DESCRIPTION]} />)

    const input = screen.getByPlaceholderText(/ask about fit/i)
    await userEvent.type(input, 'What skills am I missing for this role?')
    await userEvent.click(screen.getByRole('button', { name: /send message/i }))

    expect(sendChatMessageSpy).toHaveBeenCalledWith(
      'What skills am I missing for this role?',
      expect.any(String),
    )

    await waitFor(() =>
      expect(screen.getByText(/missing hands-on kubernetes/i)).toBeInTheDocument(),
    )

    expect(screen.getByText('Job #1')).toBeInTheDocument()
    expect(screen.getByText(/rerank 4\.21/)).toBeInTheDocument()
  })

  it('disables the input until both a resume and a job description are uploaded', () => {
    render(<ChatPanel documents={[RESUME]} />)
    expect(screen.getByPlaceholderText(/upload a resume/i)).toBeDisabled()
  })
})
