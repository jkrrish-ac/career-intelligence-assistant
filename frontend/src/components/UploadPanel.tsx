import { useCallback, useRef, useState } from 'react'
import { FileText, Trash2, UploadCloud } from 'lucide-react'
import type { DocumentMetadata, SourceType } from '../api/types'
import { ApiError } from '../api/types'
import { deleteDocument, uploadDocument } from '../api/client'
import { Card } from './ui/Card'
import { Badge } from './ui/Badge'
import { Button } from './ui/Button'

interface UploadPanelProps {
  documents: DocumentMetadata[]
  onDocumentsChanged: () => void
}

export function UploadPanel({ documents, onDocumentsChanged }: UploadPanelProps) {
  const [uploading, setUploading] = useState<SourceType | null>(null)
  const [error, setError] = useState<string | null>(null)
  const resumeInputRef = useRef<HTMLInputElement>(null)
  const jdInputRef = useRef<HTMLInputElement>(null)

  const handleUpload = useCallback(
    async (files: FileList | null, sourceType: SourceType) => {
      if (!files || files.length === 0) return
      setError(null)
      setUploading(sourceType)
      try {
        for (const file of Array.from(files)) {
          await uploadDocument(file, sourceType)
        }
        onDocumentsChanged()
      } catch (err) {
        setError(err instanceof ApiError ? err.message : 'Upload failed. Please try again.')
      } finally {
        setUploading(null)
      }
    },
    [onDocumentsChanged],
  )

  const handleDelete = useCallback(
    async (documentId: string) => {
      try {
        await deleteDocument(documentId)
        onDocumentsChanged()
      } catch (err) {
        setError(err instanceof ApiError ? err.message : 'Could not delete document.')
      }
    },
    [onDocumentsChanged],
  )

  const resumes = documents.filter((d) => d.source_type === 'resume')
  const jobDescriptions = documents.filter((d) => d.source_type === 'job_description')

  return (
    <div className="flex h-full flex-col gap-4 p-4">
      <div>
        <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">Documents</h2>
        <p className="text-xs text-[var(--color-text-secondary)]">
          Upload your resume and one or more job descriptions (PDF, DOCX, or TXT).
        </p>
      </div>

      <UploadDropzone
        label="Upload resume"
        inputRef={resumeInputRef}
        busy={uploading === 'resume'}
        onFiles={(files) => handleUpload(files, 'resume')}
      />
      <UploadDropzone
        label="Upload job description(s)"
        inputRef={jdInputRef}
        busy={uploading === 'job_description'}
        multiple
        onFiles={(files) => handleUpload(files, 'job_description')}
      />

      {error && (
        <p className="rounded-lg bg-[var(--color-danger)]/10 px-3 py-2 text-xs text-[var(--color-danger)]">
          {error}
        </p>
      )}

      <div className="flex-1 space-y-4 overflow-y-auto">
        <DocumentGroup title="Resume" badgeVariant="resume" documents={resumes} onDelete={handleDelete} />
        <DocumentGroup
          title="Job descriptions"
          badgeVariant="job_description"
          documents={jobDescriptions}
          onDelete={handleDelete}
        />

        {documents.length === 0 && (
          <p className="rounded-lg border border-dashed border-[var(--color-border)] p-4 text-center text-xs text-[var(--color-text-secondary)]">
            No documents yet — upload a resume and at least one job description to start asking
            questions.
          </p>
        )}
      </div>
    </div>
  )
}

function UploadDropzone({
  label,
  onFiles,
  inputRef,
  busy,
  multiple,
}: {
  label: string
  onFiles: (files: FileList | null) => void
  inputRef: React.RefObject<HTMLInputElement | null>
  busy: boolean
  multiple?: boolean
}) {
  const [dragging, setDragging] = useState(false)

  return (
    <Card
      className={`flex cursor-pointer flex-col items-center gap-1 border-dashed px-4 py-5 text-center transition-colors ${
        dragging ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/5' : ''
      }`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        onFiles(e.dataTransfer.files)
      }}
    >
      <UploadCloud className="h-5 w-5 text-[var(--color-text-secondary)]" />
      <span className="text-xs font-medium text-[var(--color-text-primary)]">
        {busy ? 'Uploading…' : label}
      </span>
      <span className="text-[10px] text-[var(--color-text-secondary)]">Click or drag files here</span>
      <input
        ref={inputRef}
        type="file"
        multiple={multiple}
        accept=".pdf,.docx,.txt"
        className="hidden"
        onChange={(e) => onFiles(e.target.files)}
      />
    </Card>
  )
}

function DocumentGroup({
  title,
  badgeVariant,
  documents,
  onDelete,
}: {
  title: string
  badgeVariant: 'resume' | 'job_description'
  documents: DocumentMetadata[]
  onDelete: (documentId: string) => void
}) {
  if (documents.length === 0) return null

  return (
    <div>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-[var(--color-text-secondary)]">
        {title}
      </h3>
      <ul className="space-y-2">
        {documents.map((doc) => (
          <li key={doc.document_id}>
            <Card className="flex items-center justify-between gap-2 p-3">
              <div className="flex min-w-0 items-center gap-2">
                <FileText className="h-4 w-4 shrink-0 text-[var(--color-text-secondary)]" />
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">{doc.label}</p>
                  <div className="flex items-center gap-2">
                    <Badge variant={badgeVariant}>{doc.chunk_count} chunks</Badge>
                  </div>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                aria-label={`Delete ${doc.label}`}
                onClick={() => onDelete(doc.document_id)}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </Card>
          </li>
        ))}
      </ul>
    </div>
  )
}
