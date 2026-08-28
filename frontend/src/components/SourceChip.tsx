import { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import type { SourceRef } from '../api/types'
import { cn } from '../lib/cn'

/**
 * Chip per retrieved source, expandable to show the snippet plus both the
 * fused retrieval score and the cross-encoder rerank score — this is the
 * single UI detail that shows the RAG pipeline is real (per PRD §7), not a
 * canned response.
 */
export function SourceChip({ source }: { source: SourceRef }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-background)]">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-xs"
      >
        <span className="truncate font-medium text-[var(--color-text-primary)]">{source.label}</span>
        {source.section && (
          <span className="truncate text-[var(--color-text-secondary)]">· {source.section}</span>
        )}
        <span className="ml-auto shrink-0 font-mono text-[10px] text-[var(--color-text-secondary)]">
          {source.rerank_score !== null
            ? `rerank ${source.rerank_score.toFixed(2)}`
            : `score ${source.retrieval_score.toFixed(2)}`}
        </span>
        <ChevronDown
          className={cn('h-3.5 w-3.5 shrink-0 text-[var(--color-text-secondary)] transition-transform', {
            'rotate-180': expanded,
          })}
        />
      </button>
      {expanded && (
        <div className="space-y-1 border-t border-[var(--color-border)] px-2.5 py-2 text-xs text-[var(--color-text-secondary)]">
          <p className="italic">"{source.snippet}"</p>
          <div className="flex gap-3 font-mono text-[10px]">
            <span>retrieval: {source.retrieval_score.toFixed(4)}</span>
            {source.rerank_score !== null && <span>rerank: {source.rerank_score.toFixed(4)}</span>}
          </div>
        </div>
      )}
    </div>
  )
}
