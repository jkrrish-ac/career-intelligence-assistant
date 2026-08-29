import { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import type { SourceRef } from '../api/types'
import { cn } from '../lib/cn'
import { MATCH_STRENGTH_CLASSES, MATCH_STRENGTH_LABEL, matchStrength } from '../lib/matchStrength'

/**
 * One retrieved source, shown as "Job #2 · Requirements — Strong match"
 * instead of a raw score, since a rerank logit like "4.21" means nothing to
 * someone who isn't the one who built the reranker. The exact numbers
 * aren't gone, though — they're one click away, in the expanded panel,
 * alongside the actual snippet — this is still the detail that shows the
 * RAG pipeline is real (per PRD §7), just no longer the first thing a
 * non-technical user has to decode.
 */
export function SourceChip({ source }: { source: SourceRef }) {
  const [expanded, setExpanded] = useState(false)
  const score = source.rerank_score ?? source.retrieval_score
  const strength = matchStrength(score)

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
        <span
          className={cn(
            'ml-auto shrink-0 rounded-full px-1.5 py-0.5 text-[10px] font-medium',
            MATCH_STRENGTH_CLASSES[strength],
          )}
        >
          {MATCH_STRENGTH_LABEL[strength]}
        </span>
        <ChevronDown
          className={cn('h-3.5 w-3.5 shrink-0 text-[var(--color-text-secondary)] transition-transform', {
            'rotate-180': expanded,
          })}
        />
      </button>
      {expanded && (
        <div className="space-y-1.5 border-t border-[var(--color-border)] px-2.5 py-2 text-xs text-[var(--color-text-secondary)]">
          <p className="italic">"{source.snippet}"</p>
          <div className="flex gap-3 font-mono text-[10px]">
            <span>retrieval score: {source.retrieval_score.toFixed(4)}</span>
            {source.rerank_score !== null && <span>rerank score: {source.rerank_score.toFixed(4)}</span>}
          </div>
        </div>
      )}
    </div>
  )
}
