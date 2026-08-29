export type MatchStrength = 'strong' | 'relevant' | 'weak'

/**
 * Turns a raw retrieval/rerank score into a plain-language bucket for
 * non-technical users. The cross-encoder rerank score is an unbounded
 * logit, not a calibrated probability, so this is a deliberate
 * approximation rather than a precise conversion — it reuses the same
 * threshold the backend already applies for the overall `grounded` flag
 * (`_MIN_RELEVANT_RERANK_SCORE = -2.0` in chat_service.py), just applied
 * per-source instead of per-answer.
 */
export function matchStrength(score: number): MatchStrength {
  if (score > 2) return 'strong'
  if (score > -2) return 'relevant'
  return 'weak'
}

export const MATCH_STRENGTH_LABEL: Record<MatchStrength, string> = {
  strong: 'Strong match',
  relevant: 'Relevant',
  weak: 'Weak match',
}

export const MATCH_STRENGTH_CLASSES: Record<MatchStrength, string> = {
  strong: 'bg-[var(--color-success)]/15 text-[var(--color-success)]',
  relevant: 'bg-[var(--color-accent)]/15 text-[var(--color-accent)]',
  weak: 'bg-[var(--color-warning)]/15 text-[var(--color-warning)]',
}
