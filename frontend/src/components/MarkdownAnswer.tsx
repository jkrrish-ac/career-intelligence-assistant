import ReactMarkdown from 'react-markdown'

/**
 * Renders Claude's answer as actual formatted text instead of a raw
 * preformatted block. The system prompt (see backend/app/llm/claude_client.py)
 * explicitly asks for "a one-line summary first, then a short breakdown
 * (bullet points or short sections)" — before this component existed, that
 * structure was arriving as literal `-`/`#` characters in a `white-space:
 * pre-wrap` block, which is exactly the kind of thing a normal user
 * shouldn't have to mentally parse.
 *
 * Deliberately not styled with a full typography plugin: the answers are
 * short (a summary + a handful of bullets/sections), so a few targeted
 * element overrides cover every case Claude's system prompt actually
 * produces, without pulling in a much larger set of prose defaults that
 * this chat bubble's compact size doesn't need.
 */
export function MarkdownAnswer({ content }: { content: string }) {
  return (
    <div className="space-y-2 text-sm leading-relaxed [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
      <ReactMarkdown
        components={{
          p: ({ children }) => <p className="m-0">{children}</p>,
          strong: ({ children }) => (
            <strong className="font-semibold text-[var(--color-text-primary)]">{children}</strong>
          ),
          ul: ({ children }) => <ul className="list-disc space-y-1 pl-4">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal space-y-1 pl-4">{children}</ol>,
          li: ({ children }) => <li className="pl-0.5">{children}</li>,
          h1: ({ children }) => <h3 className="text-sm font-semibold">{children}</h3>,
          h2: ({ children }) => <h3 className="text-sm font-semibold">{children}</h3>,
          h3: ({ children }) => <h3 className="text-sm font-semibold">{children}</h3>,
          code: ({ children }) => (
            <code className="rounded bg-[var(--color-background)] px-1 py-0.5 font-mono text-[0.85em]">
              {children}
            </code>
          ),
          a: ({ children, href }) => (
            <a
              href={href}
              target="_blank"
              rel="noreferrer"
              className="text-[var(--color-accent)] underline underline-offset-2"
            >
              {children}
            </a>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
