import { useCallback, useEffect, useState } from 'react'
import type { DocumentMetadata } from './api/types'
import { listDocuments } from './api/client'
import { UploadPanel } from './components/UploadPanel'
import { ChatPanel } from './components/ChatPanel'

function App() {
  const [documents, setDocuments] = useState<DocumentMetadata[]>([])
  const [loaded, setLoaded] = useState(false)

  const refreshDocuments = useCallback(async () => {
    try {
      const docs = await listDocuments()
      setDocuments(docs)
    } finally {
      setLoaded(true)
    }
  }, [])

  useEffect(() => {
    refreshDocuments()
  }, [refreshDocuments])

  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center gap-2 border-b border-[var(--color-border)] px-4 py-3">
        <h1 className="text-sm font-semibold">Career Intelligence Assistant</h1>
        <span className="text-xs text-[var(--color-text-secondary)]">
          Resume × job description analysis
        </span>
      </header>

      <main className="grid flex-1 grid-cols-[320px_1fr] overflow-hidden">
        <aside className="overflow-hidden border-r border-[var(--color-border)]">
          {loaded && <UploadPanel documents={documents} onDocumentsChanged={refreshDocuments} />}
        </aside>
        <section className="overflow-hidden">
          <ChatPanel documents={documents} />
        </section>
      </main>
    </div>
  )
}

export default App
