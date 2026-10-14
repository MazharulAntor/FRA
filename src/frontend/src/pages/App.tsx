import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, ask, getHealth, listDocuments } from '../api/client'
import ChatPanel from '../components/ChatPanel'
import Dashboard from '../components/Dashboard'
import StatusBar from '../components/StatusBar'
import UploadPanel from '../components/UploadPanel'
import type { ChartEntry, ChatMessage, DocumentInfo, HealthResponse } from '../types'

const HISTORY_LIMIT = 6
const HEALTH_POLL_MS = 10000

function initialTheme(): 'light' | 'dark' {
  try {
    const saved = localStorage.getItem('theme')
    if (saved === 'light' || saved === 'dark') return saved
  } catch {
    // ignore
  }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

let nextId = 1
const newId = () => `m${nextId++}`

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [online, setOnline] = useState(false)
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [messagesByDoc, setMessagesByDoc] = useState<Record<string, ChatMessage[]>>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [charts, setCharts] = useState<ChartEntry[]>([])
  const [selectedChartId, setSelectedChartId] = useState<string | null>(null)
  const lastQuestion = useRef('')
  const [theme, setTheme] = useState<'light' | 'dark'>(initialTheme)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      localStorage.setItem('theme', theme)
    } catch {
      // storage can be blocked; the theme still works for this visit
    }
  }, [theme])

  // Backend connection status.
  useEffect(() => {
    let active = true
    const check = () =>
      getHealth()
        .then((h) => active && (setHealth(h), setOnline(true)))
        .catch(() => active && setOnline(false))
    void check()
    const timer = setInterval(check, HEALTH_POLL_MS)
    return () => {
      active = false
      clearInterval(timer)
    }
  }, [])

  // Restore documents already on the server.
  useEffect(() => {
    listDocuments()
      .then((docs) => {
        setDocuments(docs)
        const firstReady = docs.find((d) => d.status === 'ready') ?? docs[0]
        if (firstReady) setSelectedId((cur) => cur ?? firstReady.doc_id)
      })
      .catch(() => undefined)
  }, [online])

  const onDocumentChange = useCallback((doc: DocumentInfo) => {
    setDocuments((docs) =>
      docs.some((d) => d.doc_id === doc.doc_id) ? docs.map((d) => (d.doc_id === doc.doc_id ? doc : d)) : [...docs, doc],
    )
  }, [])

  const selected = documents.find((d) => d.doc_id === selectedId)
  const messages = (selectedId && messagesByDoc[selectedId]) || []

  async function run(question: string, docId: string, before: ChatMessage[]) {
    setLoading(true)
    setError('')
    lastQuestion.current = question
    const history = before.slice(-HISTORY_LIMIT).map((m) => ({ role: m.role, content: m.content }))
    try {
      const response = await ask({ doc_id: docId, question, history })
      if (!response.answer.trim()) {
        setError('The assistant returned an empty answer. Please try again or rephrase the question.')
        return
      }
      const reply: ChatMessage = { id: newId(), role: 'assistant', content: response.answer, response }
      setMessagesByDoc((all) => ({ ...all, [docId]: [...(all[docId] ?? []), reply] }))
      if (response.chart) {
        const entry: ChartEntry = { id: reply.id, question, chart: response.chart }
        setCharts((c) => [...c, entry])
        setSelectedChartId(entry.id)
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Something went wrong. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  function onSend(question: string) {
    if (!selectedId) return
    const before = messagesByDoc[selectedId] ?? []
    const userMessage: ChatMessage = { id: newId(), role: 'user', content: question }
    setMessagesByDoc((all) => ({ ...all, [selectedId]: [...before, userMessage] }))
    void run(question, selectedId, before)
  }

  function onRetry() {
    if (!selectedId || !lastQuestion.current) return
    // The user's message is already in the list; history is everything before it.
    const all = messagesByDoc[selectedId] ?? []
    const last = all[all.length - 1]
    const before = last?.role === 'user' && last.content === lastQuestion.current ? all.slice(0, -1) : all
    void run(lastQuestion.current, selectedId, before)
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <div className="logo" aria-hidden="true">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 17l5-6 4 3 8-9" />
              <path d="M15 5h5v5" />
            </svg>
          </div>
          <div>
            <h1>AI-Powered Financial Analyst</h1>
            <p>Ask your annual report anything. Every answer is cited.</p>
          </div>
        </div>
        <div className="header-right">
          <StatusBar health={health} online={online} document={selected} />
          <button
            type="button"
            className="icon-btn"
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          >
            {theme === 'dark' ? '☀️ Light' : '🌙 Dark'}
          </button>
        </div>
      </header>
      <main className="layout">
        <div className="column column-left">
          <UploadPanel
            documents={documents}
            selectedId={selectedId}
            onSelect={setSelectedId}
            onDocumentChange={onDocumentChange}
          />
          <ChatPanel
            document={selected}
            messages={messages}
            loading={loading}
            error={error}
            onSend={onSend}
            onRetry={onRetry}
          />
        </div>
        <div className="column column-right">
          <Dashboard entries={charts} selectedId={selectedChartId} onSelect={setSelectedChartId} />
        </div>
      </main>
      <footer className="app-footer">For information only. Not financial advice.</footer>
    </div>
  )
}
