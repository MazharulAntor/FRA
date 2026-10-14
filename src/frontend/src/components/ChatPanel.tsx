import { useEffect, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import type { ChatMessage, DocumentInfo } from '../types'
import MessageBubble from './MessageBubble'

const SUGGESTIONS = [
  'What was total revenue?',
  'What was net income?',
  'What were total assets?',
  'How did revenue trend over the years?',
  'What was the net profit margin?',
]

interface Props {
  document: DocumentInfo | undefined
  messages: ChatMessage[]
  loading: boolean
  error: string
  onSend: (question: string) => void
  onRetry: () => void
}

export default function ChatPanel({ document, messages, loading, error, onSend, onRetry }: Props) {
  const [text, setText] = useState('')
  const endRef = useRef<HTMLDivElement>(null)
  const ready = document?.status === 'ready'

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: 'smooth' })
  }, [messages.length, loading, error])

  function submit(e?: FormEvent) {
    e?.preventDefault()
    const question = text.trim()
    if (!question || !ready || loading) return
    setText('')
    onSend(question)
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const placeholder = !document
    ? 'Upload a PDF to start'
    : ready
      ? 'Ask a question, e.g. What was total revenue in FY2024?'
      : 'Waiting for the document to finish processing…'

  return (
    <section className="card chat" aria-labelledby="chat-title">
      <h2 id="chat-title">
        <span className="step" aria-hidden="true">
          2
        </span>
        Ask questions
      </h2>
      <div className="messages" role="log" aria-live="polite" aria-label="Conversation">
        {messages.length === 0 && (
          <>
            <p className="muted">
              Answers come only from the report and always show where they were found.
            </p>
            <div className="suggestions" aria-label="Example questions">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  type="button"
                  className="suggestion"
                  disabled={!ready || loading}
                  onClick={() => onSend(s)}
                >
                  {s}
                </button>
              ))}
            </div>
          </>
        )}
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
        {loading && (
          <div className="bubble bubble-assistant typing" role="status">
            <span className="dots" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
            Analysing the report…
          </div>
        )}
        {error && (
          <div className="error-box" role="alert">
            <span>{error}</span>
            <button type="button" className="button button-small" onClick={onRetry}>
              Retry
            </button>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <form className="chat-form" onSubmit={submit}>
        <label htmlFor="question-input" className="visually-hidden">
          Your question
        </label>
        <textarea
          id="question-input"
          rows={2}
          value={text}
          disabled={!ready}
          placeholder={placeholder}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
        />
        <button type="submit" className="button" disabled={!ready || loading || !text.trim()}>
          Send
        </button>
      </form>
    </section>
  )
}
