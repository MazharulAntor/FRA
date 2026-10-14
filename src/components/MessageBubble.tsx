import type { ReactNode } from 'react'
import type { ChatMessage } from '../types'
import CitationChip from './CitationChip'

const MARKER = /\[([^[\]]*?),\s*p\.?\s*(\d+)\]/g

const TYPE_LABELS: Record<string, string> = {
  lookup: 'Lookup',
  calculation: 'Calculation',
  comparison: 'Comparison',
  trend: 'Trend',
  summary: 'Summary',
  advice: 'Advice (declined)',
  out_of_scope: 'Out of scope',
}

/** Replace "[file.pdf, p.45]" markers in the answer text with clickable chips. */
function renderAnswer(message: ChatMessage): ReactNode[] {
  const citations = message.response?.citations ?? []
  const parts: ReactNode[] = []
  let last = 0
  let n = 0
  for (const match of message.content.matchAll(MARKER)) {
    const index = match.index ?? 0
    parts.push(message.content.slice(last, index))
    const page = Number(match[2])
    const citation = citations.find((c) => c.page === page)
    parts.push(citation ? <CitationChip key={`c${n++}`} citation={citation} /> : match[0])
    last = index + match[0].length
  }
  parts.push(message.content.slice(last))
  return parts
}

function formatInput(value: unknown): string {
  if (value && typeof value === 'object') {
    const v = value as { value?: unknown; unit?: unknown; label?: unknown }
    if ('value' in v) return `${v.label ? `${String(v.label)}: ` : ''}${String(v.value)}${v.unit ? ` ${String(v.unit)}` : ''}`
    if (Array.isArray(value)) return value.map(formatInput).join(', ')
  }
  return String(value)
}

export default function MessageBubble({ message }: { message: ChatMessage }) {
  if (message.role === 'user') {
    return (
      <div className="bubble bubble-user">
        <span className="visually-hidden">You: </span>
        {message.content}
      </div>
    )
  }

  const response = message.response
  const ungrounded = response ? !response.grounded : false
  const calculations = response?.calculations ?? []

  return (
    <div className={ungrounded ? 'bubble bubble-assistant bubble-warning' : 'bubble bubble-assistant'}>
      <span className="visually-hidden">Assistant: </span>
      {response && (
        <div className="badges">
          <span className="badge">{TYPE_LABELS[response.question_type] ?? response.question_type}</span>
          {ungrounded && <span className="badge badge-warning">⚠ Not found in the report</span>}
        </div>
      )}
      <p className="answer-text">{renderAnswer(message)}</p>

      {calculations.length > 0 && (
        <details className="calc-details">
          <summary>Calculation details</summary>
          <ul>
            {calculations.map((c, i) => (
              <li key={i}>
                <strong>{c.name}</strong>
                {c.error ? (
                  <span className="error-text"> could not be calculated: {c.error}</span>
                ) : (
                  <>
                    {' '}
                    = {c.result}
                    {c.unit ? ` ${c.unit}` : ''}
                    <div className="calc-formula">{c.formula}</div>
                  </>
                )}
                {c.inputs && (
                  <div className="calc-formula">
                    Inputs:{' '}
                    {Object.entries(c.inputs)
                      .map(([k, v]) => `${k} = ${formatInput(v)}`)
                      .join('; ')}
                  </div>
                )}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  )
}
