import type { DocumentInfo, HealthResponse } from '../types'

interface Props {
  health: HealthResponse | null
  online: boolean
  document: DocumentInfo | undefined
}

export default function StatusBar({ health, online, document }: Props) {
  return (
    <div className="status-bar header-right" role="status" aria-live="polite">
      <span className="pill">
        <span className={online ? 'dot dot-ok' : 'dot dot-bad'} aria-hidden="true" />
        {online
          ? `Backend connected${health ? ` · ${health.model}` : ''}`
          : 'Backend offline'}
      </span>
      <span className="pill pill-doc">
        {document ? `📄 ${document.filename}` : 'No document selected'}
      </span>
    </div>
  )
}
