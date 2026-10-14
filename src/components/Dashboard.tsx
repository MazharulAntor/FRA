import type { ChartEntry } from '../types'
import ChartCard from './ChartCard'

interface Props {
  entries: ChartEntry[]
  selectedId: string | null
  onSelect: (id: string) => void
}

export default function Dashboard({ entries, selectedId, onSelect }: Props) {
  const selected = entries.find((e) => e.id === selectedId) ?? entries[entries.length - 1]
  return (
    <section className="card" aria-labelledby="dash-title">
      <h2 id="dash-title">
        <span className="step" aria-hidden="true">
          3
        </span>
        Dashboard
      </h2>
      {!selected ? (
        <div className="empty-dash">
          <div className="empty-icon" aria-hidden="true">
            📊
          </div>
          <p className="muted">
            Charts appear here when an answer compares or tracks numbers, for example “How did revenue change over the
            last 3 years?”.
          </p>
        </div>
      ) : (
        <>
          <ChartCard chart={selected.chart} />
          {entries.length > 1 && (
            <div className="chart-history">
              <h3>Earlier charts this session</h3>
              <ul>
                {[...entries].reverse().map((e) => (
                  <li key={e.id}>
                    <button
                      type="button"
                      className={e.id === selected.id ? 'history-btn active' : 'history-btn'}
                      aria-current={e.id === selected.id}
                      onClick={() => onSelect(e.id)}
                    >
                      {e.chart.title}
                      <span className="muted"> — {e.question}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </section>
  )
}
