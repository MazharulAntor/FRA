import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { ChartSpec } from '../types'
import CitationChip from './CitationChip'

const COLORS = ['#6366f1', '#0ea5e9', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6']
const AXIS = { fill: 'var(--muted)', fontSize: 12 }

/** Pick the most readable x-axis label for each point. */
function pointNames(chart: ChartSpec): string[] {
  const periods = chart.series.map((s) => s.period)
  if (periods.every(Boolean) && new Set(periods).size === periods.length) return periods as string[]
  const labels = chart.series.map((s) => s.label)
  if (new Set(labels).size === labels.length) return labels
  return chart.series.map((s) => (s.period ? `${s.label} ${s.period}` : s.label))
}

export default function ChartCard({ chart }: { chart: ChartSpec }) {
  const names = pointNames(chart)
  const unit = chart.series[0]?.unit ?? ''
  const data = chart.series.map((s, i) => ({ name: names[i], value: s.value }))
  const formatValue = (v: unknown) => `${Number(v).toLocaleString()}${unit ? ` ${unit}` : ''}`
  const summary =
    `${chart.chart_type} chart: ${chart.title}. ` + data.map((d) => `${d.name}: ${formatValue(d.value)}`).join('; ')
  const tooltipStyle = {
    background: 'var(--card-solid)',
    border: '1px solid var(--border)',
    borderRadius: 10,
    color: 'var(--text)',
  }
  const seriesName = chart.series[0]?.label ?? 'Value'

  return (
    <figure className="chart-card">
      <figcaption>
        <h3>{chart.title}</h3>
        {unit && <span className="muted">Values in {unit}</span>}
      </figcaption>

      <div className="tiles">
        {data.map((d) => (
          <div className="tile" key={d.name}>
            <div className="tile-label">{d.name}</div>
            <div className="tile-value">{Number(d.value).toLocaleString()}</div>
            {unit && <div className="tile-unit">{unit}</div>}
          </div>
        ))}
      </div>

      <div className="chart-box" role="img" aria-label={summary}>
        <ResponsiveContainer width="100%" height={280}>
          {chart.chart_type === 'pie' ? (
            <PieChart>
              <Pie data={data} dataKey="value" nameKey="name" outerRadius={95} innerRadius={45} paddingAngle={2} label>
                {data.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} stroke="var(--card-solid)" />
                ))}
              </Pie>
              <Tooltip formatter={formatValue} contentStyle={tooltipStyle} />
              <Legend />
            </PieChart>
          ) : chart.chart_type === 'line' ? (
            <LineChart data={data} margin={{ top: 12, right: 20, left: 8, bottom: 8 }}>
              <CartesianGrid stroke="var(--grid)" strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={AXIS} stroke="var(--grid)" />
              <YAxis tickFormatter={(v) => Number(v).toLocaleString()} width={80} tick={AXIS} stroke="var(--grid)" />
              <Tooltip formatter={formatValue} contentStyle={tooltipStyle} />
              <Line
                type="monotone"
                dataKey="value"
                name={seriesName}
                stroke={COLORS[0]}
                strokeWidth={3}
                dot={{ r: 5, fill: COLORS[0], strokeWidth: 0 }}
                activeDot={{ r: 7 }}
                animationDuration={700}
              />
            </LineChart>
          ) : (
            <BarChart data={data} margin={{ top: 12, right: 20, left: 8, bottom: 8 }}>
              <defs>
                <linearGradient id="barFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={COLORS[1]} />
                  <stop offset="100%" stopColor={COLORS[0]} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="var(--grid)" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" tick={AXIS} stroke="var(--grid)" />
              <YAxis tickFormatter={(v) => Number(v).toLocaleString()} width={80} tick={AXIS} stroke="var(--grid)" />
              <Tooltip formatter={formatValue} contentStyle={tooltipStyle} cursor={{ fill: 'rgb(99 102 241 / 0.08)' }} />
              <Bar dataKey="value" name={seriesName} fill="url(#barFill)" radius={[8, 8, 0, 0]} animationDuration={700} />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>

      {chart.citations.length > 0 && (
        <div className="chart-citations">
          <span className="muted">Sources: </span>
          {chart.citations.map((c) => (
            <CitationChip key={`${c.filename}-${c.page}`} citation={c} />
          ))}
        </div>
      )}
    </figure>
  )
}
