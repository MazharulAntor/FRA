// Mirrors backend/app/schemas.py

export type DocStatus = 'processing' | 'ready' | 'failed'

export interface DocumentInfo {
  doc_id: string
  filename: string
  pages: number
  status: DocStatus
  error: string | null
}

export interface Citation {
  doc_id: string
  filename: string
  page: number
  snippet: string
}

export interface ChartSeries {
  label: string
  value: number
  period: string | null
  unit: string | null
}

export type ChartType = 'bar' | 'line' | 'pie'

export interface ChartSpec {
  chart_type: ChartType
  title: string
  series: ChartSeries[]
  citations: Citation[]
}

export interface HistoryMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface AskRequest {
  doc_id: string
  question: string
  history: HistoryMessage[]
}

export interface Calculation {
  name: string
  formula?: string
  inputs?: Record<string, unknown>
  result?: number
  unit?: string | null
  error?: string
}

export interface AskResponse {
  answer: string
  citations: Citation[]
  chart: ChartSpec | null
  question_type: string
  calculations: Calculation[]
  grounded: boolean
}

export interface HealthResponse {
  status: string
  llm_provider: string
  model: string
}

// ---- frontend-only types ----

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  response?: AskResponse
}

export interface ChartEntry {
  id: string
  question: string
  chart: ChartSpec
}
