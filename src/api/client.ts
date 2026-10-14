import type {
  AskRequest,
  AskResponse,
  DocumentInfo,
  HealthResponse,
} from '../types'

const BASE_URL: string = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

/** An error with a message that is safe to show to the user. */
export class ApiError extends Error {
  status: number
  code: string

  constructor(status: number, code: string, message: string) {
    super(message)
    this.status = status
    this.code = code
  }
}

const NETWORK_MESSAGE = 'Cannot reach the server. Check that the backend is running, then try again.'

function friendlyMessage(status: number, code: string, message: string): string {
  if (status === 409 || code === 'not_ready') return 'The document is not ready yet. Please wait a moment.'
  if (status === 422 && code === 'empty_question') return 'Please type a question first.'
  if (status === 404) return 'That document was not found. Please upload it again.'
  return message || 'Something went wrong. Please try again.'
}

async function toApiError(res: Response): Promise<ApiError> {
  let code = 'http_error'
  let message = ''
  try {
    const body = (await res.json()) as { error?: { code?: string; message?: string } }
    code = body.error?.code ?? code
    message = body.error?.message ?? ''
  } catch {
    // body was not JSON
  }
  return new ApiError(res.status, code, friendlyMessage(res.status, code, message))
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE_URL}${path}`, init)
  } catch {
    throw new ApiError(0, 'network', NETWORK_MESSAGE)
  }
  if (!res.ok) throw await toApiError(res)
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}

export function listDocuments(): Promise<DocumentInfo[]> {
  return request<DocumentInfo[]>('/documents')
}

export function getDocument(docId: string): Promise<DocumentInfo> {
  return request<DocumentInfo>(`/documents/${encodeURIComponent(docId)}`)
}

/** Upload a PDF (XMLHttpRequest so we can report upload progress, 0-100). */
export function uploadDocument(file: File, onProgress: (percent: number) => void): Promise<DocumentInfo> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${BASE_URL}/documents`)
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100))
    }
    xhr.onerror = () => reject(new ApiError(0, 'network', NETWORK_MESSAGE))
    xhr.onload = () => {
      let body: unknown = null
      try {
        body = JSON.parse(xhr.responseText)
      } catch {
        // not JSON
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body as DocumentInfo)
        return
      }
      const err = (body as { error?: { code?: string; message?: string } } | null)?.error
      reject(new ApiError(xhr.status, err?.code ?? 'http_error', err?.message ?? 'Upload failed.'))
    }
    const form = new FormData()
    form.append('file', file)
    xhr.send(form)
  })
}

export function ask(body: AskRequest): Promise<AskResponse> {
  return request<AskResponse>('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}
