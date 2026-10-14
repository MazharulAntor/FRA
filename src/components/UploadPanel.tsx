import { useEffect, useRef, useState } from 'react'
import type { DragEvent } from 'react'
import { getDocument, uploadDocument } from '../api/client'
import type { DocumentInfo } from '../types'

interface Props {
  documents: DocumentInfo[]
  selectedId: string | null
  onSelect: (docId: string) => void
  onDocumentChange: (doc: DocumentInfo) => void
}

const POLL_MS = 2000

function statusText(doc: DocumentInfo): string {
  if (doc.status === 'processing') return 'Processing…'
  if (doc.status === 'failed') return `Failed: ${doc.error ?? 'could not read this PDF'}`
  return `Ready (${doc.pages} pages)`
}

export default function UploadPanel({ documents, selectedId, onSelect, onDocumentChange }: Props) {
  const [progress, setProgress] = useState<number | null>(null)
  const [uploadingName, setUploadingName] = useState('')
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  // Poll every 2 seconds while any document is still processing.
  const processingIds = documents.filter((d) => d.status === 'processing').map((d) => d.doc_id)
  const processingKey = processingIds.join(',')
  useEffect(() => {
    if (!processingKey) return
    const timer = setInterval(() => {
      processingKey.split(',').forEach((id) => {
        getDocument(id)
          .then(onDocumentChange)
          .catch(() => undefined) // keep polling; the status bar shows connection problems
      })
    }, POLL_MS)
    return () => clearInterval(timer)
  }, [processingKey, onDocumentChange])

  async function upload(file: File | undefined) {
    if (!file) return
    setError('')
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Please choose a PDF file.')
      return
    }
    setUploadingName(file.name)
    setProgress(0)
    try {
      const doc = await uploadDocument(file, setProgress)
      onDocumentChange(doc)
      onSelect(doc.doc_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed.')
    } finally {
      setProgress(null)
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setDragging(false)
    void upload(e.dataTransfer.files[0])
  }

  return (
    <section className="card" aria-labelledby="upload-title">
      <h2 id="upload-title">
        <span className="step" aria-hidden="true">
          1
        </span>
        Upload a report
      </h2>
      <div
        className={dragging ? 'dropzone dragging' : 'dropzone'}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <div className="drop-icon" aria-hidden="true">
          📄
        </div>
        <p>Drag and drop a text-based PDF here, or</p>
        <label className="button" htmlFor="pdf-input">
          Choose a PDF
        </label>
        <input
          id="pdf-input"
          ref={inputRef}
          className="visually-hidden-input"
          type="file"
          accept="application/pdf,.pdf"
          onChange={(e) => void upload(e.target.files?.[0])}
        />
      </div>

      {progress !== null && (
        <div className="progress-wrap">
          <span>
            Uploading {uploadingName}… {progress}%
          </span>
          <progress max={100} value={progress} aria-label="Upload progress" />
        </div>
      )}
      {error && (
        <p className="error-text" role="alert">
          {error}
        </p>
      )}

      {documents.length > 0 && (
        <fieldset className="doc-list">
          <legend>Your documents</legend>
          {documents.map((doc) => (
            <label key={doc.doc_id} className={doc.doc_id === selectedId ? 'doc-row selected' : 'doc-row'}>
              <input
                type="radio"
                name="document"
                checked={doc.doc_id === selectedId}
                onChange={() => onSelect(doc.doc_id)}
              />
              <span className="doc-name">{doc.filename}</span>
              <span className={`doc-status status-${doc.status}`}>{statusText(doc)}</span>
            </label>
          ))}
        </fieldset>
      )}
    </section>
  )
}
