import { useState } from 'react'
import type { Citation } from '../types'

interface Props {
  citation: Citation
}

/** Clickable chip such as "Annual_Report.pdf, p.45". Click to show the quoted text. */
export default function CitationChip({ citation }: Props) {
  const [open, setOpen] = useState(false)
  return (
    <span className="chip-wrap">
      <button
        type="button"
        className="chip"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        {citation.filename}, p.{citation.page}
      </button>
      {open && (
        <span className="chip-panel" role="note">
          <strong>
            {citation.filename}, page {citation.page}
          </strong>
          <q>{citation.snippet}</q>
        </span>
      )}
    </span>
  )
}
