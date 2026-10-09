import React from 'react';
import { X, CheckCircle, AlertTriangle, BookOpen, ExternalLink } from 'lucide-react';

export default function CitationModal({ citation, onClose }) {
  if (!citation) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <BookOpen size={18} color="var(--primary-light)" />
            <h3 style={{ fontSize: '1rem', color: 'white' }}>Statutory Passage Citation</h3>
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}>
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          {/* Metadata Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', background: 'rgba(255,255,255,0.03)', padding: '14px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Document</div>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)', marginTop: '2px' }}>{citation.document_title}</div>
            </div>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Regulatory Reference</div>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-emerald)', fontFamily: 'monospace', marginTop: '2px' }}>
                {citation.circular_number || 'Internal Policy'}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Location</div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                Page {citation.page_number} | Clause: {citation.clause_number || 'N/A'}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Section Heading</div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '2px' }}>{citation.section_heading}</div>
            </div>
          </div>

          {/* Verification Badge */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 14px', borderRadius: '6px', background: citation.verified ? 'rgba(16, 185, 129, 0.1)' : 'rgba(244, 63, 94, 0.1)', border: `1px solid ${citation.verified ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}` }}>
            {citation.verified ? (
              <>
                <CheckCircle size={16} color="var(--accent-emerald)" />
                <span style={{ fontSize: '0.8rem', color: 'var(--accent-emerald)', fontWeight: 600 }}>
                  Grounded & Verified against Ingested Source Text (Match: {Math.round(citation.similarity_match * 100)}%)
                </span>
              </>
            ) : (
              <>
                <AlertTriangle size={16} color="var(--accent-rose)" />
                <span style={{ fontSize: '0.8rem', color: 'var(--accent-rose)', fontWeight: 600 }}>
                  Unverified Citation (Potential Hallucination or Paraphrase)
                </span>
              </>
            )}
          </div>

          {/* Quoted Excerpt */}
          <div>
            <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase', marginBottom: '6px' }}>
              Verbatim Supporting Excerpt
            </div>
            <div style={{ background: '#090d16', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', padding: '16px', fontSize: '0.9rem', lineHeight: '1.6', color: '#f1f5f9', fontStyle: 'italic', borderLeft: '3px solid var(--primary)' }}>
              "{citation.quote}"
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
