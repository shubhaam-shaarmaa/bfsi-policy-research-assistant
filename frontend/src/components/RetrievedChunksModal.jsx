import React from 'react';
import { X, Layers, Database, Sparkles } from 'lucide-react';

export default function RetrievedChunksModal({ chunks, onClose }) {
  if (!chunks) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{ maxWidth: '840px' }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={18} color="var(--accent-cyan)" />
            <h3 style={{ fontSize: '1rem', color: 'white' }}>
              Retrieval Inspector: Chunks Fed to Context Window ({chunks.length})
            </h3>
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}>
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
            These chunks were selected by the Hybrid RRF Retriever (+ Reranker) and budgeted into Claude's prompt context.
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {chunks.map((chunk, idx) => (
              <div
                key={chunk.chunk_id || idx}
                style={{
                  background: 'rgba(255,255,255,0.03)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: '8px',
                  padding: '14px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '8px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '0.7rem', padding: '2px 6px', borderRadius: '4px', background: 'rgba(99, 102, 241, 0.2)', color: 'var(--primary-light)', fontWeight: 700 }}>
                      RANK #{idx + 1}
                    </span>
                    <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)' }}>
                      {chunk.document_title}
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.75rem' }}>
                    <span style={{ color: 'var(--text-dim)' }}>Source: <strong>{chunk.source}</strong></span>
                    <span style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>Score: {chunk.score}</span>
                  </div>
                </div>

                <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', display: 'flex', gap: '12px' }}>
                  <span>Ref: {chunk.circular_number || 'N/A'}</span>
                  <span>Page: {chunk.page_number}</span>
                  <span>Clause: {chunk.clause_number || 'N/A'}</span>
                  <span>ID: {chunk.chunk_id}</span>
                </div>

                <div style={{ background: '#090d16', padding: '10px 12px', borderRadius: '6px', fontSize: '0.82rem', color: '#cbd5e1', lineHeight: '1.5', fontFamily: 'monospace' }}>
                  {chunk.snippet}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
