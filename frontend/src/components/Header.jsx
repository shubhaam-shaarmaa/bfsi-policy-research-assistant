import React from 'react';
import { Shield, Activity, Database, FileText, History } from 'lucide-react';

export default function Header({ health, onOpenAuditLogs }) {
  return (
    <header className="top-header">
      <div className="brand-section">
        <div className="brand-icon">
          <Shield size={20} />
        </div>
        <div>
          <div className="brand-title">
            BFSI Policy Research Assistant
            <span className="brand-tag">RAG v1.0</span>
          </div>
        </div>
      </div>

      <div className="header-status">
        <div className="status-chip" title="System Health">
          <span className="status-dot"></span>
          <span>{health ? health.status : 'Connecting...'}</span>
        </div>

        <div className="status-chip" title="Vector Database">
          <Database size={13} />
          <span>{health ? `${health.vector_db_provider.toUpperCase()} (${health.total_vectors} vecs)` : 'Vector DB'}</span>
        </div>

        <div className="status-chip" title="Loaded Documents">
          <FileText size={13} />
          <span>{health ? `${health.total_documents} Docs` : '0 Docs'}</span>
        </div>

        <button
          onClick={onOpenAuditLogs}
          className="toggle-btn"
          style={{ display: 'flex', alignItems: 'center', gap: '6px', border: '1px solid var(--border-subtle)', padding: '5px 10px' }}
        >
          <History size={14} />
          <span>Audit Logs</span>
        </button>
      </div>
    </header>
  );
}
