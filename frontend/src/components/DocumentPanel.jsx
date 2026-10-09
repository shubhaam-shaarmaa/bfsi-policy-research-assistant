import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, Trash2, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import { uploadDocument, deleteDocument } from '../services/api';

export default function DocumentPanel({ documents, onRefresh, onSelectSampleQuestion }) {
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      const res = await uploadDocument(file);
      setUploadSuccess(`Ingested "${file.name}" (${res.total_chunks} chunks)`);
      if (fileInputRef.current) fileInputRef.current.value = '';
      onRefresh();
    } catch (err) {
      setUploadError(err.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (docId, title) => {
    if (!window.confirm(`Delete "${title}" and its vector index?`)) return;
    try {
      await deleteDocument(docId);
      onRefresh();
    } catch (err) {
      alert(`Delete failed: ${err.message}`);
    }
  };

  return (
    <aside className="doc-panel">
      {/* Upload Section */}
      <div className="panel-section">
        <div className="section-title">
          <span>Ingest Document</span>
          <span className="telemetry-tag">PDF / DOCX / TXT</span>
        </div>

        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileUpload}
          accept=".pdf,.docx,.txt,.md"
          style={{ display: 'none' }}
        />

        <div
          className="dropzone"
          onClick={() => !uploading && fileInputRef.current?.click()}
        >
          {uploading ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
              <Loader2 className="dropzone-icon animate-spin" size={24} />
              <div className="dropzone-text">Parsing & chunking text...</div>
            </div>
          ) : (
            <>
              <UploadCloud className="dropzone-icon" size={24} />
              <div className="dropzone-text">
                <strong>Click to upload</strong> RBI circular or policy SOP
              </div>
              <div className="dropzone-hint">Extracts clauses, headings, and metadata</div>
            </>
          )}
        </div>

        {uploadSuccess && (
          <div style={{ marginTop: '8px', fontSize: '0.75rem', color: 'var(--accent-emerald)', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <CheckCircle size={14} />
            <span>{uploadSuccess}</span>
          </div>
        )}

        {uploadError && (
          <div style={{ marginTop: '8px', fontSize: '0.75rem', color: 'var(--accent-rose)', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <AlertCircle size={14} />
            <span>{uploadError}</span>
          </div>
        )}
      </div>

      {/* Suggested Questions */}
      <div className="panel-section" style={{ padding: '12px 20px' }}>
        <div className="section-title">
          <span>Sample Questions</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <button
            className="toggle-btn"
            style={{ textAlign: 'left', background: 'rgba(255,255,255,0.03)', padding: '6px 8px', borderRadius: '4px' }}
            onClick={() => onSelectSampleQuestion("What is the permissible portfolio cap for Default Loss Guarantee under RBI?")}
          >
            "What is the permissible cap for Default Loss Guarantee?"
          </button>
          <button
            className="toggle-btn"
            style={{ textAlign: 'left', background: 'rgba(255,255,255,0.03)', padding: '6px 8px', borderRadius: '4px' }}
            onClick={() => onSelectSampleQuestion("What are the requirements for AI credit underwriting model governance?")}
          >
            "What are the requirements for AI credit underwriting?"
          </button>
          <button
            className="toggle-btn"
            style={{ textAlign: 'left', background: 'rgba(255,255,255,0.03)', padding: '6px 8px', borderRadius: '4px' }}
            onClick={() => onSelectSampleQuestion("What is the cutoff time for RTGS batch settlement?")}
          >
            "What is the cutoff time for RTGS batch settlement?"
          </button>
        </div>
      </div>

      {/* Document Library */}
      <div className="panel-section" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div className="section-title">
          <span>Policy Library ({documents.length})</span>
        </div>

        <div className="doc-list">
          {documents.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '24px 8px', color: 'var(--text-dim)', fontSize: '0.8rem' }}>
              No documents ingested yet.<br />Upload a circular to begin.
            </div>
          ) : (
            documents.map((doc) => (
              <div key={doc.doc_id} className="doc-card">
                <div className="doc-header">
                  <span className="doc-name">{doc.title}</span>
                  <button
                    onClick={() => handleDelete(doc.doc_id, doc.title)}
                    style={{ background: 'none', border: 'none', color: 'var(--text-dim)', cursor: 'pointer', padding: '2px' }}
                    title="Delete document"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
                <div className="doc-meta">
                  <span className="doc-badge">{doc.doc_type.toUpperCase()}</span>
                  <span>{doc.total_pages} pg</span>
                  <span>•</span>
                  <span>{doc.total_chunks} chunks</span>
                </div>
                {doc.circular_number && (
                  <div style={{ fontSize: '0.68rem', color: 'var(--accent-emerald)', fontFamily: 'monospace' }}>
                    {doc.circular_number}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </aside>
  );
}
