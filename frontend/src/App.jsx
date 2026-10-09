import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import DocumentPanel from './components/DocumentPanel';
import ChatPanel from './components/ChatPanel';
import CitationModal from './components/CitationModal';
import RetrievedChunksModal from './components/RetrievedChunksModal';
import AuditLogModal from './components/AuditLogModal';
import { getHealth, getDocuments } from './services/api';

export default function App() {
  const [health, setHealth] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [queryInput, setQueryInput] = useState('');
  const [activeCitation, setActiveCitation] = useState(null);
  const [inspectChunks, setInspectChunks] = useState(null);
  const [showAuditLogs, setShowAuditLogs] = useState(false);

  const loadData = async () => {
    try {
      const [h, d] = await Promise.all([getHealth(), getDocuments()]);
      setHealth(h);
      setDocuments(d);
    } catch (err) {
      console.error('Failed to load initial data:', err);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app-container">
      <Header
        health={health}
        onOpenAuditLogs={() => setShowAuditLogs(true)}
      />

      <div className="main-layout">
        <DocumentPanel
          documents={documents}
          onRefresh={loadData}
          onSelectSampleQuestion={(q) => setQueryInput(q)}
        />

        <ChatPanel
          queryInput={queryInput}
          setQueryInput={setQueryInput}
          onSelectCitation={(c) => setActiveCitation(c)}
          onInspectChunks={(chunks) => setInspectChunks(chunks)}
        />
      </div>

      {/* Modals */}
      <CitationModal
        citation={activeCitation}
        onClose={() => setActiveCitation(null)}
      />

      <RetrievedChunksModal
        chunks={inspectChunks}
        onClose={() => setInspectChunks(null)}
      />

      {showAuditLogs && (
        <AuditLogModal
          onClose={() => setShowAuditLogs(false)}
        />
      )}
    </div>
  );
}
