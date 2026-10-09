import React, { useState } from 'react';
import { Send, Sparkles, BookOpen, Layers, CheckCircle2, AlertTriangle, Clock, Sliders } from 'lucide-react';
import { askQuestion } from '../services/api';

export default function ChatPanel({
  queryInput,
  setQueryInput,
  onSelectCitation,
  onInspectChunks,
}) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      answer: 'Welcome to the BFSI Policy Research Assistant. You can ask compliance and regulatory questions across ingested RBI circulars, Master Directions, payment systems directives, and banking SOPs. Every claim will be backed by verified citations.',
      citations: [],
      latency_ms: 0,
      model_used: 'System',
      retrieved_chunks: [],
      has_sufficient_context: true,
    },
  ]);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState('hybrid');
  const [useReranker, setUseReranker] = useState(false);

  const handleSubmit = async (e) => {
    e?.preventDefault();
    const query = queryInput.trim();
    if (!query || loading) return;

    // Add user message
    const newMessages = [...messages, { role: 'user', content: query }];
    setMessages(newMessages);
    setQueryInput('');
    setLoading(true);

    try {
      const response = await askQuestion({
        query,
        mode,
        top_k: 4,
        use_reranker: useReranker,
      });

      setMessages([
        ...newMessages,
        {
          role: 'assistant',
          answer: response.answer,
          citations: response.citations || [],
          latency_ms: response.latency_ms,
          model_used: response.model_used,
          confidence_note: response.confidence_note,
          retrieved_chunks: response.retrieved_chunks || [],
          has_sufficient_context: response.has_sufficient_context,
          verification_passed: response.verification_passed,
        },
      ]);
    } catch (err) {
      setMessages([
        ...newMessages,
        {
          role: 'assistant',
          answer: `Error executing regulatory query: ${err.message}`,
          citations: [],
          latency_ms: 0,
          model_used: 'Error',
          has_sufficient_context: false,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-workspace">
      <div className="messages-container">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`message-bubble ${msg.role === 'user' ? 'message-user' : 'message-assistant'}`}
          >
            {msg.role === 'user' ? (
              <div>{msg.content}</div>
            ) : (
              <div>
                <div className="answer-header">
                  <div className="answer-badge">
                    <Sparkles size={14} color="var(--accent-emerald)" />
                    <span>{msg.has_sufficient_context ? 'Grounded Regulatory Answer' : 'Context Notice'}</span>
                  </div>
                  {msg.latency_ms > 0 && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span className="telemetry-tag" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Clock size={11} /> {Math.round(msg.latency_ms)}ms
                      </span>
                      <span className="telemetry-tag">{msg.model_used}</span>
                    </div>
                  )}
                </div>

                <div className="answer-body">{msg.answer}</div>

                {/* Citations Strip */}
                {msg.citations && msg.citations.length > 0 && (
                  <div className="citations-strip">
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', fontWeight: 600, marginRight: '4px' }}>
                      CITATIONS:
                    </span>
                    {msg.citations.map((c, cIdx) => (
                      <div
                        key={cIdx}
                        className="citation-chip"
                        onClick={() => onSelectCitation(c)}
                        title="Click to view verified source excerpt"
                      >
                        <BookOpen size={12} />
                        <span>{c.circular_number || c.document_title}</span>
                        <span style={{ opacity: 0.7 }}>• Pg {c.page_number}</span>
                        {c.verified && (
                          <CheckCircle2 size={12} color="var(--accent-emerald)" />
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {/* Action Bar for Retrieved Chunks */}
                {msg.retrieved_chunks && msg.retrieved_chunks.length > 0 && (
                  <div style={{ marginTop: '14px', display: 'flex', justifyContent: 'flex-end' }}>
                    <button
                      className="toggle-btn"
                      onClick={() => onInspectChunks(msg.retrieved_chunks)}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        background: 'rgba(255,255,255,0.04)',
                        border: '1px solid var(--border-subtle)',
                        fontSize: '0.72rem',
                        padding: '4px 8px',
                      }}
                    >
                      <Layers size={13} color="var(--accent-cyan)" />
                      <span>Inspect Context Chunks ({msg.retrieved_chunks.length})</span>
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="message-bubble message-assistant" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div className="status-dot animate-pulse"></div>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              Executing Hybrid RRF retrieval and synthesizing citation-grounded response...
            </span>
          </div>
        )}
      </div>

      {/* Input Controls */}
      <div className="chat-controls">
        <form onSubmit={handleSubmit} className="search-input-wrapper">
          <input
            type="text"
            className="search-input"
            value={queryInput}
            onChange={(e) => setQueryInput(e.target.value)}
            placeholder="Ask a compliance question (e.g. 'What is the permissible cap for Default Loss Guarantee?')..."
            disabled={loading}
          />
          <button type="submit" className="search-button" disabled={loading || !queryInput.trim()}>
            <Send size={15} />
            <span>Search Policy</span>
          </button>
        </form>

        <div className="mode-toggles">
          <div className="toggle-group">
            <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 600 }}>
              Retrieval Mode:
            </span>
            <button
              type="button"
              className={`toggle-btn ${mode === 'hybrid' ? 'active' : ''}`}
              onClick={() => setMode('hybrid')}
            >
              Hybrid (RRF)
            </button>
            <button
              type="button"
              className={`toggle-btn ${mode === 'dense' ? 'active' : ''}`}
              onClick={() => setMode('dense')}
            >
              Dense (Vector)
            </button>
            <button
              type="button"
              className={`toggle-btn ${mode === 'bm25' ? 'active' : ''}`}
              onClick={() => setMode('bm25')}
            >
              BM25 (Exact)
            </button>
          </div>

          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', fontSize: '0.75rem' }}>
            <input
              type="checkbox"
              checked={useReranker}
              onChange={(e) => setUseReranker(e.target.checked)}
              style={{ accentColor: 'var(--primary)' }}
            />
            <span>Cross-Encoder Reranker</span>
          </label>
        </div>
      </div>
    </div>
  );
}
