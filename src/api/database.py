"""SQLite Database Layer for Query Auditing and Document Registry.

Stores LLMOps telemetry (query text, answer, latency, token counts, retrieved chunk IDs)
and uploaded document metadata in a lightweight local SQLite database.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from config.settings import PROJECT_ROOT, settings
from src.core.logging import app_logger

DB_PATH = PROJECT_ROOT / "data" / "bfsi_assistant.db"


def get_db_connection() -> sqlite3.Connection:
    """Create a thread-safe connection to the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize database tables for queries and documents."""
    conn = get_db_connection()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query_text TEXT NOT NULL,
                answer_text TEXT NOT NULL,
                model_used TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                retrieved_chunk_ids TEXT DEFAULT '[]',
                citations_count INTEGER DEFAULT 0,
                verification_passed INTEGER DEFAULT 1,
                has_sufficient_context INTEGER DEFAULT 1,
                confidence_note TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_records (
                doc_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                issuer TEXT NOT NULL,
                circular_number TEXT,
                date TEXT,
                doc_type TEXT NOT NULL,
                source_filename TEXT NOT NULL,
                total_pages INTEGER DEFAULT 1,
                total_chunks INTEGER DEFAULT 0,
                ingested_at TEXT NOT NULL
            );
            """
        )
    conn.close()
    app_logger.info("Initialized SQLite database at %s", DB_PATH)


def log_query(
    query_text: str,
    answer_text: str,
    model_used: str,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    retrieved_chunk_ids: list[str],
    citations_count: int,
    verification_passed: bool,
    has_sufficient_context: bool,
    confidence_note: str,
) -> int:
    """Insert an executed query log into SQLite for LLMOps and audit trails."""
    conn = get_db_connection()
    now_utc = datetime.now(timezone.utc).isoformat()
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO query_logs (
                timestamp, query_text, answer_text, model_used, latency_ms,
                prompt_tokens, completion_tokens, retrieved_chunk_ids,
                citations_count, verification_passed, has_sufficient_context, confidence_note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_utc,
                query_text,
                answer_text,
                model_used,
                latency_ms,
                prompt_tokens,
                completion_tokens,
                json.dumps(retrieved_chunk_ids),
                citations_count,
                1 if verification_passed else 0,
                1 if has_sufficient_context else 0,
                confidence_note,
            ),
        )
        query_id = cursor.lastrowid or 0
    conn.close()
    return query_id


def get_query_logs(limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve recent query logs."""
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT * FROM query_logs ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        d["retrieved_chunk_ids"] = json.loads(d["retrieved_chunk_ids"])
        d["verification_passed"] = bool(d["verification_passed"])
        d["has_sufficient_context"] = bool(d["has_sufficient_context"])
        results.append(d)
    return results


def upsert_document_record(
    doc_id: str,
    title: str,
    issuer: str,
    circular_number: str | None,
    date: str | None,
    doc_type: str,
    source_filename: str,
    total_pages: int,
    total_chunks: int,
) -> None:
    """Record or update document registry metadata."""
    conn = get_db_connection()
    now_utc = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO document_records (
                doc_id, title, issuer, circular_number, date, doc_type,
                source_filename, total_pages, total_chunks, ingested_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                title,
                issuer,
                circular_number,
                date,
                doc_type,
                source_filename,
                total_pages,
                total_chunks,
                now_utc,
            ),
        )
    conn.close()


def list_documents() -> list[dict[str, Any]]:
    """Return all recorded regulatory documents."""
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM document_records ORDER BY ingested_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_document_record(doc_id: str) -> bool:
    """Delete document entry from registry."""
    conn = get_db_connection()
    with conn:
        cursor = conn.execute("DELETE FROM document_records WHERE doc_id = ?", (doc_id,))
        deleted = cursor.rowcount > 0
    conn.close()
    return deleted
