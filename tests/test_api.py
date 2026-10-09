"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
import pytest
from src.api.database import init_db
from src.api.main import app

init_db()
client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "BFSI Policy Research Assistant" in data["app_name"]


def test_documents_list_endpoint():
    response = client.get("/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_unsupported_file_upload():
    response = client.post(
        "/documents",
        files={"file": ("malicious.exe", b"binarycontent", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_ask_endpoint_grounded_response():
    response = client.post(
        "/ask",
        json={
            "query": "What are the guidelines on AI credit underwriting?",
            "mode": "hybrid",
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "latency_ms" in data
    assert "retrieved_chunks" in data
    assert data["has_sufficient_context"] is True


def test_queries_audit_log_endpoint():
    # Submit a query first
    client.post(
        "/ask",
        json={"query": "Test audit logging query", "mode": "hybrid", "top_k": 2},
    )

    response = client.get("/queries?limit=5")
    assert response.status_code == 200
    logs = response.json()
    assert len(logs) >= 1
    assert any("Test audit logging" in l["query_text"] for l in logs)
