"""API-level tests using FastAPI's TestClient.

RAGPipeline is stubbed out via the `stub_rag_singleton` fixture so these tests
never load a real embedding/LLM model (fast + deterministic). DocumentProcessor
is left real for the upload test since chunking plain text is cheap and fast.
"""
import io

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(stub_rag_singleton):
    # stub_rag_singleton must be applied BEFORE the app's lifespan runs
    # (TestClient's context manager triggers startup), so RAGPipeline() never
    # loads a real model during `with TestClient(app) as c:`.
    with TestClient(app) as c:
        yield c


def test_health_endpoint_reports_knowledge_base_stats(client, stub_rag_singleton):
    stub_rag_singleton.get_stats.return_value = {
        "total_vectors": 12,
        "indexed_documents": ["a.txt", "b.txt"],
    }

    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["knowledge_base"]["total_vectors"] == 12
    assert body["knowledge_base"]["indexed_documents"] == ["a.txt", "b.txt"]


def test_upload_document_success(client, stub_rag_singleton):
    stub_rag_singleton.add_documents.return_value = 3

    file_content = b"Some plain text content that will be chunked for testing." * 5
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("notes.txt", io.BytesIO(file_content), "text/plain")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "notes.txt"
    assert body["chunks_indexed"] == 3
    stub_rag_singleton.add_documents.assert_called_once()


def test_upload_document_rejects_unsupported_extension(client):
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("malware.exe", io.BytesIO(b"binary"), "application/octet-stream")},
    )

    assert resp.status_code == 400
    assert "not supported" in resp.json()["detail"]


def test_upload_document_enforces_size_limit(client, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 0)  # anything is "too big"

    resp = client.post(
        "/api/documents/upload",
        files={"file": ("big.txt", io.BytesIO(b"x" * 1024), "text/plain")},
    )

    assert resp.status_code == 413


def test_stats_endpoint(client, stub_rag_singleton):
    stub_rag_singleton.get_stats.return_value = {
        "total_vectors": 5,
        "indexed_documents": ["report.pdf"],
    }

    resp = client.get("/api/documents/stats")

    assert resp.status_code == 200
    assert resp.json() == {"total_vectors": 5, "indexed_documents": ["report.pdf"]}


def test_clear_endpoint(client, stub_rag_singleton):
    resp = client.delete("/api/documents/clear")

    assert resp.status_code == 200
    stub_rag_singleton.clear.assert_called_once()


def test_query_endpoint_success(client, stub_rag_singleton):
    from langchain.schema import Document

    stub_rag_singleton.query.return_value = {
        "answer": "The company was founded in 2015.",
        "sources": [
            (Document(page_content="Founded in 2015.", metadata={"source": "a.txt", "chunk_index": 0}), 0.12)
        ],
    }

    resp = client.post("/api/query", json={"question": "When was the company founded?", "top_k": 2})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "The company was founded in 2015."
    assert body["sources"][0]["source"] == "a.txt"
    assert body["sources"][0]["score"] == 0.12
    stub_rag_singleton.query.assert_called_once_with("When was the company founded?", top_k=2)


def test_query_endpoint_rejects_empty_question(client):
    resp = client.post("/api/query", json={"question": ""})

    assert resp.status_code == 422


def test_query_endpoint_truncates_long_source_content(client, stub_rag_singleton):
    from langchain.schema import Document

    long_text = "x" * 800
    stub_rag_singleton.query.return_value = {
        "answer": "answer",
        "sources": [
            (Document(page_content=long_text, metadata={"source": "a.txt", "chunk_index": 0}), 0.5)
        ],
    }

    resp = client.post("/api/query", json={"question": "question"})

    content = resp.json()["sources"][0]["content"]
    assert len(content) == 501  # 500 chars + ellipsis
    assert content.endswith("…")
