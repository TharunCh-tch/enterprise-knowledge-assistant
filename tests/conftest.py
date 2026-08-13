"""Shared pytest fixtures for the Enterprise Knowledge Assistant test suite."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.rag.pipeline import RAGPipeline

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_doc_path() -> Path:
    """Path to the small sample knowledge-base document used across tests."""
    return FIXTURES_DIR / "test_document.txt"


@pytest.fixture
def isolated_data_dirs(tmp_path, monkeypatch):
    """Redirect settings.upload_dir / settings.faiss_dir to a throwaway tmp dir
    so tests never touch (or depend on) the real data/ directory."""
    from app.core.config import settings

    upload_dir = tmp_path / "uploads"
    faiss_dir = tmp_path / "faiss_index"
    monkeypatch.setattr(type(settings), "upload_dir", property(lambda self: upload_dir))
    monkeypatch.setattr(type(settings), "faiss_dir", property(lambda self: faiss_dir))
    return {"upload_dir": upload_dir, "faiss_dir": faiss_dir}


@pytest.fixture
def bare_pipeline():
    """A RAGPipeline instance created WITHOUT running __init__ (so no real
    embedding/LLM models are loaded) and WITHOUT touching the process-wide
    singleton. Callers attach whatever mocks they need to .embeddings,
    .vector_store, ._chain, etc.
    """
    pipeline = object.__new__(RAGPipeline)
    pipeline._ready = True
    pipeline.vector_store = None
    return pipeline


@pytest.fixture
def stub_rag_singleton(monkeypatch):
    """Replace the process-wide RAGPipeline singleton with a MagicMock so that
    app.main's lifespan startup and the API routes' `RAGPipeline()` calls
    never load a real embedding/LLM model. Yields the mock for configuration.
    """
    fake = MagicMock(name="RAGPipelineStub")
    monkeypatch.setattr(RAGPipeline, "_instance", fake)
    yield fake
    monkeypatch.setattr(RAGPipeline, "_instance", None)
