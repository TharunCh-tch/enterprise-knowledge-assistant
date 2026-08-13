"""Unit tests for app.rag.pipeline.RAGPipeline's retrieval/indexing logic.

These tests never load a real embedding model or LLM — they use `bare_pipeline`
(a RAGPipeline built via object.__new__, bypassing __init__) with mocked
.embeddings / .vector_store / ._chain, so they run in well under a second.
"""
from unittest.mock import MagicMock

from langchain.schema import Document

import app.rag.pipeline as pipeline_module


def _doc(text: str, source: str, chunk_index: int) -> Document:
    return Document(page_content=text, metadata={"source": source, "chunk_index": chunk_index})


def test_query_on_empty_knowledge_base_returns_placeholder_answer(bare_pipeline):
    bare_pipeline.vector_store = None

    result = bare_pipeline.query("What is the revenue?")

    assert "empty" in result["answer"].lower()
    assert result["sources"] == []


def test_query_uses_configured_top_k_when_not_overridden(bare_pipeline, monkeypatch):
    monkeypatch.setattr(pipeline_module.settings, "TOP_K", 3)

    fake_store = MagicMock()
    fake_store.index.ntotal = 5
    fake_store.similarity_search_with_score.return_value = [
        (_doc("chunk A", "doc.txt", 0), 0.12),
    ]
    bare_pipeline.vector_store = fake_store
    bare_pipeline._chain = MagicMock()
    bare_pipeline._chain.invoke.return_value = "The revenue was $142 million."

    bare_pipeline.query("What is the revenue?")

    fake_store.similarity_search_with_score.assert_called_once_with(
        "What is the revenue?", k=3
    )


def test_query_respects_explicit_top_k_override(bare_pipeline):
    fake_store = MagicMock()
    fake_store.index.ntotal = 5
    fake_store.similarity_search_with_score.return_value = []
    bare_pipeline.vector_store = fake_store
    bare_pipeline._chain = MagicMock()
    bare_pipeline._chain.invoke.return_value = "answer"

    bare_pipeline.query("question", top_k=7)

    fake_store.similarity_search_with_score.assert_called_once_with("question", k=7)


def test_query_formats_context_and_returns_sources_with_scores(bare_pipeline):
    fake_store = MagicMock()
    fake_store.index.ntotal = 2
    fake_store.similarity_search_with_score.return_value = [
        (_doc("Revenue was $142M.", "annual_report.txt", 4), 0.0821),
        (_doc("Founded in 2015.", "annual_report.txt", 0), 0.31),
    ]
    bare_pipeline.vector_store = fake_store
    bare_pipeline._chain = MagicMock()
    bare_pipeline._chain.invoke.return_value = "  The revenue was $142 million.  "

    result = bare_pipeline.query("What is the revenue?")

    # Answer is stripped.
    assert result["answer"] == "The revenue was $142 million."

    # Sources preserve (Document, score) pairs, in retrieval order.
    assert len(result["sources"]) == 2
    doc0, score0 = result["sources"][0]
    assert doc0.page_content == "Revenue was $142M."
    assert score0 == 0.0821

    # The chain was invoked with a context string built from both chunks.
    call_kwargs = bare_pipeline._chain.invoke.call_args[0][0]
    assert "Revenue was $142M." in call_kwargs["context"]
    assert "Founded in 2015." in call_kwargs["context"]
    assert "annual_report.txt" in call_kwargs["context"]
    assert call_kwargs["question"] == "What is the revenue?"


def test_add_documents_creates_vector_store_when_none_exists(
    bare_pipeline, isolated_data_dirs, monkeypatch
):
    bare_pipeline.vector_store = None
    bare_pipeline.embeddings = MagicMock()

    fake_new_store = MagicMock()
    fake_new_store.index.ntotal = 2
    from_documents = MagicMock(return_value=fake_new_store)
    monkeypatch.setattr(pipeline_module.FAISS, "from_documents", from_documents)

    docs = [_doc("a", "f.txt", 0), _doc("b", "f.txt", 1)]
    n = bare_pipeline.add_documents(docs)

    assert n == 2
    from_documents.assert_called_once_with(docs, bare_pipeline.embeddings)
    fake_new_store.save_local.assert_called_once_with(str(isolated_data_dirs["faiss_dir"]))
    assert bare_pipeline.vector_store is fake_new_store


def test_add_documents_appends_to_existing_vector_store(bare_pipeline, isolated_data_dirs):
    existing_store = MagicMock()
    existing_store.index.ntotal = 10
    bare_pipeline.vector_store = existing_store

    docs = [_doc("c", "f2.txt", 0)]
    n = bare_pipeline.add_documents(docs)

    assert n == 1
    existing_store.add_documents.assert_called_once_with(docs)
    existing_store.save_local.assert_called_once_with(str(isolated_data_dirs["faiss_dir"]))


def test_get_stats_returns_unique_sorted_sources(bare_pipeline):
    fake_store = MagicMock()
    fake_store.index.ntotal = 3
    fake_store.docstore._dict.values.return_value = [
        _doc("x", "beta.txt", 0),
        _doc("y", "alpha.txt", 0),
        _doc("z", "alpha.txt", 1),
    ]
    bare_pipeline.vector_store = fake_store

    stats = bare_pipeline.get_stats()

    assert stats["total_vectors"] == 3
    assert stats["indexed_documents"] == ["alpha.txt", "beta.txt"]


def test_get_stats_on_empty_pipeline(bare_pipeline):
    bare_pipeline.vector_store = None

    stats = bare_pipeline.get_stats()

    assert stats == {"total_vectors": 0, "indexed_documents": []}


def test_clear_resets_vector_store_and_removes_index_files(bare_pipeline, isolated_data_dirs):
    faiss_dir = isolated_data_dirs["faiss_dir"]
    faiss_dir.mkdir(parents=True, exist_ok=True)
    (faiss_dir / "index.faiss").write_bytes(b"fake")
    (faiss_dir / "index.pkl").write_bytes(b"fake")
    bare_pipeline.vector_store = MagicMock()

    bare_pipeline.clear()

    assert bare_pipeline.vector_store is None
    assert not (faiss_dir / "index.faiss").exists()
    assert not (faiss_dir / "index.pkl").exists()
