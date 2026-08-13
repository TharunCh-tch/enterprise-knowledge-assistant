"""Real end-to-end test of the embedding + retrieval pipeline.

Unlike test_pipeline.py (which mocks embeddings/FAISS for speed), this test
loads the actual HuggingFace embedding model configured in settings, chunks a
real fixture document with DocumentProcessor, indexes the chunks in a real
FAISS store, and asserts that semantic similarity search actually retrieves
the chunk that answers the question. No LLM generation is exercised here
(that stays mocked elsewhere) to keep this test fast — it's the
embedding + retrieval path that's under real test.

Marked `integration` since it downloads/loads a real model on first run.
"""
import pytest
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.rag.document_processor import DocumentProcessor


@pytest.mark.integration
def test_real_embedding_and_retrieval_finds_relevant_chunk(fixture_doc_path):
    # 1. Real chunking of the fixture document.
    docs = DocumentProcessor().process(fixture_doc_path)
    assert len(docs) > 1, "fixture document should split into multiple chunks"

    # 2. Real embedding model (small, CPU-friendly, matches production config).
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 3. Real FAISS index built from the real embeddings.
    vector_store = FAISS.from_documents(docs, embeddings)
    assert vector_store.index.ntotal == len(docs)

    # 4. Real similarity search — question is specific to one chunk's content.
    results = vector_store.similarity_search("Who is the Chief Technology Officer?", k=1)

    assert len(results) == 1
    assert "Okonkwo" in results[0].page_content


@pytest.mark.integration
def test_real_retrieval_distinguishes_between_topics(fixture_doc_path):
    docs = DocumentProcessor().process(fixture_doc_path)
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = FAISS.from_documents(docs, embeddings)

    financial_hit = vector_store.similarity_search("annual revenue million dollars", k=1)[0]
    products_hit = vector_store.similarity_search(
        "Tell me about the machine learning platform product", k=1
    )[0]

    assert "revenue" in financial_hit.page_content.lower()
    assert "cloudsync" in products_hit.page_content.lower()
