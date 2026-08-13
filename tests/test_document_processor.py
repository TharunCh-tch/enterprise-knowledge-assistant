"""Unit tests for app.rag.document_processor.DocumentProcessor.

These tests are fast and deterministic — no embedding model or FAISS involved,
just text extraction + chunking.
"""
import pytest

from app.core.config import settings
from app.rag.document_processor import DocumentProcessor


@pytest.fixture
def processor() -> DocumentProcessor:
    return DocumentProcessor()


def test_process_txt_returns_documents_with_metadata(processor, fixture_doc_path):
    docs = processor.process(fixture_doc_path)

    assert len(docs) > 0
    for i, doc in enumerate(docs):
        assert doc.metadata["source"] == fixture_doc_path.name
        assert doc.metadata["chunk_index"] == i
        assert doc.page_content.strip() != ""


def test_process_respects_configured_chunk_size(processor, fixture_doc_path):
    docs = processor.process(fixture_doc_path)

    # RecursiveCharacterTextSplitter may slightly exceed chunk_size when a
    # single "atomic" separator unit is longer than chunk_size, but chunks
    # should stay in the same ballpark as CHUNK_SIZE, not e.g. the whole file.
    for doc in docs:
        assert len(doc.page_content) <= settings.CHUNK_SIZE * 2


def test_process_chunks_cover_the_source_content(processor, fixture_doc_path):
    docs = processor.process(fixture_doc_path)
    combined = " ".join(doc.page_content for doc in docs)

    # Spot-check a few facts from the fixture document survive chunking.
    assert "TechCorp" in combined
    assert "Sarah Mitchell" in combined
    assert "AutoML Studio" in combined


def test_process_md_file_uses_plain_text_extraction(processor, tmp_path):
    md_file = tmp_path / "notes.md"
    md_file.write_text("# Title\n\nSome **markdown** content for chunking.", encoding="utf-8")

    docs = processor.process(md_file)

    assert len(docs) == 1
    assert "markdown" in docs[0].page_content


def test_process_empty_file_raises_value_error(processor, tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("   \n\n  ", encoding="utf-8")

    with pytest.raises(ValueError, match="No extractable text"):
        processor.process(empty_file)


class _FakePage:
    """Minimal stand-in for a pdfplumber Page with just extract_text()."""

    def __init__(self, text: str):
        self._text = text

    def extract_text(self):
        return self._text


def test_process_pdf_delegates_to_pdfplumber(processor, tmp_path, monkeypatch):
    """Verify the .pdf branch calls the pdfplumber extraction path, without
    depending on a real PDF file."""
    import app.rag.document_processor as dp_module

    class FakePdf:
        pages = [_FakePage("Extracted PDF text about quarterly earnings.")]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(dp_module.pdfplumber, "open", lambda _path: FakePdf())

    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake bytes")

    docs = processor.process(pdf_path)

    assert len(docs) >= 1
    assert "quarterly earnings" in docs[0].page_content
    assert docs[0].metadata["source"] == "report.pdf"
