import logging
from pathlib import Path
from typing import List

import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def process(self, file_path: Path) -> List[Document]:
        """Extract text from a file and return a list of LangChain Documents."""
        raw_text = self._extract_text(file_path)
        if not raw_text.strip():
            raise ValueError(f"No extractable text found in '{file_path.name}'")

        chunks = self.splitter.split_text(raw_text)
        docs = [
            Document(
                page_content=chunk,
                metadata={"source": file_path.name, "chunk_index": i},
            )
            for i, chunk in enumerate(chunks)
        ]
        logger.info("'%s' → %d chunks", file_path.name, len(docs))
        return docs

    # ── private helpers ──────────────────────────────────────────────────────

    def _extract_text(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(file_path)
        return file_path.read_text(encoding="utf-8", errors="replace")

    def _extract_pdf(self, file_path: Path) -> str:
        pages: List[str] = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)
