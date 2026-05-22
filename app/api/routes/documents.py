import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import settings
from app.models.schemas import DocumentUploadResponse, KnowledgeBaseStats
from app.rag.document_processor import DocumentProcessor
from app.rag.pipeline import RAGPipeline

router = APIRouter()
logger = logging.getLogger(__name__)

_processor = DocumentProcessor()


def _pipeline() -> RAGPipeline:
    return RAGPipeline()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or text file, chunk it, embed it, and add it to the FAISS index."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{suffix}' is not supported. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.upload_dir / file.filename
    file_size = 0
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024

    try:
        with open(dest, "wb") as buf:
            while chunk := await file.read(1024 * 256):
                file_size += len(chunk)
                if file_size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {settings.MAX_FILE_SIZE_MB} MB upload limit.",
                    )
                buf.write(chunk)

        docs = _processor.process(dest)
        n = _pipeline().add_documents(docs)

        return DocumentUploadResponse(
            filename=file.filename,
            chunks_indexed=n,
            message=f"'{file.filename}' indexed successfully ({n} chunks).",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to process '%s': %s", file.filename, exc, exc_info=True)
        if dest.exists():
            dest.unlink()
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")


@router.get("/stats", response_model=KnowledgeBaseStats)
async def knowledge_base_stats():
    """Return the number of indexed vectors and the list of source documents."""
    return KnowledgeBaseStats(**_pipeline().get_stats())


@router.delete("/clear")
async def clear_knowledge_base():
    """Remove all vectors from the knowledge base."""
    _pipeline().clear()
    return {"message": "Knowledge base cleared successfully."}
