import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import QueryRequest, QueryResponse, SourceDocument
from app.rag.pipeline import RAGPipeline

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Retrieve the most relevant document chunks for the question,
    then generate a grounded answer using the LLM.
    """
    pipeline = RAGPipeline()

    try:
        result = pipeline.query(request.question, top_k=request.top_k)
    except Exception as exc:
        logger.error("Query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")

    sources = [
        SourceDocument(
            content=(
                doc.page_content[:500] + "…"
                if len(doc.page_content) > 500
                else doc.page_content
            ),
            source=doc.metadata.get("source", "Unknown"),
            chunk_index=doc.metadata.get("chunk_index", 0),
            score=round(score, 4),
        )
        for doc, score in result["sources"]
    ]

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        sources=sources,
    )
