from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: Optional[int] = Field(default=3, ge=1, le=20)


class SourceDocument(BaseModel):
    content: str
    source: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]


class KnowledgeBaseStats(BaseModel):
    total_vectors: int
    indexed_documents: List[str]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    knowledge_base: KnowledgeBaseStats
