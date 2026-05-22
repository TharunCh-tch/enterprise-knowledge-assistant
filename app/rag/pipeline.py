"""
RAG pipeline: singleton that owns embeddings, the FAISS vector store, and the LLM.
All heavy objects are created once and reused across requests.
"""
import logging
from typing import Dict, List, Optional, Tuple

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline as hf_pipeline

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_PROMPT = PromptTemplate.from_template(
    """You are an Enterprise Knowledge Assistant. Answer the question using ONLY the context below.
If the context lacks sufficient information, respond with:
"I don't have enough information in the knowledge base to answer this question."

Context:
{context}

Question: {question}

Answer:"""
)


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        idx = doc.metadata.get("chunk_index", "?")
        parts.append(f"[{src} | chunk {idx}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Pipeline singleton ─────────────────────────────────────────────────────────

class RAGPipeline:
    _instance: Optional["RAGPipeline"] = None

    def __new__(cls) -> "RAGPipeline":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    def __init__(self):
        if self._ready:
            return

        # Embeddings
        logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # FAISS index (loaded from disk if it exists)
        self.vector_store: Optional[FAISS] = None
        self._load_index()

        # LLM
        logger.info("Loading LLM: %s", settings.LLM_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(settings.LLM_MODEL)
        pipe = hf_pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=settings.LLM_MAX_NEW_TOKENS,
            truncation=True,
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

        # LCEL chain: prompt → llm → string
        self._chain = _PROMPT | self.llm | StrOutputParser()

        self._ready = True
        logger.info("RAG pipeline ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def add_documents(self, docs: List[Document]) -> int:
        """Embed and index a list of LangChain Documents."""
        settings.faiss_dir.mkdir(parents=True, exist_ok=True)
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vector_store.add_documents(docs)
        self.vector_store.save_local(str(settings.faiss_dir))
        logger.info("Indexed %d docs. Total vectors: %d", len(docs), self.vector_store.index.ntotal)
        return len(docs)

    def query(self, question: str, top_k: int = None) -> Dict:
        """Retrieve relevant chunks and generate an answer."""
        if self.vector_store is None or self.vector_store.index.ntotal == 0:
            return {
                "answer": "The knowledge base is empty. Please upload documents first.",
                "sources": [],
            }

        k = top_k or settings.TOP_K
        docs_scores: List[Tuple[Document, float]] = (
            self.vector_store.similarity_search_with_score(question, k=k)
        )
        retrieved_docs = [d for d, _ in docs_scores]
        scores = [float(s) for _, s in docs_scores]

        context = _format_docs(retrieved_docs)
        answer = self._chain.invoke({"context": context, "question": question})

        return {
            "answer": answer.strip(),
            "sources": list(zip(retrieved_docs, scores)),
        }

    def get_stats(self) -> Dict:
        """Return current knowledge base statistics."""
        total = self.vector_store.index.ntotal if self.vector_store else 0
        sources: List[str] = []
        if self.vector_store:
            seen: set = set()
            for doc in self.vector_store.docstore._dict.values():
                src = doc.metadata.get("source", "Unknown")
                if src not in seen:
                    seen.add(src)
                    sources.append(src)
        return {"total_vectors": total, "indexed_documents": sorted(sources)}

    def clear(self):
        """Delete all vectors and remove the persisted index."""
        self.vector_store = None
        for fname in ("index.faiss", "index.pkl"):
            p = settings.faiss_dir / fname
            if p.exists():
                p.unlink()
        logger.info("Knowledge base cleared.")

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_index(self):
        idx_file = settings.faiss_dir / "index.faiss"
        if not idx_file.exists():
            return
        try:
            self.vector_store = FAISS.load_local(
                str(settings.faiss_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("FAISS index loaded (%d vectors).", self.vector_store.index.ntotal)
        except Exception as exc:
            logger.warning("Could not load existing FAISS index (%s). Starting fresh.", exc)
            self.vector_store = None
