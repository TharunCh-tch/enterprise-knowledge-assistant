# Enterprise Knowledge Assistant

A production-ready Retrieval-Augmented Generation (RAG) system that lets you upload documents and query them with natural language. Built with **FastAPI**, **LangChain**, **HuggingFace Transformers**, and **FAISS**.

---

## Features

| Category | Details |
|---|---|
| **Ingestion** | Upload PDF, TXT, and Markdown files via drag-and-drop or API |
| **Chunking** | `RecursiveCharacterTextSplitter` (LangChain) with configurable size & overlap |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` — fast, 384-dim, CPU-friendly |
| **Vector store** | FAISS (`IndexFlatIP`) — persisted to disk, reloaded on restart |
| **Generation** | `google/flan-t5-base` (swappable via `.env`) wrapped in a LangChain LCEL chain |
| **API** | FastAPI with auto-generated OpenAPI docs at `/docs` |
| **Frontend** | Single-page HTML app — no framework, no build step |
| **Observability** | `/health` endpoint, structured logging |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser (HTML/JS)                        │
│   Upload ──► POST /api/documents/upload                          │
│   Query  ──► POST /api/query                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP
┌────────────────────────────▼────────────────────────────────────┐
│                        FastAPI  (app/main.py)                    │
│                                                                  │
│  /api/documents/upload                /api/query                 │
│         │                                    │                   │
│  DocumentProcessor                    RAGPipeline.query()        │
│  (pdfplumber + LangChain splitter)    │                          │
│         │                    ┌────────┴──────────────┐           │
│         ▼                    │                       │           │
│  RAGPipeline.add_documents() │  FAISS similarity     │  LCEL     │
│         │                    │  search (top-k)       │  chain    │
│         ▼                    │       │               │    │      │
│  HuggingFaceEmbeddings        │       ▼               │    ▼     │
│  (sentence-transformers)     │  Retrieved chunks     │  LLM     │
│         │                    │                       │  (flan-  │
│         ▼                    └───────────────────────┘   t5)    │
│  FAISS index (disk)                                             │
└─────────────────────────────────────────────────────────────────┘
```

### Request flow

1. **Upload** — file is saved → text extracted → split into chunks → embedded → stored in FAISS → index persisted to `data/faiss_index/`.
2. **Query** — question is embedded → top-k nearest chunks retrieved → context + question fed into LLM via prompt template → answer returned with source citations.

---

## Project Structure

```
enterprise-rag/
├── app/
│   ├── main.py                  # FastAPI app, lifespan, routing
│   ├── core/
│   │   └── config.py            # Pydantic Settings (env-configurable)
│   ├── api/
│   │   └── routes/
│   │       ├── documents.py     # Upload / stats / clear endpoints
│   │       └── query.py         # /query endpoint
│   ├── rag/
│   │   ├── document_processor.py  # Text extraction + chunking
│   │   └── pipeline.py            # RAGPipeline singleton (embed, store, generate)
│   └── models/
│       └── schemas.py           # Pydantic request/response models
├── frontend/
│   └── index.html               # Self-contained SPA (no build step)
├── data/
│   ├── uploads/                 # Saved uploaded files
│   └── faiss_index/             # Persisted FAISS index + metadata
├── .env.example                 # All configurable knobs
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- 2 GB RAM minimum (4 GB recommended)
- Internet access on first run (models are downloaded from HuggingFace Hub)

### 1. Clone & install

```bash
git clone <repo-url>
cd enterprise-rag

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure (optional)

```bash
cp .env.example .env
# Edit .env to change models, chunk size, etc.
```

### 3. Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

> **First run note:** HuggingFace will download the embedding model (~90 MB) and the LLM (~300 MB for `flan-t5-base`). This happens once and is cached in `~/.cache/huggingface/`.

Open your browser at **http://localhost:8000** for the UI, or **http://localhost:8000/docs** for the interactive API docs.

---

## API Reference

### `POST /api/documents/upload`

Upload a document and index it.

```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@my-report.pdf"
```

Response:
```json
{
  "filename": "my-report.pdf",
  "chunks_indexed": 42,
  "message": "'my-report.pdf' indexed successfully (42 chunks)."
}
```

---

### `POST /api/query`

Query the knowledge base.

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "top_k": 3}'
```

Response:
```json
{
  "question": "What are the main findings?",
  "answer": "The main findings include...",
  "sources": [
    {
      "content": "...chunk text...",
      "source": "my-report.pdf",
      "chunk_index": 7,
      "score": 0.8231
    }
  ]
}
```

---

### `GET /api/documents/stats`

Returns the number of indexed vectors and source document names.

### `DELETE /api/documents/clear`

Clears all vectors from the knowledge base.

### `GET /health`

Returns app status and knowledge base statistics.

---

## Configuration

All options are set via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `LLM_MODEL` | `google/flan-t5-base` | HuggingFace seq2seq LLM |
| `LLM_MAX_NEW_TOKENS` | `256` | Max tokens generated per answer |
| `CHUNK_SIZE` | `400` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |
| `TOP_K` | `3` | Number of chunks retrieved per query |
| `MAX_FILE_SIZE_MB` | `50` | Upload size limit |

### Upgrading the LLM

| Model | RAM | Quality |
|---|---|---|
| `google/flan-t5-base` | ~1 GB | Good for demos |
| `google/flan-t5-large` | ~2 GB | Better reasoning |
| `google/flan-t5-xl` | ~6 GB | Near-GPT-3 quality |

Set `LLM_MODEL=google/flan-t5-large` in `.env` and restart.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | [FastAPI](https://fastapi.tiangolo.com/) |
| Orchestration | [LangChain](https://www.langchain.com/) |
| Embeddings | [sentence-transformers](https://www.sbert.net/) via HuggingFace |
| LLM | [HuggingFace Transformers](https://huggingface.co/docs/transformers) |
| Vector store | [FAISS](https://faiss.ai/) (`faiss-cpu`) |
| PDF parsing | [pdfplumber](https://github.com/jsvine/pdfplumber) |
| Settings | [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |
| Frontend | Vanilla HTML / CSS / JS |
