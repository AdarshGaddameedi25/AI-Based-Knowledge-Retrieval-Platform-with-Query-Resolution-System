# AI-Based Knowledge Retrieval Platform

Production RAG system: PostgreSQL + pgvector + OpenRouter + sentence-transformers.

## Architecture

```
Document Upload
      ↓
Validate → Extract → Clean → Chunk (tiktoken, 500 tokens)
      ↓
Sentence-Transformers Embedding (all-MiniLM-L6-v2, 384-dim, local)
      ↓
PostgreSQL + pgvector (cosine similarity, IVFFlat index)
      ↓
User Query → Query Embedding → pgvector Search → Top-K Chunks
      ↓
RAG Context Builder (source delimiters + page numbers)
      ↓
OpenRouter LLM (meta-llama/llama-3.1-8b-instruct)
      ↓
Grounded Answer + Citations + Similarity Scores
```

## Prerequisites

- Python 3.10+
- PostgreSQL 18
- pgvector extension (see Installation)
- OpenRouter API key

## Installation

### 1. Clone & set up Python environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install pgvector on PostgreSQL 18 (Windows)

pgvector must be compiled from source for PostgreSQL 18:

```bash
# Install VS Build Tools first (or use winget)
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

# Then build pgvector
python scripts/build_pgvector.py
```

### 3. Configure environment

```bash
copy .env.example .env
# Edit .env with your credentials
```

`.env` required values:
```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/ai_knowledge_retrieval
OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=meta-llama/llama-3.1-8b-instruct
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 4. Create database

```bash
python scripts/create_db.py
```

### 5. Run migrations

```bash
alembic upgrade head
```

This creates all tables and enables the pgvector extension.

### 6. Start the backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Open the frontend

Navigate to: http://localhost:8000

Or open `frontend/index.html` directly in a browser and set the API base URL.

## API Reference

### GET /api/health
Returns database connectivity, pgvector status, and document/chunk counts.

### GET /api/documents
Returns list of all indexed documents with status and chunk counts.

### POST /api/documents/upload
Upload and index a document (PDF, DOCX, TXT, CSV).

```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@fmla_employee_guide.pdf"
```

Response:
```json
{
  "document_id": "...",
  "file_name": "fmla_employee_guide.pdf",
  "chunks_created": 47,
  "status": "indexed"
}
```

### POST /api/query
Query the knowledge base.

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FMLA?", "top_k": 5}'
```

Response:
```json
{
  "type": "rag_response",
  "query": "What is FMLA?",
  "answer": "FMLA (Family and Medical Leave Act) ...",
  "sources": [
    {
      "rank": 1,
      "source_file": "fmla_employee_guide.pdf",
      "page_number": 3,
      "similarity_score": 0.8923,
      "chunk_id": "..."
    }
  ],
  "retrieval_count": 5
}
```

### GET /docs
Swagger UI (interactive API documentation).

## End-to-End Demo

1. Start PostgreSQL
2. `alembic upgrade head`
3. `uvicorn backend.main:app --reload`
4. Open http://localhost:8000
5. Upload `fmla_employee_guide.pdf` via the Upload panel
6. Upload `technology_knowledge_base.csv`
7. Ask: "What is FMLA?" → Grounded answer + source page citation
8. Ask: "What is the employee maternity leave policy of Infosys?" → "I could not find an answer..."

## Running Tests

```bash
# Unit tests (no database required)
pytest tests/test_ingestion.py tests/test_rag.py tests/test_retrieval.py -v

# Integration tests (requires PostgreSQL + pgvector)
pytest tests/test_database.py -v

# All tests
pytest tests/ -v
```

## Supported File Types

| Type | Handler | Notes |
|------|---------|-------|
| PDF | pypdf | Page numbers preserved |
| DOCX | python-docx | Paragraphs extracted |
| TXT | built-in | UTF-8 with error recovery |
| CSV | pandas | Each row → searchable text |

## Database Schema

| Table | Purpose |
|-------|---------|
| documents | Document metadata + file hash for dedup |
| document_chunks | Text chunks with page numbers |
| embeddings | 384-dim vectors (pgvector) |
| queries | Query log |
| retrieval_results | Per-query retrieval log |
| responses | LLM response log |

## Limitations

- pgvector IVFFlat index requires >100 rows for optimal performance
- Embedding model is local (all-MiniLM-L6-v2, 384-dim)
- LLM via OpenRouter (rate limits apply)
- DOCX table extraction not yet implemented
- No authentication/authorization (development mode)
