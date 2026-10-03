# Milestone 4 Validation & Audit Report

## System Verification

- **FastAPI Backend**: Version 4.0.0, registered with CORS middleware and all API routes (`ingest`, `query`, `health`, `history`, `analytics`).
- **PostgreSQL & pgvector**: PostgreSQL 18 with 384-dimensional IVFFlat cosine similarity index.
- **Embedding Model**: `all-MiniLM-L6-v2` (384-d dense vectors generated locally via Sentence-Transformers).
- **LLM Integration**: OpenRouter API (`meta-llama/llama-3.1-8b-instruct`).

## Validation Execution Summary

### Live RAG & Multi-Domain Pipeline
1. **HR Domain**: Queries regarding FMLA leave eligibility and maternity leave return grounded responses with exact page citations.
2. **Technology Domain**: Queries regarding PostgreSQL pgvector setup and FastAPI deployment return grounded evidence chunks.
3. **Finance Domain**: Queries regarding travel expense reimbursement limits and corporate card policies route correctly to the Finance domain knowledge base.
4. **Knowledge Gap Handling**: Queries requesting unindexed corporate policies (e.g., Infosys maternity leave) return a controlled, non-hallucinated response ("I could not find sufficient information...") and auto-log a record into `knowledge_gaps`.

### Database Schema Validation
- `documents` table: Preserves SHA-256 hashes for deduplication.
- `document_chunks` table: Preserves chunk text and page numbers.
- `embeddings` table: Stores 384-dimensional vector embeddings.
- `query_analytics` table: Captures latency, similarity, intent, and routing decisions.
- `knowledge_gaps` table: Maintains normalized query frequencies and gap reasons.
