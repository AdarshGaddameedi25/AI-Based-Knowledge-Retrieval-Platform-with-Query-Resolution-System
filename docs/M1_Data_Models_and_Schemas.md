# M1.3 — Data Models and Schemas

## Overview

This document defines the seven primary data schemas used throughout the AI-Based Knowledge Retrieval Platform. These schemas govern how data flows between the ingestion pipeline, vector store, retrieval layer, agent layer, and API layer.

---

## Schema Relationships

```
Document
   ↓ (one-to-many)
Chunk
   ↓ (one-to-one)
Embedding
   ↓
Vector Store

Query
   ↓ (one-to-many)
Retrieval Result
   ↓ (many-to-one)
Response

ConversationMemory
   ↓ (provides context to)
Query
```

---

## 1. Document

Represents a document uploaded to the knowledge base.

| Field | Type | Description | Example |
|---|---|---|---|
| document_id | UUID (string) | Unique document identifier | `"a3f2c1d4-..."` |
| file_name | String | Original filename | `"employee_handbook.txt"` |
| file_type | String | Uppercase file extension | `"TXT"`, `"PDF"`, `"DOCX"`, `"CSV"` |
| file_size | Integer | File size in bytes | `45312` |
| upload_date | ISO 8601 DateTime | UTC timestamp of upload | `"2026-08-28T08:00:00"` |
| status | String | Processing status | `"loaded"`, `"indexed"`, `"error"` |
| text | String | Extracted full text | `"ACME Corporation Employee Handbook..."` |
| metadata | JSON / dict | Additional key-value metadata | `{"source_path": "/uploads/handbook.txt"}` |

---

## 2. Chunk

Represents a single text segment produced by the chunking process.

| Field | Type | Description | Example |
|---|---|---|---|
| chunk_id | UUID (string) | Unique chunk identifier | `"b7e1a3f2-..."` |
| document_id | UUID (string) | Parent document identifier | `"a3f2c1d4-..."` |
| chunk_index | Integer | Sequential position of chunk (0-based) | `0`, `1`, `2` |
| text | String | Raw text content of the chunk | `"Annual leave entitlement is 21 days..."` |
| token_count | Integer | Number of tokens in the chunk | `487` |
| page_number | Integer or None | Source page number (PDF only) | `3`, `None` |
| metadata | JSON / dict | Additional key-value metadata | `{"source_file": "handbook.txt"}` |

---

## 3. Embedding

Represents the vector embedding of a single chunk.

| Field | Type | Description | Example |
|---|---|---|---|
| chunk_id | UUID (string) | Chunk this embedding belongs to | `"b7e1a3f2-..."` |
| document_id | UUID (string) | Parent document | `"a3f2c1d4-..."` |
| chunk_index | Integer | Position in the document | `0` |
| source_file | String | Original filename | `"employee_handbook.txt"` |
| embedding | List[Float] | Dense vector (1536 dimensions) | `[0.021, -0.034, 0.118, ...]` |
| model | String | Embedding model name | `"text-embedding-3-small"` |
| page_number | Integer or None | Source page if available | `None` |
| metadata | JSON / dict | Additional metadata | `{"text": "Annual leave entitlement..."}` |

---

## 4. Query

Represents a user query processed by the system.

| Field | Type | Description | Example |
|---|---|---|---|
| query_id | UUID (string) | Unique query identifier | `"c9d2e5f7-..."` |
| query_text | String | Raw user input | `"How many leave days do I get?"` |
| timestamp | ISO 8601 DateTime | UTC timestamp | `"2026-08-28T09:15:00"` |
| detected_intent | String or None | Intent category | `"factual"`, `"procedural"`, `"comparative"` |
| key_terms | List[String] | Extracted key terms | `["leave", "days"]` |

---

## 5. Retrieval Result

Represents a single chunk returned by the retrieval system in response to a query.

| Field | Type | Description | Example |
|---|---|---|---|
| chunk_id | UUID (string) | Identifier of retrieved chunk | `"b7e1a3f2-..."` |
| document_id | UUID (string) | Parent document | `"a3f2c1d4-..."` |
| source_file | String | Source filename | `"leave_policy.txt"` |
| text | String | Chunk text content | `"Full-time employees are entitled to 21..."` |
| similarity_score | Float | Cosine similarity (0.0–1.0) | `0.892` |
| rank | Integer | Position in Top-K results (1-based) | `1` |
| chunk_index | Integer | Chunk position in source document | `4` |
| metadata | JSON / dict | Additional metadata from vector store | `{"model": "text-embedding-3-small"}` |

---

## 6. Response

Represents the final generated response returned to the user.

| Field | Type | Description | Example |
|---|---|---|---|
| response_id | UUID (string) | Unique response identifier | `"d4f8g1h2-..."` |
| query | String | The original user query | `"How many leave days do I get?"` |
| answer | String | LLM-generated grounded answer | `"Full-time employees are entitled to 21 working days..."` |
| citations | List[dict] | Source references used | `[{"source": "leave_policy.txt", "chunk_id": "...", "score": 0.892}]` |
| timestamp | ISO 8601 DateTime | UTC timestamp of response | `"2026-08-28T09:15:03"` |

---

## 7. Conversation Memory

Represents a stored message in the conversation history.

| Field | Type | Description | Example |
|---|---|---|---|
| role | String | Speaker role | `"user"` or `"assistant"` |
| content | String | Message text | `"How do I apply for leave?"` |
| timestamp | ISO 8601 DateTime | UTC timestamp | `"2026-08-28T09:10:00"` |

The ConversationMemoryAgent maintains a rolling window of the last N messages (default: 10). Messages beyond the window are discarded.

---

## Python Implementation Reference

All schemas are implemented as Python dataclasses in the ingestion and retrieval modules:

- `ingestion/document_loader.py` → `DocumentRecord`
- `ingestion/chunking.py` → `Chunk`
- `ingestion/embeddings.py` → `EmbeddingRecord`
- `retrieval/retriever.py` → `RetrievalResult`
- `retrieval/rag_pipeline.py` → `RAGResponse`
- `agents/conversation_memory_agent.py` → `Message`

FastAPI Pydantic models for the API layer are in `backend/models/schemas.py`.
