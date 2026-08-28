# M1.2 — System Architecture

## System Overview

The AI-Based Knowledge Retrieval Platform is a multi-agent, RAG-powered question-answering system designed to help users query private organisational knowledge bases using natural language. The platform ingests documents in multiple formats, converts them into searchable embeddings stored in a vector database, and uses an AI agent pipeline to retrieve relevant content and generate accurate, cited answers.

The system is designed to be modular and extensible, allowing each component to be independently improved or replaced as the project progresses through its milestones.

---

## Architecture Diagram

![AI-Based Knowledge Retrieval Platform — System Architecture](System_Architecture.png)

---

## Major Components

### 1. User Interface / Access Layer

The entry point for all users. In Milestone 1 this is the REST API. In future milestones a web chat interface with voice input and output will be added.

- Web Interface (Chat / Search / Upload)
- Voice Input via Web Speech API (Speech-to-Text)
- Voice Output via Web Speech API (Text-to-Speech)

### 2. Backend / API Layer

A FastAPI application exposing:
- `POST /api/v1/ingest` — Document upload and indexing
- `POST /api/v1/query` — Natural language query processing
- `GET /api/v1/health` — Service health check

The API handles authentication (planned), rate limiting (planned), and logging.

### 3. Knowledge Base Management

Manages document upload, listing, update, and deletion. In Milestone 1 upload and indexing are implemented via the ingest endpoint.

### 4. Knowledge Base Ingestion Pipeline

Processes uploaded documents through a sequential pipeline:

```
PDF / DOCX / TXT / CSV
        ↓
Text Extraction
        ↓
Cleaning / Normalization
        ↓
Chunking (500 tokens, 50 overlap)
        ↓
Embedding Generation (OpenAI text-embedding-3-small)
        ↓
Vector Database (ChromaDB)
```

Each stage is implemented in a separate Python module under `ingestion/`.

### 5. AI Agent Layer

Five specialised agents process each query:

| Agent | Role |
|---|---|
| Query Understanding Agent | Intent detection, normalisation, key term extraction |
| Retrieval Agent | Embeds query, searches vector store, returns Top-K chunks |
| Response Generation Agent | Builds LLM prompt, generates answer, prepares citations |
| Clarification Agent | Detects ambiguous queries, generates clarification questions |
| Conversation Memory Agent | Maintains history, resolves pronoun references |

### 6. Multi-Agent Orchestrator

Routes each query through the correct sequence of agents. In Milestone 1 the orchestration logic is embedded in the API query route. A standalone orchestrator class will be developed in Milestone 2.

### 7. Vector Database (ChromaDB)

Stores chunk embeddings with cosine similarity indexing. Supports Top-K similarity search and metadata filtering. The store is persisted on disk at `./data/vector_store`.

### 8. RAG Retrieval Pipeline

```
User Query → Query Embedding → Vector Search → Relevant Chunks → Context Construction → LLM → Answer + Citations
```

### 9. LLM / Foundation Model Layer

In Milestone 1: OpenAI GPT-4o-mini. The LLM layer is abstracted so it can be replaced with Llama, Mistral, or other models in later milestones.

### 10. Response and Citation Generation

The response includes the generated answer and a list of source references (file name, chunk ID, similarity score, rank).

### 11. Confidence and Transparency Module

Planned for a later milestone. Will include confidence scoring, source relevance explanation, and answer explainability.

### 12. Metadata / Document Store

ChromaDB stores chunk metadata including document_id, chunk_index, source_file, and embedding model. A relational database for full document metadata management is planned for a later milestone.

### 13. Analytics / Knowledge Gap Detection

Planned for a later milestone. Will include query logs, trend analysis, and detection of topics not covered by the current knowledge base.

### 14. Security / Monitoring / Deployment

Authentication, rate limiting, monitoring, and Docker deployment are planned for a later milestone.

---

## Knowledge Ingestion Flow

```
PDF / DOCX / TXT / CSV
        ↓
    Text Extraction
    (pypdf / python-docx / pandas / open())
        ↓
    Cleaning / Normalization
    (unicode, whitespace, special chars)
        ↓
    Chunking
    (500 tokens, 50 token overlap, cl100k_base)
        ↓
    Embedding Generation
    (OpenAI text-embedding-3-small → 1536-dim vector)
        ↓
    Vector Database
    (ChromaDB, cosine similarity, persistent)
```

Each chunk is stored with metadata: document_id, chunk_index, source_file, embedding model, and the original chunk text.

---

## Query Resolution Flow

```
User Query
    ↓
Conversation Memory Agent
(resolve pronouns, retrieve history)
    ↓
Clarification Agent
(is the query clear?)
    ↓  YES — proceed
Query Understanding Agent
(intent, key terms, normalise)
    ↓
Multi-Agent Orchestrator
    ↓
Retrieval Agent
(embed query → cosine search)
    ↓
Vector Search (ChromaDB)
    ↓
Relevant Chunks (Top-K)
    ↓
Response Generation Agent
(context + query → LLM prompt)
    ↓
LLM (GPT-4o-mini)
    ↓
Answer + Citations
```

### Clarification Agent Placement

The Clarification Agent runs before the Retrieval Agent. If it detects an ambiguous query, it returns a clarification question to the user and halts the pipeline. The user's clarified response re-enters at the top of the flow.

### Conversation Memory Agent Placement

The Conversation Memory Agent runs both at the start (to resolve references and provide history) and at the end (to store the exchange for future context).

---

## Component Implementation Status (Milestone 1)

| Component | Status |
|---|---|
| FastAPI backend | Implemented |
| Document ingestion API | Implemented |
| TXT extraction | Implemented |
| CSV extraction | Implemented |
| PDF extraction | Implemented (pypdf) |
| DOCX extraction | Implemented (python-docx) |
| Text cleaning | Implemented |
| Token-based chunking | Implemented |
| Embedding generation | Implemented (requires API key) |
| ChromaDB vector store | Implemented |
| Semantic retrieval | Implemented |
| RAG pipeline | Implemented |
| Query Understanding Agent | Implemented |
| Retrieval Agent | Implemented |
| Response Generation Agent | Implemented |
| Clarification Agent | Implemented |
| Conversation Memory Agent | Implemented |
| Multi-Agent Orchestrator | Basic (embedded in API route) |
| Web UI | Planned |
| Voice I/O | Planned |
| Authentication | Planned |
| Confidence scoring | Planned |
| Analytics | Planned |
| Docker deployment | Planned |
