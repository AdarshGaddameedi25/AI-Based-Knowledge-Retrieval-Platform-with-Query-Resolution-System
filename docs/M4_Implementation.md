# Milestone 4 Implementation Document

## System Architecture

The AI-Based Knowledge Retrieval Platform (Query Resolution System) implements a multi-agent RAG pipeline integrated with real-time query analytics and automated knowledge gap detection.

```
[User Interface / Voice Input]
             │
             ▼
[FastAPI Backend / Query Handler] ──(time.perf_counter)──► [Analytics Service]
             │                                                  │
             ▼                                                  ▼
[Agent Orchestrator]                                  [QueryAnalytics Log]
   ├── Query Understanding Agent (Intent & Domain)    [KnowledgeGaps Table]
   ├── Conversation Memory Agent (Context & Topic)
   ├── Clarification Agent (Disambiguation)
   ├── Retrieval Agent (pgvector Cosine Search)
   └── Response Generation Agent (OpenRouter Llama-3.1)
```

## Component Overview

### 1. Analytics & Telemetry Engine (`M4.1`)
- **Database Schema**: `QueryAnalytics` table records 21 metric dimensions per query turn (latency, similarity scores, retrieved chunks, intent, domain, routing path, response status).
- **Knowledge Gap Engine**: `KnowledgeGap` table automatically captures and increments frequency counts for queries returning no retrieved context or falling below application thresholds.
- **REST Endpoints**: 7 REST endpoints exposed at `/api/analytics/*` (overview, paginated queries, domain breakdown, intent breakdown, knowledge gaps, trends, common terms).

### 2. Multi-Domain Knowledge Architecture (`M4.2`)
- **Domain Coverage**: Supports **HR**, **Technology**, and **Finance** knowledge domains with automatic domain detection via keyword matching and vector filtering.
- **Finance Knowledge Base**: Dedicated sample policy document (`data/sample_documents/finance/finance_policy.txt`) covering expense reimbursement, procurement limits, and corporate card rules.

### 3. Analytics Dashboard UI (`M4.2`)
- **Frontend Dashboard**: Native responsive UI integrated into `frontend/index.html` with overview cards, query type distribution, domain breakdown, tag clouds, knowledge gap table, and query telemetry history with multi-column filtering.
