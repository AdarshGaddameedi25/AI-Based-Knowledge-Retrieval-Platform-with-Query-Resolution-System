# Milestone 2 — Technical Implementation Documentation

## Overview

Milestone 2 implements the complete multi-agent query resolution pipeline:

```
User Query
    ↓
Query Understanding Agent   (M2.1)
    ↓
Classification + Routing
    ↓
 ┌──────────────────────────────────────┐
 │                                      │
 │ routing=direct                       │ routing=clarification
 │ (greeting/small-talk)                │ (ambiguous query)
 │                                      │
 ↓                                      ↓
Direct Response               Clarification Question
                                         │
                              routing=retrieval ──────────────┐
                                                               │
                                                               ↓
                                                    Retrieval Agent   (M2.2)
                                                               ↓
                                                    Query Embedding
                                                    (all-MiniLM-L6-v2, 384-dim)
                                                               ↓
                                                    pgvector Cosine Search
                                                               ↓
                                                    Top-K + Similarity Filter
                                                               ↓
                                                    Response Generation Agent  (M2.3)
                                                               ↓
                                                    OpenRouter LLM
                                                    (meta-llama/llama-3.1-8b-instruct)
                                                               ↓
                                                    Grounded Answer + Citations
```

---

## M2.1 — Query Understanding Agent

**File:** `agents/query_understanding_agent.py`

### Classification

The agent classifies every query into one of five canonical types:

| `query_type` | `routing` | Description |
|---|---|---|
| `factual` | `retrieval` | What/who/when/where/which questions |
| `procedural` | `retrieval` | How-to / steps / process questions |
| `comparative` | `retrieval` | Difference/compare/vs questions |
| `ambiguous` | `clarification` | Unresolved pronouns with insufficient context |
| `direct` | `direct` | Greetings, small-talk, one-word queries |

### Classification Logic

1. **Greeting detection** — exact match against `GREETINGS` and `SMALL_TALK` sets → `direct`
2. **Single-word queries** → `direct`  
3. **Ambiguity detection** — queries containing unresolved pronouns (`it`, `that`, `this`, `they`, `them`, `those`, `these`) with fewer than 3 substantive key terms → `ambiguous`
4. **Pattern matching** (priority order):
   - Comparative keywords → `comparative`
   - Procedural keywords → `procedural`
   - Factual/definitional keywords → `factual`
5. **Default fallback** → `factual` with confidence 0.60

### Output Schema (`QueryAnalysis`)

```python
@dataclass
class QueryAnalysis:
    original_query: str           # raw input
    normalized_query: str         # cleaned + question-terminated
    requires_retrieval: bool      # whether to hit vector store
    query_type: str               # canonical M2 label
    routing: str                  # retrieval | clarification | direct
    detected_intent: str          # alias for query_type (backward compat)
    key_terms: List[str]          # significant non-stop-word tokens
    suggested_domain: Optional[str]  # hr | technology | legal | None
    classification_confidence: float  # 0.0–1.0
```

### Domain Detection

Keyword scoring across three domains: `hr`, `technology`, `legal`.  
The domain with the highest keyword count wins. Used as a hint for domain-filtered retrieval.

---

## M2.2 — Retrieval Agent

**Files:** `agents/retrieval_agent.py`, `retrieval/retriever.py`, `retrieval/vector_store.py`

### Embedding

- Model: `all-MiniLM-L6-v2` (sentence-transformers, local, 384-dimensional)
- Same model used for document ingestion AND query embedding (guaranteed consistency)
- Singleton pattern (`EmbeddingService`) — model loaded once, reused for all requests
- Normalization: L2-normalized embeddings (cosine similarity = dot product)

### Vector Search

```sql
SELECT
    e.chunk_id, dc.text, dc.chunk_index, dc.page_number, dc.chunk_metadata,
    d.file_name, d.domain,
    1 - (e.embedding <=> CAST(:embedding AS vector)) AS similarity_score
FROM embeddings e
JOIN document_chunks dc ON e.chunk_id = dc.chunk_id
JOIN documents d ON dc.document_id = d.document_id
WHERE d.status = 'indexed'
[AND LOWER(d.domain) = LOWER(:domain_filter)]
ORDER BY e.embedding <=> CAST(:embedding AS vector)
LIMIT :top_k
```

- `<=>` = pgvector cosine **distance** operator
- `1 - distance` = cosine **similarity** (higher = more relevant)
- `ORDER BY distance ASC` = results ordered by descending similarity
- `similarity_threshold` applied post-query as a Python filter (default: 0.3)

### Retrieval Result Schema

```python
@dataclass
class RetrievalResult:
    chunk_id: str
    document_id: str
    source_file: str
    text: str
    similarity_score: float      # 0.0–1.0 relevance signal (NOT a probability)
    rank: int
    chunk_index: int
    page_number: Optional[int]
    domain: Optional[str]
    metadata: dict
```

**Important:** `similarity_score` is a retrieval relevance signal derived from cosine distance. It is not a probability of factual correctness.

---

## M2.3 — Response Generation Agent

**File:** `agents/response_generation_agent.py`

### LLM Integration

Single authoritative path:

```
Application → httpx → https://openrouter.ai/api/v1/chat/completions → Selected LLM
```

No OpenAI SDK. No competing providers.

### System Prompt

The system prompt:
1. Enforces context-only answers ("ONLY information found in the provided context")
2. Mandates source citation
3. Provides the exact knowledge-gap phrase to use when evidence is insufficient
4. Includes a query-type hint for response style (factual/procedural/comparative)

### Multi-Turn Context

The `generate()` method accepts `conversation_history` (prior user/assistant turns) and includes them in the messages array before the current question. This enables coherent multi-turn answers.

### Knowledge-Gap Handling

```python
if not retrieval_results:
    return GeneratedResponse(
        answer="I could not find sufficient information in the available "
               "knowledge base to answer this question.",
        citations=[],
        confidence=0.0,
    )
```

The system **never calls the LLM** when no evidence is retrieved.

### Confidence

`confidence = mean(similarity_scores)` of retrieved chunks.  
This is a **retrieval relevance signal**, not an LLM certainty measure.

---

## M2.4 — Multi-Agent Orchestration

**File:** `agents/orchestrator.py`

### Pipeline

```python
Step 1: QueryUnderstandingAgent.analyze(query)
    → QueryAnalysis(query_type, routing, ...)

Step 2: ConversationMemoryAgent.resolve_references(normalized_query)
    → Pronoun substitution from prior assistant messages

Step 3: Routing decision
    if routing == "direct" → return greeting response
    if query_type == "ambiguous" → return clarification question
    if ClarificationAgent finds ambiguity → return clarification question

Step 4: RetrievalAgent.retrieve(analysis, db, top_k, domain_filter)
    → List[RetrievalResult] (empty = knowledge gap)
    (wrapped in try/except → controlled error response on failure)

Step 5: ResponseGenerationAgent.generate(query, results, query_type, history)
    → GeneratedResponse
    (wrapped in try/except → controlled error response on failure)

Step 6: ConversationMemoryAgent.add_message(user + assistant)
    → history updated for multi-turn

Step 7: Return OrchestratorResult
```

### Error Isolation

| Error | Behavior |
|-------|---------|
| Retrieval DB failure | Controlled: "Retrieval service temporarily unavailable." |
| LLM timeout/failure | Controlled: "The answer could not be generated at this time." |
| EnvironmentError (no API key) | 503 from API route |
| Unexpected orchestrator error | 500 with generic message (no internal details leaked) |

### OrchestratorResult Schema

```python
@dataclass
class OrchestratorResult:
    query: str
    answer: str
    sources: List[dict]           # [{rank, source_file, chunk_id, page_number, similarity_score, domain}]
    query_type: str               # canonical M2 label
    routing: str                  # retrieval | clarification | direct
    detected_intent: str          # alias (backward compat)
    detected_domain: Optional[str]
    key_terms: List[str]
    clarification_needed: bool
    clarification_question: str
    confidence: float
    retrieval_count: int
    classification_confidence: float
    session_id: Optional[str]
```

---

## API Contract

### POST /api/query

**Request:**
```json
{
  "query": "Who is eligible for FMLA?",
  "top_k": 5,
  "session_id": "optional-session-uuid",
  "domain_filter": "hr"
}
```

**Response:**
```json
{
  "type": "rag_response",
  "query": "Who is eligible for FMLA?",
  "query_type": "factual",
  "routing": "retrieval",
  "detected_intent": "factual",
  "detected_domain": "hr",
  "key_terms": ["eligible", "fmla"],
  "answer": "According to the FMLA Employee Guide...",
  "sources": [
    {
      "rank": 1,
      "source_file": "fmla_employee_guide.pdf",
      "chunk_id": "...",
      "chunk_index": 3,
      "page_number": 2,
      "similarity_score": 0.8923,
      "domain": "hr"
    }
  ],
  "clarification_needed": false,
  "clarification_question": "",
  "confidence": 0.8923,
  "classification_confidence": 0.9,
  "retrieval_count": 5,
  "session_id": "..."
}
```

**Response types:**

| `type` | Condition |
|--------|-----------|
| `direct` | Greeting or small-talk |
| `clarification` | Ambiguous query, clarification question in `answer` |
| `rag_response` | Knowledge query with retrieval results |

---

## Conversation Memory

**File:** `agents/conversation_memory_agent.py`

- Per-session `ConversationMemoryAgent` stored in `InMemorySessionStore`
- Max history: configurable via `MAX_SESSION_HISTORY` (default: 10 turns)
- Pronoun resolution: searches last assistant message for capitalized nouns to substitute for `it`, `that`, `this`, etc.
- Context window: last 6 turns passed to LLM for multi-turn coherence

**Session management:**
- `POST /api/query` with no `session_id` → new session created
- `POST /api/query` with existing `session_id` → history resumed
- `POST /api/query/clear-session` → history cleared

---

## Knowledge-Gap Behavior

| Condition | System behavior |
|-----------|----------------|
| No retrieval results | LLM NOT called; "I could not find sufficient information..." returned |
| Results below similarity threshold | Filtered out; if all filtered, same as no results |
| Query matches domain but low relevance | Results pass to LLM with low confidence score |
| Query completely off-domain | Empty results → knowledge-gap response |

---

## Clarification Behavior

| Condition | Trigger | Response |
|-----------|---------|---------|
| Pronoun with few key terms | `query_type=ambiguous` (QUA) | "Could you clarify what 'it' refers to?" |
| Short query with no history | `ClarificationAgent` | "Could you provide more context?" |
| "Tell me more" open-ended | `ClarificationAgent` pattern | "Could you specify which topic?" |
| Rich context despite pronoun | QUA allows retrieval to proceed | No clarification asked |

---

## Security Notes

- API keys and credentials are in `.env` (gitignored)
- All `.env` values accessed via `config/settings.py` (pydantic-settings)
- Internal error details are NOT exposed in API error responses
- File uploads: extension whitelist (pdf/docx/txt/csv) + 50MB size limit
- SQL: parameterized queries only (no string interpolation)
- Filenames: `os.path.basename()` prevents path traversal
