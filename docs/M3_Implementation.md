# Milestone 3 — Technical Implementation Documentation

## Overview

Milestone 3 builds on the complete M2 multi-agent pipeline to add four capabilities:

| Milestone | Feature | Status |
|---|---|---|
| M3.1 | Clarification Agent — two-turn clarification cycle with state | ✅ Complete |
| M3.2 | Conversation Memory — topic tracking, entity resolution, context switching | ✅ Complete |
| M3.3 | Voice Input (SpeechRecognition) + Text-to-Speech (SpeechSynthesis) | ✅ Complete |
| M3.4 | Response Transparency Panel — chunk text, citations, confidence, expandable | ✅ Complete |
| Bug | Similarity threshold fix — was hardcoded 0.0, now uses settings value | ✅ Fixed |

---

## M3.1 — Clarification Agent

**File:** `agents/clarification_agent.py`

### Architecture

M3.1 implements a **two-turn clarification lifecycle**:

```
Turn 1 (ambiguous query):
  evaluate(query) → ClarificationResult(is_ambiguous=True, question=...)
  Orchestrator stores ClarificationState in memory.set_clarification_pending()
  API returns type="clarification" + clarification question

Turn 2 (clarification response received):
  Orchestrator detects memory.has_pending_clarification()
  memory.consume_clarification() → pending ClarificationState
  refine_query(original_query, clarification_response) → refined_query
  Pipeline continues with refined_query as if it were a fresh query
  API returns type="rag_response" + refined_query field
```

### New Classes

```python
@dataclass
class ClarificationState:
    required: bool
    original_query: str
    question: str
    reason: str
    pending: bool
    clarification_response: str = ""
    refined_query: str = ""

@dataclass
class ClarificationResult:
    is_ambiguous: bool
    clarification_question: str
    original_query: str
    reason: str = ""
    state: Optional[ClarificationState] = None
```

### Ambiguity Detection (improved from M2)

| Rule | Trigger | Example |
|---|---|---|
| Open-ended follow-up | Matches follow-up patterns | "Tell me more", "Go on" |
| Pure pronoun query | ≤3 words + pronoun + <2 key terms | "What does it mean?" |
| Pronoun with no context | Pronoun present + <3 key terms + no history | "How does it work?" |
| Very short no-domain | ≤2 words + no domain signal + no history | "Benefits?" |

### Query Refinement

`refine_query(original, response)` uses pattern matching to construct natural English:

| Original | Clarification | Refined |
|---|---|---|
| "What are the requirements?" | "FMLA eligibility" | "What are the FMLA eligibility requirements?" |
| "How does it work?" | "the leave approval process" | "How does the leave approval process work?" |
| "Tell me more" | "FMLA medical certification" | "Tell me more about FMLA medical certification." |

### Targeted Question Generation

`generate_targeted_question(query, key_terms)` uses key terms to produce specific questions:

- "What are the requirements?" + terms=["FMLA"] → "Could you clarify which requirements you mean? For example: FMLA eligibility requirements, or FMLA documentation requirements?"
- "What does it mean?" + terms=["policy"] → "Could you clarify what 'it' refers to? Are you asking about policy?"

---

## M3.2 — Conversation Memory Agent

**File:** `agents/conversation_memory_agent.py`

### New State Fields

```python
active_topic: Optional[str]       # Primary topic of current conversation
last_entities: List[str]          # Named entities from last assistant response
last_source_docs: List[str]       # Document names cited last turn
active_key_terms: List[str]       # Key terms from last query
_pending_clarification: Optional[ClarificationState]  # M3.1 state
```

### Improved Reference Resolution

**Before (M2):** Replaced ALL pronouns with first capitalized noun in last reply.

**After (M3):**
1. No pronouns → return as-is
2. `last_entities` present → substitute with best entity
3. `active_topic` set → substitute with topic name
4. Fallback: first capitalized noun from last assistant reply
5. No context → return original (let ClarificationAgent handle)

### Topic Tracking

```python
memory.update_topic(
    query_key_terms=["fmla", "leave", "eligibility"],
    source_docs=["fmla_guide.pdf"],
    assistant_response="FMLA provides up to 12 weeks of leave...",
)

# Next query: "What are its advantages?"
memory.is_topic_continuation(["fmla", "advantages"]) → True  (overlap: "fmla")
memory.is_topic_continuation(["postgresql", "database"]) → False  (no overlap)

# Topic switch detected → stale context NOT sent to LLM
context = memory.get_relevant_context(["postgresql"]) → []
```

### Context Selection

| Scenario | What is sent to LLM |
|---|---|
| Topic continuation | Last 4 messages (2 turns) |
| Topic switch (new topic) | Empty list — start fresh |
| First query (no history) | Empty list |

---

## M3.3 — Voice I/O

**File:** `frontend/index.html` (JavaScript)

### Voice Input (SpeechRecognition)

**Two microphone entry points:**

1. **Workspace mic** (`#workspace-mic-btn`) — inline in the main query area. Shows live transcript as user speaks. Populates textarea in real-time. Submits query on `isFinal`.
2. **Section mic** (`#mic-toggle-btn`) — in the dedicated `#voice` section (existing M2 location, improved error handling).

**Error handling (user-friendly messages):**

| Browser Error Code | User Message |
|---|---|
| `not-allowed` | "Microphone access was denied..." |
| `no-speech` | "No speech detected. Please try again." |
| `network` | "Speech recognition requires an internet connection." |
| `audio-capture` | "No microphone found." |
| `aborted` | "Voice recording was aborted." |

### Text-to-Speech (SpeechSynthesis)

**Controls:**

| Button | Function | When visible |
|---|---|---|
| 🔊 Speak | `ttsSpeak()` — reads `lastAnswerText` | After every RAG response |
| ⏸ Pause | `ttsPause()` | While speaking |
| ▶ Resume | `ttsResume()` | While paused |
| ⏹ Stop | `ttsStop()` | While speaking |
| Voice select | English voices from `speechSynthesis.getVoices()` | Always |

**Lifecycle:**
- TTS controls are hidden by default
- Appear automatically after every grounded RAG answer
- Hidden/stopped when a new query is submitted
- Gracefully hidden if browser does not support `window.speechSynthesis`
- Chrome voices loaded asynchronously via `voiceschanged` event

---

## M3.4 — Response Transparency Panel

**Files:** `agents/orchestrator.py` + `frontend/index.html`

### Source Dict Schema (M3 additions marked)

```python
{
    "rank": 1,
    "citation": "[1]",             # M3.4 NEW — reference label
    "source_file": "fmla.pdf",
    "document_name": "fmla.pdf",   # M3.4 NEW — alias for UI
    "chunk_id": "c-abc123",        # M3.4 enhanced — was missing
    "chunk_index": 0,
    "page_number": 2,
    "similarity_score": 0.8765,
    "domain": "hr",
    "text": "FMLA provides...",    # M3.4 NEW — chunk text, max 600 chars
}
```

### Frontend Transparency Panel Features

| Feature | Status |
|---|---|
| Citation label [1], [2] | ✅ |
| Source document name | ✅ |
| Page number | ✅ |
| Chunk ID (truncated) | ✅ |
| Similarity score (4dp) | ✅ |
| Color-coded score bar (green/amber/red) | ✅ |
| Domain tag | ✅ |
| Evidence confidence header (High/Medium/Low) | ✅ |
| Expandable chunk text (click "Show text") | ✅ |
| Low-confidence card styling (red border when <40%) | ✅ |
| Source grouping (multi-chunk from same doc) | ✅ |
| No-results state | ✅ |
| Clarification pending state | ✅ |

---

## Bug Fix — Similarity Threshold

**File:** `backend/api/routes/query.py` line 64

**Before:** `similarity_threshold=0.0` — interpreted as valid threshold, config ignored.

**After:** `similarity_threshold=None` — retriever falls back to `settings.similarity_threshold = 0.3`.

**Impact:** Low-relevance chunks (score < 0.3) are now correctly filtered out. The hallucination guard described in the frontend now actually works.

---

## API Response Schema Changes (M3)

`POST /api/query` response additions:

```json
{
  "type": "rag_response | clarification | direct",
  "refined_query": "What are the FMLA eligibility requirements?",
  "clarification": {
    "required": true,
    "question": "Could you clarify which requirements you mean?",
    "pending": true
  },
  "memory": {
    "session_id": "abc-123",
    "used_context": true,
    "active_topic": "FMLA"
  },
  "sources": [
    {
      "rank": 1,
      "citation": "[1]",
      "source_file": "fmla.pdf",
      "document_name": "fmla.pdf",
      "chunk_id": "c-abc123",
      "chunk_index": 0,
      "page_number": 2,
      "similarity_score": 0.8765,
      "domain": "hr",
      "text": "FMLA provides up to 12 weeks of..."
    }
  ]
}
```

All M2 flat fields preserved for backward compatibility.

---

## No Database Changes

M3 uses only in-memory session state (consistent with M2). No migrations required.

---

## Test Coverage

| Suite | File | Tests |
|---|---|---|
| M2 regression | `tests/test_m2_milestone.py` | 35 tests — all must pass |
| M2 orchestrator | `tests/test_orchestrator.py` | 7 tests — all must pass |
| M3 | `tests/test_m3_milestone.py` | 51 tests (10 skipped browser) |

Run all tests:
```bash
pytest tests/ -v
```
