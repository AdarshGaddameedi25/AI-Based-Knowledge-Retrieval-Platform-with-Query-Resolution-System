# M1.1 — Research and Technical Understanding

## 1. Retrieval-Augmented Generation (RAG)

### What is RAG?

Retrieval-Augmented Generation is a pattern for AI question-answering that combines two stages: (1) retrieving relevant information from a knowledge base, and (2) generating a grounded answer using a Large Language Model conditioned on that information.

### Why RAG is Required

Large Language Models are trained on general web data up to a fixed knowledge cutoff date. They cannot access private organisational documents, real-time information, or specialised domain knowledge. When asked questions about internal policies, technical documentation, or proprietary data, a standalone LLM either refuses to answer or generates plausible-sounding but incorrect information (a phenomenon known as hallucination).

RAG solves this by providing the LLM with relevant text retrieved from a controlled, up-to-date document store before generating its answer.

### Traditional LLM vs RAG

| Dimension | Traditional LLM | RAG-Augmented LLM |
|---|---|---|
| Knowledge Source | Training data only | Private knowledge base + training data |
| Knowledge Cutoff | Fixed training date | Updated dynamically as documents are added |
| Hallucination Risk | High for domain-specific queries | Significantly reduced — grounded in retrieved text |
| Citability | Cannot cite sources | Cites specific document chunks |
| Private Data | No access | Full access to ingested documents |
| Cost of Updates | Full model retraining | Add new documents to vector store |

### End-to-End RAG Pipeline

```
Document Files (PDF / DOCX / TXT / CSV)
          ↓
    Text Extraction
          ↓
  Cleaning / Normalisation
          ↓
  Chunking (500 tokens / 50 overlap)
          ↓
    Embedding Generation
          ↓
     Vector Database
          ↓
     (User Query)
          ↓
  Query Embedding
          ↓
  Similarity Search (Top-K)
          ↓
   Relevant Chunks Retrieved
          ↓
  Context + Query → LLM Prompt
          ↓
   Grounded Answer + Citations
```

---

## 2. Embeddings

### What are Embeddings?

An embedding is a dense numerical vector — typically 1,536 or 3,072 floating-point numbers — that represents a piece of text in a high-dimensional space. Each dimension captures an aspect of meaning derived from the model's training.

### Why Text is Converted into Vectors

Computers cannot directly compare "how similar are these two sentences?" in text form. When text is converted to vectors, similarity becomes a mathematical operation: the cosine angle between two vectors. Sentences with similar meaning produce vectors pointing in nearly the same direction regardless of whether they share the exact same words.

### Semantic Similarity

Semantic similarity measures how close two pieces of text are in meaning, not just in wording. For example:

- "How do I request time off?" and "What is the procedure for applying for annual leave?" are semantically similar.
- Their embedding vectors will have a high cosine similarity even though they share almost no words.

This makes vector search far superior to keyword matching for a knowledge retrieval system.

---

## 3. Chunking

### Why Documents are Chunked

LLMs have a context window limit (typically 4,096 to 128,000 tokens depending on the model). A full document — especially a 50-page PDF — cannot be sent to the LLM at once. More importantly, embedding an entire document as a single vector makes retrieval imprecise: the embedding averages out all topics in the document, making it harder to retrieve the specific relevant section.

Chunking divides documents into smaller, focused text segments that can each receive their own precise embedding.

### Chunk Size

Chunk size is measured in tokens (not characters or words). One token is approximately 0.75 words in English. A chunk of 500 tokens is approximately 375 words — enough to contain a meaningful passage without losing context.

### Chunk Overlap

Overlap means that the last N tokens of one chunk are included at the beginning of the next chunk. This prevents a sentence or idea that falls at a boundary from being split in a way that loses meaning. An overlap of 50 tokens ensures continuity across adjacent chunks.

### Milestone 1 Strategy

| Parameter | Value |
|---|---|
| Chunk Size | 500 tokens |
| Chunk Overlap | 50 tokens |
| Tokeniser | cl100k_base (tiktoken) |
| Chunking Method | Fixed-size sliding window |

### Limitations of Fixed-Size Chunking

Fixed-size chunking can split a sentence mid-way when the token count threshold is reached. It does not respect semantic boundaries such as paragraph breaks, section headers, or topic changes. Later milestones will evaluate sentence-aware and semantic chunking strategies to improve retrieval quality.

---

## 4. Vector Search

### What a Vector Database Does

A vector database stores document chunk embeddings alongside their metadata. It provides specialised indexing structures (such as HNSW — Hierarchical Navigable Small World graphs) that allow fast approximate nearest-neighbour search over millions of vectors.

### Semantic Search

Semantic search finds results that match the meaning of a query rather than its exact words. The query is converted to an embedding and the database finds stored vectors that are geometrically closest.

### Similarity Search

The similarity metric used in this project is cosine similarity, which measures the cosine of the angle between two vectors:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

A score of 1.0 means the vectors are identical in direction (maximum similarity). A score of 0.0 means they are orthogonal (unrelated). Scores below 0 indicate opposing meaning.

### Top-K Retrieval

Rather than returning all matching chunks, the retriever returns the K most similar chunks ranked by similarity score. The default in this project is Top-5 (configurable via `TOP_K_RESULTS` in the environment). These Top-K chunks form the context provided to the LLM.

---

## 5. Multi-Agent Query Resolution

A multi-agent architecture uses specialised agents, each with a defined role, coordinated by an orchestrator.

### Query Understanding Agent

Analyses the incoming user query to identify intent (factual, procedural, comparative, definitional), extract key terms, normalise phrasing, and determine whether retrieval is required.

### Retrieval Agent

Converts the processed query into an embedding and queries the vector database. Returns ranked Top-K chunks with similarity scores.

### Response Generation Agent

Takes the user query and the retrieved chunks as context. Constructs a prompt and calls the LLM to generate a grounded, cited answer.

### Clarification Agent

Detects queries that are ambiguous or rely on unresolved references (pronouns such as "it", "that", or very short queries). Generates a clarifying follow-up question instead of attempting retrieval.

### Conversation Memory Agent

Maintains a rolling window of conversation history. Resolves pronoun references using prior context. Provides the conversation history as part of the LLM prompt when multi-turn dialogue is required.

### Multi-Agent Orchestrator

The orchestrator receives the incoming query and routes it through the appropriate sequence of agents:

```
User Query
    ↓
Conversation Memory Agent (resolve references)
    ↓
Clarification Agent (is query clear enough?)
    ↓  (if clear)
Query Understanding Agent (intent, key terms)
    ↓
Retrieval Agent (Top-K chunks)
    ↓
Response Generation Agent (LLM answer + citations)
    ↓
Conversation Memory Agent (store exchange)
    ↓
Response to User
```

The orchestrator in Milestone 1 is implemented in the API query route. Full standalone orchestration will be developed in a later milestone.

---

## 6. Web Speech API

### Speech-to-Text

The Web Speech API's `SpeechRecognition` interface allows a browser to capture microphone audio and convert it to text. This enables users to speak their queries rather than type them.

### Text-to-Speech

The `SpeechSynthesis` interface converts text responses back into audio. This enables voice output of AI-generated answers.

### Future Integration

In a later milestone, the web frontend will use the Web Speech API to:
1. Accept voice queries via `SpeechRecognition`.
2. Convert voice to text and send to the backend `/api/v1/query` endpoint.
3. Receive the text answer and use `SpeechSynthesis` to read it aloud.

This creates a fully voice-capable knowledge assistant without requiring a dedicated speech processing backend.

---

## 7. Technology Decisions

| Layer | Selected Technology | Reason |
|---|---|---|
| Web Framework | FastAPI | Async-capable, Pydantic integration, auto-generated OpenAPI docs |
| Embedding Model | OpenAI text-embedding-3-small | 1,536-dimension, high semantic quality, cost-effective |
| LLM | OpenAI GPT-4o-mini | Strong reasoning, JSON mode support, cost-efficient |
| Vector Database | ChromaDB | Open source, persistent local storage, cosine similarity, simple Python API |
| PDF Extraction | pypdf | Pure Python, no external binary dependencies |
| DOCX Extraction | python-docx | Official python-docx library, full paragraph extraction |
| CSV Processing | pandas | Industry standard, handles encoding issues and various CSV dialects |
| Token Counting | tiktoken | OpenAI's official tokeniser, cl100k_base encoding |
| Settings Management | pydantic-settings | Type-safe environment variable loading, .env support |
| Testing | pytest | Industry standard, fixture support, parametrised tests |

Local embedding model alternative (sentence-transformers) is included in requirements.txt for offline/no-API-key scenarios and will be integrated in a later milestone.
