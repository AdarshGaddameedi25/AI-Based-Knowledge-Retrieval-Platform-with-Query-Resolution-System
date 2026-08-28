# M1.4 — Retrieval Validation

## Overview

This document defines the retrieval validation framework for Milestone 1. It specifies the test domains, test queries, evaluation methodology, and the evaluation framework design. Automated retrieval scores require an active `OPENAI_API_KEY` and indexed documents; the framework below is ready to execute results as soon as the environment is configured.

---

## Test Domains and Documents

### Domain 1 — Human Resources (HR)

| Document | File | Topics Covered |
|---|---|---|
| Employee Handbook | `employee_handbook.txt` | Working hours, annual leave, sick leave, maternity/paternity, code of conduct, grievance |
| Leave Policy | `leave_policy.txt` | Leave types, entitlements, application procedure, forfeiture, encashment |

### Domain 2 — Technology / Information Security

| Document | File | Topics Covered |
|---|---|---|
| Cloud Security Guide | `cloud_security.txt` | IAM, MFA, encryption, VPC, Zero Trust, vulnerability management |
| Network Security Guide | `network_security.txt` | Firewalls, IDS/IPS, VPN, segmentation, wireless, NAC, penetration testing |

---

## Test Queries

### HR Domain — Factual

**Query 1:** How many annual leave days are available?

- Expected relevant document: `leave_policy.txt`, `employee_handbook.txt`
- Expected answer: 21 working days for full-time employees
- Query type: Factual

### HR Domain — Procedural

**Query 2:** How do I apply for leave?

- Expected relevant document: `leave_policy.txt`, `employee_handbook.txt`
- Expected answer: Step-by-step process via Employee Self-Service portal
- Query type: Procedural

### HR Domain — Unavailable Information

**Query 3:** What is the company's retirement age?

- Expected relevant document: None
- Expected behaviour: System should state the information is not available
- Query type: Unavailable — tests hallucination prevention

### Technology Domain — Comparative

**Query 4:** What is the difference between MFA and password authentication?

- Expected relevant document: `cloud_security.txt`
- Expected answer: Password relies on one factor; MFA adds a second verification layer
- Query type: Comparative

### Technology Domain — Factual

**Query 5:** What encryption standard is recommended for data at rest?

- Expected relevant document: `cloud_security.txt`
- Expected answer: AES-256
- Query type: Factual

### Technology Domain — Procedural

**Query 6:** What are the steps to handle a security incident?

- Expected relevant document: `cloud_security.txt`, `network_security.txt`
- Expected answer: Maintain incident response plan, define roles, test with tabletop exercises
- Query type: Procedural

---

## Evaluation Methodology

For each test query the evaluation records:

| Field | Description |
|---|---|
| Query | The exact query text |
| Expected Source | The document expected to contain the answer |
| Top-1 Result | The highest-ranked retrieved chunk |
| Top-3 Results | The three highest-ranked chunks |
| Top-5 Results | The five highest-ranked chunks |
| Top-1 Correct | Whether the correct source appeared at rank 1 |
| Top-3 Correct | Whether the correct source appeared in ranks 1–3 |
| Top-5 Correct | Whether the correct source appeared in ranks 1–5 |
| Similarity Scores | Cosine similarity scores for each returned chunk |

Accuracy is calculated as:

```
Top-K Accuracy = (Queries where correct source in Top-K) / (Total queries with known expected source)
```

Queries testing unavailable information (Query 3) are evaluated separately: the system must NOT hallucinate an answer. They are not included in the accuracy denominator.

---

## Evaluation Framework Implementation

The evaluation script is located at `tests/test_retrieval.py`. The `test_retrieval_validation_framework` test verifies the structure of the evaluation queries. Full automated evaluation with real similarity scores is executed by running the ingestion pipeline on the sample documents, then querying the retrieval system.

### To Run Full Evaluation

1. Set `OPENAI_API_KEY` in `.env`.
2. Run the ingestion script to index sample documents:
   ```bash
   python -c "
   from backend.services.ingestion_service import IngestionService
   svc = IngestionService()
   results = svc.ingest_directory('data/sample_documents')
   print(results)
   "
   ```
3. Run the retrieval evaluation:
   ```bash
   python -c "
   from retrieval.rag_pipeline import RAGPipeline
   pipeline = RAGPipeline()
   queries = [
       'How many annual leave days are available?',
       'How do I apply for leave?',
       'What is the company retirement age?',
       'What is the difference between MFA and password authentication?',
       'What encryption standard is recommended for data at rest?',
       'What are the steps to handle a security incident?',
   ]
   for q in queries:
       r = pipeline.run(q, top_k=5)
       print(f'Q: {q}')
       print(f'A: {r.answer[:200]}')
       print(f'Sources: {[s[\"source_file\"] for s in r.sources]}')
       print()
   "
   ```

---

## Evaluation Results

**Status: Not yet run — requires OPENAI_API_KEY to be configured.**

The evaluation results will be recorded here once the system is connected to the OpenAI API and the sample documents are indexed. The evaluation framework and test queries are defined and ready.

| Query | Type | Expected Source | Top-1 Correct | Top-3 Correct | Top-5 Correct | Top-1 Score |
|---|---|---|---|---|---|---|
| Annual leave days | Factual | leave_policy.txt | Pending | Pending | Pending | Pending |
| Apply for leave | Procedural | leave_policy.txt | Pending | Pending | Pending | Pending |
| Retirement age | Unavailable | None | N/A (hallucination test) | N/A | N/A | Pending |
| MFA vs password | Comparative | cloud_security.txt | Pending | Pending | Pending | Pending |
| Encryption standard | Factual | cloud_security.txt | Pending | Pending | Pending | Pending |
| Security incident steps | Procedural | cloud_security/network_security | Pending | Pending | Pending | Pending |

**Results will be updated in the next milestone after live API evaluation.**

---

## Hallucination Prevention Verification

The RAG pipeline is designed with an explicit system prompt that instructs the LLM:

> "If the context does not contain the answer, say: 'I could not find an answer to your question in the available knowledge base.' Do not invent information."

Query 3 ("What is the company's retirement age?") tests this behaviour. The expected output is a refusal to answer rather than a fabricated response. This will be verified during live evaluation.

---

## Notes on Evaluation Methodology

- Real similarity scores will depend on the embedding model, chunk size, and the specific text of the sample documents.
- Top-K accuracy is a standard IR (Information Retrieval) metric also known as Recall@K.
- Milestone 2 will add Mean Reciprocal Rank (MRR) and NDCG (Normalised Discounted Cumulative Gain) for a more comprehensive evaluation.
- The test queries and expected sources defined above were designed to cover factual, procedural, comparative, and unavailable-information scenarios across two distinct domains.
