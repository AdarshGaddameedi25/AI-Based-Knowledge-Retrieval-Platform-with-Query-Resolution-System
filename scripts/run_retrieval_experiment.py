"""
scripts/run_retrieval_experiment.py — Reproducible Global RAG Retrieval Optimization Experiment

Evaluates 3 distinct RAG retrieval configurations across a unified global searchable corpus
containing all committed repository sample documents (HR, Technology, Finance):
  - Configuration A: chunk_size=300, overlap=30, top_k=3
  - Configuration B: chunk_size=500, overlap=50, top_k=5 (Operational Baseline)
  - Configuration C: chunk_size=700, overlap=70, top_k=7

Key Methodology:
  - 100% Global Corpus Vector Search (NO pre-filtering by domain)
  - Evaluates Recall@1, Recall@3, Recall@5, MRR (Mean Reciprocal Rank), Hit Rate, Avg Similarity, and Latency
  - Fails loudly if required repository sample document files are missing (No silent dummy data)

Saves empirical benchmark results artifact to docs/results/m4_retrieval_experiment.json.
"""

import json
import time
from pathlib import Path
import numpy as np

from ingestion.chunking import TextChunker
from ingestion.embeddings import EmbeddingService


# Ground-Truth Evaluation Dataset (13 answerable queries across 5 committed repository files + 2 gap queries)
BENCHMARK_DATASET = [
    # HR Domain Ground Truth (leave_policy.txt & employee_handbook.txt)
    {
        "id": "HR-01",
        "query": "How many days of paid annual leave are full-time employees entitled to per calendar year?",
        "expected_domain": "hr",
        "expected_document": "leave_policy.txt",
        "expected_keywords": ["21 working days", "annual leave"],
        "query_type": "factual",
        "is_answerable": True,
    },
    {
        "id": "HR-02",
        "query": "What is the maternity leave duration and paid eligibility under the leave policy?",
        "expected_domain": "hr",
        "expected_document": "leave_policy.txt",
        "expected_keywords": ["16 consecutive weeks", "maternity leave"],
        "query_type": "factual",
        "is_answerable": True,
    },
    {
        "id": "HR-03",
        "query": "What is the outpatient medical leave entitlement for employees?",
        "expected_domain": "hr",
        "expected_document": "leave_policy.txt",
        "expected_keywords": ["10 working days", "medical leave"],
        "query_type": "factual",
        "is_answerable": True,
    },
    {
        "id": "HR-04",
        "query": "What is the maximum carry-over limit for unused annual leave to the next leave year?",
        "expected_domain": "hr",
        "expected_document": "leave_policy.txt",
        "expected_keywords": ["10 days", "carry-over"],
        "query_type": "procedural",
        "is_answerable": True,
    },

    # Technology Domain Ground Truth (cloud_security.txt & network_security.txt)
    {
        "id": "TECH-01",
        "query": "What encryption standard is required for cloud data at rest?",
        "expected_domain": "technology",
        "expected_document": "cloud_security.txt",
        "expected_keywords": ["AES-256", "encryption at rest"],
        "query_type": "factual",
        "is_answerable": True,
    },
    {
        "id": "TECH-02",
        "query": "What is the difference between password authentication and Multi-Factor Authentication (MFA)?",
        "expected_domain": "technology",
        "expected_document": "cloud_security.txt",
        "expected_keywords": ["MFA", "factor", "password"],
        "query_type": "comparative",
        "is_answerable": True,
    },
    {
        "id": "TECH-03",
        "query": "What minimum TLS protocol version is required for data encryption in transit?",
        "expected_domain": "technology",
        "expected_document": "cloud_security.txt",
        "expected_keywords": ["TLS 1.2", "transit"],
        "query_type": "factual",
        "is_answerable": True,
    },
    {
        "id": "TECH-04",
        "query": "What principle dictates that users should be granted only minimum permissions?",
        "expected_domain": "technology",
        "expected_document": "cloud_security.txt",
        "expected_keywords": ["Least Privilege", "permissions"],
        "query_type": "factual",
        "is_answerable": True,
    },

    # Finance Domain Ground Truth (finance_policy.txt)
    {
        "id": "FIN-01",
        "query": "What is the corporate travel expense reimbursement limit for hotel nights in standard cities?",
        "expected_domain": "finance",
        "expected_document": "finance_policy.txt",
        "expected_keywords": ["$250", "hotel"],
        "query_type": "factual",
        "is_answerable": True,
    },
    {
        "id": "FIN-02",
        "query": "What receipt submission requirement applies to expenses exceeding $25?",
        "expected_domain": "finance",
        "expected_document": "finance_policy.txt",
        "expected_keywords": ["receipts", "$25", "ExpenseCloud"],
        "query_type": "procedural",
        "is_answerable": True,
    },
    {
        "id": "FIN-03",
        "query": "What approval threshold applies to expenses between $2000 and $10000?",
        "expected_domain": "finance",
        "expected_document": "finance_policy.txt",
        "expected_keywords": ["$2,000", "Finance department"],
        "query_type": "factual",
        "is_answerable": True,
    },
    {
        "id": "FIN-04",
        "query": "What categories of personal expenses are ineligible for corporate reimbursement?",
        "expected_domain": "finance",
        "expected_document": "finance_policy.txt",
        "expected_keywords": ["ineligible", "fines", "entertainment"],
        "query_type": "factual",
        "is_answerable": True,
    },
    {
        "id": "FIN-05",
        "query": "What is the annual employee allowance for professional certifications and technical books?",
        "expected_domain": "finance",
        "expected_document": "finance_policy.txt",
        "expected_keywords": ["$2,000", "$500", "certifications"],
        "query_type": "factual",
        "is_answerable": True,
    },

    # Out-of-Domain / Knowledge Gap Ground Truth (Unanswerable)
    {
        "id": "GAP-01",
        "query": "What is the employee maternity leave policy of Infosys for 2026?",
        "expected_domain": "hr",
        "expected_document": None,
        "expected_keywords": [],
        "query_type": "gap",
        "is_answerable": False,
    },
    {
        "id": "GAP-02",
        "query": "What is the quantum computing algorithm hardware relocation allowance?",
        "expected_domain": "technology",
        "expected_document": None,
        "expected_keywords": [],
        "query_type": "gap",
        "is_answerable": False,
    },
]

CONFIGURATIONS = [
    {"name": "Config A (Compact)", "chunk_size": 300, "overlap": 30, "top_k": 3},
    {"name": "Config B (Operational Baseline)", "chunk_size": 500, "overlap": 50, "top_k": 5},
    {"name": "Config C (Large Context)", "chunk_size": 700, "overlap": 70, "top_k": 7},
]


def load_repository_documents():
    """
    Loads all real sample documents from repository.
    Fails loudly if any required sample document file is missing.
    """
    required_files = {
        "leave_policy.txt": "data/sample_documents/hr/leave_policy.txt",
        "employee_handbook.txt": "data/sample_documents/hr/employee_handbook.txt",
        "cloud_security.txt": "data/sample_documents/technology/cloud_security.txt",
        "network_security.txt": "data/sample_documents/technology/network_security.txt",
        "finance_policy.txt": "data/sample_documents/finance/finance_policy.txt",
    }
    
    docs = {}
    for filename, filepath in required_files.items():
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(
                f"[Benchmark Error] Required repository sample document missing: '{filepath}'. "
                f"Benchmark requires real committed sample document files to run reproducibly."
            )
        with open(p, "r", encoding="utf-8") as f:
            docs[filename] = f.read()

    return docs


def is_chunk_relevant(chunk_text: str, source_doc: str, item: dict) -> bool:
    """
    Checks relevance of a chunk against ground truth item.
    A chunk is relevant if it comes from the expected document AND contains at least 1 expected keyword.
    """
    if not item["is_answerable"] or item["expected_document"] is None:
        return False
        
    if source_doc != item["expected_document"]:
        return False

    chunk_lower = chunk_text.lower()
    return any(kw.lower() in chunk_lower for kw in item["expected_keywords"])


def run_experiment():
    print("=" * 80)
    print("RUNNING REPRODUCIBLE GLOBAL RAG RETRIEVAL OPTIMIZATION EXPERIMENT")
    print("=" * 80)

    docs = load_repository_documents()
    print(f"Loaded {len(docs)} committed repository sample documents for global corpus indexing.")

    embedder = EmbeddingService()
    results = []

    for cfg in CONFIGURATIONS:
        print(f"\nBenchmarking {cfg['name']} (chunk_size={cfg['chunk_size']}, top_k={cfg['top_k']})...")
        chunker = TextChunker(chunk_size=cfg["chunk_size"], chunk_overlap=cfg["overlap"])

        # Step 1: Chunk all repository documents into one global corpus
        global_chunks = []
        global_source_docs = []

        for doc_name, text in docs.items():
            chunks = chunker.chunk(text, document_id=f"doc-{doc_name}")
            for c in chunks:
                global_chunks.append(c)
                global_source_docs.append(doc_name)

        print(f"  Total Global Corpus Chunks: {len(global_chunks)}")

        # Step 2: Compute dense vector embeddings for entire global corpus
        chunk_texts = [c.text for c in global_chunks]
        corpus_embeddings = np.array(embedder.generate_embeddings(chunk_texts))

        # Step 3: Evaluate query retrieval against global corpus (100% NO domain pre-filtering)
        recalls_at_1 = []
        recalls_at_3 = []
        recalls_at_5 = []
        reciprocal_ranks = []
        similarity_scores = []
        query_latencies = []
        hits = 0

        valid_answerable_items = [q for q in BENCHMARK_DATASET if q["is_answerable"]]

        for q_item in BENCHMARK_DATASET:
            q_text = q_item["query"]

            t0 = time.perf_counter()
            q_emb = np.array(embedder.generate_embedding(q_text))

            # Global Cosine Similarity against all corpus chunks
            sims = np.dot(corpus_embeddings, q_emb)
            top_k_indices = np.argsort(sims)[::-1][:cfg["top_k"]]
            
            latency_ms = (time.perf_counter() - t0) * 1000.0
            query_latencies.append(latency_ms)

            best_sim = float(sims[top_k_indices[0]]) if len(top_k_indices) > 0 else 0.0
            similarity_scores.append(best_sim)

            if not q_item["is_answerable"]:
                continue

            # Evaluate relevance for answerable queries
            first_rel_rank = None
            relevant_in_top_1 = False
            relevant_in_top_3 = False
            relevant_in_top_5 = False

            for rank_idx, chunk_idx in enumerate(top_k_indices, start=1):
                chunk = global_chunks[chunk_idx]
                src_doc = global_source_docs[chunk_idx]

                if is_chunk_relevant(chunk.text, src_doc, q_item):
                    if first_rel_rank is None:
                        first_rel_rank = rank_idx
                    if rank_idx <= 1:
                        relevant_in_top_1 = True
                    if rank_idx <= 3:
                        relevant_in_top_3 = True
                    if rank_idx <= 5:
                        relevant_in_top_5 = True

            recalls_at_1.append(1 if relevant_in_top_1 else 0)
            recalls_at_3.append(1 if relevant_in_top_3 else 0)
            recalls_at_5.append(1 if relevant_in_top_5 else 0)

            if first_rel_rank is not None:
                hits += 1
                reciprocal_ranks.append(1.0 / first_rel_rank)
            else:
                reciprocal_ranks.append(0.0)

        n_valid = len(valid_answerable_items)
        recall_1_pct = float(np.mean(recalls_at_1)) * 100.0 if recalls_at_1 else 0.0
        recall_3_pct = float(np.mean(recalls_at_3)) * 100.0 if recalls_at_3 else 0.0
        recall_5_pct = float(np.mean(recalls_at_5)) * 100.0 if recalls_at_5 else 0.0
        mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
        hit_rate = (hits / n_valid) * 100.0 if n_valid > 0 else 0.0
        avg_sim = float(np.mean(similarity_scores)) if similarity_scores else 0.0
        avg_lat = float(np.mean(query_latencies))

        cfg_res = {
            "name": cfg["name"],
            "chunk_size": cfg["chunk_size"],
            "overlap": cfg["overlap"],
            "top_k": cfg["top_k"],
            "corpus_chunks": len(global_chunks),
            "recall_at_1_pct": round(recall_1_pct, 1),
            "recall_at_3_pct": round(recall_3_pct, 1),
            "recall_at_5_pct": round(recall_5_pct, 1),
            "mrr": round(mrr, 4),
            "hit_rate_pct": round(hit_rate, 1),
            "avg_similarity": round(avg_sim, 4),
            "avg_latency_ms": round(avg_lat, 2),
            "total_queries_eval": len(BENCHMARK_DATASET),
            "answerable_queries_eval": n_valid,
        }
        results.append(cfg_res)

        print(
            f"  Recall@1: {cfg_res['recall_at_1_pct']}% | "
            f"Recall@3: {cfg_res['recall_at_3_pct']}% | "
            f"Recall@5: {cfg_res['recall_at_5_pct']}% | "
            f"MRR: {cfg_res['mrr']} | "
            f"Hit Rate: {cfg_res['hit_rate_pct']}% | "
            f"Avg Latency: {cfg_res['avg_latency_ms']} ms"
        )

    # Save benchmark result artifact to docs/results/m4_retrieval_experiment.json
    out_dir = Path("docs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "m4_retrieval_experiment.json"

    out_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "search_scope": "global_corpus_unfiltered",
        "dataset_size": len(BENCHMARK_DATASET),
        "answerable_queries": len([q for q in BENCHMARK_DATASET if q["is_answerable"]]),
        "configurations": results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print("\n" + "=" * 80)
    print(f"EXPERIMENT COMPLETE. Results saved to: {out_file.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
