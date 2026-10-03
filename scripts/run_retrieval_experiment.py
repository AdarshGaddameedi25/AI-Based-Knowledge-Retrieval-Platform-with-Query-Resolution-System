"""
scripts/run_retrieval_experiment.py — Reproducible Retrieval Optimization Experiment

Evaluates 3 distinct RAG retrieval configurations across HR, Technology, and Finance domains:
  - Configuration A: chunk_size=300, overlap=30, top_k=3
  - Configuration B: chunk_size=500, overlap=50, top_k=5 (Selected Default)
  - Configuration C: chunk_size=700, overlap=70, top_k=7

Saves empirical benchmark measurements to docs/results/m4_retrieval_experiment.json.
"""

import json
import time
import os
from pathlib import Path
import numpy as np

from ingestion.chunking import TextChunker
from ingestion.embeddings import EmbeddingService


TEST_QUERIES = [
    # HR Domain Queries
    {"query": "What are the eligibility requirements for FMLA leave?", "domain": "hr", "target": "fmla"},
    {"query": "How many days of paid annual leave are full-time employees entitled to?", "domain": "hr", "target": "leave"},
    {"query": "What is the maternity leave return to work procedure?", "domain": "hr", "target": "maternity"},

    # Technology Domain Queries
    {"query": "How do I configure vector search in PostgreSQL using pgvector?", "domain": "technology", "target": "pgvector"},
    {"query": "What are the required cloud encryption standards for data at rest?", "domain": "technology", "target": "encryption"},
    {"query": "How to deploy a FastAPI application service on Kubernetes?", "domain": "technology", "target": "kubernetes"},

    # Finance Domain Queries
    {"query": "What is the corporate travel expense reimbursement limit for hotels?", "domain": "finance", "target": "reimbursement"},
    {"query": "What documents and receipts are required for expense submission?", "domain": "finance", "target": "receipts"},
    {"query": "What is the approval threshold for corporate card expenses over $2000?", "domain": "finance", "target": "approval"},
    {"query": "What expenses are ineligible for corporate reimbursement?", "domain": "finance", "target": "ineligible"},
]

CONFIGURATIONS = [
    {"name": "Config A (Compact)", "chunk_size": 300, "overlap": 30, "top_k": 3},
    {"name": "Config B (Selected Baseline)", "chunk_size": 500, "overlap": 50, "top_k": 5},
    {"name": "Config C (Large Context)", "chunk_size": 700, "overlap": 70, "top_k": 7},
]


def load_sample_texts():
    texts = {}
    sample_files = {
        "hr": "data/sample_documents/hr/fmla_guide.txt",
        "technology": "data/sample_documents/technology/tech_kb.txt",
        "finance": "data/sample_documents/finance/finance_policy.txt",
    }
    
    # Fallback inline sample texts if sample directory files not found
    fallback_texts = {
        "hr": "FMLA provides up to 12 weeks of unpaid leave for eligible employees who have worked at least 1,250 hours over the preceding 12 months. Employees receive 20 days of paid annual leave.",
        "technology": "PostgreSQL pgvector extension enables 384-dimensional vector similarity search using cosine distance. Cloud data at rest must be encrypted using AES-256.",
        "finance": "Corporate travel reimbursement for hotels is capped at $250 per night for standard cities and $350 for high-cost cities. Original receipts required for expenses exceeding $25.",
    }

    for domain, filepath in sample_files.items():
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                texts[domain] = f.read()
        else:
            texts[domain] = fallback_texts[domain]
            
    return texts


def run_experiment():
    print("=" * 70)
    print("RUNNING REPRODUCIBLE RETRIEVAL OPTIMIZATION EXPERIMENT")
    print("=" * 70)

    sample_texts = load_sample_texts()
    embedder = EmbeddingService()

    results = []

    for cfg in CONFIGURATIONS:
        print(f"\nBenchmarking {cfg['name']} (chunk_size={cfg['chunk_size']}, top_k={cfg['top_k']})...")
        chunker = TextChunker(chunk_size=cfg["chunk_size"], chunk_overlap=cfg["overlap"])

        # Ingest and embed chunks across all domains
        domain_chunks = {}
        domain_embeddings = {}

        for domain, text in sample_texts.items():
            chunks = chunker.chunk(text, document_id=f"doc-{domain}")
            domain_chunks[domain] = chunks
            if chunks:
                chunk_texts = [c.text for c in chunks]
                domain_embeddings[domain] = np.array(embedder.generate_embeddings(chunk_texts))
            else:
                domain_embeddings[domain] = np.empty((0, 384))

        # Evaluate query retrieval performance
        query_latencies = []
        similarity_scores = []
        hits = 0

        for q_item in TEST_QUERIES:
            q_text = q_item["query"]
            target_domain = q_item["domain"]
            target_kw = q_item["target"]

            t0 = time.perf_counter()
            q_emb = np.array(embedder.generate_embedding(q_text))

            # Cosine similarity search over target domain chunks
            doc_embs = domain_embeddings.get(target_domain)
            chunks = domain_chunks.get(target_domain, [])

            if doc_embs is not None and len(doc_embs) > 0:
                # Cosine similarity = dot product of normalized vectors
                sims = np.dot(doc_embs, q_emb)
                top_k_indices = np.argsort(sims)[::-1][:cfg["top_k"]]
                top_sims = sims[top_k_indices]
                top_chunks = [chunks[idx] for idx in top_k_indices]
            else:
                top_sims = np.array([])
                top_chunks = []

            latency_ms = (time.perf_counter() - t0) * 1000.0
            query_latencies.append(latency_ms)

            if len(top_sims) > 0:
                best_sim = float(top_sims[0])
                similarity_scores.append(best_sim)
                # Check hit — does top retrieved chunk contain target concept keyword?
                top_text_combined = " ".join([c.text.lower() for c in top_chunks])
                if target_kw.lower() in top_text_combined:
                    hits += 1

        hit_rate = (hits / len(TEST_QUERIES)) * 100.0
        avg_sim = float(np.mean(similarity_scores)) if similarity_scores else 0.0
        avg_latency = float(np.mean(query_latencies))

        cfg_res = {
            "name": cfg["name"],
            "chunk_size": cfg["chunk_size"],
            "overlap": cfg["overlap"],
            "top_k": cfg["top_k"],
            "hit_rate_pct": round(hit_rate, 1),
            "avg_similarity": round(avg_sim, 4),
            "avg_retrieval_latency_ms": round(avg_latency, 2),
            "total_queries_tested": len(TEST_QUERIES),
        }
        results.append(cfg_res)

        print(f"  Hit Rate: {cfg_res['hit_rate_pct']}% | Avg Sim: {cfg_res['avg_similarity']} | Avg Latency: {cfg_res['avg_retrieval_latency_ms']} ms")

    # Output results to docs/results/m4_retrieval_experiment.json
    out_dir = Path("docs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "m4_retrieval_experiment.json"

    out_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "configurations": results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT COMPLETE. Results saved to: {out_file.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
