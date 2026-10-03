"""Diagnostic: check raw similarity scores for leave-related queries."""
import sys
sys.path.insert(0, '.')

from ingestion.embeddings import EmbeddingService
from retrieval.vector_store import VectorStore
from backend.db.base import get_db

db = next(get_db())
embedder = EmbeddingService()
store = VectorStore()

queries = [
    "What is the employee leave policy?",
    "What is the minimum wage?",
    "annual leave policy",
    "leave entitlement days",
    "employee handbook leave",
]

for q in queries:
    emb = embedder.generate_embedding(q)
    # Use threshold=0 to see ALL scores regardless of cutoff
    hits = store.similarity_search(db, emb, top_k=5, similarity_threshold=0.0)
    print(f"\nQuery: {q}")
    if not hits:
        print("  (no results at all)")
    for h in hits[:5]:
        score = h["similarity_score"]
        source = h["source_file"]
        text = h["text"][:70].replace("\n", " ")
        print(f"  score={score:.4f}  file={source}  | {text}")

db.close()
