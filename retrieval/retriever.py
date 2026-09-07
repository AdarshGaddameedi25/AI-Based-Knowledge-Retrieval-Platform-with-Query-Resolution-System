import logging
from dataclasses import dataclass, field
from typing import Optional, List
from sqlalchemy.orm import Session

from config.settings import settings
from ingestion.embeddings import EmbeddingService
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunk_id: str
    document_id: str
    source_file: str
    text: str
    similarity_score: float
    rank: int
    chunk_index: int
    page_number: Optional[int] = None
    domain: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class Retriever:
    def __init__(self):
        self._embedder = EmbeddingService()
        self._store = VectorStore()

    def retrieve(
        self,
        query: str,
        db: Session,
        top_k: int = None,
        domain_filter: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        top_k = top_k or settings.top_k_results
        threshold = similarity_threshold if similarity_threshold is not None else settings.similarity_threshold

        logger.info(
            f"Embedding query: {query[:80]} | "
            f"domain_filter={domain_filter!r} | threshold={threshold}"
        )
        query_embedding = self._embedder.generate_embedding(query)

        hits = self._store.similarity_search(
            db,
            query_embedding,
            top_k=top_k,
            domain_filter=domain_filter,
            similarity_threshold=threshold,
        )
        top_score = f"{hits[0]['similarity_score']:.4f}" if hits else "N/A"
        logger.info(f"Retrieved {len(hits)} chunks, top score: {top_score}")

        return [
            RetrievalResult(
                chunk_id=h["chunk_id"],
                document_id=h["document_id"],
                source_file=h["source_file"],
                text=h["text"],
                similarity_score=h["similarity_score"],
                rank=h["rank"],
                chunk_index=h["chunk_index"],
                page_number=h.get("page_number"),
                domain=h.get("domain"),
                metadata=h.get("metadata", {}),
            )
            for h in hits
        ]
