import logging
from typing import List, Optional
from sqlalchemy.orm import Session

from retrieval.retriever import Retriever, RetrievalResult
from agents.query_understanding_agent import QueryAnalysis

logger = logging.getLogger(__name__)


class RetrievalAgent:
    def __init__(self, retriever: Retriever = None):
        self._retriever = retriever or Retriever()

    def retrieve(
        self,
        analysis: QueryAnalysis,
        db: Session,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        resolved_query: Optional[str] = None,
    ) -> List[RetrievalResult]:
        if not analysis.requires_retrieval:
            return []

        # Use suggested domain from query analysis if not explicitly overridden
        effective_domain = domain_filter or analysis.suggested_domain

        # Use pronoun-resolved query for vector search if available,
        # otherwise fall back to the normalized query from QueryUnderstandingAgent
        search_query = resolved_query or analysis.normalized_query

        logger.info(
            f"RetrievalAgent: query='{search_query[:80]}' "
            f"domain_filter={effective_domain!r} top_k={top_k} "
            f"threshold={similarity_threshold}"
        )

        return self._retriever.retrieve(
            search_query,
            db=db,
            top_k=top_k,
            domain_filter=effective_domain,
            similarity_threshold=similarity_threshold,
        )
