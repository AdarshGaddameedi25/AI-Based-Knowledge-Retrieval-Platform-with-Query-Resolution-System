from retrieval.retriever import Retriever, RetrievalResult
from agents.query_understanding_agent import QueryAnalysis


class RetrievalAgent:
    def __init__(self, retriever: Retriever = None):
        self._retriever = retriever or Retriever()

    def retrieve(self, analysis: QueryAnalysis, top_k: int = 5) -> list[RetrievalResult]:
        if not analysis.requires_retrieval:
            return []
        return self._retriever.retrieve(analysis.normalized_query, top_k=top_k)
