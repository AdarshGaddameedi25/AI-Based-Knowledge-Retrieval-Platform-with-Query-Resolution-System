from dataclasses import dataclass, field
from typing import Optional
import openai

from config.settings import settings
from retrieval.vector_store import VectorStore


@dataclass
class RetrievalResult:
    chunk_id: str
    document_id: str
    source_file: str
    text: str
    similarity_score: float
    rank: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class Retriever:
    def __init__(self, vector_store: VectorStore = None):
        if not settings.openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        self._client = openai.OpenAI(api_key=settings.openai_api_key)
        self._store = vector_store or VectorStore()

    def _embed_query(self, query: str) -> list[float]:
        response = self._client.embeddings.create(
            model=settings.openai_embedding_model,
            input=query,
        )
        return response.data[0].embedding

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        top_k = top_k or settings.top_k_results
        query_embedding = self._embed_query(query)
        hits = self._store.similarity_search(query_embedding, top_k=top_k, filter_metadata=filter_metadata)

        results = []
        for hit in hits:
            meta = hit.get("metadata", {})
            results.append(
                RetrievalResult(
                    chunk_id=hit["chunk_id"],
                    document_id=meta.get("document_id", ""),
                    source_file=meta.get("source_file", ""),
                    text=meta.get("text", ""),
                    similarity_score=hit["similarity_score"],
                    rank=hit["rank"],
                    chunk_index=int(meta.get("chunk_index", 0)),
                    metadata=meta,
                )
            )
        return results
