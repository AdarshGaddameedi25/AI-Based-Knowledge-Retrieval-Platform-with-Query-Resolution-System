import logging
import httpx
from dataclasses import dataclass, field
from typing import List, Optional
from sqlalchemy.orm import Session

from config.settings import settings
from retrieval.retriever import Retriever, RetrievalResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful and precise AI knowledge assistant.
Answer the user's question accurately based on the provided knowledge-base context.
Cite the relevant source document(s) when answering.
If the provided context truly does not contain enough information to answer the question, state: "I could not find an answer to your question in the available knowledge base."
Stay grounded in the provided context and avoid making up facts."""


@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: List[dict]
    context_used: str
    retrieval_results: List[RetrievalResult]
    confidence: float = 0.0


class RAGPipeline:
    def __init__(self, retriever: Retriever = None):
        if not settings.openrouter_api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set.")
        self._retriever = retriever or Retriever()
        self._headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AI Knowledge Retrieval Platform",
        }

    def _build_context(self, results: List[RetrievalResult]) -> str:
        sections = []
        for r in results:
            page_info = f" | Page: {r.page_number}" if r.page_number else ""
            sections.append(
                f"[Source {r.rank}]\n"
                f"Document: {r.source_file}{page_info}\n"
                f"Similarity: {r.similarity_score:.4f}\n"
                f"Content:\n{r.text}"
            )
        return "\n\n---\n\n".join(sections)

    def _call_llm(self, context: str, query: str) -> str:
        user_message = f"Knowledge Base Context:\n\n{context}\n\n---\n\nUser Question: {query}"

        payload = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{settings.openrouter_base_url}/chat/completions",
                headers=self._headers,
                json=payload,
            )
            response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def run(
        self,
        query: str,
        db: Session,
        top_k: int = None,
        domain_filter: Optional[str] = None,
    ) -> RAGResponse:
        results = self._retriever.retrieve(
            query,
            db=db,
            top_k=top_k,
            domain_filter=domain_filter,
        )

        if not results:
            return RAGResponse(
                query=query,
                answer="I could not find an answer to your question in the available knowledge base.",
                sources=[],
                context_used="",
                retrieval_results=[],
                confidence=0.0,
            )

        context = self._build_context(results)
        logger.info(f"Calling OpenRouter LLM ({settings.llm_model}) with {len(results)} retrieved chunks")

        answer = self._call_llm(context, query)

        # Compute confidence as mean similarity score
        scores = [r.similarity_score for r in results]
        confidence = round(sum(scores) / len(scores), 4) if scores else 0.0

        sources = [
            {
                "rank": r.rank,
                "source_file": r.source_file,
                "chunk_id": r.chunk_id,
                "chunk_index": r.chunk_index,
                "page_number": r.page_number,
                "similarity_score": round(r.similarity_score, 4),
                "domain": r.domain,
            }
            for r in results
        ]

        return RAGResponse(
            query=query,
            answer=answer,
            sources=sources,
            context_used=context,
            retrieval_results=results,
            confidence=confidence,
        )
