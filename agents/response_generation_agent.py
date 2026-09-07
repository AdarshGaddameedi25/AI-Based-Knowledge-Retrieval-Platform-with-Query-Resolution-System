import logging
import httpx
from dataclasses import dataclass, field
from typing import List, Optional

from config.settings import settings
from retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class GeneratedResponse:
    query: str
    answer: str
    citations: List[dict]
    context_chunks_used: int
    confidence: float = 0.0


class ResponseGenerationAgent:
    SYSTEM_PROMPT = (
        "You are a knowledgeable AI assistant. Answer using ONLY the context provided. "
        "If the answer is not in the context, respond with: "
        "'The requested information is not available in the knowledge base.' "
        "Always cite the source document name at the end of your answer."
    )

    def __init__(self):
        if not settings.openrouter_api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set.")
        self._headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AI Knowledge Retrieval Platform",
        }

    def generate(self, query: str, retrieval_results: List[RetrievalResult]) -> GeneratedResponse:
        if not retrieval_results:
            return GeneratedResponse(
                query=query,
                answer="The requested information is not available in the knowledge base.",
                citations=[],
                context_chunks_used=0,
                confidence=0.0,
            )

        # Build context string with source labels
        context_parts = []
        for r in retrieval_results:
            page_info = f" | Page: {r.page_number}" if r.page_number else ""
            context_parts.append(f"[{r.source_file}{page_info}]\n{r.text}")
        context = "\n\n".join(context_parts)

        # Calculate confidence from mean similarity of top results
        scores = [r.similarity_score for r in retrieval_results]
        confidence = round(sum(scores) / len(scores), 4) if scores else 0.0

        user_message = f"Context:\n{context}\n\nQuestion: {query}"

        payload = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{settings.openrouter_base_url}/chat/completions",
                    headers=self._headers,
                    json=payload,
                )
                response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"ResponseGenerationAgent LLM call failed: {e}")
            raise

        citations = [
            {
                "source": r.source_file,
                "chunk_id": r.chunk_id,
                "score": round(r.similarity_score, 4),
                "page_number": r.page_number,
                "domain": r.domain,
            }
            for r in retrieval_results
        ]

        return GeneratedResponse(
            query=query,
            answer=answer,
            citations=citations,
            context_chunks_used=len(retrieval_results),
            confidence=confidence,
        )
