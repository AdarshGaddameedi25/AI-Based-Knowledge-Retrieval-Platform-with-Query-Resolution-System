"""
ResponseGenerationAgent — Milestone 2

Generates a grounded, cited answer by calling OpenRouter LLM with:
  - Retrieved evidence chunks as the knowledge context
  - Query type hint so the LLM can tailor its response style
  - Optional conversation history for multi-turn coherence

Design principles:
  - Responses are grounded in retrieved evidence only
  - The system prompt explicitly prohibits unsupported claims
  - Insufficient evidence results in a controlled "not available" response
  - Source attribution is derived strictly from the retrieved chunks
  - Confidence is the mean retrieval similarity score (a relevance signal, NOT a probability)
"""
import logging
import httpx
from dataclasses import dataclass, field
from typing import List, Optional

from config.settings import settings
from retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)

_KNOWLEDGE_GAP_RESPONSE = (
    "I could not find sufficient information in the available knowledge base "
    "to answer this question."
)

_SYSTEM_PROMPT_TEMPLATE = """You are a precise AI knowledge assistant. Your answers must be grounded in the retrieved context provided below.

Rules:
1. Answer using ONLY information found in the provided context.
2. If the context does not contain enough information, respond with exactly: "{gap_message}"
3. Always cite the relevant source document name(s) in your answer.
4. Be concise and directly address the question.
5. Do not fabricate facts, documents, page numbers, or citations.
6. For {query_type} questions, structure your answer appropriately."""

_SYSTEM_PROMPT_DEFAULT = _SYSTEM_PROMPT_TEMPLATE.format(
    gap_message=_KNOWLEDGE_GAP_RESPONSE,
    query_type="knowledge",
)


@dataclass
class GeneratedResponse:
    """Structured output from the ResponseGenerationAgent."""
    query: str
    answer: str
    citations: List[dict]
    context_chunks_used: int
    confidence: float = 0.0


class ResponseGenerationAgent:
    """
    Calls OpenRouter LLM to generate a grounded answer from retrieved evidence.

    The single authoritative LLM integration path for this application is:
      Application → httpx → OpenRouter API → Selected LLM model

    No other LLM provider is used.
    """

    def __init__(self):
        if not settings.openrouter_api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set. Set it in .env before starting the server."
            )
        self._headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AI Knowledge Retrieval Platform",
        }

    def generate(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        query_type: str = "factual",
        conversation_history: Optional[List[dict]] = None,
    ) -> GeneratedResponse:
        """
        Generate a grounded answer.

        Args:
            query: The (resolved) user question.
            retrieval_results: Chunks retrieved from pgvector.
            query_type: Canonical M2 label (factual/procedural/comparative/ambiguous).
            conversation_history: Prior messages for multi-turn context.
                Expected format: [{"role": "user"|"assistant"|"system", "content": "..."}]
                The system message in this list is replaced by our own system prompt.

        Returns:
            GeneratedResponse with answer, citations, and confidence.
        """
        # ── Insufficient evidence path ─────────────────────────────────────
        if not retrieval_results:
            logger.info("[ResponseGenerationAgent] No retrieval results — returning knowledge gap response")
            return GeneratedResponse(
                query=query,
                answer=_KNOWLEDGE_GAP_RESPONSE,
                citations=[],
                context_chunks_used=0,
                confidence=0.0,
            )

        # ── Build context string with source labels ────────────────────────
        context_parts = []
        for r in retrieval_results:
            page_info = f" | Page: {r.page_number}" if r.page_number else ""
            section_info = f" | Domain: {r.domain}" if r.domain else ""
            context_parts.append(
                f"[Source: {r.source_file}{page_info}{section_info}]\n"
                f"Relevance: {r.similarity_score:.4f}\n"
                f"{r.text}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # ── Confidence = mean retrieval similarity (relevance signal) ──────
        scores = [r.similarity_score for r in retrieval_results]
        confidence = round(sum(scores) / len(scores), 4) if scores else 0.0

        # ── Build system prompt with query-type hint ───────────────────────
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            gap_message=_KNOWLEDGE_GAP_RESPONSE,
            query_type=query_type,
        )

        # ── Assemble messages ──────────────────────────────────────────────
        #    Message structure:
        #    [system]
        #    [prior user/assistant turns, if any — from conversation_history, excluding system]
        #    [current user turn with context + question]
        messages = [{"role": "system", "content": system_prompt}]

        # Include prior conversation turns (skip any system messages from history)
        if conversation_history:
            for msg in conversation_history:
                if msg.get("role") in ("user", "assistant"):
                    messages.append({"role": msg["role"], "content": msg["content"]})

        # Append the current question with its retrieved context
        user_message = (
            f"Retrieved Knowledge Base Context:\n\n{context}\n\n"
            f"---\n\nQuestion: {query}"
        )
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1024,
        }

        # ── Call OpenRouter ────────────────────────────────────────────────
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
        except httpx.HTTPStatusError as e:
            logger.error(
                "[ResponseGenerationAgent] OpenRouter HTTP error %s: %s",
                e.response.status_code, e.response.text[:200],
            )
            raise
        except httpx.TimeoutException:
            logger.error("[ResponseGenerationAgent] OpenRouter request timed out")
            raise
        except Exception as e:
            logger.error("[ResponseGenerationAgent] LLM call failed: %s", e)
            raise

        # ── Build structured citations from retrieved chunks ────────────────
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
