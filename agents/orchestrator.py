"""
AgentOrchestrator — Milestone 2 unified multi-agent pipeline.

Flow:
  1. QueryUnderstandingAgent  → normalize, detect intent + domain
  2. ConversationMemoryAgent  → resolve pronouns using session history
  3. ClarificationAgent       → detect ambiguity (with history context)
  4. RetrievalAgent           → domain-filtered vector search (skipped if ambiguous)
  5. ResponseGenerationAgent  → LLM answer with confidence score
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from sqlalchemy.orm import Session

from agents.query_understanding_agent import QueryUnderstandingAgent, QueryAnalysis
from agents.clarification_agent import ClarificationAgent, ClarificationResult
from agents.conversation_memory_agent import ConversationMemoryAgent
from agents.retrieval_agent import RetrievalAgent
from agents.response_generation_agent import ResponseGenerationAgent
from retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    query: str
    answer: str
    sources: List[dict]
    detected_intent: str
    detected_domain: Optional[str]
    key_terms: List[str]
    clarification_needed: bool
    clarification_question: str
    confidence: float
    retrieval_count: int
    session_id: Optional[str] = None


class AgentOrchestrator:
    def __init__(
        self,
        query_agent: QueryUnderstandingAgent = None,
        clarification_agent: ClarificationAgent = None,
        retrieval_agent: RetrievalAgent = None,
        response_agent: ResponseGenerationAgent = None,
    ):
        self._query_agent = query_agent or QueryUnderstandingAgent()
        self._clarification_agent = clarification_agent or ClarificationAgent()
        self._retrieval_agent = retrieval_agent or RetrievalAgent()
        self._response_agent = response_agent or ResponseGenerationAgent()

    def run(
        self,
        query: str,
        db: Session,
        memory: ConversationMemoryAgent,
        session_id: Optional[str] = None,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
        similarity_threshold: float = 0.0,
    ) -> OrchestratorResult:
        # ── Step 1: Query Understanding ────────────────────────────────────
        analysis: QueryAnalysis = self._query_agent.analyze(query)
        logger.info(
            f"[Orchestrator] intent={analysis.detected_intent!r} "
            f"domain={analysis.suggested_domain!r} "
            f"requires_retrieval={analysis.requires_retrieval}"
        )

        # ── Step 2: Resolve pronouns from conversation history ─────────────
        resolved_query = memory.resolve_references(analysis.normalized_query)
        if resolved_query != analysis.normalized_query:
            logger.info(f"[Orchestrator] Pronoun resolved: '{analysis.normalized_query}' → '{resolved_query}'")
            analysis.normalized_query = resolved_query

        # ── Step 3: Handle greetings / small talk (before clarification) ───────
        if not analysis.requires_retrieval:
            logger.info("[Orchestrator] No retrieval needed — direct response")
            memory.add_message("user", query)
            answer = (
                "Hello! I'm the AI Knowledge Retrieval Assistant. "
                "Ask me anything about the uploaded documents and I'll find relevant answers for you."
            )
            memory.add_message("assistant", answer)
            return OrchestratorResult(
                query=query,
                answer=answer,
                sources=[],
                detected_intent=analysis.detected_intent,
                detected_domain=None,
                key_terms=analysis.key_terms,
                clarification_needed=False,
                clarification_question="",
                confidence=1.0,
                retrieval_count=0,
                session_id=session_id,
            )

        # ── Step 4: Clarification check (only for knowledge queries) ──────────
        history = memory.get_history()
        clarification: ClarificationResult = self._clarification_agent.evaluate(resolved_query, history)

        if clarification.is_ambiguous:
            logger.info(f"[Orchestrator] Ambiguous query — requesting clarification")
            memory.add_message("user", query)
            memory.add_message(
                "assistant",
                f"[clarification needed] {clarification.clarification_question}",
            )
            return OrchestratorResult(
                query=query,
                answer=clarification.clarification_question,
                sources=[],
                detected_intent=analysis.detected_intent,
                detected_domain=analysis.suggested_domain,
                key_terms=analysis.key_terms,
                clarification_needed=True,
                clarification_question=clarification.clarification_question,
                confidence=0.0,
                retrieval_count=0,
                session_id=session_id,
            )



        # ── Step 5: Retrieval ──────────────────────────────────────────────
        results: List[RetrievalResult] = self._retrieval_agent.retrieve(
            analysis,
            db=db,
            top_k=top_k,
            domain_filter=domain_filter,
            similarity_threshold=similarity_threshold,
        )
        logger.info(f"[Orchestrator] Retrieved {len(results)} chunks")

        # ── Step 6: Response Generation ────────────────────────────────────
        generated = self._response_agent.generate(resolved_query, results)

        # ── Step 7: Update conversation memory ────────────────────────────
        memory.add_message("user", query)
        memory.add_message("assistant", generated.answer)

        sources = [
            {
                "rank": idx + 1,
                "source_file": r.source_file,
                "chunk_id": r.chunk_id,
                "chunk_index": r.chunk_index,
                "page_number": r.page_number,
                "similarity_score": round(r.similarity_score, 4),
                "domain": r.domain,
            }
            for idx, r in enumerate(results)
        ]

        return OrchestratorResult(
            query=query,
            answer=generated.answer,
            sources=sources,
            detected_intent=analysis.detected_intent,
            detected_domain=analysis.suggested_domain,
            key_terms=analysis.key_terms,
            clarification_needed=False,
            clarification_question="",
            confidence=generated.confidence,
            retrieval_count=len(results),
            session_id=session_id,
        )
