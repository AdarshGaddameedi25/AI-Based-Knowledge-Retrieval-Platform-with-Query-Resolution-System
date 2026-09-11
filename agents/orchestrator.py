"""
AgentOrchestrator — Milestone 2 unified multi-agent pipeline.

Flow:
  1. QueryUnderstandingAgent  → normalize, classify (query_type, routing), detect domain
  2. ConversationMemoryAgent  → resolve pronouns using session history
  3. ClarificationAgent       → validate ambiguity (with history context)
  4. RetrievalAgent           → domain-filtered vector search
  5. ResponseGenerationAgent  → LLM answer with conversation context + confidence score

Routing:
  query_type=direct        → routing=direct      → immediate response (no retrieval)
  query_type=ambiguous     → routing=clarification → ClarificationAgent → question returned
  query_type=factual/
             procedural/
             comparative   → routing=retrieval   → RetrievalAgent → ResponseGenerationAgent
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from sqlalchemy.orm import Session

from agents.query_understanding_agent import (
    QueryUnderstandingAgent, QueryAnalysis,
    QUERY_TYPE_AMBIGUOUS, QUERY_TYPE_DIRECT,
    ROUTING_RETRIEVAL, ROUTING_CLARIFICATION, ROUTING_DIRECT,
)
from agents.clarification_agent import ClarificationAgent, ClarificationResult
from agents.conversation_memory_agent import ConversationMemoryAgent
from agents.retrieval_agent import RetrievalAgent
from agents.response_generation_agent import ResponseGenerationAgent
from retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """Unified response object returned by the orchestrator for every query."""
    query: str
    answer: str
    sources: List[dict]

    # M2 classification fields
    query_type: str              # factual | procedural | comparative | ambiguous | direct
    routing: str                 # retrieval | clarification | direct
    detected_intent: str         # same as query_type (backward compat)
    detected_domain: Optional[str]
    key_terms: List[str]

    # Clarification state
    clarification_needed: bool
    clarification_question: str

    # Scoring
    confidence: float
    retrieval_count: int
    classification_confidence: float = 0.85

    # Session tracking
    session_id: Optional[str] = None


class AgentOrchestrator:
    """
    Coordinates the five M2 agents in a single deterministic pipeline.
    All agents are injectable for testability; defaults are singletons.
    """

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
        """
        Execute the full M2 pipeline:
        QueryUnderstanding → MemoryResolution → [Clarification | Retrieval] → Generation
        """

        # ── Step 1: Query Understanding ────────────────────────────────────
        analysis: QueryAnalysis = self._query_agent.analyze(query)
        logger.info(
            "[Orchestrator] query_type=%r routing=%r domain=%r requires_retrieval=%s",
            analysis.query_type, analysis.routing,
            analysis.suggested_domain, analysis.requires_retrieval,
        )

        # ── Step 2: Resolve pronouns from conversation history ─────────────
        resolved_query = memory.resolve_references(analysis.normalized_query)
        if resolved_query != analysis.normalized_query:
            logger.info(
                "[Orchestrator] Reference resolved: %r → %r",
                analysis.normalized_query, resolved_query,
            )
            analysis.normalized_query = resolved_query

        # ── Step 3: Direct response (greetings / small-talk only) ──────────
        #    IMPORTANT: query_type=ambiguous also has requires_retrieval=False
        #    but must NOT be treated as direct — it needs the clarification path.
        #    Only QUERY_TYPE_DIRECT (greetings, small-talk) short-circuits here.
        if analysis.query_type == QUERY_TYPE_DIRECT:
            logger.info("[Orchestrator] Routing=direct — no retrieval needed")
            answer = (
                "Hello! I'm the AI Knowledge Retrieval Assistant. "
                "Ask me anything about the uploaded documents and I'll find relevant answers for you."
            )
            memory.add_message("user", query)
            memory.add_message("assistant", answer)
            return OrchestratorResult(
                query=query,
                answer=answer,
                sources=[],
                query_type=analysis.query_type,
                routing=ROUTING_DIRECT,
                detected_intent=analysis.detected_intent,
                detected_domain=None,
                key_terms=analysis.key_terms,
                clarification_needed=False,
                clarification_question="",
                confidence=1.0,
                retrieval_count=0,
                classification_confidence=analysis.classification_confidence,
                session_id=session_id,
            )

        # ── Step 4: Clarification check ────────────────────────────────────
        #    Run for any query routed to clarification, OR for queries
        #    the QUA labelled ambiguous but did not suppress retrieval for.
        history = memory.get_history()

        # Shortcut: if QUA already classified as ambiguous, skip re-evaluation
        if analysis.query_type == QUERY_TYPE_AMBIGUOUS:
            clarification_q = self._clarification_agent._generate_clarification(resolved_query)
            logger.info("[Orchestrator] Routing=clarification (QUA detected ambiguity)")
            memory.add_message("user", query)
            memory.add_message("assistant", f"[clarification needed] {clarification_q}")
            return OrchestratorResult(
                query=query,
                answer=clarification_q,
                sources=[],
                query_type=QUERY_TYPE_AMBIGUOUS,
                routing=ROUTING_CLARIFICATION,
                detected_intent=analysis.detected_intent,
                detected_domain=analysis.suggested_domain,
                key_terms=analysis.key_terms,
                clarification_needed=True,
                clarification_question=clarification_q,
                confidence=0.0,
                retrieval_count=0,
                classification_confidence=analysis.classification_confidence,
                session_id=session_id,
            )

        # Secondary clarification check (ClarificationAgent catches pronoun queries
        # that slipped through QUA's ambiguity check due to rich key-term context)
        clarification: ClarificationResult = self._clarification_agent.evaluate(
            resolved_query, history
        )
        if clarification.is_ambiguous:
            logger.info("[Orchestrator] Routing=clarification (ClarificationAgent)")
            memory.add_message("user", query)
            memory.add_message(
                "assistant",
                f"[clarification needed] {clarification.clarification_question}",
            )
            return OrchestratorResult(
                query=query,
                answer=clarification.clarification_question,
                sources=[],
                query_type=QUERY_TYPE_AMBIGUOUS,
                routing=ROUTING_CLARIFICATION,
                detected_intent=analysis.detected_intent,
                detected_domain=analysis.suggested_domain,
                key_terms=analysis.key_terms,
                clarification_needed=True,
                clarification_question=clarification.clarification_question,
                confidence=0.0,
                retrieval_count=0,
                classification_confidence=analysis.classification_confidence,
                session_id=session_id,
            )

        # ── Step 5: Retrieval ──────────────────────────────────────────────
        results: List[RetrievalResult] = []
        retrieval_error: Optional[str] = None

        try:
            results = self._retrieval_agent.retrieve(
                analysis,
                db=db,
                top_k=top_k,
                domain_filter=domain_filter,
                similarity_threshold=similarity_threshold,
            )
            logger.info("[Orchestrator] Retrieved %d chunks", len(results))
        except Exception as e:
            logger.error("[Orchestrator] Retrieval failed: %s", e, exc_info=True)
            retrieval_error = "Retrieval service temporarily unavailable."

        if retrieval_error:
            memory.add_message("user", query)
            memory.add_message("assistant", retrieval_error)
            return OrchestratorResult(
                query=query,
                answer=retrieval_error,
                sources=[],
                query_type=analysis.query_type,
                routing=ROUTING_RETRIEVAL,
                detected_intent=analysis.detected_intent,
                detected_domain=analysis.suggested_domain,
                key_terms=analysis.key_terms,
                clarification_needed=False,
                clarification_question="",
                confidence=0.0,
                retrieval_count=0,
                classification_confidence=analysis.classification_confidence,
                session_id=session_id,
            )

        # ── Step 6: Response Generation ────────────────────────────────────
        #    Pass conversation context so the LLM can produce coherent multi-turn answers.
        conversation_context = memory.build_context_window("")  # system prompt added inside agent
        generation_error: Optional[str] = None
        generated = None

        try:
            generated = self._response_agent.generate(
                query=resolved_query,
                retrieval_results=results,
                query_type=analysis.query_type,
                conversation_history=conversation_context,
            )
        except Exception as e:
            logger.error("[Orchestrator] Response generation failed: %s", e, exc_info=True)
            generation_error = (
                "The answer could not be generated at this time. "
                "Please try again or rephrase your question."
            )

        if generation_error:
            memory.add_message("user", query)
            memory.add_message("assistant", generation_error)
            return OrchestratorResult(
                query=query,
                answer=generation_error,
                sources=[],
                query_type=analysis.query_type,
                routing=ROUTING_RETRIEVAL,
                detected_intent=analysis.detected_intent,
                detected_domain=analysis.suggested_domain,
                key_terms=analysis.key_terms,
                clarification_needed=False,
                clarification_question="",
                confidence=0.0,
                retrieval_count=len(results),
                classification_confidence=analysis.classification_confidence,
                session_id=session_id,
            )

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
            query_type=analysis.query_type,
            routing=ROUTING_RETRIEVAL,
            detected_intent=analysis.detected_intent,
            detected_domain=analysis.suggested_domain,
            key_terms=analysis.key_terms,
            clarification_needed=False,
            clarification_question="",
            confidence=generated.confidence,
            retrieval_count=len(results),
            classification_confidence=analysis.classification_confidence,
            session_id=session_id,
        )
