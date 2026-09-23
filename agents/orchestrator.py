"""
AgentOrchestrator — Milestone 3 unified multi-agent pipeline.

Flow:
  0. Clarification check      → if pending clarification, refine_query first
  1. QueryUnderstandingAgent  → normalize, classify (query_type, routing), detect domain
  2. ConversationMemoryAgent  → resolve pronouns using session history + entity tracking
  3. ClarificationAgent       → validate ambiguity (with history context + key terms)
  4. RetrievalAgent           → domain-filtered vector search
  5. ResponseGenerationAgent  → LLM answer with conversation context + confidence score
  6. Memory Update            → store entities, active topic, source docs

Routing:
  query_type=direct        → routing=direct      → immediate response (no retrieval)
  query_type=ambiguous     → routing=clarification → ClarificationAgent → question returned
  pending clarification    → refine_query()      → re-enter pipeline with refined query
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
from agents.clarification_agent import ClarificationAgent, ClarificationResult, ClarificationState
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

    # M3.1 — refined query produced after clarification cycle
    refined_query: str = ""

    # Scoring
    confidence: float = 0.0
    retrieval_count: int = 0
    classification_confidence: float = 0.85

    # Session tracking
    session_id: Optional[str] = None

    # M3.2 — memory context flag
    used_memory_context: bool = False


class AgentOrchestrator:
    """
    Coordinates the five M3 agents in a single deterministic pipeline.
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
        similarity_threshold: Optional[float] = None,
    ) -> OrchestratorResult:
        """
        Execute the full M3 pipeline:
        [ClarificationCheck] → QueryUnderstanding → MemoryResolution →
        [Clarification | Retrieval] → Generation → MemoryUpdate
        """

        # ── Step 0: Check for pending clarification ────────────────────────
        #    If the previous turn ended with a clarification question,
        #    the current user message is the clarification response.
        #    Combine original_query + response → refined_query and continue.
        refined_query: Optional[str] = None
        original_query_for_clarification: Optional[str] = None

        if memory.has_pending_clarification():
            pending: ClarificationState = memory.consume_clarification()
            refined_query = self._clarification_agent.refine_query(
                pending.original_query, query
            )
            original_query_for_clarification = pending.original_query
            logger.info(
                "[Orchestrator] Clarification resolved: original=%r response=%r refined=%r",
                pending.original_query, query, refined_query,
            )
            # Use refined query for the rest of the pipeline
            query = refined_query

        # ── Step 1: Query Understanding ────────────────────────────────────
        analysis: QueryAnalysis = self._query_agent.analyze(query)
        logger.info(
            "[Orchestrator] query_type=%r routing=%r domain=%r requires_retrieval=%s",
            analysis.query_type, analysis.routing,
            analysis.suggested_domain, analysis.requires_retrieval,
        )

        # ── Step 2: Resolve pronouns from conversation history ─────────────
        memory_resolved_query = memory.resolve_references(analysis.normalized_query)
        if memory_resolved_query != analysis.normalized_query:
            logger.info(
                "[Orchestrator] Reference resolved: %r → %r",
                analysis.normalized_query, memory_resolved_query,
            )
            analysis.normalized_query = memory_resolved_query

        # M3.2 — resolved_query holds the best query text for vector search:
        # either the clarification-refined query (from Step 0) or the
        # memory pronoun-resolved query (from Step 2 above).
        # NOTE: always assign — avoids UnboundLocalError in clarification cycle.
        resolved_query = refined_query if refined_query else memory_resolved_query

        # Determine whether prior memory context is being used
        used_memory_context = memory.is_topic_continuation(analysis.key_terms)

        # ── Step 3: Direct response (greetings / small-talk only) ──────────
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
                refined_query=refined_query or "",
                confidence=1.0,
                retrieval_count=0,
                classification_confidence=analysis.classification_confidence,
                session_id=session_id,
                used_memory_context=False,
            )

        # ── Step 4: Clarification check ────────────────────────────────────
        history = memory.get_history()

        # Shortcut: QUA classified as ambiguous — generate targeted question
        if analysis.query_type == QUERY_TYPE_AMBIGUOUS:
            clarification_q = self._clarification_agent.generate_targeted_question(
                resolved_query, analysis.key_terms
            )
            logger.info("[Orchestrator] Routing=clarification (QUA detected ambiguity)")

            # Store pending state in memory
            clar_state = ClarificationState(
                required=True,
                original_query=resolved_query,
                question=clarification_q,
                reason="Query classified as ambiguous by QueryUnderstandingAgent",
                pending=True,
            )
            memory.set_clarification_pending(clar_state)
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
                refined_query="",
                confidence=0.0,
                retrieval_count=0,
                classification_confidence=analysis.classification_confidence,
                session_id=session_id,
                used_memory_context=used_memory_context,
            )

        # Secondary clarification check (ClarificationAgent catches pronoun
        # queries that slipped through QUA's ambiguity check)
        clarification: ClarificationResult = self._clarification_agent.evaluate(
            resolved_query, history, analysis.key_terms
        )
        if clarification.is_ambiguous:
            logger.info("[Orchestrator] Routing=clarification (ClarificationAgent)")

            # Store pending state in memory
            if clarification.state:
                memory.set_clarification_pending(clarification.state)

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
                refined_query="",
                confidence=0.0,
                retrieval_count=0,
                classification_confidence=analysis.classification_confidence,
                session_id=session_id,
                used_memory_context=used_memory_context,
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
                resolved_query=resolved_query,  # M3.2 — use pronoun-resolved text for vector search
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
                refined_query=refined_query or "",
                confidence=0.0,
                retrieval_count=0,
                classification_confidence=analysis.classification_confidence,
                session_id=session_id,
                used_memory_context=used_memory_context,
            )

        # ── Step 6: Response Generation ────────────────────────────────────
        #    Pass contextually relevant conversation history (not full history)
        conversation_context = memory.get_relevant_context(analysis.key_terms)
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
                refined_query=refined_query or "",
                confidence=0.0,
                retrieval_count=len(results),
                classification_confidence=analysis.classification_confidence,
                session_id=session_id,
                used_memory_context=used_memory_context,
            )

        # ── Step 7: Update conversation memory ────────────────────────────
        memory.add_message("user", query)
        memory.add_message("assistant", generated.answer)

        # M3.2 — Update topic, entities, source docs for next turn
        source_doc_names = list({r.source_file for r in results})
        memory.update_topic(
            query_key_terms=analysis.key_terms,
            source_docs=source_doc_names,
            assistant_response=generated.answer,
        )

        # ── Step 8: Build sources with M3.4 transparency fields ────────────
        sources = [
            {
                "rank": idx + 1,
                "citation": f"[{idx + 1}]",
                "source_file": r.source_file,
                "document_name": r.source_file,
                "chunk_id": r.chunk_id,
                "chunk_index": r.chunk_index,
                "page_number": r.page_number,
                "similarity_score": round(r.similarity_score, 4),
                "domain": r.domain,
                # M3.4 — include chunk text (truncated) for transparency panel
                "text": r.text[:600] if r.text else "",
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
            refined_query=refined_query or "",
            confidence=generated.confidence,
            retrieval_count=len(results),
            classification_confidence=analysis.classification_confidence,
            session_id=session_id,
            used_memory_context=used_memory_context,
        )
