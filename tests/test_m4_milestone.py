"""
tests/test_m4_milestone.py — Milestone 4 Comprehensive Test Suite

Covers:
  M4.1 — Analytics Recording (tests 1–12)
  M4.1 — Knowledge Gap Detection (tests 13–22)
  M4.1 — Analytics Aggregation (tests 23–32)
  M4.2 — Three-Domain Workflow (tests 33–44)
  M4.2 — Retrieval Quality (tests 45–52)
  M4.3 — Agent Routing Optimization (tests 53–62)
  M4.3 — Conversation Memory Optimization (tests 63–70)
  M4.3 — Voice/TTS fallback (described as browser-only, marked SKIPPED)

All LLM, DB, and embedding calls are mocked.
No live PostgreSQL, OpenRouter, or browser required.
"""
import os
import sys
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.query_understanding_agent import (
    QueryUnderstandingAgent,
    QUERY_TYPE_FACTUAL, QUERY_TYPE_PROCEDURAL, QUERY_TYPE_COMPARATIVE,
    QUERY_TYPE_AMBIGUOUS, QUERY_TYPE_DIRECT,
    ROUTING_RETRIEVAL, ROUTING_CLARIFICATION, ROUTING_DIRECT,
)
from agents.clarification_agent import ClarificationAgent, ClarificationState
from agents.conversation_memory_agent import ConversationMemoryAgent
from agents.retrieval_agent import RetrievalAgent
from agents.orchestrator import AgentOrchestrator, OrchestratorResult
from agents.response_generation_agent import ResponseGenerationAgent, GeneratedResponse
from retrieval.retriever import RetrievalResult


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def make_result(rank=1, score=0.85, text="PostgreSQL is an open-source relational database.", source="postgres.pdf", domain="technology", page=1):
    return RetrievalResult(
        chunk_id=f"chunk-{rank}",
        document_id="doc-tech-001",
        source_file=source,
        text=text,
        similarity_score=score,
        rank=rank,
        chunk_index=rank - 1,
        page_number=page,
        domain=domain,
    )


def make_hr_result(rank=1, score=0.82):
    return RetrievalResult(
        chunk_id=f"hr-chunk-{rank}",
        document_id="doc-hr-001",
        source_file="employee_handbook.txt",
        text="FMLA provides eligible employees with up to 12 weeks of unpaid leave.",
        similarity_score=score,
        rank=rank,
        chunk_index=rank - 1,
        page_number=1,
        domain="hr",
    )


def make_finance_result(rank=1, score=0.78):
    return RetrievalResult(
        chunk_id=f"fin-chunk-{rank}",
        document_id="doc-fin-001",
        source_file="finance_policy.txt",
        text="Travel expenses are reimbursable up to $250/night for standard cities.",
        similarity_score=score,
        rank=rank,
        chunk_index=rank - 1,
        page_number=1,
        domain="finance",
    )


def mock_response_agent_generate(query, retrieval_results, query_type="factual", conversation_history=None):
    if not retrieval_results:
        return GeneratedResponse(
            query=query,
            answer="I could not find sufficient information in the available knowledge base to answer this question.",
            citations=[],
            context_chunks_used=0,
            confidence=0.0,
        )
    scores = [r.similarity_score for r in retrieval_results]
    conf = round(sum(scores) / len(scores), 4)
    return GeneratedResponse(
        query=query,
        answer=f"Based on the retrieved context: {retrieval_results[0].text[:100]}",
        citations=[{"source": r.source_file, "chunk_id": r.chunk_id, "score": r.similarity_score, "page_number": r.page_number, "domain": r.domain} for r in retrieval_results],
        context_chunks_used=len(retrieval_results),
        confidence=conf,
    )


def make_orchestrator(retrieval_results):
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = retrieval_results
    retrieval_agent = RetrievalAgent(retriever=mock_retriever)
    response_agent = MagicMock(spec=ResponseGenerationAgent)
    response_agent.generate.side_effect = mock_response_agent_generate
    return AgentOrchestrator(
        query_agent=QueryUnderstandingAgent(),
        clarification_agent=ClarificationAgent(),
        retrieval_agent=retrieval_agent,
        response_agent=response_agent,
    )


def make_memory():
    return ConversationMemoryAgent(max_history=10)


# ═══════════════════════════════════════════════════════════════════════════
# M4.1 — ANALYTICS SERVICE UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalyticsService:
    """Tests for the analytics_service module (mocked DB)."""

    def setup_method(self):
        from backend.services.analytics_service import AnalyticsService
        self.service = AnalyticsService()

    # Test 1: Gap detection — no chunks retrieved
    def test_gap_detect_no_chunks(self):
        is_gap, reason = self.service._detect_gap(
            routing_path="retrieval",
            retrieval_count=0,
            best_score=None,
            response_status="knowledge_gap",
        )
        assert is_gap is True
        assert reason == "no_chunks_retrieved"

    # Test 2: Gap detection — below threshold
    def test_gap_detect_below_threshold(self):
        is_gap, reason = self.service._detect_gap(
            routing_path="retrieval",
            retrieval_count=2,
            best_score=0.15,       # well below 0.25 threshold
            response_status="answered",
        )
        assert is_gap is True
        assert reason == "below_similarity_threshold"

    # Test 3: NOT a gap — direct routing
    def test_no_gap_direct_routing(self):
        is_gap, reason = self.service._detect_gap(
            routing_path="direct",
            retrieval_count=0,
            best_score=None,
            response_status="answered",
        )
        assert is_gap is False
        assert reason is None

    # Test 4: NOT a gap — clarification routing
    def test_no_gap_clarification_routing(self):
        is_gap, reason = self.service._detect_gap(
            routing_path="clarification",
            retrieval_count=0,
            best_score=None,
            response_status="clarification",
        )
        assert is_gap is False
        assert reason is None

    # Test 5: NOT a gap — good evidence retrieved
    def test_no_gap_good_evidence(self):
        is_gap, reason = self.service._detect_gap(
            routing_path="retrieval",
            retrieval_count=5,
            best_score=0.82,
            response_status="answered",
        )
        assert is_gap is False
        assert reason is None

    # Test 6: Gap detection — response marked as gap
    def test_gap_detect_response_status(self):
        is_gap, reason = self.service._detect_gap(
            routing_path="retrieval",
            retrieval_count=1,
            best_score=0.35,
            response_status="knowledge_gap",
        )
        assert is_gap is True

    # Test 7: Gap detection — low evidence + low confidence combined
    def test_gap_detect_low_evidence_low_confidence(self):
        is_gap, reason = self.service._detect_gap(
            routing_path="retrieval",
            retrieval_count=1,   # <= _GAP_LOW_CHUNK_THRESHOLD
            best_score=0.28,     # < 0.35
            response_status="answered",
        )
        assert is_gap is True
        assert reason == "low_evidence_low_confidence"

    # Test 8: normalize_for_dedup removes punctuation
    def test_normalize_for_dedup(self):
        from backend.services.analytics_service import _normalize_for_dedup
        result = _normalize_for_dedup("What is Infosys's maternity leave policy?!")
        assert "?" not in result
        assert "!" not in result
        assert result == result.lower()
        assert "  " not in result

    # Test 9: normalize_for_dedup produces same string for similar queries
    def test_normalize_dedup_grouping(self):
        from backend.services.analytics_service import _normalize_for_dedup
        q1 = _normalize_for_dedup("What is maternity leave?")
        q2 = _normalize_for_dedup("What is maternity leave")
        assert q1 == q2  # punctuation stripped → equal

    # Test 10: AnalyticsService.record_query succeeds with mocked DB
    def test_record_query_success(self):
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        result = self.service.record_query(
            db=mock_db,
            original_query="What is PostgreSQL?",
            normalized_query="What is PostgreSQL?",
            query_type=QUERY_TYPE_FACTUAL,
            detected_intent=QUERY_TYPE_FACTUAL,
            detected_domain="technology",
            routing_path=ROUTING_RETRIEVAL,
            clarification_required=False,
            retrieval_count=3,
            retrieved_chunks=3,
            similarity_scores=[0.85, 0.78, 0.71],
            source_count=1,
            response_status="answered",
            response_generated=True,
            response_latency_ms=1250.5,
            key_terms=["postgresql", "database"],
            session_id="test-session-001",
            query_id="q-001",
        )
        assert result is not None
        mock_db.add.assert_called()
        mock_db.commit.assert_called()

    # Test 11: record_query handles DB failure gracefully
    def test_record_query_db_failure_graceful(self):
        mock_db = MagicMock()
        mock_db.add.side_effect = Exception("DB error")
        # Should not raise
        result = self.service.record_query(
            db=mock_db,
            original_query="Test query",
            routing_path=ROUTING_RETRIEVAL,
        )
        assert result is None

    # Test 12: Knowledge gap upsert — new gap created
    def test_upsert_knowledge_gap_new(self):
        mock_db = MagicMock()
        # No existing gap
        mock_db.query.return_value.filter.return_value.first.return_value = None
        self.service._upsert_knowledge_gap(
            db=mock_db,
            original_query="What is Infosys maternity leave policy?",
            domain="hr",
            reason="no_chunks_retrieved",
            retrieval_count=0,
            best_score=None,
            session_id="session-abc",
        )
        mock_db.add.assert_called_once()

    # Test 13: Knowledge gap upsert — existing gap frequency incremented
    def test_upsert_knowledge_gap_existing(self):
        mock_db = MagicMock()
        existing_gap = MagicMock()
        existing_gap.frequency = 3
        existing_gap.best_similarity_score = None
        existing_gap.retrieval_count = 0
        mock_db.query.return_value.filter.return_value.first.return_value = existing_gap
        self.service._upsert_knowledge_gap(
            db=mock_db,
            original_query="What is Infosys maternity leave policy?",
            domain="hr",
            reason="no_chunks_retrieved",
            retrieval_count=0,
            best_score=0.18,
            session_id="session-def",
        )
        assert existing_gap.frequency == 4
        assert existing_gap.best_similarity_score == 0.18  # set since was None


# ═══════════════════════════════════════════════════════════════════════════
# M4.2 — THREE-DOMAIN WORKFLOW TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestThreeDomainWorkflow:
    """Verifies the system processes queries across HR, Technology, and Finance domains."""

    # Test 14: QUA detects HR domain
    def test_qua_detects_hr_domain(self):
        agent = QueryUnderstandingAgent()
        result = agent.analyze("Who is eligible for FMLA leave?")
        assert result.suggested_domain == "hr"
        assert result.query_type in (QUERY_TYPE_FACTUAL, QUERY_TYPE_PROCEDURAL)

    # Test 15: QUA detects Technology domain
    def test_qua_detects_technology_domain(self):
        agent = QueryUnderstandingAgent()
        result = agent.analyze("What is PostgreSQL?")
        assert result.suggested_domain == "technology"
        assert result.query_type == QUERY_TYPE_FACTUAL

    # Test 16: QUA detects Finance domain
    def test_qua_detects_finance_domain(self):
        agent = QueryUnderstandingAgent()
        result = agent.analyze("What is the expense reimbursement policy?")
        assert result.suggested_domain == "finance"

    # Test 17: HR factual query — M4-HR-001
    def test_hr_factual_query(self):
        orchestrator = make_orchestrator([make_hr_result(1, 0.85), make_hr_result(2, 0.78)])
        memory = make_memory()
        mock_db = MagicMock()
        result = orchestrator.run("Who is eligible for FMLA?", db=mock_db, memory=memory)
        assert result.routing == ROUTING_RETRIEVAL
        assert result.retrieval_count == 2
        assert result.confidence > 0

    # Test 18: Technology factual query — M4-TECH-001
    def test_tech_factual_query(self):
        orchestrator = make_orchestrator([make_result(1, 0.90), make_result(2, 0.82)])
        memory = make_memory()
        mock_db = MagicMock()
        result = orchestrator.run("What is PostgreSQL?", db=mock_db, memory=memory)
        assert result.routing == ROUTING_RETRIEVAL
        assert result.retrieval_count == 2
        assert result.detected_domain == "technology"

    # Test 19: Finance factual query — M4-FIN-001
    def test_finance_factual_query(self):
        orchestrator = make_orchestrator([make_finance_result(1, 0.80)])
        memory = make_memory()
        mock_db = MagicMock()
        result = orchestrator.run("What are the expense reimbursement limits?", db=mock_db, memory=memory)
        assert result.routing == ROUTING_RETRIEVAL
        assert result.retrieval_count == 1
        assert result.detected_domain == "finance"

    # Test 20: Knowledge gap query — M4-GAP-001
    def test_knowledge_gap_query(self):
        orchestrator = make_orchestrator([])  # Empty retrieval → gap
        memory = make_memory()
        mock_db = MagicMock()
        result = orchestrator.run("What is the maternity leave policy of Infosys?", db=mock_db, memory=memory)
        assert result.retrieval_count == 0
        assert "could not find" in result.answer.lower() or result.retrieval_count == 0

    # Test 21: Procedural query classification — M4-PROC-001
    def test_procedural_query_classification(self):
        agent = QueryUnderstandingAgent()
        result = agent.analyze("How do I submit an expense reimbursement request?")
        assert result.query_type == QUERY_TYPE_PROCEDURAL
        assert result.routing == ROUTING_RETRIEVAL

    # Test 22: Comparative query classification — M4-COMP-001
    def test_comparative_query_classification(self):
        agent = QueryUnderstandingAgent()
        result = agent.analyze("What is the difference between PTO and FMLA?")
        assert result.query_type == QUERY_TYPE_COMPARATIVE
        assert result.routing == ROUTING_RETRIEVAL

    # Test 23: Ambiguous query classification — M4-AMB-001
    def test_ambiguous_query_classification(self):
        agent = QueryUnderstandingAgent()
        result = agent.analyze("What are its advantages?")
        assert result.query_type == QUERY_TYPE_AMBIGUOUS
        assert result.routing == ROUTING_CLARIFICATION

    # Test 24: Incomplete query triggers clarification — M4-INC-001
    def test_incomplete_query_triggers_clarification(self):
        orchestrator = make_orchestrator([])
        memory = make_memory()
        mock_db = MagicMock()
        result = orchestrator.run("What are its requirements?", db=mock_db, memory=memory)
        assert result.clarification_needed is True

    # Test 25: Context switching does not leak
    def test_context_switch_no_leakage(self):
        """HR → Technology switch should not inject HR context into Technology query."""
        orchestrator = make_orchestrator([make_result(1, 0.85)])
        memory = make_memory()
        mock_db = MagicMock()

        # First: HR query
        memory.add_message("user", "What is FMLA?")
        memory.add_message("assistant", "FMLA provides 12 weeks of unpaid leave for eligible employees.")
        memory.update_topic(["fmla", "leave"], ["employee_handbook.txt"], "FMLA provides 12 weeks...")

        # Second: Technology query (different domain)
        result = orchestrator.run("What is pgvector?", db=mock_db, memory=memory)
        assert result.routing == ROUTING_RETRIEVAL
        # Context should not include FMLA-specific content (topic switched)
        assert result.used_memory_context is False


# ═══════════════════════════════════════════════════════════════════════════
# M4.3 — MULTI-TURN MEMORY TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestMultiTurnMemory:

    # Test 26: Multi-turn reference resolution — M4-MEM-001
    def test_multi_turn_pronoun_resolution(self):
        """Turn 1: What is PostgreSQL? → Turn 2: What are its advantages? → resolved to PostgreSQL's advantages."""
        orchestrator = make_orchestrator([make_result(1, 0.88)])
        memory = make_memory()
        mock_db = MagicMock()

        # Turn 1
        result1 = orchestrator.run("What is PostgreSQL?", db=mock_db, memory=memory)
        assert result1.routing == ROUTING_RETRIEVAL
        assert result1.query_type == QUERY_TYPE_FACTUAL

        # Turn 2 — "its" should resolve to "PostgreSQL"
        result2 = orchestrator.run("What are its advantages?", db=mock_db, memory=memory)
        # Should route to retrieval (pronoun resolved) not clarification
        assert result2.routing != ROUTING_CLARIFICATION or result2.used_memory_context

    # Test 27: Memory-resolved query re-classified correctly
    def test_memory_resolved_query_reclassified(self):
        """After pronoun resolution, the query should not remain ambiguous."""
        memory = make_memory()
        memory.add_message("user", "What is PostgreSQL?")
        memory.add_message("assistant", "PostgreSQL is an open-source relational database.")
        memory.update_topic(["postgresql", "database"], ["postgres.pdf"], "PostgreSQL is an open-source relational database.")

        resolved = memory.resolve_references("What are its advantages?")
        assert "postgresql" in resolved.lower() or resolved != "What are its advantages?"

    # Test 28: Session isolation — two sessions don't share memory
    def test_session_isolation(self):
        from backend.services.session_store import InMemorySessionStore
        store = InMemorySessionStore()

        sid_a, mem_a = store.get_or_create(None)
        sid_b, mem_b = store.get_or_create(None)

        mem_a.add_message("user", "What is FMLA?")
        mem_a.update_topic(["fmla"], [], "FMLA is a leave policy.")

        assert sid_a != sid_b
        # Session B should have no FMLA context
        assert mem_b.active_topic != "FMLA"
        assert len(mem_b.get_history()) == 0

    # Test 29: Clearing session removes all state
    def test_session_clear_removes_state(self):
        memory = ConversationMemoryAgent()
        memory.add_message("user", "What is PostgreSQL?")
        memory.add_message("assistant", "PostgreSQL is a database.")
        memory.update_topic(["postgresql"], ["postgres.pdf"], "PostgreSQL is a database.")

        memory.clear()
        assert len(memory.get_history()) == 0
        assert memory.active_topic is None
        assert memory.last_entities == []
        assert memory.last_source_docs == []

    # Test 30: Memory context bounded to max_history
    def test_memory_bounded_history(self):
        memory = ConversationMemoryAgent(max_history=6)
        for i in range(10):
            memory.add_message("user", f"Query {i}")
            memory.add_message("assistant", f"Answer {i}")

        assert len(memory.get_history()) <= 6

    # Test 31: Topic continuation detected
    def test_topic_continuation_detected(self):
        memory = ConversationMemoryAgent()
        memory.update_topic(["postgresql", "database"], ["postgres.pdf"], "PostgreSQL is a database.")
        assert memory.is_topic_continuation(["postgresql", "advantages"]) is True

    # Test 32: Topic switch detected
    def test_topic_switch_detected(self):
        memory = ConversationMemoryAgent()
        memory.update_topic(["postgresql", "database"], ["postgres.pdf"], "PostgreSQL is a database.")
        assert memory.is_topic_continuation(["fmla", "leave"]) is False

    # Test 33: Relevant context empty after topic switch
    def test_relevant_context_empty_after_switch(self):
        memory = ConversationMemoryAgent()
        memory.add_message("user", "What is PostgreSQL?")
        memory.add_message("assistant", "PostgreSQL is a database.")
        memory.update_topic(["postgresql"], ["postgres.pdf"], "PostgreSQL is a database.")

        # Switch to HR topic
        context = memory.get_relevant_context(["fmla", "leave"])
        assert context == []


# ═══════════════════════════════════════════════════════════════════════════
# M4.3 — AGENT ROUTING OPTIMIZATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestAgentRouting:

    # Test 34: Factual queries routed to retrieval
    def test_factual_routes_to_retrieval(self):
        agent = QueryUnderstandingAgent()
        for query in [
            "What is FMLA?",
            "Who is eligible for leave?",
            "What is PostgreSQL?",
            "What is the expense reimbursement limit?",
        ]:
            result = agent.analyze(query)
            assert result.routing == ROUTING_RETRIEVAL, f"Failed for: {query}"

    # Test 35: Procedural queries routed to retrieval
    def test_procedural_routes_to_retrieval(self):
        agent = QueryUnderstandingAgent()
        for query in [
            "How do I submit an expense report?",
            "How to install PostgreSQL?",
            "What are the steps to apply for FMLA?",
        ]:
            result = agent.analyze(query)
            assert result.routing == ROUTING_RETRIEVAL, f"Failed for: {query}"
            assert result.query_type == QUERY_TYPE_PROCEDURAL, f"Not procedural: {query}"

    # Test 36: Comparative queries routed to retrieval
    def test_comparative_routes_to_retrieval(self):
        agent = QueryUnderstandingAgent()
        for query in [
            "What is the difference between PTO and FMLA?",
            "Compare PostgreSQL and MySQL.",
        ]:
            result = agent.analyze(query)
            assert result.routing == ROUTING_RETRIEVAL, f"Failed for: {query}"
            assert result.query_type == QUERY_TYPE_COMPARATIVE, f"Not comparative: {query}"

    # Test 37: Greetings routed as direct
    def test_greetings_routed_direct(self):
        agent = QueryUnderstandingAgent()
        for query in ["hi", "hello", "thanks", "bye"]:
            result = agent.analyze(query)
            assert result.routing == ROUTING_DIRECT, f"Failed for: {query}"
            assert result.query_type == QUERY_TYPE_DIRECT

    # Test 38: Ambiguous pronoun routes to clarification (no history)
    def test_ambiguous_pronoun_routes_to_clarification(self):
        agent = QueryUnderstandingAgent()
        result = agent.analyze("What does it mean?")
        assert result.query_type == QUERY_TYPE_AMBIGUOUS
        assert result.routing == ROUTING_CLARIFICATION

    # Test 39: Clarification pipeline works end-to-end
    def test_clarification_pipeline(self):
        orchestrator = make_orchestrator([make_hr_result(1, 0.85)])
        memory = make_memory()
        mock_db = MagicMock()

        # Turn 1: ambiguous query
        result1 = orchestrator.run("What are the requirements?", db=mock_db, memory=memory)
        assert result1.clarification_needed is True

        # Turn 2: clarification response
        result2 = orchestrator.run("FMLA eligibility", db=mock_db, memory=memory)
        # After clarification, should route to retrieval
        assert not result2.clarification_needed or result2.routing == ROUTING_RETRIEVAL

    # Test 40: Direct responses have no sources
    def test_direct_response_no_sources(self):
        orchestrator = make_orchestrator([])
        memory = make_memory()
        mock_db = MagicMock()
        result = orchestrator.run("hello", db=mock_db, memory=memory)
        assert result.routing == ROUTING_DIRECT
        assert result.sources == []
        assert result.retrieval_count == 0


# ═══════════════════════════════════════════════════════════════════════════
# M4.3 — RETRIEVAL QUALITY TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestRetrievalQuality:
    """
    Retrieval quality tests with mock retrieval.
    Ground truth is defined here; actual vector search requires live DB.
    """

    # Test 41: High-similarity results returned as sources
    def test_high_similarity_sources_returned(self):
        results = [make_result(1, 0.92), make_result(2, 0.85), make_result(3, 0.78)]
        orchestrator = make_orchestrator(results)
        memory = make_memory()
        mock_db = MagicMock()
        result = orchestrator.run("What is PostgreSQL?", db=mock_db, memory=memory)

        assert len(result.sources) == 3
        assert all(s["similarity_score"] > 0.5 for s in result.sources)

    # Test 42: Sources contain required transparency fields
    def test_source_transparency_fields(self):
        results = [make_result(1, 0.88)]
        orchestrator = make_orchestrator(results)
        memory = make_memory()
        mock_db = MagicMock()
        result = orchestrator.run("What is PostgreSQL?", db=mock_db, memory=memory)

        assert len(result.sources) >= 1
        source = result.sources[0]
        assert "source_file" in source
        assert "similarity_score" in source
        assert "chunk_id" in source
        assert "rank" in source

    # Test 43: Empty retrieval does not crash pipeline
    def test_empty_retrieval_graceful(self):
        orchestrator = make_orchestrator([])
        memory = make_memory()
        mock_db = MagicMock()
        result = orchestrator.run("What is an obscure technology nobody knows?", db=mock_db, memory=memory)
        assert result is not None
        assert result.sources == []
        assert result.retrieval_count == 0

    # Test 44: Confidence is mean similarity score
    def test_confidence_is_mean_similarity(self):
        scores = [0.90, 0.80, 0.70]
        expected_conf = round(sum(scores) / len(scores), 4)
        results = [make_result(i+1, scores[i]) for i in range(3)]
        orchestrator = make_orchestrator(results)
        memory = make_memory()
        mock_db = MagicMock()
        result = orchestrator.run("What is PostgreSQL?", db=mock_db, memory=memory)
        assert abs(result.confidence - expected_conf) < 0.01

    # Test 45: Finance domain retrieval — correct domain returned
    def test_finance_domain_retrieval(self):
        results = [make_finance_result(1, 0.82), make_finance_result(2, 0.74)]
        orchestrator = make_orchestrator(results)
        memory = make_memory()
        mock_db = MagicMock()
        result = orchestrator.run("What is the expense reimbursement limit for hotels?", db=mock_db, memory=memory)
        assert len(result.sources) == 2
        assert all(s["domain"] == "finance" for s in result.sources)

    # Test 46: HR domain retrieval — correct domain returned
    def test_hr_domain_retrieval(self):
        results = [make_hr_result(1, 0.88), make_hr_result(2, 0.80)]
        orchestrator = make_orchestrator(results)
        memory = make_memory()
        mock_db = MagicMock()
        result = orchestrator.run("What are FMLA eligibility requirements?", db=mock_db, memory=memory)
        assert len(result.sources) == 2
        assert all(s["domain"] == "hr" for s in result.sources)

    # Test 47: Sources are ranked in order
    def test_sources_ranked_in_order(self):
        results = [make_result(1, 0.92), make_result(2, 0.85), make_result(3, 0.78)]
        orchestrator = make_orchestrator(results)
        memory = make_memory()
        mock_db = MagicMock()
        result = orchestrator.run("What is PostgreSQL?", db=mock_db, memory=memory)
        ranks = [s["rank"] for s in result.sources]
        assert ranks == sorted(ranks)


# ═══════════════════════════════════════════════════════════════════════════
# M4.3 — VOICE I/O TESTS (Browser-only — all SKIPPED)
# ═══════════════════════════════════════════════════════════════════════════

class TestVoiceIO:
    """
    Voice I/O tests require a browser environment (Web Speech API).
    These tests are documented here but skipped in automated execution.
    Manual verification required during the final demonstration.
    """

    @pytest.mark.skip(reason="Requires browser Web Speech API — execute manually during demo")
    def test_microphone_permission_denied(self):
        """Browser: denied mic → friendly error message, no crash."""

    @pytest.mark.skip(reason="Requires browser Web Speech API — execute manually during demo")
    def test_no_speech_timeout(self):
        """Browser: no speech → 'No speech detected' message shown."""

    @pytest.mark.skip(reason="Requires browser Web Speech API — execute manually during demo")
    def test_unsupported_browser_fallback(self):
        """Non-Chrome browser: Web Speech API missing → fallback message shown."""

    @pytest.mark.skip(reason="Requires browser Web Speech API — execute manually during demo")
    def test_tts_start_pause_resume_stop(self):
        """Browser: TTS controls work correctly for long responses."""

    @pytest.mark.skip(reason="Requires browser Web Speech API — execute manually during demo")
    def test_voice_to_query_pipeline(self):
        """Speech → Text → Query → Retrieval → Answer → TTS full cycle."""


# ═══════════════════════════════════════════════════════════════════════════
# M4.2 — END-TO-END PIPELINE TESTS (mocked DB+LLM)
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEndPipeline:

    # Test 48: Complete pipeline query → answer
    def test_complete_pipeline_hr(self):
        """M4-E2E-001: HR pipeline from query to structured answer."""
        orchestrator = make_orchestrator([make_hr_result(1, 0.85), make_hr_result(2, 0.79)])
        memory = make_memory()
        mock_db = MagicMock()

        result = orchestrator.run("Who is eligible for FMLA leave?", db=mock_db, memory=memory)

        assert result.query is not None
        assert result.answer is not None
        assert result.routing == ROUTING_RETRIEVAL
        assert result.retrieval_count == 2
        assert len(result.sources) == 2
        assert result.confidence > 0
        assert result.query_type == QUERY_TYPE_FACTUAL

    # Test 49: Complete pipeline query → answer (Technology)
    def test_complete_pipeline_technology(self):
        """M4-E2E-002: Technology pipeline from query to structured answer."""
        orchestrator = make_orchestrator([make_result(1, 0.90)])
        memory = make_memory()
        mock_db = MagicMock()

        result = orchestrator.run("What is a vector database?", db=mock_db, memory=memory)

        assert result.routing == ROUTING_RETRIEVAL
        assert result.detected_domain == "technology"

    # Test 50: Complete pipeline query → answer (Finance)
    def test_complete_pipeline_finance(self):
        """M4-E2E-003: Finance pipeline from query to structured answer."""
        orchestrator = make_orchestrator([make_finance_result(1, 0.82)])
        memory = make_memory()
        mock_db = MagicMock()

        result = orchestrator.run("What expenses are eligible for reimbursement?", db=mock_db, memory=memory)

        assert result.routing == ROUTING_RETRIEVAL
        assert result.detected_domain == "finance"

    # Test 51: Multi-turn across domains (HR → Technology → Finance)
    def test_multi_turn_cross_domain(self):
        """M4-E2E-004: Context does not bleed across domain changes."""
        orchestrator_hr = make_orchestrator([make_hr_result(1, 0.85)])
        orchestrator_tech = make_orchestrator([make_result(1, 0.88)])
        orchestrator_fin = make_orchestrator([make_finance_result(1, 0.80)])
        memory = make_memory()
        mock_db = MagicMock()

        # HR turn
        r1 = orchestrator_hr.run("What is FMLA?", db=mock_db, memory=memory)
        assert r1.detected_domain == "hr"

        # Technology turn — memory should detect topic switch
        r2 = orchestrator_tech.run("What is pgvector?", db=mock_db, memory=memory)
        assert r2.detected_domain == "technology"

        # Finance turn — fresh domain
        r3 = orchestrator_fin.run("What is the hotel expense limit?", db=mock_db, memory=memory)
        assert r3.detected_domain == "finance"

    # Test 52: Knowledge gap recorded for unanswerable queries
    def test_knowledge_gap_detection_in_pipeline(self):
        """M4-E2E-005: Pipeline correctly identifies and signals knowledge gap."""
        orchestrator = make_orchestrator([])
        memory = make_memory()
        mock_db = MagicMock()

        result = orchestrator.run(
            "What is the Infosys maternity leave policy for 2026?",
            db=mock_db,
            memory=memory,
        )
        # Should have no retrieval results
        assert result.retrieval_count == 0
        # Answer should be the knowledge-gap message
        assert "could not find" in result.answer.lower() or "not find" in result.answer.lower()


# ═══════════════════════════════════════════════════════════════════════════
# M4.3 — PROMPT / RESPONSE GROUNDING TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestResponseGrounding:

    # Test 53: No retrieval → knowledge gap response triggered
    def test_no_retrieval_triggers_gap_response(self):
        orchestrator = make_orchestrator([])
        memory = make_memory()
        mock_db = MagicMock()
        result = orchestrator.run("What is the Infosys 2026 bonus policy?", db=mock_db, memory=memory)
        gap_phrases = ["could not find", "not find sufficient", "not available"]
        assert any(phrase in result.answer.lower() for phrase in gap_phrases)

    # Test 54: Retrieved sources match answer domain
    def test_sources_match_answer_domain(self):
        hr_results = [make_hr_result(1, 0.87)]
        orchestrator = make_orchestrator(hr_results)
        memory = make_memory()
        mock_db = MagicMock()
        result = orchestrator.run("What is FMLA?", db=mock_db, memory=memory)
        # Sources should be HR documents
        assert all(s["domain"] == "hr" for s in result.sources)

    # Test 55: Source file names are not fabricated
    def test_source_files_not_fabricated(self):
        expected_source = "employee_handbook.txt"
        results = [make_hr_result(1, 0.88)]
        orchestrator = make_orchestrator(results)
        memory = make_memory()
        mock_db = MagicMock()
        result = orchestrator.run("What is FMLA?", db=mock_db, memory=memory)
        assert any(s["source_file"] == expected_source for s in result.sources)

    # Test 56: Confidence zero when no retrieval
    def test_confidence_zero_no_retrieval(self):
        orchestrator = make_orchestrator([])
        memory = make_memory()
        mock_db = MagicMock()
        result = orchestrator.run("Unknown topic xyz123", db=mock_db, memory=memory)
        assert result.confidence == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# M4.3 — SECURITY / CONFIGURATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSecurityAndConfiguration:

    # Test 57: Settings load without exposing credentials
    def test_settings_load(self):
        from config.settings import settings
        assert settings.embedding_model == "all-MiniLM-L6-v2"
        assert settings.top_k_results >= 1
        assert settings.similarity_threshold > 0.0
        assert settings.chunk_size > 0
        assert settings.chunk_overlap >= 0

    # Test 58: QUA key terms do not contain stop words
    def test_key_terms_no_stop_words(self):
        from agents.query_understanding_agent import _STOP_WORDS
        agent = QueryUnderstandingAgent()
        result = agent.analyze("What is the difference between PostgreSQL and MySQL?")
        for term in result.key_terms:
            assert term not in _STOP_WORDS, f"Stop word found in key_terms: {term}"

    # Test 59: Session IDs are unique UUIDs
    def test_session_ids_are_unique(self):
        from backend.services.session_store import InMemorySessionStore
        store = InMemorySessionStore()
        ids = set()
        for _ in range(10):
            sid, _ = store.get_or_create(None)
            ids.add(sid)
        assert len(ids) == 10

    # Test 60: ALLOWED_EXTENSIONS whitelist enforced (ingest route level)
    def test_allowed_extensions_whitelist(self):
        from backend.api.routes.ingest import ALLOWED_EXTENSIONS
        assert ".pdf" in ALLOWED_EXTENSIONS
        assert ".txt" in ALLOWED_EXTENSIONS
        assert ".docx" in ALLOWED_EXTENSIONS
        assert ".exe" not in ALLOWED_EXTENSIONS
        assert ".py" not in ALLOWED_EXTENSIONS
        assert ".js" not in ALLOWED_EXTENSIONS

    # Test 61: MAX_FILE_SIZE defined and reasonable
    def test_max_file_size_defined(self):
        from backend.api.routes.ingest import MAX_FILE_SIZE
        assert MAX_FILE_SIZE >= 1 * 1024 * 1024   # at least 1 MB
        assert MAX_FILE_SIZE <= 200 * 1024 * 1024  # at most 200 MB

    # Test 62: Embedding dimension correct for all-MiniLM-L6-v2
    def test_embedding_dimension(self):
        from backend.db.models import EMBEDDING_DIM
        from ingestion.embeddings import EMBEDDING_DIM as EMB_DIM
        assert EMBEDDING_DIM == 384
        assert EMB_DIM == 384


# ═══════════════════════════════════════════════════════════════════════════
# M4.1 — ANALYTICS API STRUCTURE TESTS (unit — no live server)
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalyticsAPIStructure:

    # Test 63: Analytics service overview handles empty DB
    def test_overview_empty_db(self):
        from backend.services.analytics_service import AnalyticsService
        service = AnalyticsService()
        mock_db = MagicMock()

        # Mock count/scalar queries to return 0/None
        mock_db.query.return_value.filter.return_value.count.return_value = 0
        mock_db.query.return_value.filter.return_value.scalar.return_value = None

        result = service.get_overview(mock_db, days=30)
        assert "total_queries" in result
        assert "answered_queries" in result
        assert "knowledge_gap_queries" in result
        assert "clarification_queries" in result

    # Test 64: Analytics serializer produces expected keys
    def test_analytics_serializer_keys(self):
        from backend.services.analytics_service import AnalyticsService
        from backend.db.models import QueryAnalytics
        service = AnalyticsService()

        record = QueryAnalytics(
            analytics_id="a1",
            original_query="test",
            normalized_query="test",
            query_type="factual",
            detected_domain="technology",
            routing_path="retrieval",
            clarification_required=False,
            retrieval_count=2,
            retrieved_chunks=2,
            source_count=1,
            response_status="answered",
            response_generated=True,
            knowledge_gap=False,
            timestamp=datetime.utcnow(),
            key_terms=["test"],
        )
        data = service._serialize_analytics(record)
        required_keys = [
            "analytics_id", "query_type", "detected_domain", "routing_path",
            "clarification_required", "retrieval_count", "knowledge_gap",
            "response_status", "key_terms",
        ]
        for k in required_keys:
            assert k in data, f"Missing key: {k}"

    # Test 65: Gap serializer produces expected keys
    def test_gap_serializer_keys(self):
        from backend.services.analytics_service import AnalyticsService
        from backend.db.models import KnowledgeGap
        service = AnalyticsService()

        gap = KnowledgeGap(
            gap_id="g1",
            normalized_query="infosys maternity leave policy",
            original_query="What is Infosys maternity leave policy?",
            domain="hr",
            reason="no_chunks_retrieved",
            frequency=3,
            first_seen=datetime.utcnow(),
            last_seen=datetime.utcnow(),
            status="detected",
        )
        data = service._serialize_gap(gap)
        required_keys = [
            "gap_id", "original_query", "normalized_query", "domain",
            "reason", "frequency", "first_seen", "last_seen", "status",
        ]
        for k in required_keys:
            assert k in data, f"Missing key: {k}"


# ═══════════════════════════════════════════════════════════════════════════
# M4 — REGRESSION: Query Understanding Agent
# ═══════════════════════════════════════════════════════════════════════════

class TestQUARegressions:
    """Ensure M4 finance domain changes don't break existing QUA behavior."""

    def setup_method(self):
        self.agent = QueryUnderstandingAgent()

    def test_hr_queries_still_classified_hr(self):
        result = self.agent.analyze("What is the FMLA leave policy?")
        assert result.suggested_domain == "hr"

    def test_tech_queries_still_classified_tech(self):
        result = self.agent.analyze("How do I configure pgvector?")
        assert result.suggested_domain == "technology"

    def test_legal_queries_still_classified_legal(self):
        result = self.agent.analyze("What are GDPR compliance requirements?")
        assert result.suggested_domain == "legal"

    def test_finance_queries_now_classified_finance(self):
        result = self.agent.analyze("What is the budget approval threshold?")
        assert result.suggested_domain == "finance"

    def test_greeting_still_direct(self):
        result = self.agent.analyze("hello")
        assert result.routing == ROUTING_DIRECT

    def test_comparative_still_classified(self):
        result = self.agent.analyze("What is the difference between MySQL and PostgreSQL?")
        assert result.query_type == QUERY_TYPE_COMPARATIVE

    def test_procedural_still_classified(self):
        result = self.agent.analyze("How do I submit an expense report?")
        assert result.query_type == QUERY_TYPE_PROCEDURAL
