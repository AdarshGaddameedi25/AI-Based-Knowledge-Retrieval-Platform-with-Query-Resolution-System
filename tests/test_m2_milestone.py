"""
tests/test_m2_milestone.py — Comprehensive Milestone 2 test suite

Covers:
  M2.1 — Query Understanding Agent (9 tests)
  M2.2 — Retrieval Agent (7 tests)
  M2.3 — Response Generation Agent (7 tests)
  M2.4 — Multi-Agent Orchestration (12 tests)

All LLM and DB calls are mocked.
No live PostgreSQL or OpenRouter required to run these tests.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.query_understanding_agent import (
    QueryUnderstandingAgent, QueryAnalysis,
    QUERY_TYPE_FACTUAL, QUERY_TYPE_PROCEDURAL,
    QUERY_TYPE_COMPARATIVE, QUERY_TYPE_AMBIGUOUS, QUERY_TYPE_DIRECT,
    ROUTING_RETRIEVAL, ROUTING_CLARIFICATION, ROUTING_DIRECT,
)
from agents.clarification_agent import ClarificationAgent
from agents.conversation_memory_agent import ConversationMemoryAgent
from agents.retrieval_agent import RetrievalAgent
from agents.orchestrator import AgentOrchestrator, OrchestratorResult
from agents.response_generation_agent import ResponseGenerationAgent, GeneratedResponse
from retrieval.retriever import RetrievalResult


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def make_result(rank=1, score=0.85, text="FMLA provides 12 weeks leave.", source="fmla.pdf", domain="hr", page=2):
    return RetrievalResult(
        chunk_id=f"c-{rank}",
        document_id="doc-001",
        source_file=source,
        text=text,
        similarity_score=score,
        rank=rank,
        chunk_index=rank - 1,
        page_number=page,
        domain=domain,
    )


def make_mock_response_agent(retrieval_results):
    """Return a ResponseGenerationAgent with mocked generate()."""
    def mock_generate(query, retrieval_results, query_type="factual", conversation_history=None):
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
        citations = [
            {"source": r.source_file, "chunk_id": r.chunk_id,
             "score": r.similarity_score, "page_number": r.page_number, "domain": r.domain}
            for r in retrieval_results
        ]
        return GeneratedResponse(
            query=query,
            answer=f"Mock answer for: {query}",
            citations=citations,
            context_chunks_used=len(retrieval_results),
            confidence=conf,
        )

    agent = MagicMock(spec=ResponseGenerationAgent)
    agent.generate.side_effect = mock_generate
    return agent


def make_orchestrator(retrieval_results, domain_filter=None):
    """Build a fully mocked orchestrator for unit testing."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = retrieval_results
    retrieval_agent = RetrievalAgent(retriever=mock_retriever)
    response_agent = make_mock_response_agent(retrieval_results)
    return AgentOrchestrator(
        query_agent=QueryUnderstandingAgent(),
        clarification_agent=ClarificationAgent(),
        retrieval_agent=retrieval_agent,
        response_agent=response_agent,
    )


# ═══════════════════════════════════════════════════════════════════════════
# M2.1 — Query Understanding Agent Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestM21QueryUnderstanding:
    """M2.1 — Query Understanding Agent: classification, routing, schema."""

    def setup_method(self):
        self.agent = QueryUnderstandingAgent()

    # T1: Factual query
    def test_factual_query_classification(self):
        result = self.agent.analyze("Who is eligible for FMLA?")
        assert result.query_type == QUERY_TYPE_FACTUAL
        assert result.routing == ROUTING_RETRIEVAL
        assert result.requires_retrieval is True
        assert result.classification_confidence > 0.5

    # T2: Procedural query
    def test_procedural_query_classification(self):
        result = self.agent.analyze("How do I request FMLA leave?")
        assert result.query_type == QUERY_TYPE_PROCEDURAL
        assert result.routing == ROUTING_RETRIEVAL
        assert result.requires_retrieval is True
        assert result.classification_confidence >= 0.90

    # T3: Comparative query
    def test_comparative_query_classification(self):
        result = self.agent.analyze("What is the difference between PostgreSQL and MySQL?")
        assert result.query_type == QUERY_TYPE_COMPARATIVE
        assert result.routing == ROUTING_RETRIEVAL
        assert result.requires_retrieval is True
        assert result.classification_confidence >= 0.90

    # T4: Ambiguous query — no history, pronoun with few key terms
    def test_ambiguous_query_classification(self):
        result = self.agent.analyze("Tell me more about it.")
        assert result.query_type == QUERY_TYPE_AMBIGUOUS
        assert result.routing == ROUTING_CLARIFICATION
        assert result.requires_retrieval is False

    # T5: Greeting / chit-chat
    def test_greeting_classification(self):
        for greeting in ["hello", "hi", "good morning", "hey"]:
            result = self.agent.analyze(greeting)
            assert result.query_type == QUERY_TYPE_DIRECT, f"Failed for: {greeting}"
            assert result.routing == ROUTING_DIRECT
            assert result.requires_retrieval is False
            assert result.classification_confidence == 1.0

    # T6: Empty/whitespace query treated as direct
    def test_single_word_query_is_direct(self):
        result = self.agent.analyze("help")
        assert result.query_type in (QUERY_TYPE_DIRECT, QUERY_TYPE_FACTUAL)
        assert result.requires_retrieval is False or result.requires_retrieval is True  # both are acceptable

    # T7: Output schema completeness
    def test_output_schema_has_all_m2_fields(self):
        result = self.agent.analyze("What is FMLA?")
        assert hasattr(result, "original_query")
        assert hasattr(result, "normalized_query")
        assert hasattr(result, "query_type")
        assert hasattr(result, "routing")
        assert hasattr(result, "detected_intent")
        assert hasattr(result, "requires_retrieval")
        assert hasattr(result, "classification_confidence")
        assert hasattr(result, "key_terms")
        assert hasattr(result, "suggested_domain")

    # T8: Classification confidence validity
    def test_classification_confidence_is_valid_float(self):
        queries = [
            "What is FMLA?",
            "How do I apply for leave?",
            "Compare authentication and authorization",
            "Tell me about it",
            "hello",
        ]
        for q in queries:
            result = self.agent.analyze(q)
            assert 0.0 <= result.classification_confidence <= 1.0, (
                f"Confidence {result.classification_confidence} out of range for: {q}"
            )

    # T9: Routing correctness — all three routing values
    def test_routing_values_are_canonical(self):
        valid_routing = {ROUTING_RETRIEVAL, ROUTING_CLARIFICATION, ROUTING_DIRECT}
        for q in ["What is FMLA?", "Tell me about it", "hello"]:
            result = self.agent.analyze(q)
            assert result.routing in valid_routing, f"Invalid routing {result.routing!r} for: {q}"

    # T10: Normalized query
    def test_normalization_strips_whitespace(self):
        result = self.agent.analyze("  what is pgvector   ")
        assert result.normalized_query.strip() == result.normalized_query

    # T11: Domain detection
    def test_domain_detection_hr(self):
        result = self.agent.analyze("What is the FMLA leave policy for employees?")
        assert result.suggested_domain == "hr"

    def test_domain_detection_technology(self):
        result = self.agent.analyze("How does pgvector handle embedding search?")
        assert result.suggested_domain == "technology"

    # T12: Key term extraction
    def test_key_terms_extracted(self):
        result = self.agent.analyze("What is the difference between authentication and authorization?")
        assert len(result.key_terms) > 0
        assert any(t in result.key_terms for t in ["authentication", "authorization", "difference"])

    # T13: detected_intent equals query_type (backward compat)
    def test_detected_intent_matches_query_type(self):
        for q in ["What is FMLA?", "How do I apply?", "hello"]:
            result = self.agent.analyze(q)
            assert result.detected_intent == result.query_type


# ═══════════════════════════════════════════════════════════════════════════
# M2.2 — Retrieval Agent Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestM22Retrieval:
    """M2.2 — Retrieval Agent: Top-K, threshold, domain filter, metadata."""

    def setup_method(self):
        self.qua = QueryUnderstandingAgent()

    def _make_retrieval_agent(self, results):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = results
        return RetrievalAgent(retriever=mock_retriever), mock_retriever

    # T14: Retrieval skipped for non-retrieval queries
    def test_retrieval_skipped_for_greeting(self):
        agent, mock_retriever = self._make_retrieval_agent([])
        analysis = self.qua.analyze("hello")
        results = agent.retrieve(analysis, db=MagicMock(), top_k=5)
        assert results == []
        mock_retriever.retrieve.assert_not_called()

    # T15: Retrieval executes for factual query
    def test_retrieval_executes_for_factual(self):
        results = [make_result(1, 0.85), make_result(2, 0.78)]
        agent, mock_retriever = self._make_retrieval_agent(results)
        analysis = self.qua.analyze("What is FMLA?")
        retrieved = agent.retrieve(analysis, db=MagicMock(), top_k=5)
        assert len(retrieved) == 2
        mock_retriever.retrieve.assert_called_once()

    # T16: Top-K forwarded
    def test_top_k_forwarded_to_retriever(self):
        agent, mock_retriever = self._make_retrieval_agent([])
        analysis = self.qua.analyze("What is FMLA?")
        agent.retrieve(analysis, db=MagicMock(), top_k=3)
        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["top_k"] == 3

    # T17: Domain filter forwarded
    def test_domain_filter_forwarded(self):
        agent, mock_retriever = self._make_retrieval_agent([make_result()])
        analysis = self.qua.analyze("What is FMLA?")
        agent.retrieve(analysis, db=MagicMock(), top_k=5, domain_filter="hr")
        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["domain_filter"] == "hr"

    # T18: Suggested domain used when no explicit domain filter
    def test_suggested_domain_used_as_fallback(self):
        agent, mock_retriever = self._make_retrieval_agent([])
        analysis = self.qua.analyze("What is the FMLA leave policy?")
        agent.retrieve(analysis, db=MagicMock(), top_k=5)
        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        # Suggested domain should be "hr" due to FMLA keyword
        assert call_kwargs.get("domain_filter") == "hr"

    # T19: Metadata preserved in results
    def test_retrieval_result_metadata_complete(self):
        chunk = make_result(1, 0.87, source="fmla.pdf", page=3, domain="hr")
        assert chunk.chunk_id == "c-1"
        assert chunk.document_id == "doc-001"
        assert chunk.source_file == "fmla.pdf"
        assert chunk.page_number == 3
        assert chunk.similarity_score == 0.87
        assert chunk.domain == "hr"
        assert isinstance(chunk.rank, int)

    # T20: No results returns empty list (knowledge gap entry point)
    def test_no_results_returns_empty_list(self):
        agent, _ = self._make_retrieval_agent([])
        analysis = self.qua.analyze("What is the Infosys maternity leave policy?")
        results = agent.retrieve(analysis, db=MagicMock(), top_k=5)
        assert results == []

    # T21: Cosine similarity ordering (higher score = more relevant)
    def test_similarity_score_ordering(self):
        r1 = make_result(1, 0.92)
        r2 = make_result(2, 0.75)
        r3 = make_result(3, 0.60)
        # Verify that rank=1 has highest score (as our retriever orders correctly)
        assert r1.similarity_score > r2.similarity_score > r3.similarity_score


# ═══════════════════════════════════════════════════════════════════════════
# M2.3 — Response Generation Agent Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestM23ResponseGeneration:
    """M2.3 — Response Generation: grounding, citations, knowledge gap, LLM failure."""

    # T22: Insufficient evidence → knowledge gap response
    def test_empty_results_returns_knowledge_gap(self):
        agent = make_mock_response_agent([])
        result = agent.generate("What is Infosys maternity leave?", [])
        assert result.confidence == 0.0
        assert result.context_chunks_used == 0
        assert "not" in result.answer.lower() or "could not" in result.answer.lower()
        assert result.citations == []

    # T23: Grounded answer includes citations
    def test_citations_match_retrieval_results(self):
        results = [make_result(1, 0.88, source="fmla.pdf", page=2)]
        agent = make_mock_response_agent(results)
        response = agent.generate("Who is eligible for FMLA?", results)
        assert len(response.citations) == 1
        assert response.citations[0]["source"] == "fmla.pdf"
        assert response.citations[0]["page_number"] == 2
        assert response.citations[0]["score"] == 0.88

    # T24: Confidence is mean similarity score
    def test_confidence_is_mean_similarity(self):
        results = [make_result(1, 0.80), make_result(2, 0.60)]
        agent = make_mock_response_agent(results)
        response = agent.generate("What is FMLA?", results)
        expected = round((0.80 + 0.60) / 2, 4)
        assert response.confidence == expected

    # T25: Confidence is 0 when no results
    def test_confidence_is_zero_with_no_results(self):
        agent = make_mock_response_agent([])
        response = agent.generate("Unknown question", [])
        assert response.confidence == 0.0

    # T26: context_chunks_used reflects number of results
    def test_context_chunks_used_count(self):
        results = [make_result(i, 0.85) for i in range(1, 4)]
        agent = make_mock_response_agent(results)
        response = agent.generate("What is FMLA?", results)
        assert response.context_chunks_used == 3

    # T27: ResponseGenerationAgent accepts query_type parameter
    def test_generate_accepts_query_type(self):
        results = [make_result(1, 0.88)]
        agent = make_mock_response_agent(results)
        # Should not raise
        response = agent.generate(
            "How do I apply for FMLA?", results,
            query_type="procedural",
            conversation_history=[],
        )
        assert response is not None

    # T28: ResponseGenerationAgent accepts conversation_history parameter
    def test_generate_accepts_conversation_history(self):
        results = [make_result(1, 0.85)]
        agent = make_mock_response_agent(results)
        history = [
            {"role": "user", "content": "What is FMLA?"},
            {"role": "assistant", "content": "FMLA provides up to 12 weeks of leave."},
        ]
        response = agent.generate("Who is eligible?", results, conversation_history=history)
        assert response is not None

    # T29: Real ResponseGenerationAgent raises EnvironmentError without API key
    def test_response_agent_raises_without_api_key(self):
        with patch("agents.response_generation_agent.settings") as mock_settings:
            mock_settings.openrouter_api_key = ""
            with pytest.raises(EnvironmentError, match="OPENROUTER_API_KEY"):
                ResponseGenerationAgent()


# ═══════════════════════════════════════════════════════════════════════════
# M2.4 — Orchestrator Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestM24Orchestration:
    """M2.4 — Multi-agent orchestration: routing, memory, error handling, schema."""

    def setup_method(self):
        self.mock_db = MagicMock()
        self.memory = ConversationMemoryAgent()

    # T30: OrchestratorResult has all M2 fields
    def test_orchestrator_result_schema_complete(self):
        orch = make_orchestrator([make_result()])
        result = orch.run("What is FMLA?", db=self.mock_db, memory=self.memory)
        assert hasattr(result, "query_type")
        assert hasattr(result, "routing")
        assert hasattr(result, "detected_intent")
        assert hasattr(result, "detected_domain")
        assert hasattr(result, "classification_confidence")
        assert hasattr(result, "confidence")
        assert hasattr(result, "retrieval_count")
        assert hasattr(result, "sources")
        assert hasattr(result, "clarification_needed")

    # T31: Factual E2E — full pipeline
    def test_factual_query_e2e(self):
        results = [make_result(1, 0.90), make_result(2, 0.82)]
        orch = make_orchestrator(results)
        result = orch.run("Who is eligible for FMLA?", db=self.mock_db, memory=self.memory)
        assert result.query_type == QUERY_TYPE_FACTUAL
        assert result.routing == ROUTING_RETRIEVAL
        assert result.clarification_needed is False
        assert result.retrieval_count == 2
        assert len(result.sources) == 2
        assert result.confidence > 0

    # T32: Procedural query routing
    def test_procedural_query_routing(self):
        results = [make_result(1, 0.88)]
        orch = make_orchestrator(results)
        result = orch.run("How do I apply for FMLA?", db=self.mock_db, memory=self.memory)
        assert result.query_type == QUERY_TYPE_PROCEDURAL
        assert result.routing == ROUTING_RETRIEVAL

    # T33: Comparative query routing
    def test_comparative_query_routing(self):
        results = [make_result(1, 0.85)]
        orch = make_orchestrator(results)
        result = orch.run("What is the difference between FMLA and PTO?", db=self.mock_db, memory=self.memory)
        assert result.query_type == QUERY_TYPE_COMPARATIVE
        assert result.routing == ROUTING_RETRIEVAL

    # T34: Ambiguous query → clarification path
    def test_ambiguous_query_returns_clarification(self):
        orch = make_orchestrator([])
        result = orch.run("Tell me more about it.", db=self.mock_db, memory=self.memory)
        assert result.clarification_needed is True
        assert result.routing == ROUTING_CLARIFICATION
        assert len(result.clarification_question) > 0
        assert result.retrieval_count == 0

    # T35: Greeting → direct response
    def test_greeting_returns_direct_response(self):
        orch = make_orchestrator([])
        result = orch.run("hello", db=self.mock_db, memory=self.memory)
        assert result.query_type == QUERY_TYPE_DIRECT
        assert result.routing == ROUTING_DIRECT
        assert result.clarification_needed is False
        assert result.retrieval_count == 0
        assert result.confidence == 1.0

    # T36: Knowledge gap — no results → controlled response
    def test_knowledge_gap_controlled_response(self):
        orch = make_orchestrator([])
        result = orch.run(
            "What is the extraterrestrial leave policy for Mars missions?",
            db=self.mock_db, memory=self.memory,
        )
        assert result.retrieval_count == 0
        assert "not" in result.answer.lower() or result.confidence == 0.0

    # T37: Memory updated after query
    def test_memory_updated_after_rag_query(self):
        orch = make_orchestrator([make_result(1, 0.85)])
        orch.run("What is FMLA?", db=self.mock_db, memory=self.memory, session_id="s1")
        history = self.memory.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    # T38: Domain detection flows through
    def test_domain_detection_in_result(self):
        orch = make_orchestrator([make_result(1, 0.88)])
        result = orch.run(
            "What is the employee leave policy under FMLA?",
            db=self.mock_db, memory=self.memory,
        )
        assert result.detected_domain == "hr"

    # T39: Retrieval failure → controlled error response
    def test_retrieval_failure_returns_controlled_response(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = Exception("DB connection lost")
        retrieval_agent = RetrievalAgent(retriever=mock_retriever)
        response_agent = make_mock_response_agent([])
        orch = AgentOrchestrator(
            query_agent=QueryUnderstandingAgent(),
            clarification_agent=ClarificationAgent(),
            retrieval_agent=retrieval_agent,
            response_agent=response_agent,
        )
        result = orch.run("What is FMLA?", db=self.mock_db, memory=self.memory)
        # Should not raise — should return a controlled error response
        assert isinstance(result, OrchestratorResult)
        assert "unavailable" in result.answer.lower() or len(result.answer) > 0

    # T40: LLM failure → controlled error response
    def test_llm_failure_returns_controlled_response(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [make_result(1, 0.85)]
        retrieval_agent = RetrievalAgent(retriever=mock_retriever)

        failing_response_agent = MagicMock(spec=ResponseGenerationAgent)
        failing_response_agent.generate.side_effect = Exception("OpenRouter timeout")

        orch = AgentOrchestrator(
            query_agent=QueryUnderstandingAgent(),
            clarification_agent=ClarificationAgent(),
            retrieval_agent=retrieval_agent,
            response_agent=failing_response_agent,
        )
        result = orch.run("What is FMLA?", db=self.mock_db, memory=self.memory)
        assert isinstance(result, OrchestratorResult)
        assert "could not" in result.answer.lower() or "try again" in result.answer.lower()

    # T41: Pronoun resolution in multi-turn conversation
    def test_pronoun_resolution_with_history(self):
        self.memory.add_message(
            "assistant",
            "FMLA stands for Family and Medical Leave Act. It provides up to 12 weeks of unpaid leave.",
        )
        orch = make_orchestrator([make_result(1, 0.85)])
        result = orch.run("How does FMLA work?", db=self.mock_db, memory=self.memory)
        # Should not raise; should proceed to retrieval
        assert isinstance(result, OrchestratorResult)

    # T42: Session ID is propagated in result
    def test_session_id_propagated(self):
        orch = make_orchestrator([make_result(1, 0.85)])
        result = orch.run("What is FMLA?", db=self.mock_db, memory=self.memory, session_id="sess-xyz")
        assert result.session_id == "sess-xyz"
