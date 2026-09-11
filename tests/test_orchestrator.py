"""
Tests for AgentOrchestrator — Milestone 2 unified pipeline.
All LLM calls and DB interactions are mocked.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.query_understanding_agent import QueryUnderstandingAgent, QueryAnalysis
from agents.clarification_agent import ClarificationAgent
from agents.conversation_memory_agent import ConversationMemoryAgent
from agents.retrieval_agent import RetrievalAgent
from agents.orchestrator import AgentOrchestrator, OrchestratorResult
from retrieval.retriever import RetrievalResult


def make_result(rank=1, score=0.85, text="FMLA provides 12 weeks leave.", source="fmla.pdf", domain="hr"):
    return RetrievalResult(
        chunk_id=f"c-{rank}",
        document_id="doc-001",
        source_file=source,
        text=text,
        similarity_score=score,
        rank=rank,
        chunk_index=rank - 1,
        page_number=None,
        domain=domain,
    )


# ── QueryUnderstandingAgent tests ─────────────────────────────────────────────

def test_qua_domain_detection_hr():
    agent = QueryUnderstandingAgent()
    result = agent.analyze("What is the FMLA leave policy for employees?")
    assert result.suggested_domain == "hr"


def test_qua_domain_detection_technology():
    agent = QueryUnderstandingAgent()
    result = agent.analyze("How does pgvector handle vector embedding search?")
    assert result.suggested_domain == "technology"


def test_qua_domain_detection_none():
    agent = QueryUnderstandingAgent()
    result = agent.analyze("hello")
    assert result.suggested_domain is None


def test_qua_requires_retrieval_for_knowledge_query():
    agent = QueryUnderstandingAgent()
    result = agent.analyze("What is the EEOC workplace discrimination policy?")
    assert result.requires_retrieval is True
    assert result.detected_intent in {"factual", "definitional", "general"}


def test_qua_no_retrieval_for_greeting():
    agent = QueryUnderstandingAgent()
    result = agent.analyze("good morning")
    assert result.requires_retrieval is False
    assert result.suggested_domain is None


# ── RetrievalAgent tests ──────────────────────────────────────────────────────

def test_retrieval_agent_skips_if_no_retrieval_needed():
    mock_retriever = MagicMock()
    agent = RetrievalAgent(retriever=mock_retriever)
    analysis = QueryUnderstandingAgent().analyze("hello")
    mock_db = MagicMock()
    results = agent.retrieve(analysis, db=mock_db, top_k=5)
    assert results == []
    mock_retriever.retrieve.assert_not_called()


def test_retrieval_agent_passes_domain_filter():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [make_result()]
    agent = RetrievalAgent(retriever=mock_retriever)
    analysis = QueryUnderstandingAgent().analyze("What is FMLA?")
    mock_db = MagicMock()
    results = agent.retrieve(analysis, db=mock_db, top_k=3, domain_filter="hr")
    mock_retriever.retrieve.assert_called_once()
    call_kwargs = mock_retriever.retrieve.call_args.kwargs
    assert call_kwargs["domain_filter"] == "hr"


# ── Orchestrator integration tests ───────────────────────────────────────────

def _make_orchestrator_with_mocked_llm(retrieval_results):
    """Helper: build orchestrator with mocked retriever and mocked LLM."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = retrieval_results

    retrieval_agent = RetrievalAgent(retriever=mock_retriever)

    with patch("agents.response_generation_agent.ResponseGenerationAgent.__init__", return_value=None):
        resp_agent = object.__new__(__import__("agents.response_generation_agent", fromlist=["ResponseGenerationAgent"]).ResponseGenerationAgent)
        resp_agent._headers = {}

    # Patch the LLM HTTP call
    from agents.response_generation_agent import ResponseGenerationAgent, GeneratedResponse

    def mock_generate(query, retrieval_results, query_type="factual", conversation_history=None):
        if not retrieval_results:
            return GeneratedResponse(query=query, answer="Not found.", citations=[], context_chunks_used=0, confidence=0.0)
        scores = [r.similarity_score for r in retrieval_results]
        conf = round(sum(scores) / len(scores), 4)
        return GeneratedResponse(
            query=query,
            answer=f"Answer for: {query}",
            citations=[{"source": r.source_file, "chunk_id": r.chunk_id, "score": r.similarity_score, "page_number": None, "domain": r.domain} for r in retrieval_results],
            context_chunks_used=len(retrieval_results),
            confidence=conf,
        )

    resp_agent.generate = mock_generate

    return AgentOrchestrator(
        query_agent=QueryUnderstandingAgent(),
        clarification_agent=ClarificationAgent(),
        retrieval_agent=retrieval_agent,
        response_agent=resp_agent,
    )


def test_orchestrator_greeting_returns_direct_response():
    orch = _make_orchestrator_with_mocked_llm([])
    memory = ConversationMemoryAgent()
    mock_db = MagicMock()

    result = orch.run("hello", db=mock_db, memory=memory, session_id="s1")
    assert result.clarification_needed is False
    assert result.retrieval_count == 0
    assert result.confidence == 1.0
    assert "assistant" in result.answer.lower() or "knowledge" in result.answer.lower()


def test_orchestrator_returns_clarification_for_ambiguous():
    orch = _make_orchestrator_with_mocked_llm([])
    memory = ConversationMemoryAgent()
    mock_db = MagicMock()

    result = orch.run("What does it mean?", db=mock_db, memory=memory, session_id="s2")
    assert result.clarification_needed is True
    assert len(result.clarification_question) > 0
    assert result.retrieval_count == 0


def test_orchestrator_full_rag_pipeline():
    results = [make_result(1, 0.90), make_result(2, 0.80)]
    orch = _make_orchestrator_with_mocked_llm(results)
    memory = ConversationMemoryAgent()
    mock_db = MagicMock()

    result = orch.run(
        "What is the FMLA leave policy?",
        db=mock_db,
        memory=memory,
        session_id="s3",
    )
    assert result.clarification_needed is False
    assert result.retrieval_count == 2
    assert result.confidence > 0
    assert len(result.sources) == 2
    assert result.sources[0]["source_file"] == "fmla.pdf"


def test_orchestrator_detects_domain():
    results = [make_result(1, 0.88)]
    orch = _make_orchestrator_with_mocked_llm(results)
    memory = ConversationMemoryAgent()
    mock_db = MagicMock()

    result = orch.run(
        "What is the employee leave policy under FMLA?",
        db=mock_db,
        memory=memory,
    )
    assert result.detected_domain == "hr"


def test_orchestrator_updates_memory():
    results = [make_result(1, 0.85)]
    orch = _make_orchestrator_with_mocked_llm(results)
    memory = ConversationMemoryAgent()
    mock_db = MagicMock()

    orch.run("What is FMLA?", db=mock_db, memory=memory, session_id="s4")
    history = memory.get_history()
    assert len(history) == 2  # user + assistant
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_orchestrator_pronoun_resolution():
    results = [make_result(1, 0.87)]
    orch = _make_orchestrator_with_mocked_llm(results)
    memory = ConversationMemoryAgent()
    memory.add_message("assistant", "FMLA provides up to 12 weeks of unpaid family leave.")
    mock_db = MagicMock()

    # "it" should be resolved using prior assistant message
    result = orch.run("How does it work?", db=mock_db, memory=memory, session_id="s5")
    # The query is ambiguous due to "it" — ClarificationAgent should catch it first
    # OR pronoun resolution converts it before clarification check
    # Either way it shouldn't crash
    assert isinstance(result, OrchestratorResult)


def test_orchestrator_no_retrieval_returns_knowledge_gap():
    orch = _make_orchestrator_with_mocked_llm([])
    memory = ConversationMemoryAgent()
    mock_db = MagicMock()

    result = orch.run(
        "What is the maternity leave policy of Infosys?",
        db=mock_db,
        memory=memory,
    )
    # With empty retrieval, answer should indicate not found
    assert "not" in result.answer.lower() or result.confidence == 0.0
