import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_mock_result(rank=1, score=0.85, text="FMLA grants up to 12 weeks of leave.", source="fmla_employee_guide.pdf", page=3):
    from retrieval.retriever import RetrievalResult
    return RetrievalResult(
        chunk_id=f"chunk-{rank}",
        document_id="doc-001",
        source_file=source,
        text=text,
        similarity_score=score,
        rank=rank,
        chunk_index=rank - 1,
        page_number=page,
    )


def test_rag_no_results_returns_knowledge_gap():
    from retrieval.rag_pipeline import RAGPipeline
    with patch("retrieval.rag_pipeline.RAGPipeline._call_llm") as mock_llm:
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline._retriever = MagicMock()
        pipeline._retriever.retrieve.return_value = []
        pipeline._headers = {}

        mock_db = MagicMock()
        response = pipeline.run("What is the maternity leave policy of Infosys?", db=mock_db)

        assert "could not find" in response.answer.lower()
        assert response.sources == []
        mock_llm.assert_not_called()


def test_rag_builds_context_with_sources():
    from retrieval.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._retriever = MagicMock()
    pipeline._headers = {}

    results = [
        make_mock_result(1, 0.91, "FMLA provides up to 12 weeks unpaid leave.", "fmla.pdf", 2),
        make_mock_result(2, 0.83, "Eligible employees must have worked 12 months.", "fmla.pdf", 4),
    ]

    context = pipeline._build_context(results)
    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "Page: 2" in context
    assert "Page: 4" in context
    assert "FMLA provides" in context
    assert "0.9100" in context


def test_rag_returns_sources_in_response():
    from retrieval.rag_pipeline import RAGPipeline
    with patch.object(RAGPipeline, "_call_llm", return_value="FMLA provides 12 weeks of leave.") as mock_llm:
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline._retriever = MagicMock()
        pipeline._headers = {}

        results = [make_mock_result(1, 0.9)]
        pipeline._retriever.retrieve.return_value = results

        mock_db = MagicMock()
        response = pipeline.run("What is FMLA?", db=mock_db)

        assert len(response.sources) == 1
        assert response.sources[0]["source_file"] == "fmla_employee_guide.pdf"
        assert response.sources[0]["page_number"] == 3
        assert response.sources[0]["similarity_score"] == 0.9
        mock_llm.assert_called_once()


def test_rag_confidence_is_zero_when_no_results():
    """M2: confidence must be 0.0 when no chunks retrieved."""
    from retrieval.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._retriever = MagicMock()
    pipeline._retriever.retrieve.return_value = []
    pipeline._headers = {}
    mock_db = MagicMock()
    response = pipeline.run("Unknown topic", db=mock_db)
    assert response.confidence == 0.0


def test_rag_confidence_is_nonzero_with_results():
    """M2: confidence is mean similarity score when results exist."""
    from retrieval.rag_pipeline import RAGPipeline
    with patch.object(RAGPipeline, "_call_llm", return_value="Answer."):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline._retriever = MagicMock()
        pipeline._headers = {}

        results = [
            make_mock_result(1, 0.80),
            make_mock_result(2, 0.60),
        ]
        pipeline._retriever.retrieve.return_value = results
        mock_db = MagicMock()
        response = pipeline.run("What is FMLA?", db=mock_db)

        expected_confidence = round((0.80 + 0.60) / 2, 4)
        assert response.confidence == expected_confidence


def test_rag_domain_filter_forwarded_to_retriever():
    """M2: domain_filter parameter is passed through to retriever."""
    from retrieval.rag_pipeline import RAGPipeline
    with patch.object(RAGPipeline, "_call_llm", return_value="Answer."):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline._retriever = MagicMock()
        pipeline._retriever.retrieve.return_value = [make_mock_result(1, 0.85)]
        pipeline._headers = {}
        mock_db = MagicMock()
        pipeline.run("What is FMLA?", db=mock_db, domain_filter="hr")
        call_kwargs = pipeline._retriever.retrieve.call_args.kwargs
        assert call_kwargs.get("domain_filter") == "hr"


def test_query_agent_requires_retrieval_for_factual():
    from agents.query_understanding_agent import QueryUnderstandingAgent
    agent = QueryUnderstandingAgent()
    result = agent.analyze("What is FMLA?")
    assert result.requires_retrieval is True
    assert result.detected_intent in {"factual", "definitional", "general"}


def test_query_agent_no_retrieval_for_greeting():
    from agents.query_understanding_agent import QueryUnderstandingAgent
    agent = QueryUnderstandingAgent()
    result = agent.analyze("hello")
    assert result.requires_retrieval is False


def test_query_agent_extracts_key_terms():
    from agents.query_understanding_agent import QueryUnderstandingAgent
    agent = QueryUnderstandingAgent()
    result = agent.analyze("What is the difference between authentication and authorization?")
    assert len(result.key_terms) > 0
    terms = result.key_terms
    assert any(t in terms for t in ["authentication", "authorization", "difference"])


def test_clarification_agent_flags_ambiguous_pronoun():
    from agents.clarification_agent import ClarificationAgent
    agent = ClarificationAgent()
    result = agent.evaluate("What does it mean?")
    assert result.is_ambiguous is True


def test_clarification_agent_clear_query():
    from agents.clarification_agent import ClarificationAgent
    agent = ClarificationAgent()
    result = agent.evaluate("What is the FMLA leave eligibility criteria?")
    assert result.is_ambiguous is False


def test_retrieval_evaluation_framework():
    eval_queries = [
        {"query": "What is FMLA?", "expected_source": "fmla_employee_guide.pdf", "query_type": "factual"},
        {"query": "What workplace rights does EEOC protect?", "expected_source": "eeoc_workplace_rights.pdf", "query_type": "factual"},
        {"query": "What is pgvector?", "expected_source": "technology_knowledge_base.csv", "query_type": "factual"},
        {"query": "What is the maternity leave policy of Infosys?", "expected_source": None, "query_type": "unavailable"},
    ]
    for q in eval_queries:
        assert "query" in q
        assert "expected_source" in q
        assert "query_type" in q
    assert len(eval_queries) == 4
    unavailable = [q for q in eval_queries if q["query_type"] == "unavailable"]
    assert len(unavailable) == 1
    assert unavailable[0]["expected_source"] is None

    from retrieval.rag_pipeline import RAGPipeline
    with patch("retrieval.rag_pipeline.RAGPipeline._call_llm") as mock_llm:
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline._retriever = MagicMock()
        pipeline._retriever.retrieve.return_value = []
        pipeline._headers = {}

        mock_db = MagicMock()
        response = pipeline.run("What is the maternity leave policy of Infosys?", db=mock_db)

        assert "could not find" in response.answer.lower()
        assert response.sources == []
        mock_llm.assert_not_called()


def test_rag_builds_context_with_sources():
    from retrieval.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._retriever = MagicMock()
    pipeline._headers = {}

    results = [
        make_mock_result(1, 0.91, "FMLA provides up to 12 weeks unpaid leave.", "fmla.pdf", 2),
        make_mock_result(2, 0.83, "Eligible employees must have worked 12 months.", "fmla.pdf", 4),
    ]

    context = pipeline._build_context(results)
    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "Page: 2" in context
    assert "Page: 4" in context
    assert "FMLA provides" in context
    assert "0.9100" in context


def test_rag_returns_sources_in_response():
    from retrieval.rag_pipeline import RAGPipeline
    with patch.object(RAGPipeline, "_call_llm", return_value="FMLA provides 12 weeks of leave.") as mock_llm:
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline._retriever = MagicMock()
        pipeline._headers = {}

        results = [make_mock_result(1, 0.9)]
        pipeline._retriever.retrieve.return_value = results

        mock_db = MagicMock()
        response = pipeline.run("What is FMLA?", db=mock_db)

        assert len(response.sources) == 1
        assert response.sources[0]["source_file"] == "fmla_employee_guide.pdf"
        assert response.sources[0]["page_number"] == 3
        assert response.sources[0]["similarity_score"] == 0.9
        mock_llm.assert_called_once()


def test_query_agent_requires_retrieval_for_factual():
    from agents.query_understanding_agent import QueryUnderstandingAgent
    agent = QueryUnderstandingAgent()
    result = agent.analyze("What is FMLA?")
    assert result.requires_retrieval is True
    assert result.detected_intent in {"factual", "definitional", "general"}


def test_query_agent_no_retrieval_for_greeting():
    from agents.query_understanding_agent import QueryUnderstandingAgent
    agent = QueryUnderstandingAgent()
    result = agent.analyze("hello")
    assert result.requires_retrieval is False


def test_query_agent_extracts_key_terms():
    from agents.query_understanding_agent import QueryUnderstandingAgent
    agent = QueryUnderstandingAgent()
    result = agent.analyze("What is the difference between authentication and authorization?")
    assert len(result.key_terms) > 0
    terms = result.key_terms
    assert any(t in terms for t in ["authentication", "authorization", "difference"])


def test_clarification_agent_flags_ambiguous_pronoun():
    from agents.clarification_agent import ClarificationAgent
    agent = ClarificationAgent()
    result = agent.evaluate("What does it mean?")
    assert result.is_ambiguous is True


def test_clarification_agent_clear_query():
    from agents.clarification_agent import ClarificationAgent
    agent = ClarificationAgent()
    result = agent.evaluate("What is the FMLA leave eligibility criteria?")
    assert result.is_ambiguous is False


def test_retrieval_evaluation_framework():
    eval_queries = [
        {"query": "What is FMLA?", "expected_source": "fmla_employee_guide.pdf", "query_type": "factual"},
        {"query": "What workplace rights does EEOC protect?", "expected_source": "eeoc_workplace_rights.pdf", "query_type": "factual"},
        {"query": "What is pgvector?", "expected_source": "technology_knowledge_base.csv", "query_type": "factual"},
        {"query": "What is the maternity leave policy of Infosys?", "expected_source": None, "query_type": "unavailable"},
    ]
    for q in eval_queries:
        assert "query" in q
        assert "expected_source" in q
        assert "query_type" in q
    assert len(eval_queries) == 4
    unavailable = [q for q in eval_queries if q["query_type"] == "unavailable"]
    assert len(unavailable) == 1
    assert unavailable[0]["expected_source"] is None
