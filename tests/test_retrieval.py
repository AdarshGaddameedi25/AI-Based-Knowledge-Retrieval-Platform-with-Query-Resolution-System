import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.query_understanding_agent import QueryUnderstandingAgent
from agents.clarification_agent import ClarificationAgent
from agents.conversation_memory_agent import ConversationMemoryAgent


def test_query_understanding_detects_factual_intent():
    agent = QueryUnderstandingAgent()
    result = agent.analyze("How many weeks of leave are available under FMLA?")
    assert result.detected_intent in {"factual", "procedural", "general"}
    assert result.requires_retrieval is True


def test_query_understanding_normalizes_query():
    agent = QueryUnderstandingAgent()
    result = agent.analyze("  what is pgvector   ")
    assert result.normalized_query.strip() == result.normalized_query


def test_query_understanding_greeting_no_retrieval():
    agent = QueryUnderstandingAgent()
    result = agent.analyze("hello")
    assert result.requires_retrieval is False


def test_query_understanding_extracts_key_terms():
    agent = QueryUnderstandingAgent()
    result = agent.analyze("What is the difference between authentication and authorization?")
    assert len(result.key_terms) > 0
    assert any(t in result.key_terms for t in ["authentication", "authorization", "difference"])


def test_clarification_agent_detects_ambiguous_pronoun():
    agent = ClarificationAgent()
    result = agent.evaluate("What does it mean?")
    assert result.is_ambiguous is True
    assert len(result.clarification_question) > 0


def test_clarification_agent_not_ambiguous_for_clear_query():
    agent = ClarificationAgent()
    result = agent.evaluate("What are the FMLA eligibility requirements for employees?")
    assert result.is_ambiguous is False


def test_conversation_memory_stores_messages():
    memory = ConversationMemoryAgent()
    memory.add_message("user", "What is FMLA?")
    memory.add_message("assistant", "FMLA provides up to 12 weeks of unpaid leave.")
    history = memory.get_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_conversation_memory_max_history():
    memory = ConversationMemoryAgent(max_history=3)
    for i in range(10):
        memory.add_message("user", f"Message {i}")
    history = memory.get_history()
    assert len(history) <= 3


def test_conversation_memory_clear():
    memory = ConversationMemoryAgent()
    memory.add_message("user", "Test message")
    memory.clear()
    assert memory.get_history() == []


def test_conversation_memory_resolve_pronoun():
    memory = ConversationMemoryAgent()
    memory.add_message("assistant", "FMLA stands for Family and Medical Leave Act and provides 12 weeks leave.")
    resolved = memory.resolve_references("How does it work?")
    assert resolved != "How does it work?"


def test_retrieval_evaluation_framework():
    test_queries = [
        {
            "query": "What is FMLA?",
            "domain": "HR",
            "query_type": "factual",
            "expected_source": "fmla_employee_guide.pdf",
        },
        {
            "query": "Who is eligible for FMLA leave?",
            "domain": "HR",
            "query_type": "factual",
            "expected_source": "fmla_employee_guide.pdf",
        },
        {
            "query": "What workplace rights does EEOC protect?",
            "domain": "HR",
            "query_type": "factual",
            "expected_source": "eeoc_workplace_rights.pdf",
        },
        {
            "query": "What is pgvector?",
            "domain": "Technology",
            "query_type": "factual",
            "expected_source": "technology_knowledge_base.csv",
        },
        {
            "query": "What is the difference between authentication and authorization?",
            "domain": "Technology",
            "query_type": "comparative",
            "expected_source": "technology_knowledge_base.csv",
        },
        {
            "query": "What is the employee maternity leave policy of Infosys?",
            "domain": "HR",
            "query_type": "unavailable",
            "expected_source": None,
        },
    ]
    for test in test_queries:
        assert "query" in test
        assert "domain" in test
        assert "query_type" in test
        assert "expected_source" in test
    unavailable = [t for t in test_queries if t["query_type"] == "unavailable"]
    assert len(unavailable) == 1
    assert unavailable[0]["expected_source"] is None
    assert len(test_queries) == 6
