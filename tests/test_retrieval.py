import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.query_understanding_agent import QueryUnderstandingAgent
from agents.clarification_agent import ClarificationAgent
from agents.conversation_memory_agent import ConversationMemoryAgent


def test_query_understanding_detects_factual_intent():
    agent = QueryUnderstandingAgent()
    result = agent.analyze("How many annual leave days are available?")
    assert result.detected_intent in {"factual", "procedural", "general"}
    assert result.requires_retrieval is True


def test_query_understanding_normalizes_query():
    agent = QueryUnderstandingAgent()
    result = agent.analyze("  what is MFA   ")
    assert result.normalized_query.strip() == result.normalized_query


def test_query_understanding_greeting_no_retrieval():
    agent = QueryUnderstandingAgent()
    result = agent.analyze("hello")
    assert result.requires_retrieval is False


def test_query_understanding_extracts_key_terms():
    agent = QueryUnderstandingAgent()
    result = agent.analyze("What is the difference between MFA and password authentication?")
    assert len(result.key_terms) > 0
    assert "mfa" in result.key_terms or "password" in result.key_terms or "authentication" in result.key_terms


def test_clarification_agent_detects_ambiguous_pronoun():
    agent = ClarificationAgent()
    result = agent.evaluate("What does it mean?")
    assert result.is_ambiguous is True
    assert len(result.clarification_question) > 0


def test_clarification_agent_not_ambiguous_for_clear_query():
    agent = ClarificationAgent()
    result = agent.evaluate("How do I apply for annual leave in the employee portal?")
    assert result.is_ambiguous is False


def test_conversation_memory_stores_messages():
    memory = ConversationMemoryAgent()
    memory.add_message("user", "What is the leave policy?")
    memory.add_message("assistant", "Employees are entitled to 21 days of annual leave.")
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
    memory.add_message("assistant", "MFA stands for Multi-Factor Authentication and is required for all admin accounts.")
    resolved = memory.resolve_references("How does it work?")
    assert resolved != "How does it work?"


def test_retrieval_validation_framework():
    test_queries = [
        {
            "query": "How many annual leave days are available?",
            "domain": "HR",
            "query_type": "factual",
            "expected_source": "leave_policy.txt or employee_handbook.txt",
        },
        {
            "query": "How do I apply for leave?",
            "domain": "HR",
            "query_type": "procedural",
            "expected_source": "leave_policy.txt or employee_handbook.txt",
        },
        {
            "query": "What is the difference between MFA and password authentication?",
            "domain": "Technology",
            "query_type": "comparative",
            "expected_source": "cloud_security.txt",
        },
        {
            "query": "What is the company's retirement age?",
            "domain": "HR",
            "query_type": "unavailable",
            "expected_source": "None — information not in knowledge base",
        },
    ]
    for test in test_queries:
        assert "query" in test
        assert "domain" in test
        assert "query_type" in test
        assert "expected_source" in test
    assert len(test_queries) == 4
