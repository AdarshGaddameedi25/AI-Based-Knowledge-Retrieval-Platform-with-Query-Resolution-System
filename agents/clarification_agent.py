import re
from dataclasses import dataclass


AMBIGUITY_PATTERNS = [
    re.compile(r"\bit\b", re.I),
    re.compile(r"\bthat\b", re.I),
    re.compile(r"\bthis\b", re.I),
    re.compile(r"\bthe (previous|last|above)\b", re.I),
    re.compile(r"\bsame\b", re.I),
    re.compile(r"^(tell me more|more details|explain|go on|continue)[\.\?]?$", re.I),
]


@dataclass
class ClarificationResult:
    is_ambiguous: bool
    clarification_question: str
    original_query: str


class ClarificationAgent:
    def evaluate(self, query: str, conversation_history: list[dict] = None) -> ClarificationResult:
        ambiguous = self._is_ambiguous(query, conversation_history)
        clarification_q = ""
        if ambiguous:
            clarification_q = self._generate_clarification(query)
        return ClarificationResult(
            is_ambiguous=ambiguous,
            clarification_question=clarification_q,
            original_query=query,
        )

    def _is_ambiguous(self, query: str, history: list[dict] = None) -> bool:
        stripped = query.strip().lower()
        if len(stripped.split()) <= 2 and not history:
            return True
        for pattern in AMBIGUITY_PATTERNS:
            if pattern.search(query):
                return True
        return False

    def _generate_clarification(self, query: str) -> str:
        stripped = query.strip().lower()
        if re.search(r"\bit\b|\bthat\b|\bthis\b", query, re.I):
            return "Could you please clarify what 'it' or 'that' refers to in your question?"
        if re.match(r"^(tell me more|more details|explain|go on|continue)", stripped):
            return "Could you specify which topic or section you would like more information about?"
        if len(stripped.split()) <= 2:
            return f"Your query '{query}' seems short. Could you provide more context or details?"
        return "Could you please rephrase or add more detail to your question?"
