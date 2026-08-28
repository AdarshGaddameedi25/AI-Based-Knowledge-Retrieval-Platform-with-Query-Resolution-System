import re
from dataclasses import dataclass


@dataclass
class QueryAnalysis:
    original_query: str
    normalized_query: str
    requires_retrieval: bool
    detected_intent: str
    key_terms: list[str]


GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
SMALL_TALK = {"how are you", "what can you do", "who are you"}

INTENT_PATTERNS = {
    "factual": re.compile(r"\b(what|who|when|where|how many|how much|which)\b", re.I),
    "procedural": re.compile(r"\b(how (do|to|can)|steps|process|procedure|guide)\b", re.I),
    "comparative": re.compile(r"\b(difference|compare|vs|versus|better|worse|between)\b", re.I),
    "definitional": re.compile(r"\b(define|what is|what are|meaning of|explain)\b", re.I),
}


class QueryUnderstandingAgent:
    def analyze(self, query: str) -> QueryAnalysis:
        normalized = self._normalize(query)
        requires_retrieval = self._needs_retrieval(normalized)
        intent = self._detect_intent(normalized)
        key_terms = self._extract_key_terms(normalized)

        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized,
            requires_retrieval=requires_retrieval,
            detected_intent=intent,
            key_terms=key_terms,
        )

    def _normalize(self, query: str) -> str:
        query = query.strip()
        query = re.sub(r"\s+", " ", query)
        if query and not query.endswith("?"):
            if any(kw in query.lower() for kw in ["what", "how", "when", "where", "who", "why"]):
                query = query + "?"
        return query

    def _needs_retrieval(self, query: str) -> bool:
        lower = query.lower().strip()
        if lower in GREETINGS or lower in SMALL_TALK:
            return False
        if len(query.split()) < 2:
            return False
        return True

    def _detect_intent(self, query: str) -> str:
        for intent, pattern in INTENT_PATTERNS.items():
            if pattern.search(query):
                return intent
        return "general"

    def _extract_key_terms(self, query: str) -> list[str]:
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "i", "you", "he", "she", "it", "we", "they", "what", "how",
            "when", "where", "who", "why", "which", "that", "this", "of",
            "in", "on", "at", "to", "for", "with", "and", "or", "but",
        }
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
        return [t for t in tokens if t not in stop_words]
