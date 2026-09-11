"""
QueryUnderstandingAgent — Milestone 2

Analyzes a raw user query and produces a structured QueryAnalysis containing:
  - normalized_query: cleaned, question-terminated text
  - query_type: canonical M2 label (factual | procedural | comparative | ambiguous | direct)
  - detected_intent: internal label (same value as query_type for M2 compatibility)
  - routing: where the orchestrator should route this query (retrieval | clarification | direct)
  - classification_confidence: float 0.0–1.0
  - requires_retrieval: bool
  - key_terms: list of significant tokens
  - suggested_domain: hr | technology | legal | None
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Canonical query-type labels
# ---------------------------------------------------------------------------
QUERY_TYPE_FACTUAL = "factual"
QUERY_TYPE_PROCEDURAL = "procedural"
QUERY_TYPE_COMPARATIVE = "comparative"
QUERY_TYPE_AMBIGUOUS = "ambiguous"
QUERY_TYPE_DIRECT = "direct"          # greetings / small-talk

# Canonical routing labels
ROUTING_RETRIEVAL = "retrieval"
ROUTING_CLARIFICATION = "clarification"
ROUTING_DIRECT = "direct"


@dataclass
class QueryAnalysis:
    """Structured output of the QueryUnderstandingAgent."""
    original_query: str
    normalized_query: str
    requires_retrieval: bool

    # Canonical M2 classification
    query_type: str                    # factual | procedural | comparative | ambiguous | direct
    routing: str                       # retrieval | clarification | direct

    # Internal detail fields (kept for backward compatibility)
    detected_intent: str               # same as query_type
    key_terms: List[str]
    suggested_domain: Optional[str] = None
    classification_confidence: float = 0.85


# ---------------------------------------------------------------------------
# Match sets — kept lowercase for fast lookup
# ---------------------------------------------------------------------------
GREETINGS = {
    "hi", "hello", "hey",
    "good morning", "good afternoon", "good evening",
    "good night",
}
SMALL_TALK = {
    "how are you", "what can you do", "who are you",
    "thanks", "thank you", "bye", "goodbye",
}

# ---------------------------------------------------------------------------
# Intent patterns — evaluated in PRIORITY ORDER (most specific first)
# ---------------------------------------------------------------------------
INTENT_PATTERNS = [
    (QUERY_TYPE_COMPARATIVE, re.compile(
        r"\b(difference|compare|vs\b|versus|better|worse|between|distinction|contrast|similarities)\b", re.I
    ), 0.95),
    (QUERY_TYPE_PROCEDURAL, re.compile(
        r"\b(how (do|to|can|should|would|does|did)|steps|process|procedure|guide|instructions|implement|configure|set up|install)\b", re.I
    ), 0.95),
    (QUERY_TYPE_FACTUAL, re.compile(
        r"\b(define|definition of|meaning of|what is meant by|explain|what|who|when|where|how many|how much|which|list|tell me about)\b", re.I
    ), 0.90),
]

# Ambiguity signals — pronouns without resolved antecedents, vague open-ended phrases
AMBIGUITY_PATTERNS = re.compile(
    r"\b(it|that|this|they|them|those|these|the (previous|last|above))\b"
    r"|^(tell me more|more details|explain|go on|continue|and|also)\W*$",
    re.I,
)

# Domain keyword scoring
DOMAIN_KEYWORDS = {
    "hr": {
        "leave", "fmla", "pto", "vacation", "sick", "policy", "employee", "hr",
        "human resources", "payroll", "benefits", "eeoc", "disability", "hiring",
        "onboarding", "termination", "performance", "appraisal", "workforce",
        "maternity", "paternity", "salary", "compensation", "discrimination",
    },
    "technology": {
        "api", "software", "database", "server", "cloud", "docker", "kubernetes",
        "python", "javascript", "react", "node", "sql", "nosql", "vector",
        "pgvector", "embedding", "machine learning", "ai", "llm", "rag",
        "authentication", "authorization", "oauth", "jwt", "security", "devops",
        "microservices", "architecture", "deployment", "ci", "cd", "pipeline",
    },
    "legal": {
        "law", "legal", "contract", "agreement", "compliance", "regulation",
        "statute", "jurisdiction", "liability", "court", "litigation", "ip",
        "intellectual property", "patent", "trademark", "copyright", "gdpr",
        "hipaa", "privacy", "audit",
    },
}

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "i", "you", "he", "she", "it", "we", "they", "what", "how",
    "when", "where", "who", "why", "which", "that", "this", "of",
    "in", "on", "at", "to", "for", "with", "and", "or", "but",
    "also", "about", "tell", "me", "please", "just",
})


class QueryUnderstandingAgent:
    """
    Classifies a raw user query into a canonical type and generates routing
    information for the multi-agent orchestrator.

    Classification hierarchy (evaluated in order):
      1. Greeting / small-talk → query_type=direct, routing=direct
      2. Ambiguous pronoun / vague reference → query_type=ambiguous, routing=clarification
      3. Comparative → query_type=comparative, routing=retrieval
      4. Procedural → query_type=procedural, routing=retrieval
      5. Factual / definitional → query_type=factual, routing=retrieval
      6. Default → query_type=factual, routing=retrieval (with lower confidence)
    """

    def analyze(self, query: str) -> QueryAnalysis:
        normalized = self._normalize(query)
        lower = normalized.lower().strip()

        # ── Step 1: Greeting / small-talk ──────────────────────────────────
        if lower in GREETINGS or lower in SMALL_TALK:
            return QueryAnalysis(
                original_query=query,
                normalized_query=normalized,
                requires_retrieval=False,
                query_type=QUERY_TYPE_DIRECT,
                routing=ROUTING_DIRECT,
                detected_intent=QUERY_TYPE_DIRECT,
                key_terms=[],
                suggested_domain=None,
                classification_confidence=1.0,
            )

        # Single-word or very short non-domain queries are weak signals
        tokens = normalized.split()
        if len(tokens) < 2:
            return QueryAnalysis(
                original_query=query,
                normalized_query=normalized,
                requires_retrieval=False,
                query_type=QUERY_TYPE_DIRECT,
                routing=ROUTING_DIRECT,
                detected_intent=QUERY_TYPE_DIRECT,
                key_terms=tokens,
                suggested_domain=None,
                classification_confidence=0.80,
            )

        # ── Step 2: Ambiguity detection ────────────────────────────────────
        #    Queries containing unresolved pronouns without rich domain signals
        #    are flagged as ambiguous so the orchestrator can route to clarification.
        domain = self._detect_domain(lower)
        key_terms = self._extract_key_terms(normalized)
        if self._is_ambiguous(normalized, key_terms):
            return QueryAnalysis(
                original_query=query,
                normalized_query=normalized,
                requires_retrieval=False,
                query_type=QUERY_TYPE_AMBIGUOUS,
                routing=ROUTING_CLARIFICATION,
                detected_intent=QUERY_TYPE_AMBIGUOUS,
                key_terms=key_terms,
                suggested_domain=domain,
                classification_confidence=0.70,
            )

        # ── Step 3: Intent classification ─────────────────────────────────
        query_type, confidence = self._classify_intent(normalized)

        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized,
            requires_retrieval=True,
            query_type=query_type,
            routing=ROUTING_RETRIEVAL,
            detected_intent=query_type,
            key_terms=key_terms,
            suggested_domain=domain,
            classification_confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, query: str) -> str:
        """Strip extra whitespace; optionally append '?' for question stems."""
        query = query.strip()
        query = re.sub(r"\s+", " ", query)
        if query and not query.endswith("?"):
            if any(kw in query.lower() for kw in [
                "what", "how", "when", "where", "who", "why", "which"
            ]):
                query = query + "?"
        return query

    def _is_ambiguous(self, query: str, key_terms: List[str]) -> bool:
        """
        A query is ambiguous when it contains unresolved pronouns or vague
        open-ended phrases AND does not provide enough substantive content to
        resolve against the knowledge base.
        """
        if not AMBIGUITY_PATTERNS.search(query):
            return False
        # If the query has at least 3 meaningful key terms, the pronoun is
        # likely accompanied by enough context to attempt retrieval.
        return len(key_terms) < 3

    def _classify_intent(self, query: str) -> tuple[str, float]:
        """
        Returns (query_type, confidence).
        Evaluates INTENT_PATTERNS in priority order; falls back to factual.
        """
        for query_type, pattern, confidence in INTENT_PATTERNS:
            if pattern.search(query):
                return query_type, confidence
        # Default: treat as factual with lower confidence
        return QUERY_TYPE_FACTUAL, 0.60

    def _detect_domain(self, query_lower: str) -> Optional[str]:
        scores: dict[str, int] = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[domain] = score
        if not scores:
            return None
        return max(scores, key=lambda d: scores[d])

    def _extract_key_terms(self, query: str) -> List[str]:
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
        return [t for t in tokens if t not in _STOP_WORDS]
