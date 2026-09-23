"""
ClarificationAgent — Milestone 3.1

Detects ambiguous, incomplete, or underspecified queries and manages the
two-turn clarification lifecycle:

  Turn 1 (ambiguous query):
    evaluate() → ClarificationResult(is_ambiguous=True, question=..., state=...)

  Turn 2 (clarification response received):
    refine_query(original, response) → refined_query string
    → re-enters normal pipeline

Design principles:
  - Only request clarification when genuinely necessary
  - Do NOT ask for clarification on clear queries with ≥3 substantive key terms
  - Generate targeted questions using key terms from the query
  - Preserve original query at all times
  - refine_query() synthesises original + clarification into a coherent query
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Structured clarification state
# ---------------------------------------------------------------------------

@dataclass
class ClarificationState:
    """
    Structured state for a pending clarification interaction.

    pending=True  → waiting for user's clarification response
    pending=False → clarification resolved; refined_query is ready
    """
    required: bool
    original_query: str
    question: str
    reason: str
    pending: bool
    clarification_response: str = ""
    refined_query: str = ""


# ---------------------------------------------------------------------------
# Ambiguity signals
# ---------------------------------------------------------------------------

# Pronouns/phrases that are genuinely ambiguous only when query has too few terms
_PRONOUN_PATTERNS = re.compile(
    r"\b(it|its|that|this|they|them|their|those|these|the (previous|last|above))\b",
    re.I,
)

# Open-ended follow-up phrases — always ambiguous
_FOLLOW_UP_PATTERNS = re.compile(
    r"^(tell me more|more details?|explain|go on|continue|and|also)[\.\?]?\s*$",
    re.I,
)

# Vague relative qualifiers — "relevant", "appropriate", "best", etc.
# These make a query underspecified without a domain anchor.
_VAGUE_QUALIFIER_PATTERN = re.compile(
    r"\b(relevant|appropriate|suitable|best|good|important|useful|key|main|"
    r"common|typical|standard|proper|right|correct|necessary|related|similar|"
    r"effective|efficient|ideal|recommended|possible|available)\b",
    re.I,
)

# Short queries lacking context (≤2 words, no strong domain signal)
_MIN_MEANINGFUL_WORDS = 3


# ---------------------------------------------------------------------------
# ClarificationResult (returned from evaluate())
# ---------------------------------------------------------------------------

@dataclass
class ClarificationResult:
    is_ambiguous: bool
    clarification_question: str
    original_query: str
    reason: str = ""
    state: Optional[ClarificationState] = None


# ---------------------------------------------------------------------------
# ClarificationAgent
# ---------------------------------------------------------------------------

class ClarificationAgent:
    """
    Two-phase clarification manager:

    Phase 1 — evaluate(query, history) → ClarificationResult
    Phase 2 — refine_query(original, response) → str
    """

    def evaluate(
        self,
        query: str,
        conversation_history: Optional[List[dict]] = None,
        key_terms: Optional[List[str]] = None,
    ) -> ClarificationResult:
        """
        Evaluate whether a query needs clarification.

        Args:
            query: The (pronoun-resolved) user query.
            conversation_history: Prior conversation turns.
            key_terms: Substantive terms extracted by QueryUnderstandingAgent.

        Returns:
            ClarificationResult with is_ambiguous, question, reason, and state.
        """
        reason, is_ambiguous = self._assess_ambiguity(
            query,
            key_terms or [],
            conversation_history or [],
        )

        question = ""
        state = None

        if is_ambiguous:
            question = self.generate_targeted_question(query, key_terms or [])
            state = ClarificationState(
                required=True,
                original_query=query,
                question=question,
                reason=reason,
                pending=True,
            )

        return ClarificationResult(
            is_ambiguous=is_ambiguous,
            clarification_question=question,
            original_query=query,
            reason=reason,
            state=state,
        )

    def refine_query(self, original_query: str, clarification_response: str) -> str:
        """
        Combine the original query with the user's clarification response
        into a coherent, self-contained refined query.

        Examples:
          "What are the requirements?" + "FMLA eligibility"
          → "What are the eligibility requirements for FMLA?"

          "How does it work?" + "the leave approval process"
          → "How does the leave approval process work?"

          "Tell me more" + "FMLA medical certification"
          → "Tell me more about FMLA medical certification"
        """
        original = original_query.strip().rstrip("?").strip()
        response = clarification_response.strip().rstrip(".").strip()

        if not response:
            return original_query

        original_lower = original.lower()
        response_lower = response.lower()

        # Pattern: "What are the X?" + "Y Z" → "What are the Y Z for/about Z?"
        what_are = re.match(
            r"^what (are|is|were|was) the (.+)$", original_lower
        )
        if what_are:
            article = what_are.group(1)
            noun = what_are.group(2)
            # Check if response sounds like a specifier
            if len(response.split()) <= 5:
                return f"What {article} the {response} {noun}?"
            return f"What {article} {response}?"

        # Pattern: "How does it/this/that work?" + "X" → "How does X work?"
        how_does = re.match(
            r"^how (does|do|did|can|should|would) (it|this|that|they)(.*)$",
            original_lower,
        )
        if how_does:
            verb = how_does.group(1)
            rest = how_does.group(3).strip()
            if rest:
                return f"How {verb} {response} {rest}?"
            return f"How {verb} {response} work?"

        # Pattern: open-ended "Tell me more / explain"
        if _FOLLOW_UP_PATTERNS.match(original_lower):
            return f"Tell me more about {response}."

        # Pattern: very short original (1–2 words)
        if len(original.split()) <= 2:
            return f"What is {response}?"

        # Generic: append clarification to original
        # Remove pronoun subject from original if present
        refined = _PRONOUN_PATTERNS.sub(response, original)
        if refined == original:
            # No substitution happened — append as context
            return f"{original} — specifically about {response}?"

        return refined.strip().rstrip("?") + "?"

    def generate_targeted_question(
        self,
        query: str,
        key_terms: List[str],
    ) -> str:
        """
        Generate a specific clarification question tailored to the query
        content and detected key terms.
        """
        query_lower = query.strip().lower()

        # Follow-up / open-ended
        if _FOLLOW_UP_PATTERNS.match(query_lower):
            if key_terms:
                return (
                    f"Could you specify which topic you'd like more information about? "
                    f"For example, are you referring to {', '.join(key_terms[:2])}?"
                )
            return "Could you specify which topic or section you would like more information about?"

        # Pure pronoun query — very short
        words = query_lower.split()
        if len(words) <= 3 and _PRONOUN_PATTERNS.search(query_lower):
            if key_terms:
                return (
                    f"Could you clarify what '{words[0]}' refers to? "
                    f"Are you asking about {key_terms[0]}?"
                )
            return "Could you clarify what you are referring to in your question?"

        # "What are the requirements / benefits / steps?"
        vague_nouns = {
            "requirements", "benefits", "steps", "process", "policy",
            "rules", "guidelines", "procedure", "criteria", "conditions",
        }
        for noun in vague_nouns:
            if noun in query_lower:
                examples = _build_domain_examples(noun, key_terms)
                if examples:
                    return (
                        f"Could you clarify which {noun} you mean? "
                        f"For example: {examples}?"
                    )
                return f"Could you clarify which {noun} you are asking about — please add more context."

        # Pronoun present but query has some context
        if _PRONOUN_PATTERNS.search(query):
            if key_terms:
                return (
                    f"Could you clarify what 'it' or 'this' refers to? "
                    f"Are you asking about {key_terms[0]}?"
                )
            return "Could you clarify what 'it' or 'this' refers to in your question?"

        # Vague qualifier — "relevant", "best", "suitable", etc.
        if _VAGUE_QUALIFIER_PATTERN.search(query_lower):
            qualifier_match = _VAGUE_QUALIFIER_PATTERN.search(query_lower)
            qualifier = qualifier_match.group(0) if qualifier_match else "relevant"
            if key_terms:
                # Filter out the qualifier word itself from key_terms for the example
                content_terms = [t for t in key_terms if not _VAGUE_QUALIFIER_PATTERN.match(t)]
                if content_terms:
                    return (
                        f"Could you clarify what '{qualifier}' means in context? "
                        f"For example, are you asking about {content_terms[0]} "
                        f"in a specific domain (HR, Technology, Legal)?"
                    )
            return (
                f"Could you provide more context? '{qualifier.capitalize()}' "
                f"depends on a specific topic or domain — which area are you asking about?"
            )

        # Generic fallback
        return "Could you please provide more context or rephrase your question?"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assess_ambiguity(
        self,
        query: str,
        key_terms: List[str],
        history: List[dict],
    ) -> tuple[str, bool]:
        """
        Returns (reason, is_ambiguous).

        Decision logic:
          1. Open-ended follow-up phrases → always ambiguous
          2. Pure pronoun query (≤3 words, pronoun present, no key terms) → ambiguous
          3. Pronoun present AND fewer than 3 key terms AND no prior history → ambiguous
          4. Vague noun query with only the vague noun as the key term → ambiguous
          5. ≥3 key terms with domain signal → not ambiguous (enough context)
          6. Rich history (prior context available) → resolve via memory, not clarification
        """
        query_lower = query.strip().lower()
        stripped = query_lower

        # Rule 1: open-ended follow-up
        if _FOLLOW_UP_PATTERNS.match(stripped):
            return "open-ended follow-up with no specific topic", True

        # Rule 2: very short pure-pronoun query (no meaningful terms)
        if len(stripped.split()) <= 3 and _PRONOUN_PATTERNS.search(query):
            if len(key_terms) < 2:
                return "short pronoun query with insufficient context", True

        # Rule 3: pronoun present, few key terms, no prior conversation
        if _PRONOUN_PATTERNS.search(query):
            if len(key_terms) < 3 and not history:
                return "pronoun reference with no prior context", True

        # Rule 4: vague noun without domain specificity
        # Queries like "What are the requirements?" with key_terms=["requirements"]
        # are underspecified when the ONLY key term is the vague noun itself.
        _VAGUE_NOUNS = {
            "requirements", "benefits", "steps", "process", "policy",
            "rules", "guidelines", "procedure", "criteria", "conditions",
            "details", "information", "stuff", "things", "points",
            "agents", "methods", "approaches", "techniques", "tools",
            "features", "options", "solutions", "types", "ways",
        }
        # If all key terms are vague nouns and no history exists → ambiguous
        if key_terms and all(t.lower() in _VAGUE_NOUNS for t in key_terms):
            if not history:
                return "query contains only vague nouns without domain context", True

        # Rule 4b: vague qualifier adjective without enough domain context
        # Catches queries like "What are relevant agents?" or "What are the best tools?"
        # where the qualifier (relevant / best) makes the noun underspecified.
        if _VAGUE_QUALIFIER_PATTERN.search(query_lower):
            if len(key_terms) <= 2 and not history:
                return "query uses a relative qualifier without sufficient domain context", True

        # Rule 5: single word / two words with no domain signal
        if len(stripped.split()) <= 2 and not history and not key_terms:
            return "query too short to resolve without context", True

        return "", False

    # Kept for backward compatibility with orchestrator line 150 usage
    def _generate_clarification(self, query: str) -> str:
        return self.generate_targeted_question(query, [])


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _build_domain_examples(noun: str, key_terms: List[str]) -> str:
    """Build 1–2 example phrases based on detected key terms and noun."""
    term = key_terms[0].upper() if key_terms else None
    if noun == "requirements":
        if term:
            return f"{term} eligibility requirements, or {term} documentation requirements"
        return "eligibility requirements, documentation requirements, or compliance requirements"
    if noun == "benefits":
        if term:
            return f"{term} employee benefits, or {term} medical benefits"
        return "employee benefits, financial benefits, or health benefits"
    if noun in ("steps", "process", "procedure"):
        if term:
            return f"the {term} application process, or the approval process"
        return "the application process, or the approval process"
    if noun == "policy":
        if term:
            return f"the {term} leave policy, or the {term} conduct policy"
        return "the leave policy, the conduct policy, or another specific policy"
    return ""
