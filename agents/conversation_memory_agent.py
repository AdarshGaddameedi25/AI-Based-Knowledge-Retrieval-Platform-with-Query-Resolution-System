"""
ConversationMemoryAgent — Milestone 3.2

Manages per-session conversation memory with:
  - Bounded message history (default: 10 turns)
  - Improved pronoun/coreference resolution using entity tracking
  - Active topic tracking for context continuity and topic switching
  - Pending clarification state management (M3.1 integration)
  - Relevant context selection (avoids sending stale/irrelevant history)

Design principles:
  - Memory is session-scoped (isolated, no cross-session bleed)
  - Do NOT blindly send entire history to LLM — use relevance check
  - Keep conversation memory separate from permanent knowledge base
  - Support topic continuation AND context switching
  - Clarification state is stored here (per session) and consumed once
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.clarification_agent import ClarificationState


@dataclass
class Message:
    role: str       # "user" | "assistant" | "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ConversationMemoryAgent:
    """
    Per-session conversation memory with topic tracking and reference resolution.
    """

    def __init__(self, max_history: int = 10):
        self._history: List[Message] = []
        self.max_history = max_history

        # M3.2 — Topic and entity state
        self.active_topic: Optional[str] = None          # e.g. "FMLA", "PostgreSQL"
        self.last_entities: List[str] = []               # proper nouns from last assistant response
        self.last_source_docs: List[str] = []            # document names cited last turn
        self.active_key_terms: List[str] = []            # key terms from last query

        # M3.1 — Clarification state (one pending at a time per session)
        self._pending_clarification: Optional["ClarificationState"] = None

    # -----------------------------------------------------------------------
    # Core message storage
    # -----------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """Append a message and enforce max_history bound."""
        self._history.append(Message(role=role, content=content))
        if len(self._history) > self.max_history:
            # Drop oldest messages but always keep the most recent ones
            self._history = self._history[-self.max_history:]

    def get_history(self) -> List[dict]:
        """Return full history as list of dicts."""
        return [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp}
            for m in self._history
        ]

    def clear(self) -> None:
        """Clear all session state."""
        self._history = []
        self.active_topic = None
        self.last_entities = []
        self.last_source_docs = []
        self.active_key_terms = []
        self._pending_clarification = None

    # -----------------------------------------------------------------------
    # M3.2 — Topic and entity tracking
    # -----------------------------------------------------------------------

    def update_topic(
        self,
        query_key_terms: List[str],
        source_docs: Optional[List[str]] = None,
        assistant_response: Optional[str] = None,
    ) -> None:
        """
        Update the active topic after a successful retrieval+generation turn.

        Args:
            query_key_terms: Substantive tokens from the resolved query.
            source_docs: Document names cited in this turn's sources.
            assistant_response: The generated answer (used for entity extraction).
        """
        if query_key_terms:
            # Active topic = most frequent/prominent term from key_terms
            self.active_key_terms = query_key_terms[:10]
            self.active_topic = query_key_terms[0] if query_key_terms else None

        if source_docs:
            self.last_source_docs = source_docs[:5]

        if assistant_response:
            self.last_entities = self._extract_entities(assistant_response)

    def is_topic_continuation(self, query_key_terms: List[str]) -> bool:
        """
        Return True if the new query's key terms have meaningful overlap
        with the active topic key terms.

        This determines whether prior context is relevant to the new query.
        """
        if not self.active_key_terms or not query_key_terms:
            return False
        active_set = set(t.lower() for t in self.active_key_terms)
        new_set = set(t.lower() for t in query_key_terms)
        overlap = active_set & new_set
        return len(overlap) >= 1  # At least one shared term = continuation

    # -----------------------------------------------------------------------
    # M3.2 — Reference resolution (improved)
    # -----------------------------------------------------------------------

    _PRONOUNS = re.compile(
        r"\b(it|its|that|this|they|them|their|those|these|the (previous|last|above))\b",
        re.I,
    )

    def resolve_references(self, query: str) -> str:
        """
        Resolve pronouns and vague references using tracked session context.

        Priority order (most -> least reliable):
          1. active_key_terms — substantive terms from the user's last query.
             e.g. ["annual", "leave", "days"] -> "annual leave".
             Best for vector search as they directly mirror indexed content.
          2. active_topic — first key term from last query (single-word fallback).
          3. last_entities — proper nouns from last LLM response, filtered to
             exclude known boilerplate words (company names, document titles).
          4. Capitalised noun from last assistant message (broad fallback).
          5. Return original — let ClarificationAgent request clarification.
        """
        if not self._PRONOUNS.search(query):
            return query

        # Words that appear in LLM responses but are NOT useful referents
        _BOILERPLATE = {
            "ACME", "CORPORATION", "POLICY", "DOCUMENT", "REFERENCE",
            "VERSION", "ACCORDING", "SECTION", "BASED",
        }

        # Strategy 1: Use active_key_terms as a phrase (most semantically useful)
        # e.g. ["annual", "leave", "days"] -> "annual leave"
        if self.active_key_terms:
            phrase_terms = [
                t for t in self.active_key_terms[:3]
                if len(t) > 2 and t.lower() not in
                {"the", "are", "was", "has", "for", "and", "its", "it"}
            ]
            if phrase_terms:
                phrase = " ".join(phrase_terms)
                resolved = self._PRONOUNS.sub(phrase, query)
                return resolved

        # Strategy 2: active_topic (first key term)
        if self.active_topic:
            resolved = self._PRONOUNS.sub(self.active_topic, query)
            return resolved

        # Strategy 3: last_entities filtered — skip boilerplate proper nouns
        if self.last_entities:
            filtered = [e for e in self.last_entities if e.upper() not in _BOILERPLATE]
            if filtered:
                resolved = self._PRONOUNS.sub(filtered[0], query)
                return resolved

        # Strategy 4: Fallback — first capitalised noun from last assistant message
        last_assistant = next(
            (m.content for m in reversed(self._history) if m.role == "assistant"),
            None,
        )
        if last_assistant:
            if "[clarification needed]" not in last_assistant:
                nouns = re.findall(r"\b[A-Z][a-z]{3,}\b", last_assistant)
                useful = [n for n in nouns if n.upper() not in _BOILERPLATE]
                if useful:
                    resolved = self._PRONOUNS.sub(useful[0], query)
                    return resolved

        # No context to resolve against — return original
        return query

    # -----------------------------------------------------------------------
    # Context window for LLM
    # -----------------------------------------------------------------------

    def build_context_window(self, system_prompt: str) -> List[dict]:
        """
        Build a message list for the LLM including system prompt and
        relevant recent history (last 6 messages, excluding system messages).

        The system_prompt argument is accepted for backward compatibility
        but is NOT prepended here (ResponseGenerationAgent supplies its own).
        """
        messages: List[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for m in self._history[-6:]:
            # Exclude internal clarification markers from LLM context
            if m.role == "assistant" and m.content.startswith("[clarification needed]"):
                # Include a cleaned version so LLM knows a clarification was asked
                cleaned = m.content.replace("[clarification needed] ", "")
                messages.append({"role": "assistant", "content": cleaned})
            else:
                messages.append({"role": m.role, "content": m.content})

        return messages

    def get_relevant_context(self, query_key_terms: List[str]) -> List[dict]:
        """
        Return only contextually relevant history messages.

        If the new query is a topic continuation → include prior turns.
        If topic has switched → return empty (don't contaminate new topic).
        Always returns at most last 4 turns regardless.
        """
        if self.is_topic_continuation(query_key_terms):
            # Include last 4 messages (2 turns)
            return [
                {"role": m.role, "content": m.content}
                for m in self._history[-4:]
                if not (m.role == "assistant" and m.content.startswith("[clarification needed]"))
            ]
        # Topic switched — do not send stale context
        return []

    # -----------------------------------------------------------------------
    # M3.1 — Clarification state management
    # -----------------------------------------------------------------------

    def set_clarification_pending(self, state: "ClarificationState") -> None:
        """Store a pending clarification state for this session."""
        self._pending_clarification = state

    def consume_clarification(self) -> Optional["ClarificationState"]:
        """
        Return and clear the pending clarification state.
        Returns None if no clarification is pending.
        """
        state = self._pending_clarification
        self._pending_clarification = None
        return state

    def has_pending_clarification(self) -> bool:
        """Check whether a clarification is awaiting a user response."""
        return self._pending_clarification is not None and self._pending_clarification.pending

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract proper nouns and key technical terms from text.
        Returns up to 5 candidate entities for pronoun resolution.
        """
        # Match capitalised words (proper nouns) of length ≥ 4
        capitalized = re.findall(r"\b[A-Z][A-Za-z]{3,}\b", text)
        # Match all-caps acronyms (e.g., FMLA, SQL, API)
        acronyms = re.findall(r"\b[A-Z]{2,6}\b", text)

        seen = set()
        entities: List[str] = []
        for term in capitalized + acronyms:
            if term not in seen:
                seen.add(term)
                entities.append(term)
            if len(entities) >= 5:
                break
        return entities
