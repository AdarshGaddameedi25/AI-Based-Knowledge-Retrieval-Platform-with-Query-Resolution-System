"""
In-memory session store for per-session ConversationMemoryAgent instances.

Each unique session_id gets its own isolated ConversationMemoryAgent so that
multi-turn conversations maintain context without cross-session bleed.
"""
import logging
import uuid
from typing import Dict, Optional

from agents.conversation_memory_agent import ConversationMemoryAgent
from config.settings import settings

logger = logging.getLogger(__name__)


class InMemorySessionStore:
    def __init__(self):
        self._sessions: Dict[str, ConversationMemoryAgent] = {}

    def get_or_create(self, session_id: Optional[str] = None) -> tuple[str, ConversationMemoryAgent]:
        """
        Return (session_id, memory) — creating a new session if session_id is None or unknown.
        """
        if session_id and session_id in self._sessions:
            logger.debug(f"Session resumed: {session_id}")
            return session_id, self._sessions[session_id]

        # Generate a new session_id if not provided or not found
        new_id = session_id or str(uuid.uuid4())
        memory = ConversationMemoryAgent(max_history=settings.max_session_history)
        self._sessions[new_id] = memory
        logger.info(f"Session created: {new_id}")
        return new_id, memory

    def clear(self, session_id: str) -> bool:
        """Clear a session's conversation history. Returns True if session existed."""
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            logger.info(f"Session cleared: {session_id}")
            return True
        return False

    def delete(self, session_id: str) -> bool:
        """Fully remove a session from the store."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session deleted: {session_id}")
            return True
        return False

    def active_session_count(self) -> int:
        return len(self._sessions)


# Application-level singleton — shared across all requests
session_store = InMemorySessionStore()
