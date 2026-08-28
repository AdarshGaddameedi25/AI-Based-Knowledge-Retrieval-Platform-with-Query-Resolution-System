import re
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ConversationMemoryAgent:
    def __init__(self, max_history: int = 10):
        self._history: list[Message] = []
        self.max_history = max_history

    def add_message(self, role: str, content: str) -> None:
        self._history.append(Message(role=role, content=content))
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

    def get_history(self) -> list[dict]:
        return [{"role": m.role, "content": m.content, "timestamp": m.timestamp} for m in self._history]

    def resolve_references(self, query: str) -> str:
        pronouns = re.compile(r"\b(it|that|this|they|them|those|these)\b", re.I)
        if not pronouns.search(query):
            return query
        if len(self._history) < 1:
            return query
        last_assistant = next(
            (m.content for m in reversed(self._history) if m.role == "assistant"),
            None,
        )
        if not last_assistant:
            return query
        first_nouns = re.findall(r"\b[A-Z][a-z]{3,}\b", last_assistant)
        if first_nouns:
            resolved = pronouns.sub(first_nouns[0], query)
            return resolved
        return query

    def build_context_window(self, system_prompt: str) -> list[dict]:
        messages = [{"role": "system", "content": system_prompt}]
        for m in self._history[-6:]:
            messages.append({"role": m.role, "content": m.content})
        return messages

    def clear(self) -> None:
        self._history = []
