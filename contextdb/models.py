"""Core data models for the context database.

Everything an agent does is recorded as an :class:`Event` on a timeline.
Events are grouped into sessions (one session == one conversation/run).
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


class EventType(str, enum.Enum):
    """The kinds of things we log in an agentic loop."""

    USER_MESSAGE = "user_message"
    ASSISTANT_RESPONSE = "assistant_response"
    SYSTEM_PROMPT = "system_prompt"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    FUNCTION_CALL = "function_call"
    # Context pulled in from prior turns / prior conversations / memory.
    CONTEXT_INJECTION = "context_injection"
    # A state change the agent caused in the world (file edit, db write...).
    CHANGE = "change"
    ERROR = "error"


def _now_ms() -> float:
    return time.time() * 1000.0


@dataclass
class TokenUsage:
    """Token accounting for a single event."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total": self.total,
        }


@dataclass
class ContextRef:
    """A reference to context that was carried into the current turn.

    This is the heart of the "what context was sent" question: it records
    *where* a piece of context came from (an earlier event/session/memory)
    and how big it was.
    """

    source_kind: str  # "previous_event" | "previous_session" | "memory" | "rag" | "file"
    source_id: Optional[str] = None  # event id / session id / doc id
    summary: str = ""  # human-readable description of the carried context
    tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Event:
    """A single recorded moment in an agent's life."""

    type: EventType
    session_id: str
    name: str = ""  # tool/function name, or a short label
    content: Any = None  # the payload: text, args, etc.
    input_data: Any = None  # for tool/function calls: the arguments
    output_data: Any = None  # for tool/function calls: the return value
    tokens: TokenUsage = field(default_factory=TokenUsage)
    duration_ms: Optional[float] = None
    parent_id: Optional[str] = None  # for nesting (a tool call inside a turn)
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    context_refs: list[ContextRef] = field(default_factory=list)

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp_ms: float = field(default_factory=_now_ms)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        d["tokens"] = self.tokens.to_dict()
        d["context_refs"] = [c.to_dict() for c in self.context_refs]
        return d
