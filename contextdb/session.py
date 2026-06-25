"""Sessions and the ambient "current session" used by the decorators.

A :class:`Session` represents one conversation/agent run. It is the unit the
insights layer reasons about ("how did this run go?", "what context was
carried into it?").

The module keeps a :class:`contextvars.ContextVar` so decorators and helper
loggers can find the active session without it being passed around explicitly.
This also makes nesting work: a tool call logged inside a turn automatically
gets the turn's event as its ``parent_id``.
"""

from __future__ import annotations

import contextvars
import uuid
from typing import Any, Optional

from .models import ContextRef, Event, EventType, TokenUsage
from .store import EventStore
from .tokens import count_tokens

# The active session and the id of the currently-open parent event (a turn,
# usually). Both are contextvar-based so they work across threads/async tasks.
_current_session: contextvars.ContextVar[Optional["Session"]] = contextvars.ContextVar(
    "contextdb_current_session", default=None
)
_current_parent: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "contextdb_current_parent", default=None
)


def get_current_session() -> Optional["Session"]:
    return _current_session.get()


def get_current_parent() -> Optional[str]:
    return _current_parent.get()


class Session:
    """One conversation / agent run."""

    def __init__(
        self,
        store: EventStore,
        session_id: Optional[str] = None,
        *,
        parent_session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.store = store
        self.id = session_id or uuid.uuid4().hex
        # Lets you chain runs: "this conversation continues that one".
        self.parent_session_id = parent_session_id
        self.metadata = metadata or {}
        self._token = None  # contextvar reset token

    # -- activation (so decorators/loggers find this session) -------------

    def __enter__(self) -> "Session":
        self._token = _current_session.set(self)
        return self

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            _current_session.reset(self._token)
            self._token = None

    # -- low-level logging -------------------------------------------------

    def log(self, event: Event) -> Event:
        if event.parent_id is None:
            event.parent_id = get_current_parent()
        return self.store.add(event)

    # -- convenience loggers ----------------------------------------------

    def log_user_message(self, text: str, **metadata: Any) -> Event:
        return self.log(
            Event(
                type=EventType.USER_MESSAGE,
                session_id=self.id,
                content=text,
                tokens=TokenUsage(input_tokens=count_tokens(text)),
                metadata=metadata,
            )
        )

    def log_system_prompt(self, text: str, **metadata: Any) -> Event:
        return self.log(
            Event(
                type=EventType.SYSTEM_PROMPT,
                session_id=self.id,
                content=text,
                tokens=TokenUsage(input_tokens=count_tokens(text)),
                metadata=metadata,
            )
        )

    def log_response(self, text: str, **metadata: Any) -> Event:
        return self.log(
            Event(
                type=EventType.ASSISTANT_RESPONSE,
                session_id=self.id,
                content=text,
                tokens=TokenUsage(output_tokens=count_tokens(text)),
                metadata=metadata,
            )
        )

    def log_thinking(self, text: str, **metadata: Any) -> Event:
        return self.log(
            Event(
                type=EventType.THINKING,
                session_id=self.id,
                content=text,
                tokens=TokenUsage(output_tokens=count_tokens(text)),
                metadata=metadata,
            )
        )

    def log_change(self, description: str, *, target: str = "", **metadata: Any) -> Event:
        md = {"target": target, **metadata}
        return self.log(
            Event(
                type=EventType.CHANGE,
                session_id=self.id,
                name=target,
                content=description,
                metadata=md,
            )
        )

    def log_context(
        self,
        refs: list[ContextRef],
        *,
        label: str = "context_injection",
        **metadata: Any,
    ) -> Event:
        """Record context carried into this turn from elsewhere.

        This answers "what context was sent": each :class:`ContextRef` says
        where a chunk came from (a prior event, a prior session, memory, RAG)
        and how many tokens it added.
        """

        total = sum(r.tokens for r in refs)
        return self.log(
            Event(
                type=EventType.CONTEXT_INJECTION,
                session_id=self.id,
                name=label,
                content=[r.summary for r in refs],
                tokens=TokenUsage(input_tokens=total),
                context_refs=refs,
                metadata=metadata,
            )
        )

    def turn(self, label: str = "turn") -> "_Turn":
        """Open a parent scope that nested events attach to."""

        return _Turn(self, label)


class _Turn:
    """Context manager that groups nested events under a single parent event."""

    def __init__(self, session: Session, label: str) -> None:
        self.session = session
        self.label = label
        self._parent_token = None
        self.event: Optional[Event] = None

    def __enter__(self) -> Event:
        # Checkpoint: block here if the dashboard paused us, then pull in any
        # context the dashboard queued for the agent's next turn.
        from .control import get_control

        control = get_control()
        control.gate(f"turn:{self.label}")
        pending = control.drain_context()
        if pending:
            self.session.log_context(
                [ContextRef(**p) for p in pending], label="dashboard_patch"
            )

        self.event = self.session.log(
            Event(
                type=EventType.FUNCTION_CALL,
                session_id=self.session.id,
                name=self.label,
                metadata={"is_turn": True},
            )
        )
        self._parent_token = _current_parent.set(self.event.id)
        return self.event

    def __exit__(self, *exc) -> None:
        if self._parent_token is not None:
            _current_parent.reset(self._parent_token)
            self._parent_token = None
