"""contextdb — an in-memory observability DB for LLM agents.

Records everything an agent does (user messages, thinking, tool calls,
responses, world changes) on a per-session timeline, with first-class
tracking of *context carried in from previous turns/conversations* and the
token cost of all of it. Then it gives you insights over that log.

Quickstart
----------
    import contextdb as cdb

    @cdb.log_tool
    def search(q): ...

    with cdb.session() as s:
        s.log_user_message("find docs about retries")
        s.log_context([cdb.ContextRef("previous_session", "abc", "last run summary", tokens=120)])
        s.log_thinking("I should call search")
        search("retries")
        s.log_response("Here is what I found ...")

    print(cdb.insights().session_report(s.id).to_dict())
"""

from __future__ import annotations

from typing import Any, Optional

from .control import (
    AgentAborted,
    ControlPlane,
    PatchedFault,
    checkpoint,
    get_control,
    set_control,
)
from .dashboard import serve
from .decorators import get_store, log_function, log_tool, set_store
from .insights import Insights, SessionReport, ToolStat
from .models import ContextRef, Event, EventType, TokenUsage
from .session import Session, get_current_session
from .store import EventStore
from .tokens import count_tokens, using_tiktoken

__all__ = [
    "Session",
    "EventStore",
    "Event",
    "EventType",
    "TokenUsage",
    "ContextRef",
    "Insights",
    "SessionReport",
    "ToolStat",
    "log_tool",
    "log_function",
    "get_store",
    "set_store",
    "get_current_session",
    "count_tokens",
    "using_tiktoken",
    "session",
    "insights",
    "serve",
    "ControlPlane",
    "AgentAborted",
    "PatchedFault",
    "get_control",
    "set_control",
    "checkpoint",
]

__version__ = "0.1.0"


def session(
    session_id: Optional[str] = None,
    *,
    store: Optional[EventStore] = None,
    parent_session_id: Optional[str] = None,
    **metadata: Any,
) -> Session:
    """Open a new session on the default store (or a given one)."""

    return Session(
        store or get_store(),
        session_id=session_id,
        parent_session_id=parent_session_id,
        metadata=metadata,
    )


def insights(store: Optional[EventStore] = None) -> Insights:
    """Analytics over the default store (or a given one)."""

    return Insights(store or get_store())
