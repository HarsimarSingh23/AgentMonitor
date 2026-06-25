"""Decorators and the global default store.

The headline ergonomic: drop ``@log_tool`` on any agent tool / ``@log_function``
on any function and every call is recorded — arguments, return value, duration,
token cost, and errors — with zero changes to call sites.
"""

from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Callable, Optional, TypeVar

from .control import PROCEED, RAISE, RETURN, PatchedFault, get_control
from .models import Event, EventType, TokenUsage
from .session import Session, get_current_session
from .store import EventStore
from .tokens import count_tokens

F = TypeVar("F", bound=Callable[..., Any])

# A process-wide default store so the decorators work out of the box.
_default_store = EventStore()
# A fallback session used when a decorated call happens with no active session.
_implicit_session: Optional[Session] = None


def get_store() -> EventStore:
    """The process-wide default event store."""

    return _default_store


def set_store(store: EventStore) -> None:
    global _default_store, _implicit_session
    _default_store = store
    _implicit_session = None


def _resolve_session() -> Session:
    """Find the active session, or lazily create an implicit one."""

    sess = get_current_session()
    if sess is not None:
        return sess
    global _implicit_session
    if _implicit_session is None:
        _implicit_session = Session(_default_store, session_id="implicit")
    return _implicit_session


def _truncate(value: Any, max_chars: int) -> Any:
    if max_chars <= 0:
        return value
    text = value if isinstance(value, str) else repr(value)
    if len(text) > max_chars:
        return text[:max_chars] + f"... <truncated {len(text) - max_chars} chars>"
    return value


def _make_event(
    event_type: EventType,
    name: str,
    args: tuple,
    kwargs: dict,
    record_io: bool,
    max_chars: int,
) -> tuple[Event, Any]:
    session = _resolve_session()
    input_payload: Any = None
    in_tokens = 0
    if record_io:
        input_payload = {
            "args": [_truncate(a, max_chars) for a in args],
            "kwargs": {k: _truncate(v, max_chars) for k, v in kwargs.items()},
        }
        in_tokens = count_tokens(list(args)) + count_tokens(kwargs)
    event = Event(
        type=event_type,
        session_id=session.id,
        name=name,
        input_data=input_payload,
        tokens=TokenUsage(input_tokens=in_tokens),
    )
    return event, session


def _finalize(
    event: Event,
    session: Session,
    start: float,
    result: Any,
    error: Optional[BaseException],
    record_io: bool,
    max_chars: int,
) -> None:
    event.duration_ms = (time.perf_counter() - start) * 1000.0
    if error is not None:
        event.error = f"{type(error).__name__}: {error}"
    elif record_io:
        event.output_data = _truncate(result, max_chars)
        event.tokens.output_tokens = count_tokens(result)
    session.log(event)


def _wrap(
    func: F,
    event_type: EventType,
    name: Optional[str],
    record_io: bool,
    max_chars: int,
) -> F:
    label = name or getattr(func, "__name__", "anonymous")

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            control = get_control()
            control.maybe_break(label)  # auto-pause if a breakpoint is set
            control.gate(f"tool:{label}")  # block here while paused
            action, payload = control.intercept(label)  # live patch?
            event, session = _make_event(
                event_type, label, args, kwargs, record_io, max_chars
            )
            start = time.perf_counter()
            error: Optional[BaseException] = None
            result: Any = None
            try:
                if action == RETURN:
                    event.metadata["patched"] = "return"
                    result = payload
                elif action == RAISE:
                    event.metadata["patched"] = "raise"
                    raise PatchedFault(payload)
                else:
                    result = await func(*args, **kwargs)
                return result
            except BaseException as exc:  # noqa: BLE001 - re-raised after logging
                error = exc
                raise
            finally:
                _finalize(event, session, start, result, error, record_io, max_chars)

        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        control = get_control()
        control.maybe_break(label)  # auto-pause if a breakpoint is set
        control.gate(f"tool:{label}")  # block here while paused
        action, payload = control.intercept(label)  # live patch?
        event, session = _make_event(
            event_type, label, args, kwargs, record_io, max_chars
        )
        start = time.perf_counter()
        error: Optional[BaseException] = None
        result: Any = None
        try:
            if action == RETURN:
                event.metadata["patched"] = "return"
                result = payload
            elif action == RAISE:
                event.metadata["patched"] = "raise"
                raise PatchedFault(payload)
            else:
                result = func(*args, **kwargs)
            return result
        except BaseException as exc:  # noqa: BLE001 - re-raised after logging
            error = exc
            raise
        finally:
            _finalize(event, session, start, result, error, record_io, max_chars)

    return sync_wrapper  # type: ignore[return-value]


def log_tool(
    _func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    record_io: bool = True,
    max_chars: int = 2000,
) -> Any:
    """Decorator: record every call to an agent *tool*.

    Usage::

        @log_tool
        def search(query: str) -> list[str]: ...

        @log_tool(name="web.search", max_chars=500)
        def search(query: str) -> list[str]: ...
    """

    def deco(func: F) -> F:
        return _wrap(func, EventType.TOOL_CALL, name, record_io, max_chars)

    return deco(_func) if callable(_func) else deco


def log_function(
    _func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    record_io: bool = True,
    max_chars: int = 2000,
) -> Any:
    """Decorator: record every call to a generic function (same as :func:`log_tool`
    but tagged as ``FUNCTION_CALL`` so insights can separate tools from plumbing)."""

    def deco(func: F) -> F:
        return _wrap(func, EventType.FUNCTION_CALL, name, record_io, max_chars)

    return deco(_func) if callable(_func) else deco
