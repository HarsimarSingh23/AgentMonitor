"""Live monitoring dashboard.

A zero-dependency HTTP server (stdlib only) that streams every event from an
:class:`~contextdb.store.EventStore` to the browser over Server-Sent Events
(SSE) the moment it is logged. Open the page and watch your agent work, event
by event, in real time.

Usage::

    import contextdb as cdb
    cdb.serve()              # starts the dashboard, opens the browser
    # ... run your agent; events stream in live ...
"""

from __future__ import annotations

import json
import os
import queue
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .store import EventStore

_HTML_PATH = os.path.join(os.path.dirname(__file__), "static", "dashboard.html")


class _Broker:
    """Fans store events *and* control-plane state out to all SSE clients.

    Each queued item is ``(channel, json_str)`` where channel is "event" or
    "control"; the SSE writer formats them as default vs. named SSE events.
    """

    def __init__(self, store: EventStore, control) -> None:
        self.store = store
        self.control = control
        self._clients: set[queue.Queue] = set()
        self._lock = threading.Lock()
        store.subscribe(self._on_event)
        control.subscribe(self._on_control)

    def _broadcast(self, channel: str, payload) -> None:
        item = (channel, json.dumps(payload, default=str))
        with self._lock:
            clients = list(self._clients)
        for q in clients:
            q.put(item)

    def _on_event(self, event) -> None:
        self._broadcast("event", event.to_dict())

    def _on_control(self, state) -> None:
        self._broadcast("control", state)

    def register(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._clients.add(q)
        return q

    def unregister(self, q: queue.Queue) -> None:
        with self._lock:
            self._clients.discard(q)

    def snapshot(self) -> list[tuple[str, str]]:
        items = [("event", json.dumps(e.to_dict(), default=str)) for e in self.store.all()]
        items.append(("control", json.dumps(self.control.state(), default=str)))
        return items


def _make_handler(broker: _Broker):
    class Handler(BaseHTTPRequestHandler):
        # Silence the default per-request stderr logging.
        def log_message(self, *args) -> None:  # noqa: D401
            pass

        def _send_html(self) -> None:
            try:
                with open(_HTML_PATH, "rb") as fh:
                    body = fh.read()
            except FileNotFoundError:
                body = b"<h1>dashboard.html not found</h1>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, obj) -> None:
            body = json.dumps(obj, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream_events(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            def _write(channel: str, data: str) -> None:
                prefix = "" if channel == "event" else f"event: {channel}\n"
                self.wfile.write(f"{prefix}data: {data}\n\n".encode("utf-8"))

            q = broker.register()
            try:
                # Replay everything logged so far so a late-joining browser
                # still sees the full run. The client de-dupes by event id.
                for channel, data in broker.snapshot():
                    _write(channel, data)
                self.wfile.flush()

                while True:
                    try:
                        channel, data = q.get(timeout=15)
                        _write(channel, data)
                    except queue.Empty:
                        # Heartbeat comment keeps the connection (and dead-peer
                        # detection) alive.
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                broker.unregister(q)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send_html()
            elif path == "/events":
                self._stream_events()
            elif path == "/summary":
                from .insights import Insights

                self._send_json(Insights(broker.store).summary())
            elif path == "/control/state":
                self._send_json(broker.control.state())
            else:
                self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path not in ("/control", "/ingest"):
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self.send_error(400, "invalid JSON")
                return
            if path == "/control":
                self._send_json(_dispatch_control(broker.control, body))
            else:  # /ingest — push an externally-produced event into the store
                ev = _event_from_payload(body, broker.store)
                self._send_json({"ok": True, "id": ev.id})

    return Handler


def _event_from_payload(body: dict, store: EventStore):
    """Build and store an Event from a JSON payload (used by /ingest).

    Lets external producers — Claude Code hooks, log shippers, other languages —
    feed the live dashboard over plain HTTP. Token counts are estimated from the
    payload when not supplied.
    """

    from .models import Event, EventType, TokenUsage
    from .tokens import count_tokens

    try:
        etype = EventType(body.get("type", "function_call"))
    except ValueError:
        etype = EventType.FUNCTION_CALL

    content = body.get("content")
    input_data = body.get("input_data")
    output_data = body.get("output_data")
    in_tok = body.get("input_tokens")
    out_tok = body.get("output_tokens")
    if in_tok is None:
        in_tok = count_tokens(input_data) if input_data is not None else count_tokens(
            content if etype in (EventType.USER_MESSAGE, EventType.SYSTEM_PROMPT,
                                 EventType.CONTEXT_INJECTION) else None
        )
    if out_tok is None:
        out_tok = count_tokens(output_data) if output_data is not None else count_tokens(
            content if etype in (EventType.ASSISTANT_RESPONSE, EventType.THINKING) else None
        )

    ev = Event(
        type=etype,
        session_id=body.get("session_id", "ingested"),
        name=body.get("name", ""),
        content=content,
        input_data=input_data,
        output_data=output_data,
        tokens=TokenUsage(input_tokens=int(in_tok or 0), output_tokens=int(out_tok or 0)),
        parent_id=body.get("parent_id"),
        error=body.get("error"),
        metadata=body.get("metadata") or {},
    )
    return store.add(ev)


def _coerce_value(value):
    """Let the dashboard send a JSON value or a string for a patched return."""

    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, json.JSONDecodeError):
            return value
    return value


def _dispatch_control(control, body: dict) -> dict:
    """Apply a dashboard command to the control plane; return the new state."""

    cmd = body.get("cmd")
    if cmd == "pause":
        control.pause(body.get("reason", "paused via dashboard"))
    elif cmd == "resume":
        control.resume()
    elif cmd == "step":
        control.step(int(body.get("n", 1)))
    elif cmd == "abort":
        control.abort(body.get("reason", "aborted via dashboard"))
    elif cmd == "clear_abort":
        control.clear_abort()
    elif cmd == "patch_tool":
        control.patch_tool(
            body["name"],
            body.get("action", "return"),
            value=_coerce_value(body.get("value")),
            error=body.get("error"),
        )
    elif cmd == "clear_patch":
        control.clear_patch(body.get("name"))
    elif cmd == "breakpoint":
        control.set_breakpoint(body["name"], bool(body.get("on", True)))
    elif cmd == "queue_context":
        control.queue_context(
            body.get("source_kind", "dashboard_patch"),
            body.get("summary", ""),
            tokens=body.get("tokens", 0),
            source_id=body.get("source_id"),
        )
    return control.state()


def serve(
    store: Optional[EventStore] = None,
    *,
    port: int = 8765,
    host: str = "127.0.0.1",
    open_browser: bool = True,
    block: bool = False,
) -> ThreadingHTTPServer:
    """Start the live dashboard.

    By default this is non-blocking (runs in a daemon thread) so your agent
    code can keep running and emitting events. Pass ``block=True`` to serve
    forever in the foreground.
    """

    from .control import get_control
    from .decorators import get_store

    store = store if store is not None else get_store()
    broker = _Broker(store, get_control())
    httpd = ThreadingHTTPServer((host, port), _make_handler(broker))
    url = f"http://{host}:{port}/"

    if block:
        if open_browser:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        print(f"contextdb dashboard serving at {url} (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.shutdown()
        return httpd

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print(f"contextdb dashboard live at {url}")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    return httpd
