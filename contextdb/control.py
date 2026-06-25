"""Live control plane — drive a running agent from the dashboard.

Monitoring is one-directional (agent -> store -> browser). This module adds the
*return* channel: commands from the dashboard mutate a :class:`ControlPlane`,
and the agent **cooperatively checks in** with it at checkpoints that our
decorators / ``session.turn()`` already provide.

What you can do live:
  * pause / resume / single-step the agent
  * set a breakpoint on a tool name (auto-pause right before it runs)
  * patch a tool: force its return value, or inject a fault (change scenario)
  * queue a context injection that lands in the agent's next turn
  * abort the run

Important: you cannot forcibly freeze arbitrary Python from outside the
process. The agent only stops where it checks in. Instrumented code (our
decorators + turns) checks in automatically; for hand-written loops, call
``cdb.checkpoint("label")`` wherever you want to be interruptible.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional


class AgentAborted(RuntimeError):
    """Raised inside the agent when the dashboard requests an abort."""


class PatchedFault(RuntimeError):
    """Raised in place of a tool call when a 'raise' patch is active."""


# Sentinels for tool interception decisions.
PROCEED = "proceed"
RETURN = "return"
RAISE = "raise"


class ControlPlane:
    """Thread-safe, dashboard-driven control state for one process."""

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.RLock())
        self._mode = "running"  # running | paused | stepping
        self._step_budget = 0
        self._abort = False
        self._waiting_at: Optional[str] = None
        self._reason = ""
        self._tool_patches: dict[str, dict[str, Any]] = {}
        self._breakpoints: set[str] = set()
        self._pending_context: list[dict[str, Any]] = []
        self._subs: list[Callable[[dict], None]] = []

    # -- observers ---------------------------------------------------------

    def subscribe(self, cb: Callable[[dict], None]) -> Callable[[], None]:
        self._subs.append(cb)
        return lambda: self._subs.remove(cb) if cb in self._subs else None

    def _notify(self) -> None:
        st = self.state()
        for cb in list(self._subs):
            try:
                cb(st)
            except Exception:  # noqa: BLE001 - a bad observer must not wedge control
                pass

    def state(self) -> dict[str, Any]:
        with self._cond:
            return {
                "mode": self._mode,
                "waiting_at": self._waiting_at,
                "abort": self._abort,
                "reason": self._reason,
                "patches": {k: dict(v) for k, v in self._tool_patches.items()},
                "breakpoints": sorted(self._breakpoints),
                "pending_context": list(self._pending_context),
            }

    # -- commands (called by the dashboard) --------------------------------

    def pause(self, reason: str = "paused via dashboard") -> None:
        with self._cond:
            self._mode = "paused"
            self._reason = reason
        self._notify()

    def resume(self) -> None:
        with self._cond:
            self._mode = "running"
            self._waiting_at = None
            self._reason = ""
            self._cond.notify_all()
        self._notify()

    def step(self, n: int = 1) -> None:
        with self._cond:
            self._mode = "stepping"
            self._step_budget = max(1, int(n))
            self._cond.notify_all()
        self._notify()

    def abort(self, reason: str = "aborted via dashboard") -> None:
        with self._cond:
            self._abort = True
            self._reason = reason
            self._cond.notify_all()
        self._notify()

    def clear_abort(self) -> None:
        with self._cond:
            self._abort = False
            self._reason = ""
        self._notify()

    def patch_tool(
        self, name: str, action: str, value: Any = None, error: Optional[str] = None
    ) -> None:
        with self._cond:
            self._tool_patches[name] = {"action": action, "value": value, "error": error}
        self._notify()

    def clear_patch(self, name: Optional[str] = None) -> None:
        with self._cond:
            if name is None:
                self._tool_patches.clear()
            else:
                self._tool_patches.pop(name, None)
        self._notify()

    def set_breakpoint(self, name: str, on: bool = True) -> None:
        with self._cond:
            if on:
                self._breakpoints.add(name)
            else:
                self._breakpoints.discard(name)
        self._notify()

    def queue_context(
        self,
        source_kind: str,
        summary: str,
        tokens: int = 0,
        source_id: Optional[str] = None,
    ) -> None:
        with self._cond:
            self._pending_context.append(
                {
                    "source_kind": source_kind,
                    "source_id": source_id,
                    "summary": summary,
                    "tokens": int(tokens or 0),
                }
            )
        self._notify()

    # -- agent-facing check-ins --------------------------------------------

    def drain_context(self) -> list[dict[str, Any]]:
        """Pop all queued context injections (called at the start of a turn)."""

        with self._cond:
            items = self._pending_context
            self._pending_context = []
        if items:
            self._notify()
        return items

    def maybe_break(self, name: str) -> None:
        """If ``name`` has a breakpoint, pause before it runs."""

        with self._cond:
            hit = name in self._breakpoints
            if hit:
                self._mode = "paused"
                self._reason = f"breakpoint @ {name}"
        if hit:
            self._notify()

    def gate(self, label: str) -> None:
        """Block here while paused. Honors step + abort.

        This is *the* checkpoint. Returns immediately when running; blocks while
        paused; consumes one step when stepping; raises on abort.
        """

        with self._cond:
            while True:
                if self._abort:
                    raise AgentAborted(self._reason or "aborted via dashboard")
                if self._mode == "running":
                    return
                if self._mode == "stepping" and self._step_budget > 0:
                    self._step_budget -= 1
                    if self._step_budget == 0:
                        self._mode = "paused"
                        self._reason = "stepped"
                    return
                # paused (or stepping with no budget left): announce + wait
                if self._waiting_at != label:
                    self._waiting_at = label
                    self._notify()
                self._cond.wait(timeout=1.0)

    def intercept(self, name: str) -> tuple[str, Any]:
        """Decide what should happen to a tool call: proceed / return / raise."""

        with self._cond:
            patch = self._tool_patches.get(name)
            patch = dict(patch) if patch else None
        if not patch:
            return (PROCEED, None)
        action = patch.get("action")
        if action == RETURN:
            return (RETURN, patch.get("value"))
        if action == RAISE:
            return (RAISE, patch.get("error") or f"injected fault in {name}")
        return (PROCEED, None)


# Process-wide default control plane.
_default_control = ControlPlane()


def get_control() -> ControlPlane:
    return _default_control


def set_control(control: ControlPlane) -> None:
    global _default_control
    _default_control = control


def checkpoint(label: str = "checkpoint") -> None:
    """Manual check-in for hand-written agent loops (pause/step/abort point)."""

    _default_control.gate(label)
