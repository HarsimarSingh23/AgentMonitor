"""Adapters that feed real agent activity into contextdb.

  * ``anthropic_sdk``       — log Messages API responses incl. thinking blocks
  * ``claude_code_hooks``   — ship live Claude Code hook events to the dashboard
  * ``transcript_import``   — replay a Claude Code .jsonl transcript for analysis

Submodules are imported lazily (``from contextdb.adapters import anthropic_sdk``)
so running a single adapter as ``python -m`` stays fast and warning-free on the
hot path (the Claude Code hook runs on every tool call).
"""

from __future__ import annotations

__all__ = ["anthropic_sdk", "claude_code_hooks", "transcript_import"]
