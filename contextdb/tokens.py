"""Token counting.

Uses ``tiktoken`` when it is installed (accurate for OpenAI-family models),
otherwise falls back to a cheap heuristic (~4 chars per token) so the
framework has zero hard dependencies.
"""

from __future__ import annotations

import json
from typing import Any

try:  # pragma: no cover - optional dependency
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # noqa: BLE001 - any import/runtime failure -> heuristic
    _ENC = None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def count_tokens(value: Any) -> int:
    """Best-effort token count for any value.

    Strings are counted directly; other objects are JSON-stringified first.
    """

    text = _stringify(value)
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    # Heuristic: ~4 characters per token, minimum 1 token for non-empty text.
    return max(1, len(text) // 4)


def using_tiktoken() -> bool:
    """Whether accurate token counting is available."""

    return _ENC is not None
