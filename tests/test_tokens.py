"""Tests for the token-counting helpers."""

from __future__ import annotations

import contextdb as cdb
from contextdb.tokens import count_tokens, using_tiktoken


def reset():
    from contextdb.store import EventStore
    from contextdb.control import ControlPlane

    cdb.set_store(EventStore())
    cdb.set_control(ControlPlane())


def test_empty_and_none_are_zero():
    reset()
    assert count_tokens("") == 0
    assert count_tokens(None) == 0


def test_nonempty_string_is_positive():
    reset()
    assert count_tokens("hello world") > 0


def test_counts_grow_with_length():
    reset()
    short = count_tokens("a short bit of text")
    long = count_tokens("a short bit of text " * 50)
    assert long > short


def test_non_string_values_are_stringified():
    reset()
    # dicts / lists are JSON-stringified before counting, so they're non-zero.
    assert count_tokens({"a": 1, "b": [1, 2, 3]}) > 0
    assert count_tokens([1, 2, 3, 4, 5]) > 0


def test_unserialisable_object_falls_back_to_repr():
    reset()

    class Weird:
        def __repr__(self):
            return "weird-object-repr"

    # Should not raise; str() fallback keeps a positive count.
    assert count_tokens(Weird()) > 0


def test_using_tiktoken_returns_bool():
    reset()
    assert isinstance(using_tiktoken(), bool)
