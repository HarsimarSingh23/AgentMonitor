"""Shared pytest fixtures.

Resets the process-wide default store and control plane before every test so
the global singletons don't leak state across tests.
"""

import pytest

import contextdb as cdb
from contextdb.control import ControlPlane
from contextdb.store import EventStore


@pytest.fixture(autouse=True)
def reset_state():
    cdb.set_store(EventStore())
    cdb.set_control(ControlPlane())
    yield
    cdb.set_store(EventStore())
    cdb.set_control(ControlPlane())
