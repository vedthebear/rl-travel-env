"""Shared pytest setup.

Force scripted persona mode for every test by clearing OPENROUTER_API_KEY
at session start. Tests must not hit any network — they should be fast,
deterministic, and runnable in CI without secrets.
"""

from __future__ import annotations

import os

import pytest


# autouse=True means pytest runs this fixture once per session without any
# test explicitly requesting it. So the function looks unused but isn't —
# it's how we guarantee no test ever hits the OpenRouter API.
@pytest.fixture(autouse=True, scope="session")
def _no_llm_calls():
    """Hard-disable the LLM persona path for the whole test session."""
    os.environ.pop("OPENROUTER_API_KEY", None)
    yield
