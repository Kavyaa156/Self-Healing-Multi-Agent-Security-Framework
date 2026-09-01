"""
tests/conftest.py

Shared pytest fixtures for the P4 test suite (real-pipeline integration).

`workflow` builds ONE real MultiAgentWorkflow (P1) per test session --
with no GROQ_API_KEY set, it runs fully offline using P1's deterministic
fallback responses, so these tests need no network access and no API key.

`fake_validation_store` is a lightweight stand-in for P3's real
ValidationStore (Chroma + sentence-transformers). It matches the exact
shape compute_semantic_accuracy() expects back from a real Chroma query
(a dict with a "distances" key), so the REAL compute_semantic_accuracy()
function is still exercised -- only the embedding backend is faked, the
same way you'd fake any external service in a unit test. This keeps the
test suite fast and offline; it is NOT a substitute for validating S
against the real store (do that manually via
reliability/seed_validation_store.py + main.py, which need internet once).
"""

import pytest

from agents.workflow import MultiAgentWorkflow


@pytest.fixture(scope="session")
def workflow():
    return MultiAgentWorkflow()


class FakeValidationStore:
    """
    Mimics ValidationStore.query()'s return shape. Distance semantics
    match reliability/semantic_accuracy.py's own convention: 0.0 = a
    perfect match (S -> 1.0), 2.0 = totally unrelated (S -> 0.0).

    Anything containing the words injected by fault_injection.py's F1
    ("weather", "injected") is treated as off-topic/ungrounded (S near
    0). Everything else is treated as grounded (S near 0.9), matching a
    clean audit-domain response.
    """

    def query(self, text: str, top_k: int = 3) -> dict:
        lowered = text.lower()
        if "weather" in lowered or "injected" in lowered:
            distances = [1.9, 1.95, 2.0][:top_k]
        else:
            distances = [0.2, 0.25, 0.3][:top_k]
        return {"distances": [distances]}


@pytest.fixture
def fake_validation_store():
    return FakeValidationStore()