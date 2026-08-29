"""
Person 3 — Reliability Evaluation Module
seed_validation_store.py — one-time setup script that builds and seeds
the real ValidationStore used by compute_semantic_accuracy() in
production/demo runs.

This is NOT run on every task — it's run once (or whenever the
reference content in validation_seed_data.py changes) to populate the
persistent Chroma store at ./chroma_store/. After seeding, S scoring
just queries the already-populated store.

NEEDS INTERNET ACCESS the first time it's run on a given machine:
SentenceTransformer("all-MiniLM-L6-v2") downloads its weights (~90MB)
from Hugging Face on first use, then caches them locally. After that
first run, this script (and all S scoring) works fully offline.

Usage (run from the project root):
    python -m reliability.seed_validation_store

(Running it as "python reliability/seed_validation_store.py" instead
will fail with "ModuleNotFoundError: No module named 'reliability'",
because Python only adds the script's own folder to sys.path when run
directly, not the project root above it. The sys.path fallback below
handles that case too, so either form works.)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reliability.semantic_accuracy import ValidationStore
from reliability.validation_seed_data import SECURITY_AUDIT_REFERENCE_DOCS


def seed_store(persist_directory: str = "./chroma_store") -> ValidationStore:
    """
    Build (or reopen) the persistent validation store and seed it with
    the security-audit reference documents. Safe to run more than
    once — Chroma's add() will raise on duplicate IDs, so this checks
    the collection's existing count first and skips re-seeding if it's
    already populated with the expected number of documents.
    """
    store = ValidationStore(persist_directory=persist_directory)

    existing_count = store._collection.count()
    expected_count = len(SECURITY_AUDIT_REFERENCE_DOCS)

    if existing_count >= expected_count:
        print(
            f"Validation store already has {existing_count} documents "
            f"(expected {expected_count}) — skipping re-seed. Delete "
            f"the '{persist_directory}' folder first if you want to "
            f"rebuild from scratch."
        )
        return store

    print(f"Seeding validation store with {expected_count} reference documents...")
    store.add_documents(SECURITY_AUDIT_REFERENCE_DOCS)
    print(f"Done. Store now has {store._collection.count()} documents.")
    return store


if __name__ == "__main__":
    seed_store()