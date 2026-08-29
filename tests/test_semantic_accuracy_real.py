"""
Person 3 — Reliability Evaluation Module
tests/test_semantic_accuracy_real.py — runs compute_semantic_accuracy()
against the REAL seeded validation store and the REAL
telemetry_events.jsonl sample, instead of the synthetic mocks used in
test_semantic_accuracy.py.

This is a manual/integration check, not part of the fast automated
suite — it requires internet access on first run (to download
all-MiniLM-L6-v2) and takes longer since it does real embedding
inference. Run it directly:

    python tests/test_semantic_accuracy_real.py

Expected behavior: task_sec_101's three real events (an audit-planning
thought, a tool call, and a final security-audit report) should score
noticeably higher on S than an obviously off-topic event, since the
validation store is seeded with real access-control/security reference
content that the audit report content should semantically match.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.events import ExecutionEvent
from reliability.semantic_accuracy import compute_semantic_accuracy
from reliability.seed_validation_store import seed_store

SAMPLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "telemetry_events.jsonl",
)


def load_sample_events() -> list[ExecutionEvent]:
    events = []
    with open(SAMPLE_PATH) as f:
        for line in f:
            events.append(ExecutionEvent(**json.loads(line)))
    return events


def main():
    print("Seeding/opening the real validation store (needs internet on first run)...")
    store = seed_store()

    print("\nLoading real sample events from telemetry_events.jsonl...\n")
    events = load_sample_events()

    print("=== S scores on real task_sec_101 events ===")
    for event in events:
        S = compute_semantic_accuracy(event, store, top_k=3)
        preview = event.content[:70].replace("\n", " ")
        print(f"[{event.event_type:>12}] S={S:.3f}  content: {preview}...")

    print("\n=== S score on an off-topic control event (sanity check) ===")
    off_topic_event = ExecutionEvent(
        task_id="control",
        agent_id="executor",
        step_id=99,
        event_type="final_answer",
        content="The weather in Mumbai today is sunny with a light breeze "
        "and a high of 32 degrees Celsius.",
    )
    S_off_topic = compute_semantic_accuracy(off_topic_event, store, top_k=3)
    print(f"[off-topic] S={S_off_topic:.3f}  content: {off_topic_event.content[:70]}...")

    print(
        "\nSanity check: the real security-audit events should generally "
        "score higher on S than the off-topic weather event, since the "
        "validation store is seeded with security/access-control content."
    )


if __name__ == "__main__":
    main()
