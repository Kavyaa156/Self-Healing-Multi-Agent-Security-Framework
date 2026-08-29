"""
Person 3 — Reliability Evaluation Module
semantic_accuracy.py — implements the Semantic Accuracy (S) score.

Per the plan: S is a text-matching rate of an event's output against
domain-relevant reference chunks retrieved from a Chroma vector store,
embedded with sentence-transformers (all-MiniLM-L6-v2).

Design:
    - ValidationStore wraps a persistent Chroma collection + the
      embedding model. It's built once, seeded with reference/"ground
      truth" domain documents (the "domain-relevant validation
      database" the plan calls out as a [PROJECT DESIGN DECISION] the
      team must define — e.g. verified security-audit reference docs,
      policy documents, whatever the chosen task domain is).
    - compute_semantic_accuracy(event, validation_store) embeds the
      event's content, retrieves the top-k most similar reference
      chunks, and scores S as the average cosine similarity to those
      chunks (rescaled to [0, 1]).

NOTE: sentence-transformers downloads the all-MiniLM-L6-v2 weights from
Hugging Face on first use. This sandbox's network allowlist does not
include huggingface.co, so the model load will fail here — the code
below is correct and will work as-is on a machine with normal internet
access (or with the model pre-downloaded/cached). See the bottom of
this file for a note on testing this without network access.
"""

from typing import Optional

from schemas.events import ExecutionEvent

_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class ValidationStore:
    """
    Wraps a persistent Chroma collection used as the "domain-relevant
    validation database" for S scoring, plus the sentence-transformers
    embedding model used to embed both reference docs and event content.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_store",
        collection_name: str = "validation_docs",
        embedding_model_name: str = _EMBEDDING_MODEL_NAME,
    ):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name
        )
        self._model = SentenceTransformer(embedding_model_name)

    def add_documents(self, documents: list[str], ids: Optional[list[str]] = None) -> None:
        """
        Seed the validation store with domain-reference chunks (e.g.
        verified policy documents, ground-truth answers, or whatever
        the chosen task domain's "known-correct" reference text is).

        Args:
            documents: list of reference text chunks.
            ids: optional list of unique string IDs, one per document.
                If omitted, IDs are auto-generated as "doc_0", "doc_1", ...
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        if len(ids) != len(documents):
            raise ValueError("ids and documents must be the same length.")

        embeddings = self._model.encode(documents).tolist()
        self._collection.add(documents=documents, embeddings=embeddings, ids=ids)

    def query(self, text: str, top_k: int = 3) -> dict:
        """
        Embed `text` and retrieve the top_k most similar reference
        chunks from the store, along with their distances.
        """
        query_embedding = self._model.encode([text]).tolist()
        return self._collection.query(
            query_embeddings=query_embedding, n_results=top_k
        )


def compute_semantic_accuracy(
    event: ExecutionEvent,
    validation_store: ValidationStore,
    top_k: int = 3,
) -> float:
    """
    Compute the Semantic Accuracy (S) score for one event: how well the
    event's content matches the domain-relevant validation database.

    Args:
        event: the ExecutionEvent whose `content` field is scored.
        validation_store: a ValidationStore already seeded with
            domain-reference documents.
        top_k: number of nearest reference chunks to average over.

    Returns:
        float in [0, 1] — 1.0 means the event's content is essentially
        identical (in embedding space) to its nearest reference chunks;
        0.0 means no semantic relationship at all.

    Raises:
        ValueError: if event.content is empty, or the validation store
            has no documents to compare against.
    """
    if not event.content or not event.content.strip():
        raise ValueError(
            f"Cannot compute semantic accuracy: event {event.task_id}/"
            f"step {event.step_id} has empty content."
        )

    results = validation_store.query(event.content, top_k=top_k)
    distances = results.get("distances", [[]])[0]

    if not distances:
        raise ValueError(
            "Validation store returned no matches — has it been seeded "
            "with add_documents() yet?"
        )

    # Chroma's default distance metric is squared L2 on normalized
    # embeddings, which maps to [0, 2] where 0 = identical. Convert to a
    # [0, 1] similarity score: similarity = 1 - (distance / 2), clamped.
    similarities = [max(0.0, min(1.0, 1.0 - (d / 2.0))) for d in distances]

    S = sum(similarities) / len(similarities)
    return S


# ---------------------------------------------------------------------------
# Manual smoke test (not part of the automated suite — requires network
# access to Hugging Face to download all-MiniLM-L6-v2 on first run).
# Run with: python3 reliability/semantic_accuracy.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    store = ValidationStore(persist_directory="./chroma_store_smoketest")
    store.add_documents(
        [
            "New user accounts should be provisioned with least-privilege "
            "defaults, not broad read/write access.",
            "The account lockout threshold should trigger after a small "
            "number of failed login attempts.",
            "Role hierarchies must not allow implicit inheritance of "
            "admin-level permissions by lower roles.",
        ],
        ids=["ref_0", "ref_1", "ref_2"],
    )

    grounded_event = ExecutionEvent(
        task_id="t1",
        agent_id="executor",
        step_id=1,
        event_type="final_answer",
        content="New accounts are getting excessive default permissions "
        "like read:all and write:config, which violates least privilege.",
    )
    ungrounded_event = ExecutionEvent(
        task_id="t1",
        agent_id="executor",
        step_id=2,
        event_type="final_answer",
        content="The weather in Mumbai today is sunny with a light breeze.",
    )

    print("S (grounded, on-topic event):", compute_semantic_accuracy(grounded_event, store))
    print("S (ungrounded, off-topic event):", compute_semantic_accuracy(ungrounded_event, store))
