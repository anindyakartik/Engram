"""Retrieval: relevant lessons outrank irrelevant ones; scope filtering works."""

from __future__ import annotations

from engram.memory.lesson import Lesson
from engram.memory.retrieval import retrieve
from engram.memory.store import MemoryStore


class FakeEmbedClient:
    """Deterministic embedder: returns a preset vector per exact text."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.mapping[t] for t in texts]


def _store_with_lessons() -> MemoryStore:
    store = MemoryStore()
    status = Lesson(id="L0001", content="status is integer-coded", scope="global")
    status.embedding = [1.0, 0.0, 0.0]
    soft = Lesson(id="L0002", content="filter soft-deleted rows", scope="global")
    soft.embedding = [0.0, 1.0, 0.0]
    cents = Lesson(
        id="L0003", content="prices are cents", scope="task_type", scope_key="status_revenue"
    )
    cents.embedding = [0.0, 0.0, 1.0]
    for lesson in (status, soft, cents):
        store.add(lesson)
    return store


def test_relevant_ranks_first_and_threshold() -> None:
    store = _store_with_lessons()
    query = "how many orders have a given status"
    client = FakeEmbedClient({query: [0.95, 0.05, 0.0]})  # closest to the status lesson

    out = retrieve(store, query, "status_count", client, k=3, threshold=0.1)
    assert out[0].lesson.id == "L0001"  # status lesson wins
    # The cents lesson is task_type-scoped to status_revenue, so it is filtered out.
    assert all(r.lesson.id != "L0003" for r in out)


def test_scope_filter_admits_matching_task_type() -> None:
    store = _store_with_lessons()
    query = "revenue for a status"
    client = FakeEmbedClient({query: [0.0, 0.0, 1.0]})  # closest to the cents lesson

    out = retrieve(store, query, "status_revenue", client, k=3, threshold=0.1)
    ids = [r.lesson.id for r in out]
    assert "L0003" in ids and out[0].lesson.id == "L0003"


def test_threshold_excludes_dissimilar() -> None:
    store = _store_with_lessons()
    query = "totally unrelated question"
    client = FakeEmbedClient({query: [0.1, 0.1, 0.1]})  # low cosine to all axis vectors

    out = retrieve(store, query, "status_count", client, k=3, threshold=0.9)
    assert out == []


def test_utility_aware_retrieval_excludes_harmful() -> None:
    store = _store_with_lessons()
    harmful = store.get("L0001")  # the status lesson
    # Make it proven net-harmful: retrieved 4x, helped 1 / hurt 3 -> utility < 0.
    for success in (True, False, False, False):
        harmful.record_retrieval(success)
    assert harmful.utility < 0

    query = "how many orders have a given status"
    client = FakeEmbedClient({query: [1.0, 0.0, 0.0]})  # most similar to the harmful lesson

    # Naive retrieval (use_utility=False) still injects the harmful lesson.
    naive = retrieve(store, query, "status_count", client, k=3, threshold=0.1, use_utility=False)
    assert any(r.lesson.id == "L0001" for r in naive)

    # Curated retrieval excludes it once proven harmful.
    curated = retrieve(store, query, "status_count", client, k=3, threshold=0.1, use_utility=True)
    assert all(r.lesson.id != "L0001" for r in curated)


def test_utility_aware_keeps_unproven_lessons() -> None:
    """A negative-but-under-evidence lesson is NOT excluded (needs a fair trial)."""
    store = _store_with_lessons()
    fresh = store.get("L0001")
    fresh.record_retrieval(False)  # one bad outcome, below the evidence threshold
    query = "status question"
    client = FakeEmbedClient({query: [1.0, 0.0, 0.0]})
    out = retrieve(store, query, "status_count", client, k=3, threshold=0.1, use_utility=True)
    assert any(r.lesson.id == "L0001" for r in out)


def test_topk_limits_results() -> None:
    store = _store_with_lessons()
    # Make all three in-scope (global) and similar.
    store.get("L0003").scope = "global"
    query = "q"
    client = FakeEmbedClient({query: [1.0, 1.0, 1.0]})
    out = retrieve(store, query, "anything", client, k=2, threshold=0.0)
    assert len(out) == 2
