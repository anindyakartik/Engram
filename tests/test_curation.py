"""Curation: dedup merges, utility-based pruning, consolidation, naive vs curated.

These tests pin the behaviours the whole result rests on. A fake client provides
deterministic embeddings (so similarity is controlled) and a canned consolidation
principle (so no network is needed).
"""

from __future__ import annotations

from engram.agent.reflect import Candidate
from engram.core.llm import LLMResult
from engram.memory import curation
from engram.memory.lesson import Lesson
from engram.memory.retrieval import Retrieved
from engram.memory.store import MemoryStore


class FakeClient:
    """Deterministic embedder + canned consolidation principle."""

    def __init__(self, vectors: dict[str, list[float]], principle: str = "GENERAL PRINCIPLE") -> None:
        self.vectors = vectors
        self.principle = principle

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Unknown texts (e.g. the synthesised principle) get a neutral vector.
        return [self.vectors.get(t, [0.5, 0.5, 0.5]) for t in texts]

    def generate(self, *, system, contents, tools, force_tool=True) -> LLMResult:
        return LLMResult(tool_calls=[{"name": "record_principle", "args": {"principle": self.principle}}])


def test_dedup_merges_near_duplicates() -> None:
    store = MemoryStore()
    client = FakeClient(
        {
            "status is an integer code": [1.0, 0.0, 0.0],
            "status column uses integer codes": [0.99, 0.01, 0.0],  # ~identical
        }
    )
    a = ingest_one(store, client, "status is an integer code", provenance="t1")
    assert len(store) == 1

    # Near-identical candidate is merged, not added.
    curation.ingest_candidates(
        store,
        [Candidate("status column uses integer codes", "global", "")],
        client,
        step=2,
        provenance="t2",
        curate=True,
    )
    assert len(store) == 1
    merged = store.get(a.id)
    assert merged.version == 2
    assert "t2" in merged.provenance  # provenance grew instead of duplicating


def test_naive_accumulation_keeps_duplicates() -> None:
    store = MemoryStore()
    client = FakeClient(
        {
            "status is an integer code": [1.0, 0.0, 0.0],
            "status column uses integer codes": [0.99, 0.01, 0.0],
        }
    )
    ingest_one(store, client, "status is an integer code", curate=False)
    ingest_one(store, client, "status column uses integer codes", curate=False)
    assert len(store) == 2  # no dedup -> both kept


def test_low_utility_pruned_after_fair_trial() -> None:
    store = MemoryStore()
    bad = Lesson(id=store.next_id(), content="wrong lesson", embedding=[1.0, 0.0, 0.0])
    store.add(bad)
    # Retrieved 5 times, always followed by failure.
    curation.record_utility([Retrieved(bad, 0.9)] * 5, success=False)
    removed = curation.prune(store)
    assert bad.id in removed and store.get(bad.id) is None


def test_high_utility_never_pruned() -> None:
    store = MemoryStore()
    good = Lesson(id=store.next_id(), content="good lesson", embedding=[1.0, 0.0, 0.0])
    store.add(good)
    for _ in range(9):
        curation.record_utility([Retrieved(good, 0.9)], success=True)
    curation.record_utility([Retrieved(good, 0.9)], success=False)
    assert good.utility > 0
    assert curation.prune(store) == []
    assert store.get(good.id) is good


def test_consolidation_reduces_count_and_keeps_coverage() -> None:
    store = MemoryStore()
    # Three related-but-distinct same-scope lessons whose pairwise cosine sits in the
    # consolidation band (>= 0.80) but below the dedup threshold (< 0.90), so they are
    # NOT merged at ingest and remain to be consolidated.
    vectors = {
        "always filter is_deleted = 0": [1.0, 0.0, 0.0],
        "exclude soft-deleted orders": [0.86, 0.51, 0.0],
        "drop rows where is_deleted = 1": [0.86, 0.255, 0.4417],
    }
    client = FakeClient(vectors, principle="Always exclude soft-deleted orders (is_deleted=0).")
    for text in vectors:
        lesson = ingest_one(store, client, text)
        # give the members a track record to be inherited
        curation.record_utility([Retrieved(lesson, 0.9)], success=True)
    assert len(store) == 3

    created = curation.maybe_consolidate(store, client, step=10)
    assert created == 1
    assert len(store) == 1  # three specifics -> one principle
    principle = store.all()[0]
    assert principle.source == "consolidation"
    assert principle.content == "Always exclude soft-deleted orders (is_deleted=0)."
    assert principle.helped_count == 3  # inherited the cluster's verified track record
    assert len(principle.provenance) >= 3


def test_consolidation_needs_minimum_cluster() -> None:
    store = MemoryStore()
    # In-band similar (0.80-0.90) but only 2 members (< min cluster of 3).
    vectors = {"a": [1.0, 0.0, 0.0], "b": [0.86, 0.51, 0.0]}
    client = FakeClient(vectors)
    for text in vectors:
        ingest_one(store, client, text)
    assert curation.maybe_consolidate(store, client, step=5) == 0
    assert len(store) == 2


# --- helper ---------------------------------------------------------------- #
def ingest_one(store, client, content, *, scope="global", provenance="", curate=True) -> Lesson:
    added = curation.ingest_candidates(
        store, [Candidate(content, scope, "")], client, step=1, provenance=provenance, curate=curate
    )
    return added[0] if added else None
