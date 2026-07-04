"""MemoryStore + Lesson: utility scoring, pruning eligibility, persistence."""

from __future__ import annotations

from engram.memory.lesson import Lesson
from engram.memory.store import MemoryStore


def _lesson(store: MemoryStore, content: str, **kw) -> Lesson:
    return store.add(Lesson(id=store.next_id(), content=content, **kw))


def test_utility_and_pruning_thresholds() -> None:
    lo = Lesson(id="L1", content="x")
    assert lo.utility == 0.0  # no evidence -> neutral
    assert not lo.prunable

    # Retrieved 5x, always failed -> negative utility, eligible to prune.
    bad = Lesson(id="L2", content="bad")
    for _ in range(5):
        bad.record_retrieval(success=False)
    assert bad.utility < 0
    assert bad.prunable

    # Retrieved 6x, mostly helped -> positive utility, never prunable.
    good = Lesson(id="L3", content="good")
    for success in (True, True, True, True, False, True):
        good.record_retrieval(success)
    assert good.utility > 0
    assert not good.prunable

    # A single positive observation is shrunk, not extreme.
    fresh = Lesson(id="L4", content="fresh")
    fresh.record_retrieval(True)
    assert 0 < fresh.utility < 0.6
    assert not fresh.prunable  # too few retrievals to judge


def test_scope_applicability() -> None:
    g = Lesson(id="L1", content="g", scope="global")
    t = Lesson(id="L2", content="t", scope="task_type", scope_key="status_count")
    u = Lesson(id="L3", content="u", scope="user", scope_key="alice")
    assert g.applies_to("anything")
    assert t.applies_to("status_count") and not t.applies_to("year_count")
    assert u.applies_to("x", user="alice") and not u.applies_to("x", user="bob")


def test_store_crud_and_ids() -> None:
    store = MemoryStore()
    a = _lesson(store, "a")
    b = _lesson(store, "b")
    assert a.id == "L0001" and b.id == "L0002"
    assert len(store) == 2
    assert store.get("L0001") is a
    store.remove("L0001")
    assert store.get("L0001") is None and len(store) == 1


def test_persist_load_roundtrip(tmp_path) -> None:
    store = MemoryStore()
    lesson = _lesson(store, "status is integer-coded", scope="task_type", scope_key="status_count")
    lesson.embedding = [0.1, 0.2, 0.3]
    for s in (True, True, False):
        lesson.record_retrieval(s)

    path = tmp_path / "mem.jsonl"
    store.persist(path)
    loaded = MemoryStore.load(path)

    r = loaded.get("L0001")
    assert r is not None
    assert r.content == "status is integer-coded"
    assert r.scope == "task_type" and r.scope_key == "status_count"
    assert r.embedding == [0.1, 0.2, 0.3]
    assert (r.helped_count, r.hurt_count, r.retrieved_count) == (2, 1, 3)
    # Id counter is restored so new ids do not collide.
    assert loaded.next_id() == "L0002"


def test_snapshot_summary(tmp_path) -> None:
    store = MemoryStore()
    _lesson(store, "g1", scope="global")
    _lesson(store, "t1", scope="task_type", scope_key="year_count", source="consolidation")
    store.step = 12
    summary = store.snapshot(tmp_path / "snap.json")
    assert summary["step"] == 12 and summary["count"] == 2
    assert summary["by_scope"] == {"global": 1, "task_type": 1}
    assert summary["by_source"] == {"reflection": 1, "consolidation": 1}
    # Snapshots omit embeddings for readability.
    assert all("embedding" not in row for row in summary["lessons"])
