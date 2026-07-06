"""Eval harness: checkpoint structure, condition behaviour, held-out purity.

Uses a fake domain (string verifier) and a fake client that solves a task only when
a learned note is present - so memory-on conditions improve while the control stays
flat - all offline.
"""

from __future__ import annotations

import config
from engram.core.llm import LLMResult
from engram.domains.base import Task
from engram.eval import harness
from engram.memory.lesson import Lesson
from engram.memory.store import MemoryStore

NOTE_MARKER = "You have learned the following"


class FakeDomain:
    """Two train + two eval tasks; verifier passes iff the answer is 'CORRECT'."""

    def describe(self) -> str:
        return "SCHEMA"

    def train_pool(self) -> list[Task]:
        return [
            Task("train:a", "train question A", "x", "train", "ref"),
            Task("train:b", "train question B", "x", "train", "ref"),
        ]

    def eval_pool(self) -> list[Task]:
        return [
            Task("eval:a", "eval question A", "x", "eval", "ref"),
            Task("eval:b", "eval question B", "x", "eval", "ref"),
        ]

    def verify(self, task: Task, answer: str) -> bool:
        return answer == "CORRECT"


class FakeClient:
    """Answers CORRECT only when a learned note is present; reflects on failure."""

    def stats(self) -> dict:
        return {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]  # everything is mutually similar

    def generate(self, *, system, contents, tools, force_tool=True) -> LLMResult:
        name = tools[0]["name"] if tools else ""
        if name == "submit_sql":
            answer = "CORRECT" if NOTE_MARKER in contents else "WRONG"
            return LLMResult(tool_calls=[{"name": name, "args": {"query": answer}}])
        if name == "record_lessons":
            return LLMResult(
                tool_calls=[{"name": name, "args": {"lessons": [{"content": "the magic rule", "scope": "global"}]}}]
            )
        if name == "record_principle":
            return LLMResult(tool_calls=[{"name": name, "args": {"principle": "principle"}}])
        return LLMResult()


def _run(condition: str, n_train=6, ce=2):
    return harness.run_condition(
        FakeDomain(), condition, seed=1, client=FakeClient(), n_train=n_train, checkpoint_every=ce, eval_size=2
    )


def test_checkpoint_structure() -> None:
    res = _run("engram", n_train=6, ce=2)
    # step 0 plus one per checkpoint_every.
    assert [c.step for c in res.checkpoints] == [0, 2, 4, 6]


def test_no_memory_is_flat_control() -> None:
    res = _run("no_memory")
    assert all(c.success_rate == 0.0 for c in res.checkpoints)
    assert all(c.memory_size == 0 for c in res.checkpoints)


def test_memory_conditions_improve() -> None:
    for cond in ("naive", "engram"):
        res = _run(cond)
        assert res.checkpoints[0].success_rate == 0.0  # nothing learned yet
        assert res.checkpoints[-1].success_rate == 1.0  # learned to solve held-out
        assert res.checkpoints[-1].memory_size >= 1


def test_seeded_stream_is_deterministic() -> None:
    pool = FakeDomain().train_pool()
    s1 = [t.id for t in harness.seeded_stream(pool, 10, seed=42)]
    s2 = [t.id for t in harness.seeded_stream(pool, 10, seed=42)]
    s3 = [t.id for t in harness.seeded_stream(pool, 10, seed=7)]
    assert s1 == s2 and s1 != s3


def test_held_out_purity() -> None:
    """Evaluation must never create or mutate lessons."""
    res = _run("engram")
    # No lesson may have been created during the eval phase (step > n_train).
    for row in res.final_snapshot["lessons"]:
        assert row["created_step"] <= 6
    # _evaluate itself is read-only w.r.t. the store.
    store = MemoryStore()
    store.add(Lesson(id=store.next_id(), content="x", embedding=[1.0, 0.0, 0.0]))
    before = len(store)
    harness._evaluate(FakeDomain(), store, FakeClient(), True, FakeDomain().eval_pool(), step=99, raw_seen=0)
    assert len(store) == before


def test_reflect_on_success_flag(monkeypatch) -> None:
    """With REFLECT_ON_SUCCESS off, solved training tasks spawn no new lessons."""
    monkeypatch.setattr(config, "REFLECT_ON_SUCCESS", False)
    res = _run("engram", n_train=6, ce=2)
    # Only the first (unaided) attempt fails and reflects; memory stays a single
    # merged lesson thereafter.
    assert res.checkpoints[-1].memory_size == 1
