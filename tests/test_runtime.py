"""Agent runtime: correct/incorrect answers, and memory-on vs memory-off behaviour."""

from __future__ import annotations

import pytest

from engram.agent.runtime import run_attempt
from engram.core.llm import LLMResult
from engram.domains.text_to_sql import TextToSQLDomain
from engram.memory.lesson import Lesson
from engram.memory.store import MemoryStore


class FakeClient:
    """Returns a fixed SQL answer and captures the last prompt; trivial embedder."""

    def __init__(self, answer_sql: str) -> None:
        self.answer_sql = answer_sql
        self.last_prompt = ""

    def generate(self, *, system, contents, tools, force_tool=True) -> LLMResult:
        self.last_prompt = contents
        return LLMResult(tool_calls=[{"name": "submit_sql", "args": {"query": self.answer_sql}}])

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


@pytest.fixture(scope="module")
def domain() -> TextToSQLDomain:
    return TextToSQLDomain()


def _task(domain, tid):
    for t in domain.train_pool() + domain.eval_pool():
        if t.id == tid:
            return t
    raise KeyError(tid)


def test_correct_answer_marks_success(domain) -> None:
    task = _task(domain, "train:status_count:pending")
    client = FakeClient(task.reference)  # the reference SQL is correct
    trace, retrieved = run_attempt(domain, task, client, memory_on=False)
    assert trace.success is True
    assert trace.answer == task.reference
    assert trace.retrieved_lesson_ids == []
    assert trace.pool == "train"


def test_wrong_answer_marks_failure(domain) -> None:
    task = _task(domain, "train:status_count:pending")
    client = FakeClient("SELECT COUNT(*) FROM orders WHERE status = 0")  # missing soft-delete
    trace, _ = run_attempt(domain, task, client, memory_on=False)
    assert trace.success is False
    assert trace.error is not None


def test_memory_on_injects_lessons(domain) -> None:
    task = _task(domain, "train:status_count:pending")
    store = MemoryStore()
    lesson = Lesson(id=store.next_id(), content="orders.is_deleted=1 rows must be excluded")
    lesson.embedding = [1.0, 0.0, 0.0]
    store.add(lesson)

    client = FakeClient(task.reference)
    trace, retrieved = run_attempt(domain, task, client, store, memory_on=True)
    assert "is_deleted=1 rows must be excluded" in client.last_prompt
    assert [r.lesson.id for r in retrieved] == ["L0001"]
    assert trace.retrieved_lesson_ids == ["L0001"]


def test_memory_off_ignores_store(domain) -> None:
    task = _task(domain, "train:status_count:pending")
    store = MemoryStore()
    lesson = Lesson(id=store.next_id(), content="should not appear")
    lesson.embedding = [1.0, 0.0, 0.0]
    store.add(lesson)

    client = FakeClient(task.reference)
    trace, retrieved = run_attempt(domain, task, client, store, memory_on=False)
    assert "should not appear" not in client.last_prompt
    assert retrieved == [] and trace.retrieved_lesson_ids == []
