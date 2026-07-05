"""Reflection: parses candidate lessons and assigns scope keys correctly."""

from __future__ import annotations

from engram.agent.reflect import reflect
from engram.core.llm import LLMResult
from engram.core.trace import AttemptTrace


class FakeReflectClient:
    def __init__(self, lessons: list[dict]) -> None:
        self._lessons = lessons
        self.last_prompt = ""

    def generate(self, *, system, contents, tools, force_tool=True) -> LLMResult:
        self.last_prompt = contents
        return LLMResult(tool_calls=[{"name": "record_lessons", "args": {"lessons": self._lessons}}])


def _trace(success: bool) -> AttemptTrace:
    return AttemptTrace(
        task_id="train:status_count:pending",
        task_type="status_count",
        question="How many orders are pending?",
        answer="SELECT COUNT(*) FROM orders WHERE status = 'pending'",
        success=success,
    )


def test_reflect_parses_and_scopes() -> None:
    client = FakeReflectClient(
        [
            {"content": "status is an integer code, not text", "scope": "global"},
            {"content": "pending-style counts need status decoding", "scope": "task_type"},
        ]
    )
    out = reflect(_trace(False), "SCHEMA", client)
    assert len(out) == 2
    assert out[0].scope == "global" and out[0].scope_key == ""
    assert out[1].scope == "task_type" and out[1].scope_key == "status_count"
    # The verdict (incorrect) is surfaced to the reflector.
    assert "INCORRECT" in client.last_prompt


def test_reflect_caps_and_skips_empty() -> None:
    client = FakeReflectClient(
        [
            {"content": "a", "scope": "global"},
            {"content": "  ", "scope": "global"},  # skipped (empty)
            {"content": "b", "scope": "global"},
            {"content": "c", "scope": "global"},
            {"content": "d", "scope": "global"},
        ]
    )
    out = reflect(_trace(False), "SCHEMA", client, max_lessons=3)
    assert [c.content for c in out] == ["a", "b", "c"]


def test_reflect_handles_no_lessons() -> None:
    client = FakeReflectClient([])
    assert reflect(_trace(True), "SCHEMA", client) == []
