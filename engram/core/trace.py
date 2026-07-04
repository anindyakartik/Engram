"""AttemptTrace: the full record of one attempt at one task.

A trace is what reflection reads and what the harness logs. It captures everything
needed to explain an outcome and to attribute lesson utility: which lessons were in
context, what the agent produced, and the verifier's hard verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AttemptTrace:
    """One attempt at one task.

    Attributes:
        task_id: The attempted task's id.
        task_type: The task's category (for scoped memory attribution).
        question: The natural-language question.
        retrieved_lesson_ids: Ids of lessons injected into the agent's context.
        answer: The agent's final answer (a SQL string here).
        success: The deterministic verifier's verdict.
        pool: "train" or "eval" - eval traces never produce stored lessons.
        error: Optional note (e.g. the SQL error), for reflection/debugging.
    """

    task_id: str
    task_type: str
    question: str
    retrieved_lesson_ids: list[str] = field(default_factory=list)
    answer: str = ""
    success: bool = False
    pool: str = "train"
    error: str | None = None

    def to_dict(self) -> dict:
        """Return a JSON-serialisable view of the trace."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "question": self.question,
            "retrieved_lesson_ids": list(self.retrieved_lesson_ids),
            "answer": self.answer,
            "success": self.success,
            "pool": self.pool,
            "error": self.error,
        }
