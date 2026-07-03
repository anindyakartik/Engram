"""Domain interface: the plug-in contract the memory engine is written against.

Everything in engram.memory, engram.agent, and engram.eval is domain
agnostic and depends only on this interface. Swapping the task domain (text-to-SQL
here) for another one with a deterministic oracle should not touch the core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Task:
    """A single task instance.

    Attributes:
        id: Stable unique identifier.
        question: The natural-language question posed to the agent.
        task_type: Category used for scoped memory (e.g. "status_count").
        pool: "train" or "eval". Eval tasks NEVER produce stored lessons.
        reference: The known-correct solution (SQL here). Used to compute ground
            truth and to test the verifier; never shown to the agent.
    """

    id: str
    question: str
    task_type: str
    pool: str
    reference: str = field(repr=False)


@runtime_checkable
class Domain(Protocol):
    """A task domain with a deterministic, programmatic verifier."""

    def describe(self) -> str:
        """Return the environment description shown to the agent (e.g. the DDL)."""
        ...

    def train_pool(self) -> list[Task]:
        """Return the training tasks (the agent may learn from these)."""
        ...

    def eval_pool(self) -> list[Task]:
        """Return the held-out tasks (used only for measurement)."""
        ...

    def verify(self, task: Task, answer: str) -> bool:
        """Return True iff answer solves task per the deterministic oracle."""
        ...
