"""Agent runtime: attempt one task, optionally conditioned on retrieved memory.

The loop is deliberately identical for the baseline and the memory-augmented agent
so the comparison is clean: the only difference is whether lessons are retrieved and
injected. The agent works schema-only: it sees the DDL (and any learned notes) and
submits exactly one SQL query. It never sees whether its answer is correct, so the
hidden conventions can only come from memory, not from in-attempt exploration.
"""

from __future__ import annotations

from engram.core.llm import LLMClient
from engram.core.trace import AttemptTrace
from engram.domains.base import Domain, Task
from engram.memory.retrieval import Retrieved, retrieve
from engram.memory.store import MemoryStore

# Neutral system prompt, IDENTICAL for the baseline and the memory agent, so the
# only difference between conditions is the presence of retrieved notes (injected
# into the user prompt). Keeping convention hints out of the system prompt keeps the
# no-memory baseline honest.
SYSTEM = (
    "You are a careful data analyst working with a SQLite database. "
    "You are given ONLY the schema (no sample data, no documentation). "
    "Write exactly ONE SQLite SELECT query that answers the question and submit it by "
    "calling submit_sql. Do not explain your answer."
)

SUBMIT_TOOL = {
    "name": "submit_sql",
    "description": "Submit the final SQL SELECT query that answers the question.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A single SQLite SELECT statement."}
        },
        "required": ["query"],
    },
}


def _notes_block(retrieved: list[Retrieved]) -> str:
    if not retrieved:
        return ""
    lines = "\n".join(f"- {r.lesson.content}" for r in retrieved)
    return (
        "You have learned the following real, non-obvious conventions about THIS "
        "database from past attempts. They override your default assumptions — apply "
        f"any that are relevant:\n{lines}\n\n"
    )


def build_prompt(schema: str, question: str, retrieved: list[Retrieved]) -> str:
    """Assemble the user prompt from schema, learned notes, and the question."""
    return f"Schema:\n{schema}\n\n{_notes_block(retrieved)}Question: {question}"


def run_attempt(
    domain: Domain,
    task: Task,
    client: LLMClient,
    store: MemoryStore | None = None,
    *,
    memory_on: bool = True,
    use_utility: bool = False,
    user: str = "",
) -> tuple[AttemptTrace, list[Retrieved]]:
    """Attempt task once and return its trace plus the lessons retrieved.

    Args:
        domain: The task domain (provides the schema and the verifier).
        task: The task to attempt.
        client: LLM client.
        store: Memory store (required when memory_on).
        memory_on: If True, retrieve and inject lessons; else run as the baseline.
        use_utility: If True, use utility-aware retrieval (Engram/curated only).
        user: Optional user id for user-scoped retrieval.

    Returns:
        (trace, retrieved). The retrieved lessons are returned so the caller can
        attribute utility and drive reflection.
    """
    retrieved: list[Retrieved] = []
    if memory_on and store is not None and len(store) > 0:
        retrieved = retrieve(
            store, task.question, task.task_type, client, user=user, use_utility=use_utility
        )

    prompt = build_prompt(domain.describe(), task.question, retrieved)
    result = client.generate(system=SYSTEM, contents=prompt, tools=[SUBMIT_TOOL], force_tool=True)
    answer = result.first_arg("query")

    success = domain.verify(task, answer)
    trace = AttemptTrace(
        task_id=task.id,
        task_type=task.task_type,
        question=task.question,
        retrieved_lesson_ids=[r.lesson.id for r in retrieved],
        answer=answer,
        success=success,
        pool=task.pool,
        error=None if success else "verifier: result-set mismatch or invalid SQL",
    )
    return trace, retrieved
