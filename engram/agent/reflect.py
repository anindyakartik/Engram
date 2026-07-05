"""Reflection: turn an attempt (especially a failure) into candidate lessons.

Reflection is generation, not measurement, so using the LLM here is allowed. Its
output is a hypothesis, not a verdict. Critically, reflection is NOT shown the ground
truth or the correct query; on a failure it can only hypothesise which hidden
convention it likely violated. Those hypotheses are then validated empirically: a
good lesson correlates with future successes and survives curation; a bad one loses
utility and is pruned. That is what makes the loop self-correcting rather than
credulous.

Candidate lessons are proposed here but never stored directly. They pass through
memory.curation first (dedup / merge / scope), so memory stays compact.
"""

from __future__ import annotations

from dataclasses import dataclass

from engram.core.llm import LLMClient
from engram.core.trace import AttemptTrace

REFLECT_SYSTEM = (
    "You are an expert data analyst reviewing one attempt to answer a question with "
    "SQL against a SQLite database whose conventions are UNDOCUMENTED. Extract concise, "
    "reusable lessons about the database's hidden conventions that will help on FUTURE, "
    "different questions — e.g. how a column is really encoded, a filter that must "
    "always be applied, a unit or date representation, or a join that is required. "
    "Focus on general rules, NOT facts specific to this question's numbers. If the "
    "attempt FAILED, hypothesise the single most likely convention that was violated.\n"
    "Rules for good lessons:\n"
    "- Each lesson must assert ONE specific, concrete convention. Do NOT hedge with "
    "'or' / list alternatives — commit to the single most likely cause.\n"
    "- State only what the tables and columns in THIS attempt justify. Do NOT speculate "
    "that OTHER tables or columns (not involved here) share a property.\n"
    "- Prefer one sharp lesson over several vague ones; 0 lessons is fine.\n"
    "Record 0 to 3 lessons via record_lessons. Mark a lesson 'global' if it is true of "
    "the whole database, or 'task_type' if it only applies to this kind of question."
)

RECORD_TOOL = {
    "name": "record_lessons",
    "description": "Record the reusable lessons learned from this attempt (0 to 3).",
    "parameters": {
        "type": "object",
        "properties": {
            "lessons": {
                "type": "array",
                "description": "The lessons; may be empty if nothing generalisable was learned.",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "One actionable, general insight about the database.",
                        },
                        "scope": {
                            "type": "string",
                            "description": "'global' (whole database) or 'task_type' (this kind of question only).",
                        },
                    },
                    "required": ["content", "scope"],
                },
            }
        },
        "required": ["lessons"],
    },
}


@dataclass
class Candidate:
    """A proposed lesson, pre-curation."""

    content: str
    scope: str  # "global" | "task_type"
    scope_key: str  # task_type when scope == "task_type", else ""


def _verdict_text(trace: AttemptTrace) -> str:
    if trace.success:
        return "The verifier judged this answer CORRECT."
    return f"The verifier judged this answer INCORRECT ({trace.error})."


def reflect(trace: AttemptTrace, schema: str, client: LLMClient, *, max_lessons: int = 3) -> list[Candidate]:
    """Propose candidate lessons from an attempt trace.

    Args:
        trace: The attempt to reflect on.
        schema: The database schema (DDL) shown to the agent.
        client: LLM client.
        max_lessons: Hard cap on candidates returned.

    Returns:
        A list of Candidate lessons (possibly empty).
    """
    prompt = (
        f"Schema:\n{schema}\n\n"
        f"Question: {trace.question}\n"
        f"The analyst's SQL answer was:\n{trace.answer or '(no query produced)'}\n\n"
        f"{_verdict_text(trace)}\n\n"
        "What general, reusable lesson(s) about this database's hidden conventions "
        "should be remembered for future questions?"
    )
    result = client.generate(
        system=REFLECT_SYSTEM, contents=prompt, tools=[RECORD_TOOL], force_tool=True
    )
    raw = result.first_tool_args().get("lessons", []) if result.tool_calls else []

    candidates: list[Candidate] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        scope = "task_type" if str(item.get("scope", "")).lower() == "task_type" else "global"
        candidates.append(
            Candidate(
                content=content,
                scope=scope,
                scope_key=trace.task_type if scope == "task_type" else "",
            )
        )
        if len(candidates) >= max_lessons:
            break
    return candidates
