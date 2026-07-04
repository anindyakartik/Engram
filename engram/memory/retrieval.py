"""Semantic top-k retrieval with scope filtering.

Given the current task, embed its question, keep only lessons whose scope applies,
and return the top-k by cosine similarity above a threshold. These lessons are what
gets injected into the agent's context as guidance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from engram.core.llm import LLMClient
from engram.memory.lesson import Lesson
from engram.memory.store import MemoryStore


@dataclass
class Retrieved:
    """A lesson paired with its similarity to the current query."""

    lesson: Lesson
    score: float


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def embed_query(client: LLMClient, text: str) -> list[float]:
    """Embed a single query string."""
    return client.embed([text])[0]


def retrieve(
    store: MemoryStore,
    query: str,
    task_type: str,
    client: LLMClient,
    *,
    user: str = "",
    k: int | None = None,
    threshold: float | None = None,
    use_utility: bool = False,
) -> list[Retrieved]:
    """Return up to k in-scope lessons most similar to query.

    Args:
        store: The memory store to search.
        query: The natural-language question of the current task.
        task_type: Current task type, used for scope filtering.
        client: LLM client (for embedding the query).
        user: Optional user id for user-scoped lessons.
        k: Max lessons to return (defaults to config.RETRIEVAL_K).
        threshold: Minimum cosine similarity (defaults to config).
        use_utility: If True (Engram/curated), exclude lessons that have proven
            net-harmful (enough retrievals and negative utility). Naive retrieval
            leaves this off, so it keeps injecting bad lessons - that is the point
            of the ablation.

    Returns:
        A list of Retrieved sorted by descending similarity.
    """
    k = config.RETRIEVAL_K if k is None else k
    threshold = config.SIM_THRESHOLD_RETRIEVAL if threshold is None else threshold

    candidates = [
        lesson
        for lesson in store.all()
        if lesson.embedding is not None and lesson.applies_to(task_type, user)
    ]
    if use_utility:
        candidates = [
            lesson
            for lesson in candidates
            if not (lesson.retrieved_count >= config.UTILITY_EVIDENCE_MIN and lesson.utility < 0)
        ]
    if not candidates:
        return []

    q = np.asarray(embed_query(client, query), dtype=float)
    scored = [Retrieved(lesson, _cosine(q, np.asarray(lesson.embedding, dtype=float))) for lesson in candidates]
    scored = [r for r in scored if r.score >= threshold]
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:k]
