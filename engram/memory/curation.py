"""Memory curation: dedup, utility tracking, consolidation, pruning.

Raw reflection produces a lot of lessons, many redundant or wrong. Without curation
the store bloats and retrieval quality drops. Four operations keep it useful, all
driven by measured utility and embedding similarity:

- dedup on ingest: a candidate near-identical to an existing same-scope lesson is
  merged into it instead of appended.
- utility tracking: each retrieval updates the lesson's helped/hurt counts from the
  attempt's verified outcome (no LLM in this path).
- consolidation: clusters of overlapping lessons are fused into one principle that
  inherits the cluster's track record.
- pruning: a lesson retrieved enough times with poor utility is dropped.

The curate flag lets the harness run the same code path with curation off, which is
the naive-accumulation baseline.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

import config
from engram.agent.reflect import Candidate
from engram.core.llm import LLMClient
from engram.memory.lesson import Lesson
from engram.memory.retrieval import Retrieved, _cosine
from engram.memory.store import MemoryStore

CONSOLIDATE_SYSTEM = (
    "You merge several specific, overlapping lessons about a single database into ONE "
    "concise, general principle that covers all of them without losing actionable "
    "detail. Return exactly one principle via record_principle."
)

PRINCIPLE_TOOL = {
    "name": "record_principle",
    "description": "Record the single general principle that subsumes the given lessons.",
    "parameters": {
        "type": "object",
        "properties": {"principle": {"type": "string", "description": "One general, actionable principle."}},
        "required": ["principle"],
    },
}


# --------------------------------------------------------------------------- #
# Utility tracking
# --------------------------------------------------------------------------- #
def record_utility(retrieved: list[Retrieved], success: bool) -> None:
    """Attribute an attempt's verified outcome to every lesson it retrieved."""
    for r in retrieved:
        r.lesson.record_retrieval(success)


# --------------------------------------------------------------------------- #
# Ingest (with dedup) - the entry point for candidate lessons
# --------------------------------------------------------------------------- #
def ingest_candidates(
    store: MemoryStore,
    candidates: list[Candidate],
    client: LLMClient,
    step: int,
    *,
    provenance: str = "",
    curate: bool = True,
) -> list[Lesson]:
    """Embed and add candidate lessons, deduplicating when curate is True.

    Args:
        store: Memory store to add to.
        candidates: Proposed lessons from reflection.
        client: LLM client (for embedding lesson content).
        step: Logical training step (recorded as created_step).
        provenance: Tag (e.g. the source task id) recorded on new/merged lessons.
        curate: If False, every candidate is appended (naive accumulation baseline).

    Returns:
        The lessons newly added (merges are not included).
    """
    if not candidates:
        return []
    embeddings = client.embed([c.content for c in candidates])
    added: list[Lesson] = []
    for cand, emb in zip(candidates, embeddings, strict=True):
        if curate:
            dup = _find_duplicate(store, cand, emb)
            if dup is not None:
                _merge_into(dup, provenance)
                continue
        lesson = Lesson(
            id=store.next_id(),
            content=cand.content,
            scope=cand.scope,
            scope_key=cand.scope_key,
            embedding=emb,
            created_step=step,
            provenance=[provenance] if provenance else [],
            source="reflection",
        )
        store.add(lesson)
        added.append(lesson)
    return added


def _find_duplicate(store: MemoryStore, cand: Candidate, emb: list[float]) -> Lesson | None:
    """Return the most similar same-scope lesson above the dedup threshold, if any."""
    vec = np.asarray(emb, dtype=float)
    best: Lesson | None = None
    best_sim = config.DEDUP_SIM_THRESHOLD
    for lesson in store.all():
        if lesson.embedding is None or lesson.scope != cand.scope or lesson.scope_key != cand.scope_key:
            continue
        sim = _cosine(vec, np.asarray(lesson.embedding, dtype=float))
        if sim >= best_sim:
            best, best_sim = lesson, sim
    return best


def _merge_into(existing: Lesson, provenance: str) -> None:
    """Fold a duplicate candidate into an existing lesson (keep the proven text)."""
    existing.version += 1
    if provenance and provenance not in existing.provenance:
        existing.provenance.append(provenance)


# --------------------------------------------------------------------------- #
# Pruning
# --------------------------------------------------------------------------- #
def prune(store: MemoryStore) -> list[str]:
    """Remove lessons that have had a fair trial but failed to earn their place.

    Returns:
        The ids of pruned lessons.
    """
    removed = [lesson.id for lesson in store.all() if lesson.prunable]
    for lesson_id in removed:
        store.remove(lesson_id)
    return removed


# --------------------------------------------------------------------------- #
# Consolidation
# --------------------------------------------------------------------------- #
def maybe_consolidate(store: MemoryStore, client: LLMClient, step: int) -> int:
    """Fuse clusters of overlapping lessons into single general principles.

    Returns:
        The number of consolidated principles created.
    """
    n_created = 0
    for cluster in _clusters(store):
        principle = _synthesize(client, [lesson.content for lesson in cluster])
        if not principle:
            continue
        emb = client.embed([principle])[0]
        consolidated = Lesson(
            id=store.next_id(),
            content=principle,
            scope=cluster[0].scope,
            scope_key=cluster[0].scope_key,
            embedding=emb,
            created_step=step,
            # Inherit the cluster's verified track record so the principle is not
            # treated as untested - it stands on the evidence of its members.
            retrieved_count=sum(m.retrieved_count for m in cluster),
            helped_count=sum(m.helped_count for m in cluster),
            hurt_count=sum(m.hurt_count for m in cluster),
            provenance=sorted({p for m in cluster for p in m.provenance} | {m.id for m in cluster}),
            version=max(m.version for m in cluster) + 1,
            source="consolidation",
        )
        for member in cluster:
            store.remove(member.id)
        store.add(consolidated)
        n_created += 1
    return n_created


def _clusters(store: MemoryStore) -> list[list[Lesson]]:
    """Greedily cluster same-scope lessons that are mutually similar enough."""
    groups: dict[tuple[str, str], list[Lesson]] = defaultdict(list)
    for lesson in store.all():
        if lesson.embedding is not None:
            groups[(lesson.scope, lesson.scope_key)].append(lesson)

    clusters: list[list[Lesson]] = []
    for members in groups.values():
        used: set[str] = set()
        for seed in members:
            if seed.id in used:
                continue
            seed_vec = np.asarray(seed.embedding, dtype=float)
            cluster = [seed]
            used.add(seed.id)
            for other in members:
                if other.id in used:
                    continue
                if _cosine(seed_vec, np.asarray(other.embedding, dtype=float)) >= config.CONSOLIDATE_SIM_THRESHOLD:
                    cluster.append(other)
                    used.add(other.id)
            if len(cluster) >= config.CONSOLIDATE_MIN_CLUSTER:
                clusters.append(cluster)
    return clusters


def _synthesize(client: LLMClient, contents: list[str]) -> str:
    """Ask the LLM to fuse several lesson texts into one general principle."""
    bullet = "\n".join(f"- {c}" for c in contents)
    result = client.generate(
        system=CONSOLIDATE_SYSTEM,
        contents=f"Lessons to merge into one principle:\n{bullet}",
        tools=[PRINCIPLE_TOOL],
        force_tool=True,
    )
    return str(result.first_tool_args().get("principle", "")).strip()
