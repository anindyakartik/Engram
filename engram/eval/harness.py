"""Eval harness: training stream + held-out checkpoints across three conditions.

Methodology (the part that makes the result mean something):

* Disjoint pools. The agent trains on the training stream and is measured on the
  held-out eval pool. Eval attempts RETRIEVE memory (that is the point) but never
  write to it - no reflection, no utility update, no ingest - so the eval set is
  never learned from.
* Identical code path. All three conditions run the same attempt loop; they
  differ only in memory_on (retrieve/store) and curate (dedup/consolidate/
  prune). no_memory is the control, naive accumulates every lesson, engram
  curates. A credible headline requires engram > naive > no_memory.
* Checkpoints. Success on the full held-out pool is measured before training
  (step 0) and every checkpoint_every tasks, yielding the improvement curve.
* Reproducible. The training stream is a seeded sample; with committed cassettes,
  the same seed reproduces the same stream, lessons, and curve offline.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

import config
from engram.agent.reflect import reflect
from engram.agent.runtime import run_attempt
from engram.core.llm import LLMClient
from engram.domains.base import Domain, Task
from engram.memory import curation
from engram.memory.store import MemoryStore


@dataclass
class Checkpoint:
    """One measurement point on the improvement curve."""

    step: int
    success_rate: float
    n_eval: int
    memory_size: int
    raw_lessons_seen: int  # cumulative candidate lessons proposed (pre-curation)


@dataclass
class RunResult:
    """The outcome of one (condition, seed) run."""

    condition: str
    seed: int
    checkpoints: list[Checkpoint] = field(default_factory=list)
    final_snapshot: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def seeded_stream(pool: list[Task], n: int, seed: int) -> list[Task]:
    """Return a deterministic training stream of length n sampled from pool.

    Sampling is with replacement so a short pool can feed a long stream; repetition
    is realistic and reinforces learning. The order is fully determined by seed.
    """
    rng = random.Random(seed)
    return [rng.choice(pool) for _ in range(n)]


def _evaluate(
    domain: Domain,
    store: MemoryStore,
    client: LLMClient,
    memory_on: bool,
    eval_tasks: list[Task],
    step: int,
    raw_seen: int,
    use_utility: bool = False,
) -> Checkpoint:
    """Measure held-out success. Read-only w.r.t. memory (no learning here)."""
    n_ok = 0
    for task in eval_tasks:
        trace, _ = run_attempt(domain, task, client, store, memory_on=memory_on, use_utility=use_utility)
        n_ok += int(trace.success)
    return Checkpoint(
        step=step,
        success_rate=n_ok / len(eval_tasks),
        n_eval=len(eval_tasks),
        memory_size=len(store),
        raw_lessons_seen=raw_seen,
    )


def run_condition(
    domain: Domain,
    condition: str,
    seed: int,
    client: LLMClient,
    *,
    n_train: int | None = None,
    checkpoint_every: int | None = None,
    eval_size: int | None = None,
    verbose: bool = False,
) -> RunResult:
    """Run one condition for one seed and return its improvement curve.

    Args:
        domain: The task domain.
        condition: "no_memory" | "naive" | "engram".
        seed: Seed controlling the training-stream order.
        client: LLM client (record/replay).
        n_train / checkpoint_every / eval_size: overrides for the defaults in config.

    Returns:
        A RunResult with per-checkpoint held-out success and a final memory snapshot.
    """
    n_train = config.N_TRAIN if n_train is None else n_train
    checkpoint_every = config.CHECKPOINT_EVERY if checkpoint_every is None else checkpoint_every
    eval_size = config.EVAL_SET_SIZE if eval_size is None else eval_size

    memory_on = condition in ("naive", "engram")
    curate = condition == "engram"
    schema = domain.describe()
    eval_tasks = domain.eval_pool()[:eval_size]
    stream = seeded_stream(domain.train_pool(), n_train, seed)

    store = MemoryStore()
    result = RunResult(condition=condition, seed=seed)
    raw_seen = 0

    # Utility-aware retrieval is an Engram-only curation feature.
    use_utility = curate

    # Step 0: measure before any training.
    cp0 = _evaluate(domain, store, client, memory_on, eval_tasks, 0, raw_seen, use_utility)
    result.checkpoints.append(cp0)
    if verbose:
        print(f"    step   0: held-out {cp0.success_rate:.0%}  (memory={cp0.memory_size})")

    for i, task in enumerate(stream, start=1):
        store.step = i
        trace, retrieved = run_attempt(
            domain, task, client, store, memory_on=memory_on, use_utility=use_utility
        )

        if memory_on:
            curation.record_utility(retrieved, trace.success)
            if not trace.success or config.REFLECT_ON_SUCCESS:
                candidates = reflect(trace, schema, client)
                raw_seen += len(candidates)
                curation.ingest_candidates(
                    store, candidates, client, step=i, provenance=task.id, curate=curate
                )
            if curate:
                if i % config.PRUNE_EVERY == 0:
                    curation.prune(store)
                if i % config.CONSOLIDATE_EVERY == 0:
                    curation.maybe_consolidate(store, client, step=i)
                    curation.prune(store)

        if i % checkpoint_every == 0:
            cp = _evaluate(domain, store, client, memory_on, eval_tasks, i, raw_seen, use_utility)
            result.checkpoints.append(cp)
            if verbose:
                print(
                    f"    step {i:3d}: held-out {cp.success_rate:.0%}  "
                    f"(memory={cp.memory_size}, raw seen={cp.raw_lessons_seen})"
                )

    result.final_snapshot = store.snapshot(config.RUNS_DIR / f"snapshot_{condition}_seed{seed}.json")
    result.stats = client.stats()
    return result
