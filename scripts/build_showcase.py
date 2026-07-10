"""Build results/showcase.json: concrete held-out cases where memory flips a wrong
answer into a right one. Runs entirely from committed cassettes (no API key), so the
showcase reproduces offline exactly like the rest of the result.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config
from engram.agent.reflect import reflect
from engram.agent.runtime import run_attempt
from engram.core.llm import LLMClient
from engram.domains.text_to_sql import TextToSQLDomain
from engram.eval.harness import seeded_stream
from engram.memory import curation
from engram.memory.store import MemoryStore

SEED = 1


def build_engram_store(domain: TextToSQLDomain, client: LLMClient) -> MemoryStore:
    """Reproduce the seed-1 engram memory by replaying the training loop."""
    store = MemoryStore()
    schema = domain.describe()
    for i, task in enumerate(seeded_stream(domain.train_pool(), config.N_TRAIN, SEED), start=1):
        store.step = i
        trace, retrieved = run_attempt(domain, task, client, store, memory_on=True, use_utility=True)
        curation.record_utility(retrieved, trace.success)
        if not trace.success or config.REFLECT_ON_SUCCESS:
            cands = reflect(trace, schema, client)
            curation.ingest_candidates(store, cands, client, step=i, provenance=task.id, curate=True)
        if i % config.PRUNE_EVERY == 0:
            curation.prune(store)
        if i % config.CONSOLIDATE_EVERY == 0:
            curation.maybe_consolidate(store, client, step=i)
            curation.prune(store)
    return store


def main() -> int:
    client = LLMClient(mode="replay")
    domain = TextToSQLDomain()
    store = build_engram_store(domain, client)

    flips, base_pass, eng_pass = [], 0, 0
    for task in domain.eval_pool():
        base_trace, _ = run_attempt(domain, task, client, store, memory_on=False)
        eng_trace, retrieved = run_attempt(domain, task, client, store, memory_on=True, use_utility=True)
        base_pass += int(base_trace.success)
        eng_pass += int(eng_trace.success)
        if (not base_trace.success) and eng_trace.success:
            flips.append(
                {
                    "task_type": task.task_type,
                    "question": task.question,
                    "base_sql": base_trace.answer,
                    "engram_sql": eng_trace.answer,
                    "lessons": [r.lesson.content for r in retrieved],
                }
            )

    # One representative flip per convention, for a varied story.
    seen, picked = set(), []
    for f in flips:
        if f["task_type"] not in seen:
            seen.add(f["task_type"])
            picked.append(f)

    out = {
        "n_eval": len(domain.eval_pool()),
        "base_pass": base_pass,
        "engram_pass": eng_pass,
        "n_flips": len(flips),
        "examples": picked[:4],
    }
    path = config.ROOT / "results" / "showcase.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}: base {base_pass}/{out['n_eval']}, engram {eng_pass}/{out['n_eval']}, "
          f"{len(flips)} flips, {len(out['examples'])} shown, live_calls={client.stats()['live_calls']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
