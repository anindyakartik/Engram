"""Central configuration for Engram.

Every tunable lives here so experiments are reproducible and defensible: the same
config + the same seed yields the same training stream, the same lessons, and the
same improvement curve. Values are grouped by concern. See README.md for the
rationale behind the curation thresholds.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
CASSETTE_DIR = ROOT / "cassettes"  # committed; enables offline replay
RUNS_DIR = ROOT / "runs"  # git-ignored generated artifacts
DATA_DIR = ROOT / "data"

# --------------------------------------------------------------------------- #
# LLM / SDK
# --------------------------------------------------------------------------- #
# One constant for the agent model so the whole project swaps in one place.
GEMINI_MODEL = "gemini-flash-lite-latest"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768  # requested output dimensionality for embeddings

# Deterministic generation. Low temperature + fixed seed so live runs reproduce
# and cassettes stay stable.
TEMPERATURE = 0.0
LLM_SEED = 7

# LLM_MODE: how core/llm.py resolves calls.
#   auto   -> replay if a cassette exists, else call live and record it
#   replay -> replay only; error if a cassette is missing (fully offline, no key)
#   record -> always call live and (over)write the cassette
#   live   -> always call live, do not touch cassettes
LLM_MODE = os.environ.get("ENGRAM_LLM_MODE", "auto")

# Free-tier friendliness: shared token bucket + exponential backoff on HTTP 429.
RATE_LIMIT_RPM = 12  # requests per minute budget (Flash-Lite free tier is ~15)
BACKOFF_SCHEDULE = (1, 2, 4, 8)  # seconds; retried in order on 429/transient errors

# Published free-tier-era rates ($ per 1M tokens) for cost accounting only.
PRICE_INPUT_PER_1M = 0.10
PRICE_OUTPUT_PER_1M = 0.40
PRICE_EMBED_PER_1M = 0.15

# --------------------------------------------------------------------------- #
# Global experiment seed
# --------------------------------------------------------------------------- #
SEED = 7

# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
RETRIEVAL_K = 4  # max lessons injected into an attempt
SIM_THRESHOLD_RETRIEVAL = 0.55  # cosine floor; below this a lesson is not injected

# Utility-aware retrieval (an Engram-only curation feature): once a lesson has had
# this many retrievals and its measured utility is still negative, it is proven
# net-harmful and excluded from future retrieval - so bad lessons stop being injected
# long before they would be pruned. Naive accumulation does NOT use this.
UTILITY_EVIDENCE_MIN = 3

# --------------------------------------------------------------------------- #
# Curation thresholds (the core mechanism - every value is defended in README)
# --------------------------------------------------------------------------- #
DEDUP_SIM_THRESHOLD = 0.90  # >= this vs an existing lesson -> merge, don't add
CONSOLIDATE_SIM_THRESHOLD = 0.80  # cluster tightness for consolidating specifics
CONSOLIDATE_MIN_CLUSTER = 3  # need this many similar lessons before consolidating
CONSOLIDATE_EVERY = 10  # run consolidation pass every N training tasks

# Pruning: a lesson must earn its place. Only consider pruning once it has had a
# fair number of retrieval opportunities, then drop it if its utility is poor.
PRUNE_MIN_RETRIEVED = 5  # min retrievals before a lesson is eligible for pruning
PRUNE_UTILITY_FLOOR = 0.0  # utility score at/below which an eligible lesson is pruned
PRUNE_EVERY = 3  # run a (cheap, no-LLM) prune pass every N training steps (engram)

# Utility scoring: a Wilson-style shrunk estimate of helped-rate centered at 0.
# helped bumps up, hurt bumps down; see memory/curation.py for the exact formula.
UTILITY_PRIOR = 1.0  # pseudo-count smoothing so a single sample can't dominate

# --------------------------------------------------------------------------- #
# Eval harness parameters. Pilot vs full are just parameter sets (see below).
# --------------------------------------------------------------------------- #
N_TRAIN = 30  # training-stream length (pilot default)
CHECKPOINT_EVERY = 6  # measure held-out success every N training tasks
N_SEEDS = 2  # seeds per condition (pilot default)
EVAL_SET_SIZE = 22  # held-out tasks measured per checkpoint (full eval pool)

# Reflection budget: reflect on failed training attempts (successes reinforce
# existing lessons via utility rather than spawning new ones - keeps memory compact).
REFLECT_ON_SUCCESS = False

# Conditions compared on the identical held-out set.
CONDITIONS = ("no_memory", "naive", "engram")

# The full run (scaled up after the pilot passes the result gate). The eval pool is
# fixed at 22 tasks, so the full run scales the training stream and seed count.
FULL = {
    "N_TRAIN": 60,
    "CHECKPOINT_EVERY": 12,
    "N_SEEDS": 3,
    "EVAL_SET_SIZE": 22,
}


def apply_full_scale() -> None:
    """Overwrite the pilot parameters with the full-run parameters in-process."""
    global N_TRAIN, CHECKPOINT_EVERY, N_SEEDS, EVAL_SET_SIZE
    N_TRAIN = FULL["N_TRAIN"]
    CHECKPOINT_EVERY = FULL["CHECKPOINT_EVERY"]
    N_SEEDS = FULL["N_SEEDS"]
    EVAL_SET_SIZE = FULL["EVAL_SET_SIZE"]
