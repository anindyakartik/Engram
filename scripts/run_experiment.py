"""Run the Engram experiment: baseline + naive + full Engram, and build the curve.

One command produces the whole headline result. On first run (LLM_MODE=auto) live
calls are recorded to committed cassettes; afterwards - and on any fresh clone with
no API key (LLM_MODE=replay) - it reproduces the identical curve offline.

    python scripts/run_experiment.py            # pilot (config defaults)
    python scripts/run_experiment.py --full     # full run (more train tasks + seeds)
    ENGRAM_LLM_MODE=replay python scripts/run_experiment.py   # offline reproduction
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

import config
from engram.core.llm import LLMClient
from engram.domains.text_to_sql import TextToSQLDomain
from engram.eval.harness import run_condition
from engram.eval.report import write_report

load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Engram experiment.")
    parser.add_argument("--full", action="store_true", help="use the full-scale parameters")
    parser.add_argument("--outdir", default=str(config.ROOT / "results"), help="output directory")
    args = parser.parse_args()

    if args.full:
        config.apply_full_scale()

    client = LLMClient()
    domain = TextToSQLDomain()
    outdir = pathlib.Path(args.outdir)

    print(
        f"Engram experiment | mode={client.mode} | "
        f"train={config.N_TRAIN} checkpoint_every={config.CHECKPOINT_EVERY} "
        f"seeds={config.N_SEEDS} eval={config.EVAL_SET_SIZE}"
    )
    print(f"conditions: {', '.join(config.CONDITIONS)}\n")

    results = []
    for condition in config.CONDITIONS:
        for seed in range(1, config.N_SEEDS + 1):
            print(f"--- {condition} | seed {seed} ---")
            res = run_condition(domain, condition, seed, client, verbose=True)
            results.append(res)
            curve = " ".join(f"{c.step}:{c.success_rate:.0%}" for c in res.checkpoints)
            print(f"    curve: {curve}  (final memory={res.checkpoints[-1].memory_size})\n")

    headline = write_report(results, outdir)
    _print_headline(headline, client, outdir)
    return 0


def _print_headline(h: dict, client: LLMClient, outdir: pathlib.Path) -> None:
    print("=" * 68)
    print("HEADLINE RESULTS (held-out, never learned from)")
    print("=" * 68)
    ci = h.get("engram_ci", [None, None])
    print(
        f"  Held-out success: {h['baseline_pct']}% (no memory) -> "
        f"{h['engram_pct']}% (Engram)   +{h['improvement_pts']} pts, no fine-tuning"
    )
    if ci[0] is not None:
        print(f"                    Engram 95% CI: [{ci[0]}%, {ci[1]}%]")
    print(
        f"  Curation matters: Engram {h['engram_pct']}% vs naive-accumulation "
        f"{h.get('naive_pct')}%  (+{h.get('engram_minus_naive_pts')} pts)"
    )
    print(
        f"  Memory stayed compact: {h.get('raw_lessons_seen')} raw lessons -> "
        f"{h.get('curated_memory_size')} curated (naive kept {h.get('naive_memory_size')})"
    )
    print(f"  Ordering holds (engram > naive > no_memory): {h.get('ordering_holds')}")
    print("-" * 68)
    print(f"  LLM usage: {client.stats()}")
    print(f"  Artifacts: {outdir/'curve.png'}, {outdir/'report.json'}")
    print("=" * 68)


if __name__ == "__main__":
    sys.exit(main())
