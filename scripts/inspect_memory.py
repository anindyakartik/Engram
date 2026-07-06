"""Print the lessons the agent taught itself - the "look what it learned" artifact.

Reads the committed results/report.json (which embeds each run's final memory
snapshot) and prints the curated lessons for the Engram condition, ranked by measured
utility. Needs no API key.

    python scripts/inspect_memory.py
    python scripts/inspect_memory.py --condition naive --seed 1
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config

PRIOR = config.UTILITY_PRIOR


def _utility(row: dict) -> float:
    if "utility" in row:
        return float(row["utility"])
    helped, hurt = row.get("helped_count", 0), row.get("hurt_count", 0)
    return (helped - hurt) / (helped + hurt + PRIOR)


def _find_run(report: dict, condition: str, seed: int) -> dict | None:
    for run in report.get("runs", []):
        if run["condition"] == condition and run["seed"] == seed:
            return run
    # Fall back to the first run of the condition.
    for run in report.get("runs", []):
        if run["condition"] == condition:
            return run
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the agent's curated memory.")
    parser.add_argument("--report", default=str(config.ROOT / "results" / "report.json"))
    parser.add_argument("--condition", default="engram")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    path = pathlib.Path(args.report)
    if not path.exists():
        print(f"No report at {path}. Run: python scripts/run_experiment.py")
        return 1

    report = json.loads(path.read_text())
    run = _find_run(report, args.condition, args.seed)
    if run is None:
        print(f"No '{args.condition}' run found in report.")
        return 1

    snap = run["final_snapshot"]
    lessons = sorted(snap["lessons"], key=_utility, reverse=True)

    print("=" * 74)
    print(f"CURATED MEMORY - condition={args.condition} seed={args.seed}")
    print(f"{snap['count']} lessons  |  scopes={snap['by_scope']}  |  sources={snap['by_source']}")
    print("=" * 74)
    for lesson in lessons:
        util = _utility(lesson)
        scope = lesson["scope"] + (f":{lesson['scope_key']}" if lesson["scope_key"] else "")
        badge = "consolidated" if lesson["source"] == "consolidation" else "reflected"
        print(f"\n- {lesson['content']}")
        print(
            f"    scope={scope}  utility={util:+.2f}  "
            f"retrieved={lesson['retrieved_count']} (helped={lesson['helped_count']}, "
            f"hurt={lesson['hurt_count']})  [{badge}]"
        )
    print("\n" + "=" * 74)
    headline = report.get("headline", {})
    if headline:
        print(
            f"Held-out: {headline.get('baseline_pct')}% -> {headline.get('engram_pct')}%  |  "
            f"raw lessons seen: {headline.get('raw_lessons_seen')} -> curated: {headline.get('curated_memory_size')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
