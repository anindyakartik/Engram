"""Reporting: aggregate multi-seed runs into curves, confidence intervals, plots.

Produces the machine-readable report.json (means + 95% CIs per condition per
checkpoint, plus headline numbers) and the improvement-curve figure that headlines
the README. All measurement comes from the deterministic verifier; nothing here
involves an LLM.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from engram.eval.harness import RunResult

# Two-sided t multipliers for a 95% CI, indexed by degrees of freedom (n_seeds - 1).
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}

# Colorblind-safe, theme-neutral condition colors.
_COLORS = {"no_memory": "#6b7280", "naive": "#f59e0b", "engram": "#2563eb"}
_LABELS = {"no_memory": "No memory (control)", "naive": "Naive accumulation", "engram": "Engram (curated)"}


def _ci_halfwidth(values: list[float]) -> float:
    """95% CI half-width from the sample (t-based; 0 for a single value)."""
    n = len(values)
    if n < 2:
        return 0.0
    sd = float(np.std(values, ddof=1))
    t = _T95.get(n - 1, 1.96)
    return t * sd / np.sqrt(n)


def aggregate(results: list[RunResult]) -> dict:
    """Aggregate per-(condition, seed) runs into per-condition curves with CIs."""
    by_cond: dict[str, list[RunResult]] = defaultdict(list)
    for r in results:
        by_cond[r.condition].append(r)

    agg: dict[str, dict] = {}
    for cond, runs in by_cond.items():
        steps = [c.step for c in runs[0].checkpoints]
        means, los, his, mem, raw = [], [], [], [], []
        for idx in range(len(steps)):
            rates = [run.checkpoints[idx].success_rate for run in runs]
            m = float(np.mean(rates))
            hw = _ci_halfwidth(rates)
            means.append(m)
            los.append(max(0.0, m - hw))
            his.append(min(1.0, m + hw))
            mem.append(float(np.mean([run.checkpoints[idx].memory_size for run in runs])))
            raw.append(float(np.mean([run.checkpoints[idx].raw_lessons_seen for run in runs])))
        agg[cond] = {
            "steps": steps,
            "mean": means,
            "ci_lo": los,
            "ci_hi": his,
            "memory_size": mem,
            "raw_lessons_seen": raw,
            "n_seeds": len(runs),
        }
    return agg


def summarize(agg: dict) -> dict:
    """Compute the headline numbers from an aggregate."""
    out: dict = {}

    def final(cond: str, key: str = "mean") -> float:
        return agg[cond][key][-1] if cond in agg else float("nan")

    if "no_memory" in agg:
        out["baseline_pct"] = round(100 * final("no_memory"), 1)
    if "engram" in agg:
        out["engram_pct"] = round(100 * final("engram"), 1)
        out["engram_ci"] = [round(100 * agg["engram"]["ci_lo"][-1], 1), round(100 * agg["engram"]["ci_hi"][-1], 1)]
        if "no_memory" in agg:
            out["improvement_pts"] = round(100 * (final("engram") - final("no_memory")), 1)
    if "naive" in agg:
        out["naive_pct"] = round(100 * final("naive"), 1)
        out["engram_minus_naive_pts"] = round(100 * (final("engram") - final("naive")), 1)
    if "engram" in agg:
        out["raw_lessons_seen"] = round(agg["engram"]["raw_lessons_seen"][-1], 1)
        out["curated_memory_size"] = round(agg["engram"]["memory_size"][-1], 1)
    if "naive" in agg:
        out["naive_memory_size"] = round(agg["naive"]["memory_size"][-1], 1)
    out["ordering_holds"] = (
        "engram" in agg
        and "naive" in agg
        and "no_memory" in agg
        and final("engram") > final("naive") > final("no_memory")
    )
    return out


def plot_curve(agg: dict, path: Path) -> None:
    """Plot the held-out improvement curve for all conditions with CI bands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in ("no_memory", "naive", "engram"):
        if cond not in agg:
            continue
        d = agg[cond]
        steps = d["steps"]
        mean = np.array(d["mean"]) * 100
        lo = np.array(d["ci_lo"]) * 100
        hi = np.array(d["ci_hi"]) * 100
        color = _COLORS[cond]
        ax.plot(steps, mean, marker="o", color=color, label=_LABELS[cond], linewidth=2)
        ax.fill_between(steps, lo, hi, color=color, alpha=0.15)

    ax.set_xlabel("Training tasks seen")
    ax.set_ylabel("Held-out success rate (%)")
    ax.set_title("Engram: held-out improvement with no fine-tuning")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    n_seeds = agg.get("engram", {}).get("n_seeds", 1)
    fig.text(0.99, 0.01, f"mean of {n_seeds} seeds, 95% CI bands", ha="right", va="bottom", fontsize=8, color="#6b7280")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_compaction(agg: dict, path: Path) -> None:
    """Plot raw lessons proposed vs curated memory size over training (engram)."""
    if "engram" not in agg:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = agg["engram"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(d["steps"], d["raw_lessons_seen"], marker="o", color="#9ca3af", label="Raw lessons proposed (cumulative)")
    ax.plot(d["steps"], d["memory_size"], marker="o", color="#2563eb", label="Curated memory size")
    if "naive" in agg:
        ax.plot(agg["naive"]["steps"], agg["naive"]["memory_size"], marker="o", color="#f59e0b", label="Naive memory size")
    ax.set_xlabel("Training tasks seen")
    ax.set_ylabel("Number of lessons")
    ax.set_title("Curation keeps memory compact")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(results: list[RunResult], outdir: Path) -> dict:
    """Aggregate, plot, and persist the report. Returns the headline summary."""
    outdir.mkdir(parents=True, exist_ok=True)
    agg = aggregate(results)
    headline = summarize(agg)
    (outdir / "report.json").write_text(
        json.dumps({"headline": headline, "aggregate": agg, "runs": [r.to_dict() for r in results]}, indent=2)
    )
    plot_curve(agg, outdir / "curve.png")
    plot_compaction(agg, outdir / "compaction.png")
    return headline
