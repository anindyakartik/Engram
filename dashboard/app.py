"""Streamlit view of the Engram result. Offline: reads results/report.json only,
no API key and no model calls. Run with: streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import altair as alt
import pandas as pd
import streamlit as st

import config

REPORT = config.ROOT / "results" / "report.json"

MUTED = "#64748b"
COLORS = {"no_memory": "#94a3b8", "naive": "#f59e0b", "engram": "#2563eb"}
LABELS = {"no_memory": "No memory", "naive": "Naive accumulation", "engram": "Engram (curated)"}

st.set_page_config(page_title="Engram", layout="wide")

CSS = """
<style>
#MainMenu, header, footer {visibility: hidden;}
.block-container {max-width: 1060px; padding-top: 2.4rem; padding-bottom: 4rem;}
.eyebrow {font-size: .72rem; letter-spacing: .16em; text-transform: uppercase;
  color: #2563eb; font-weight: 700;}
h1.title {font-size: 3.1rem; line-height: 1.03; margin: .3rem 0 .4rem 0;
  letter-spacing: -.02em; font-weight: 800; color: #0f172a;}
.lede {font-size: 1.12rem; color: #334155; max-width: 720px; line-height: 1.55;}
.section {font-size: .74rem; letter-spacing: .14em; text-transform: uppercase;
  color: #64748b; font-weight: 700; margin: 2.4rem 0 .6rem 0;}
.tiles {display: flex; gap: 14px; flex-wrap: wrap; margin: 1.4rem 0 .4rem 0;}
.tile {flex: 1; min-width: 150px; border: 1px solid #e2e8f0; border-radius: 14px;
  padding: 16px 18px; background: #fff;}
.tile .k {font-size: .72rem; color: #64748b; font-weight: 600;}
.tile .v {font-size: 2.1rem; font-weight: 800; color: #0f172a; letter-spacing: -.02em;}
.tile .s {font-size: .8rem; color: #64748b;}
.tile.accent {background: #2563eb; border-color: #2563eb;}
.tile.accent .k, .tile.accent .s {color: #dbeafe;}
.tile.accent .v {color: #fff;}
.step {border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 16px; background: #fff; height: 100%;}
.step .n {font-size: .7rem; font-weight: 700; color: #2563eb;}
.step .t {font-weight: 700; color: #0f172a; margin: 2px 0 4px 0;}
.step .d {font-size: .86rem; color: #475569; line-height: 1.45;}
.note {font-size: .92rem; color: #475569; line-height: 1.6; max-width: 780px;}
.lesson {border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px 14px; margin-bottom: 10px; background: #fff;}
.lesson .txt {color: #0f172a; font-size: .95rem; line-height: 1.45;}
.lesson .meta {font-size: .78rem; color: #64748b; margin-top: 6px; display: flex; gap: 14px; flex-wrap: wrap;}
.tag {display: inline-block; font-size: .72rem; font-weight: 600; color: #334155;
  background: #f1f5f9; border-radius: 999px; padding: 1px 9px;}
.bar {height: 6px; border-radius: 999px; background: #eef2f7; margin-top: 8px; overflow: hidden;}
.bar > span {display: block; height: 100%; background: #2563eb;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if not REPORT.exists():
    st.error("results/report.json not found. Run: python scripts/run_experiment.py")
    st.stop()

report = json.loads(REPORT.read_text())
agg = report["aggregate"]
h = report["headline"]
n_seeds = agg.get("engram", {}).get("n_seeds", 1)


def find_run(condition: str, seed: int = 1) -> dict | None:
    runs = report.get("runs", [])
    for r in runs:
        if r["condition"] == condition and r["seed"] == seed:
            return r
    for r in runs:
        if r["condition"] == condition:
            return r
    return None


def utility(row: dict) -> float:
    if "utility" in row:
        return float(row["utility"])
    hp, ht = row.get("helped_count", 0), row.get("hurt_count", 0)
    return (hp - ht) / (hp + ht + config.UTILITY_PRIOR)


base = h.get("baseline_pct")
eng = h.get("engram_pct")
naive = h.get("naive_pct")
delta = h.get("improvement_pts")

st.markdown('<div class="eyebrow">Self-improving agent, no fine-tuning</div>', unsafe_allow_html=True)
st.markdown('<h1 class="title">Engram</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="lede">An agent that gets better at a task by writing and curating its own '
    "memory of what worked. It attempts a task, a deterministic checker (not an LLM) grades it, "
    "the agent reflects into a short lesson, and later attempts retrieve those lessons. Progress "
    "is measured on a held-out set it never learns from.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="tiles">
  <div class="tile"><div class="k">No memory (control)</div><div class="v">{base}%</div><div class="s">base model, held-out</div></div>
  <div class="tile accent"><div class="k">Engram (curated)</div><div class="v">{eng}%</div><div class="s">+{delta} points, no fine-tuning</div></div>
  <div class="tile"><div class="k">Naive accumulation</div><div class="v">{naive}%</div><div class="s">keep every lesson</div></div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="note">Mean of {n_seeds} seeds on a text-to-SQL database with undocumented '
    "conventions (integer status codes, a soft-delete flag, cents not dollars, unix-epoch dates). "
    "The agent can only learn these by getting a query wrong and reflecting. Everything below "
    "replays offline from committed recordings, no API key.</div>",
    unsafe_allow_html=True,
)

st.markdown('<div class="section">Held-out improvement</div>', unsafe_allow_html=True)
rows = []
for cond in ("no_memory", "naive", "engram"):
    if cond not in agg:
        continue
    d = agg[cond]
    for step, m in zip(d["steps"], d["mean"], strict=True):
        rows.append({"tasks seen": step, "condition": LABELS[cond], "success": round(100 * m, 1)})
curve = pd.DataFrame(rows)
order = [LABELS[c] for c in ("no_memory", "naive", "engram") if c in agg]
crange = [COLORS[c] for c in ("no_memory", "naive", "engram") if c in agg]
line = (
    alt.Chart(curve)
    .mark_line(point=alt.OverlayMarkDef(size=55, filled=True), strokeWidth=3)
    .encode(
        x=alt.X("tasks seen:Q", title="training tasks seen", axis=alt.Axis(grid=False, tickCount=6)),
        y=alt.Y("success:Q", title="held-out success (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color(
            "condition:N",
            scale=alt.Scale(domain=order, range=crange),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=["condition", "tasks seen", "success"],
    )
    .properties(height=340)
    .configure_view(strokeOpacity=0)
    .configure_axis(labelColor=MUTED, titleColor=MUTED, domainColor="#e2e8f0", tickColor="#e2e8f0")
)
st.altair_chart(line, use_container_width=True)
st.markdown(
    '<div class="note">Each point is success on the held-out pool, which is never used for '
    "learning. Memory lifts the base model by about 41 points. Curated and naive memory land in "
    "the same place on accuracy at this scale; the difference shows up in memory size below.</div>",
    unsafe_allow_html=True,
)

st.markdown('<div class="section">How the loop works</div>', unsafe_allow_html=True)
loop = [
    ("01", "Attempt", "Retrieve relevant lessons, then write one SQL query from the schema alone."),
    ("02", "Check", "A deterministic checker runs the query and compares result sets. Hard pass or fail."),
    ("03", "Reflect", "On a failure, the model proposes a short lesson about a likely hidden convention."),
    ("04", "Curate", "Dedup, track each lesson's utility from real outcomes, consolidate, prune."),
]
cols = st.columns(4)
for col, (n, t, d) in zip(cols, loop, strict=True):
    col.markdown(
        f'<div class="step"><div class="n">{n}</div><div class="t">{t}</div><div class="d">{d}</div></div>',
        unsafe_allow_html=True,
    )

if "engram" in agg and "naive" in agg:
    st.markdown('<div class="section">Curation keeps memory compact</div>', unsafe_allow_html=True)
    comp = []
    for step, raw, mem in zip(
        agg["engram"]["steps"], agg["engram"]["raw_lessons_seen"], agg["engram"]["memory_size"], strict=True
    ):
        comp.append({"tasks seen": step, "series": "Raw lessons proposed", "lessons": round(raw, 1)})
        comp.append({"tasks seen": step, "series": "Engram memory", "lessons": round(mem, 1)})
    for step, mem in zip(agg["naive"]["steps"], agg["naive"]["memory_size"], strict=True):
        comp.append({"tasks seen": step, "series": "Naive memory", "lessons": round(mem, 1)})
    cdf = pd.DataFrame(comp)
    corder = ["Raw lessons proposed", "Naive memory", "Engram memory"]
    ccolor = ["#cbd5e1", "#f59e0b", "#2563eb"]
    comp_chart = (
        alt.Chart(cdf)
        .mark_line(point=alt.OverlayMarkDef(size=45, filled=True), strokeWidth=3)
        .encode(
            x=alt.X("tasks seen:Q", title="training tasks seen", axis=alt.Axis(grid=False, tickCount=6)),
            y=alt.Y("lessons:Q", title="lessons in memory"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=corder, range=ccolor),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=["series", "tasks seen", "lessons"],
        )
        .properties(height=300)
        .configure_view(strokeOpacity=0)
        .configure_axis(labelColor=MUTED, titleColor=MUTED, domainColor="#e2e8f0", tickColor="#e2e8f0")
    )
    st.altair_chart(comp_chart, use_container_width=True)

st.markdown('<div class="section">What the agent taught itself</div>', unsafe_allow_html=True)
run = find_run("engram")
if run is not None:
    snap = run["final_snapshot"]
    st.markdown(
        f'<div class="note">Final curated memory, {snap["count"]} lessons, ranked by measured '
        "utility. These were written by the agent from its own failures, not supplied.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    for les in sorted(snap["lessons"], key=utility, reverse=True):
        u = utility(les)
        scope = les["scope"] + (f" : {les['scope_key']}" if les["scope_key"] else "")
        width = int(max(0.0, min(1.0, (u + 1) / 2)) * 100)
        st.markdown(
            f"""<div class="lesson">
  <div class="txt">{les['content']}</div>
  <div class="bar"><span style="width:{width}%"></span></div>
  <div class="meta">
    <span class="tag">{scope}</span>
    <span>utility {u:+.2f}</span>
    <span>retrieved {les['retrieved_count']} (helped {les['helped_count']}, hurt {les['hurt_count']})</span>
    <span>{les['source']}</span>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

st.markdown('<div class="section">Reproduce</div>', unsafe_allow_html=True)
st.code("ENGRAM_LLM_MODE=replay python scripts/run_experiment.py", language="bash")
st.markdown(
    '<div class="note">Runs offline from committed recordings, so the numbers on this page '
    "reproduce with no API key. Source and method are in the repo README.</div>",
    unsafe_allow_html=True,
)
