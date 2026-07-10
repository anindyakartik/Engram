"""Streamlit view of the Engram result. Offline: reads results/report.json and
results/showcase.json only, no API key and no model calls.

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from html import escape

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import altair as alt
import pandas as pd
import streamlit as st

import config

RESULTS = config.ROOT / "results"
MUTED = "#64748b"
COLORS = {"no_memory": "#94a3b8", "naive": "#f59e0b", "engram": "#2563eb"}
LABELS = {"no_memory": "No memory", "naive": "Naive accumulation", "engram": "Engram (curated)"}

st.set_page_config(page_title="Engram", layout="wide")

CSS = """
<style>
#MainMenu, header, footer {visibility: hidden;}
html, body, [class*="css"] {font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;}
.block-container {max-width: 1080px; padding-top: 2rem; padding-bottom: 5rem;}
:root {--ink:#0f172a; --muted:#64748b; --line:#e6eaf0; --blue:#2563eb; --green:#16a34a; --red:#dc2626;}

.hero {background: linear-gradient(135deg,#f8faff 0%,#eef4ff 100%);
  border:1px solid #e3ebfb; border-radius:22px; padding:34px 34px 30px 34px; margin-bottom:8px;}
.eyebrow {font-size:.72rem; letter-spacing:.18em; text-transform:uppercase; color:var(--blue); font-weight:700;}
.title {font-size:3.4rem; line-height:1.0; margin:.35rem 0 .5rem 0; letter-spacing:-.03em; font-weight:800; color:var(--ink);}
.lede {font-size:1.14rem; color:#334155; max-width:730px; line-height:1.55;}
.bigstat {display:flex; align-items:baseline; gap:14px; margin-top:22px; flex-wrap:wrap;}
.bigstat .from {font-size:1.5rem; font-weight:700; color:var(--muted);}
.bigstat .arrow {color:#94a3b8; font-size:1.3rem;}
.bigstat .to {font-size:3.2rem; font-weight:850; color:var(--blue); letter-spacing:-.03em; line-height:1;}
.bigstat .cap {font-size:.95rem; color:#475569; margin-left:6px;}
.badges {margin-top:14px; display:flex; gap:8px; flex-wrap:wrap;}
.badge {font-size:.74rem; font-weight:600; color:#334155; background:#fff; border:1px solid var(--line);
  border-radius:999px; padding:4px 11px;}

.section {font-size:.75rem; letter-spacing:.15em; text-transform:uppercase; color:var(--muted);
  font-weight:700; margin:2.8rem 0 .3rem 0;}
.sub {font-size:.96rem; color:#475569; line-height:1.6; max-width:800px; margin-bottom:.6rem;}

.tiles {display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:1.2rem 0 .3rem 0;}
.tile {border:1px solid var(--line); border-radius:16px; padding:18px 20px; background:#fff;
  box-shadow:0 1px 2px rgba(15,23,42,.04);}
.tile .k {font-size:.74rem; color:var(--muted); font-weight:600;}
.tile .v {font-size:2.3rem; font-weight:800; color:var(--ink); letter-spacing:-.02em;}
.tile .s {font-size:.8rem; color:var(--muted);}
.tile.accent {background:var(--blue); border-color:var(--blue);}
.tile.accent .k,.tile.accent .s {color:#dbeafe;} .tile.accent .v {color:#fff;}

.demo {border:1px solid var(--line); border-radius:16px; background:#fff; padding:18px 20px; margin-bottom:14px;
  box-shadow:0 1px 2px rgba(15,23,42,.04);}
.demo .q {font-weight:700; color:var(--ink); font-size:1.02rem;}
.demo .tag {display:inline-block; font-size:.68rem; font-weight:700; letter-spacing:.04em; text-transform:uppercase;
  color:#3730a3; background:#eef2ff; border-radius:6px; padding:2px 8px; margin-bottom:8px;}
.ba {display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px;}
@media (max-width:760px){.ba{grid-template-columns:1fr;} .tiles{grid-template-columns:1fr;}}
.side .lab {font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; margin-bottom:5px;}
.side.bad .lab {color:var(--red);} .side.good .lab {color:var(--green);}
.sql {font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.82rem; line-height:1.5;
  padding:11px 13px; border-radius:10px; white-space:pre-wrap; word-break:break-word;}
.side.bad .sql {background:#fef2f2; border:1px solid #fecaca; color:#7f1d1d;}
.side.good .sql {background:#f0fdf4; border:1px solid #bbf7d0; color:#14532d;}
.learned {margin-top:12px; font-size:.9rem; color:#1e293b; background:#f8fafc; border-left:3px solid var(--blue);
  border-radius:0 8px 8px 0; padding:9px 13px;}
.learned b {color:var(--blue);}

.steps {display:grid; grid-template-columns:repeat(4,1fr); gap:12px;}
@media (max-width:760px){.steps{grid-template-columns:1fr 1fr;}}
.step {border:1px solid var(--line); border-radius:14px; padding:15px 16px; background:#fff; height:100%;}
.step .n {font-size:.72rem; font-weight:800; color:var(--blue);}
.step .t {font-weight:700; color:var(--ink); margin:3px 0 5px 0;}
.step .d {font-size:.85rem; color:#475569; line-height:1.45;}

.lesson {border:1px solid var(--line); border-radius:12px; padding:13px 15px; margin-bottom:10px; background:#fff;}
.lesson .txt {color:var(--ink); font-size:.95rem; line-height:1.45;}
.lesson .meta {font-size:.78rem; color:var(--muted); margin-top:7px; display:flex; gap:16px; flex-wrap:wrap;}
.chip {display:inline-block; font-size:.72rem; font-weight:600; color:#334155; background:#f1f5f9;
  border-radius:999px; padding:1px 9px;}
.bar {height:6px; border-radius:999px; background:#eef2f7; margin-top:9px; overflow:hidden;}
.bar > span {display:block; height:100%; background:linear-gradient(90deg,#60a5fa,#2563eb);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if not (RESULTS / "report.json").exists():
    st.error("results/report.json not found. Run: python scripts/run_experiment.py")
    st.stop()

report = json.loads((RESULTS / "report.json").read_text())
show = json.loads((RESULTS / "showcase.json").read_text()) if (RESULTS / "showcase.json").exists() else None
agg = report["aggregate"]
h = report["headline"]
n_seeds = agg.get("engram", {}).get("n_seeds", 1)


def find_run(condition: str, seed: int = 1) -> dict | None:
    for r in report.get("runs", []):
        if r["condition"] == condition and r["seed"] == seed:
            return r
    for r in report.get("runs", []):
        if r["condition"] == condition:
            return r
    return None


def utility(row: dict) -> float:
    if "utility" in row:
        return float(row["utility"])
    hp, ht = row.get("helped_count", 0), row.get("hurt_count", 0)
    return (hp - ht) / (hp + ht + config.UTILITY_PRIOR)


# ---------- hero ----------
st.markdown(
    f"""
<div class="hero">
  <div class="eyebrow">Memory-augmented agent &middot; no fine-tuning</div>
  <div class="title">Engram</div>
  <div class="lede">An agent that gets better at a task by writing and curating its own memory of what
  worked. It tries a task, a deterministic checker (not an LLM) grades it, it reflects into a short
  lesson, and later tries retrieve those lessons. Everything is measured on a held-out set it never
  learns from.</div>
  <div class="bigstat">
    <span class="from">{h.get('baseline_pct')}%</span>
    <span class="arrow">&rarr;</span>
    <span class="to">{h.get('engram_pct')}%</span>
    <span class="cap">held-out success, +{h.get('improvement_pts')} points, no weight updates</span>
  </div>
  <div class="badges">
    <span class="badge">deterministic checker, no LLM judge</span>
    <span class="badge">held-out generalization</span>
    <span class="badge">reproduces offline, no API key</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="tiles">
  <div class="tile"><div class="k">No memory (control)</div><div class="v">{h.get('baseline_pct')}%</div><div class="s">base model, held-out</div></div>
  <div class="tile accent"><div class="k">Engram (curated)</div><div class="v">{h.get('engram_pct')}%</div><div class="s">+{h.get('improvement_pts')} points</div></div>
  <div class="tile"><div class="k">Naive accumulation</div><div class="v">{h.get('naive_pct')}%</div><div class="s">keep every lesson</div></div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- watch it learn ----------
if show and show.get("examples"):
    st.markdown('<div class="section">Watch it learn</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub">Held-out questions the base model got wrong, and the same questions after the '
        f"agent taught itself the database's hidden rules. On this run it went from solving "
        f"<b>{show['base_pass']} of {show['n_eval']}</b> held-out tasks to <b>{show['engram_pass']} of "
        f"{show['n_eval']}</b>; {show['n_flips']} flipped from wrong to right. The SQL below is the agent's "
        "actual output.</div>",
        unsafe_allow_html=True,
    )
    for ex in show["examples"]:
        lesson = ex["lessons"][0] if ex["lessons"] else "a rule learned from an earlier failure"
        st.markdown(
            f"""
<div class="demo">
  <div class="tag">{escape(ex['task_type'])}</div>
  <div class="q">{escape(ex['question'])}</div>
  <div class="ba">
    <div class="side bad"><div class="lab">Base model &middot; wrong</div><div class="sql">{escape(ex['base_sql'])}</div></div>
    <div class="side good"><div class="lab">Engram &middot; correct</div><div class="sql">{escape(ex['engram_sql'])}</div></div>
  </div>
  <div class="learned"><b>Lesson it wrote:</b> {escape(lesson)}</div>
</div>
""",
            unsafe_allow_html=True,
        )

# ---------- curve ----------
st.markdown('<div class="section">Held-out improvement</div>', unsafe_allow_html=True)
rows = []
for cond in ("no_memory", "naive", "engram"):
    if cond in agg:
        for step, m in zip(agg[cond]["steps"], agg[cond]["mean"], strict=True):
            rows.append({"tasks seen": step, "condition": LABELS[cond], "success": round(100 * m, 1)})
curve = pd.DataFrame(rows)
order = [LABELS[c] for c in ("no_memory", "naive", "engram") if c in agg]
crange = [COLORS[c] for c in ("no_memory", "naive", "engram") if c in agg]
scale = alt.Scale(domain=order, range=crange)
enc_x = alt.X("tasks seen:Q", title="training tasks seen", axis=alt.Axis(grid=False, tickCount=6))
enc_y = alt.Y("success:Q", title="held-out success (%)", scale=alt.Scale(domain=[0, 100]))
color = alt.Color("condition:N", scale=scale, legend=alt.Legend(orient="top", title=None))
line = alt.Chart(curve).mark_line(point=alt.OverlayMarkDef(size=60, filled=True), strokeWidth=3).encode(
    x=enc_x, y=enc_y, color=color, tooltip=["condition", "tasks seen", "success"]
)
last = curve.sort_values("tasks seen").groupby("condition", as_index=False).last()
labels = alt.Chart(last).mark_text(align="left", dx=8, fontSize=12, fontWeight="bold").encode(
    x="tasks seen:Q", y="success:Q", text=alt.Text("success:Q", format=".0f"), color=color
)
chart = (line + labels).properties(height=360).configure_view(strokeOpacity=0).configure_axis(
    labelColor=MUTED, titleColor=MUTED, domainColor="#e2e8f0", tickColor="#e2e8f0", labelFontSize=12
)
st.altair_chart(chart, use_container_width=True)
st.markdown(
    f'<div class="sub">Mean of {n_seeds} seeds. Each point is success on the held-out pool, never used '
    "for learning. Memory lifts the base model by about 41 points. Curated and naive memory tie on "
    "accuracy at this scale; the difference is memory size, below.</div>",
    unsafe_allow_html=True,
)

# ---------- loop ----------
st.markdown('<div class="section">How the loop works</div>', unsafe_allow_html=True)
loop = [
    ("01", "Attempt", "Retrieve relevant lessons, then write one SQL query from the schema alone."),
    ("02", "Check", "A deterministic checker runs the query and compares result sets. Hard pass or fail."),
    ("03", "Reflect", "On a failure, the model proposes a short lesson about a likely hidden rule."),
    ("04", "Curate", "Dedup, score each lesson's utility from real outcomes, consolidate, prune."),
]
st.markdown(
    '<div class="steps">'
    + "".join(
        f'<div class="step"><div class="n">{n}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
        for n, t, d in loop
    )
    + "</div>",
    unsafe_allow_html=True,
)

# ---------- compaction ----------
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
    corder, ccolor = ["Raw lessons proposed", "Naive memory", "Engram memory"], ["#cbd5e1", "#f59e0b", "#2563eb"]
    comp_chart = alt.Chart(cdf).mark_line(point=alt.OverlayMarkDef(size=50, filled=True), strokeWidth=3).encode(
        x=alt.X("tasks seen:Q", title="training tasks seen", axis=alt.Axis(grid=False, tickCount=6)),
        y=alt.Y("lessons:Q", title="lessons in memory"),
        color=alt.Color("series:N", scale=alt.Scale(domain=corder, range=ccolor), legend=alt.Legend(orient="top", title=None)),
        tooltip=["series", "tasks seen", "lessons"],
    ).properties(height=300).configure_view(strokeOpacity=0).configure_axis(
        labelColor=MUTED, titleColor=MUTED, domainColor="#e2e8f0", tickColor="#e2e8f0"
    )
    st.altair_chart(comp_chart, use_container_width=True)
    st.markdown(
        '<div class="sub">Same accuracy, smaller memory. Dedup merges duplicate lessons, so curated memory '
        "stays compact while naive accumulation keeps every restatement. This is where curation pays off at "
        "this scale; the accuracy payoff grows with longer runs.</div>",
        unsafe_allow_html=True,
    )

# ---------- lessons ----------
st.markdown('<div class="section">What the agent taught itself</div>', unsafe_allow_html=True)
run = find_run("engram")
if run is not None:
    snap = run["final_snapshot"]
    st.markdown(
        f'<div class="sub">Final curated memory, {snap["count"]} lessons, ranked by measured utility. '
        "Written by the agent from its own failures, not supplied.</div>",
        unsafe_allow_html=True,
    )
    for les in sorted(snap["lessons"], key=utility, reverse=True):
        u = utility(les)
        scope = les["scope"] + (f" : {les['scope_key']}" if les["scope_key"] else "")
        width = int(max(0.0, min(1.0, (u + 1) / 2)) * 100)
        st.markdown(
            f"""<div class="lesson">
  <div class="txt">{escape(les['content'])}</div>
  <div class="bar"><span style="width:{width}%"></span></div>
  <div class="meta"><span class="chip">{escape(scope)}</span><span>utility {u:+.2f}</span>
  <span>retrieved {les['retrieved_count']} (helped {les['helped_count']}, hurt {les['hurt_count']})</span>
  <span>{escape(les['source'])}</span></div>
</div>""",
            unsafe_allow_html=True,
        )

# ---------- reproduce ----------
st.markdown('<div class="section">Reproduce</div>', unsafe_allow_html=True)
st.code("ENGRAM_LLM_MODE=replay python scripts/run_experiment.py", language="bash")
st.markdown(
    '<div class="sub">Every number and query on this page replays from committed recordings, so it '
    "reproduces with no API key. Method and source are in the repository README.</div>",
    unsafe_allow_html=True,
)
