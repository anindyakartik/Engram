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

# Palette: validated with the categorical checker (lightness band, chroma floor,
# CVD separation, contrast) for the two hued series. The control series is a true
# neutral, kept out of the hue rotation on purpose and carried by a dashed line
# plus a direct label, not by color alone.
PAPER = "#f7f3ea"
INK = "#1c1a17"
INK_SOFT = "#5b5548"
HAIR = "#ddd2b6"
BLUE = "#2a5aad"   # engram
RUST = "#c05a2a"   # naive
GRAY = "#948c78"   # no memory (control, dashed)
GOOD = "#1c6b3f"
BAD = "#b3261e"

COLORS = {"no_memory": GRAY, "naive": RUST, "engram": BLUE}

st.set_page_config(page_title="Engram", page_icon=None, layout="wide")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,440;9..144,560;9..144,680&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {{
  --paper: {PAPER}; --ink: {INK}; --ink-soft: {INK_SOFT}; --hair: {HAIR};
  --blue: {BLUE}; --rust: {RUST}; --gray: {GRAY}; --good: {GOOD}; --bad: {BAD};
  --serif: "Fraunces", "Iowan Old Style", Georgia, serif;
  --body: "Source Serif 4", Georgia, "Times New Roman", serif;
  --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}}

#MainMenu, header, footer {{visibility: hidden;}}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
  background: var(--paper) !important;
}}
.block-container {{max-width: 940px; padding-top: 3rem; padding-bottom: 6rem;}}
* {{ box-sizing: border-box; }}

/* ---- masthead ---- */
.spine {{display:flex; align-items:center; gap:10px; margin-bottom:.6rem;}}
.spine .bar {{width:22px; height:3px; background:var(--blue);}}
.eyebrow {{font-family:var(--mono); font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--ink-soft);}}
.wordmark {{font-family:var(--serif); font-optical-sizing:auto; font-weight:680; font-size:4.6rem;
  line-height:.92; letter-spacing:-.01em; color:var(--ink); margin:.1rem 0 .9rem 0;}}
.lede {{font-family:var(--body); font-size:1.22rem; line-height:1.62; color:#2c2924; max-width:680px;}}
.rule {{border:none; border-top:1px solid var(--hair); margin:1.9rem 0;}}
.rule.thick {{border-top:2px solid var(--ink); margin-bottom:2px;}}

/* ---- headline stat ---- */
.statline {{display:flex; align-items:baseline; gap:20px; flex-wrap:wrap; margin: 1.6rem 0 .3rem 0;}}
.statline .n {{font-family:var(--serif); font-weight:600; font-size:4.4rem; line-height:1;
  letter-spacing:-.02em; font-variant-numeric: oldstyle-nums;}}
.statline .from {{color:var(--ink-soft);}}
.statline .to {{color:var(--blue);}}
.statline .arrow {{font-family:var(--body); font-size:1.7rem; color:var(--hair); font-weight:400;}}
.statcap {{font-family:var(--mono); font-size:.82rem; color:var(--ink-soft); letter-spacing:.02em;
  margin-top:.2rem;}}
.statcap b {{color:var(--ink);}}

.badgerow {{display:flex; gap:22px; flex-wrap:wrap; margin:1.5rem 0 .4rem 0;}}
.badgerow span {{font-family:var(--mono); font-size:.74rem; color:var(--ink-soft);}}
.badgerow span::before {{content:"\\2013\\2002"; color:var(--blue);}}

/* ---- three-figure row ---- */
.figrow {{display:flex; margin:2.2rem 0 0 0; border-top:1px solid var(--hair);}}
.fig {{flex:1; padding:16px 22px 4px 0; border-left:1px solid var(--hair);}}
.fig:first-child {{border-left:none; padding-left:0;}}
.fig .k {{font-family:var(--mono); font-size:.72rem; letter-spacing:.06em; text-transform:uppercase;
  color:var(--ink-soft);}}
.fig .v {{font-family:var(--serif); font-weight:600; font-size:2.5rem; letter-spacing:-.01em;
  font-variant-numeric: oldstyle-nums; margin:.15rem 0;}}
.fig.accent .v {{color:var(--blue);}}
.fig .s {{font-family:var(--body); font-size:.92rem; color:var(--ink-soft); font-style:italic;}}

/* ---- section headers ---- */
.section {{display:flex; align-items:baseline; gap:10px; margin:3.4rem 0 .3rem 0;}}
.section .idx {{font-family:var(--mono); font-size:.78rem; color:var(--blue);}}
.section .ttl {{font-family:var(--serif); font-weight:600; font-size:1.5rem; color:var(--ink);
  letter-spacing:-.01em;}}
.section-rule {{border:none; border-top:1px solid var(--hair); margin:.35rem 0 1.1rem 0;}}
.sub {{font-family:var(--body); font-size:1.02rem; color:#3a362f; line-height:1.62; max-width:760px;
  margin-bottom:1.1rem;}}
.sub b {{color:var(--ink);}}

/* ---- watch it learn: figure cards ---- */
.demo {{padding:20px 0 22px 0; border-top:1px solid var(--hair);}}
.demo .figtag {{font-family:var(--mono); font-size:.74rem; letter-spacing:.05em; color:var(--ink-soft);
  text-transform:uppercase; margin-bottom:.5rem;}}
.demo .figtag b {{color:var(--blue);}}
.demo .q {{font-family:var(--body); font-style:italic; font-size:1.14rem; color:var(--ink); line-height:1.5;
  margin-bottom:1rem;}}
.ba {{display:grid; grid-template-columns:1fr 1fr; gap:20px;}}
@media (max-width:760px){{.ba{{grid-template-columns:1fr;}} .figrow{{flex-direction:column;}}
  .fig{{border-left:none; border-top:1px solid var(--hair); padding:14px 0;}} .fig:first-child{{border-top:none;}}}}
.side {{border-left:3px solid var(--bad); padding-left:14px;}}
.side.good {{border-left-color:var(--good);}}
.side .lab {{font-family:var(--mono); font-size:.72rem; font-weight:600; letter-spacing:.04em;
  text-transform:uppercase; color:var(--bad); margin-bottom:6px;}}
.side.good .lab {{color:var(--good);}}
.sql {{font-family:var(--mono); font-size:.83rem; line-height:1.55; color:var(--ink); white-space:pre-wrap;
  word-break:break-word;}}
.learned {{margin-top:14px; padding-left:14px; border-left:3px solid var(--blue); font-family:var(--body);
  font-style:italic; font-size:.96rem; color:#2c2924; line-height:1.5;}}
.learned b {{font-style:normal; font-family:var(--mono); font-size:.72rem; text-transform:uppercase;
  letter-spacing:.04em; color:var(--blue); display:block; margin-bottom:4px;}}

/* ---- chart color key ---- */
.key {{display:flex; gap:22px; flex-wrap:wrap; margin-bottom:.7rem;}}
.key span {{font-family:var(--mono); font-size:.78rem; color:var(--ink-soft); display:flex; align-items:center; gap:7px;}}
.key .sw {{width:16px; height:2px; display:inline-block;}}
.key .sw.dash {{background: repeating-linear-gradient(90deg, var(--gray) 0 5px, transparent 5px 8px);}}

/* ---- loop, numbered like footnotes ---- */
.steps {{display:grid; grid-template-columns:repeat(4,1fr); gap:26px; margin-top:.6rem;}}
@media (max-width:760px){{.steps{{grid-template-columns:1fr 1fr;}}}}
.step {{border-top:2px solid var(--ink); padding-top:12px;}}
.step .n {{font-family:var(--serif); font-weight:600; font-size:1.5rem; color:var(--blue);
  font-variant-numeric: oldstyle-nums;}}
.step .t {{font-family:var(--serif); font-weight:600; color:var(--ink); font-size:1.08rem; margin:2px 0 6px 0;}}
.step .d {{font-family:var(--body); font-size:.92rem; color:#3a362f; line-height:1.5;}}

/* ---- lessons, as a glossary ---- */
.lesson {{padding:16px 0; border-top:1px solid var(--hair);}}
.lesson .txt {{font-family:var(--body); color:var(--ink); font-size:1.02rem; line-height:1.5;}}
.lesson .meta {{font-family:var(--mono); font-size:.74rem; color:var(--ink-soft); margin-top:8px;
  display:flex; gap:16px; flex-wrap:wrap; align-items:center;}}
.ubar {{width:90px; height:4px; background:var(--hair); position:relative; display:inline-block;}}
.ubar > span {{position:absolute; left:0; top:0; height:100%; background:var(--blue);}}

/* ---- reproduce, terminal ---- */
.term {{background:var(--ink); color:#f4efe2; font-family:var(--mono); font-size:.88rem;
  padding:16px 18px; border-radius:3px; line-height:1.6;}}
.term .p {{color:var(--rust);}}
.colophon {{font-family:var(--mono); font-size:.76rem; color:var(--ink-soft); margin-top:1.6rem;}}
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


def section(idx: str, title: str) -> None:
    st.markdown(
        f'<div class="section"><span class="idx">{idx}</span><span class="ttl">{title}</span></div>'
        '<hr class="section-rule" />',
        unsafe_allow_html=True,
    )


# ---------- masthead ----------
st.markdown(
    """
<div class="spine"><span class="bar"></span><span class="eyebrow">memory-augmented agent &middot; no fine-tuning</span></div>
<div class="wordmark">Engram</div>
<div class="lede">An agent that gets better at a task by writing and curating its own memory of what
worked. It tries a task, a deterministic checker (not an LLM) grades it, it reflects into a short
lesson, and later tries retrieve those lessons. Everything below is measured on a held-out set it
never learns from.</div>
<hr class="rule thick" />
""",
    unsafe_allow_html=True,
)

# ---------- headline stat ----------
st.markdown(
    f"""
<div class="statline">
  <span class="n from">{h.get('baseline_pct')}<small style="font-size:.5em;">%</small></span>
  <span class="arrow">&#10230;</span>
  <span class="n to">{h.get('engram_pct')}<small style="font-size:.5em;">%</small></span>
</div>
<div class="statcap">held-out success rate &middot; <b>+{h.get('improvement_pts')} points</b> &middot; zero weight updates</div>
<div class="badgerow">
  <span>deterministic checker, no LLM judge</span>
  <span>held-out generalization</span>
  <span>reproduces offline, no API key</span>
</div>
<div class="figrow">
  <div class="fig"><div class="k">No memory</div><div class="v">{h.get('baseline_pct')}%</div><div class="s">base model, held-out</div></div>
  <div class="fig accent"><div class="k">Engram, curated</div><div class="v">{h.get('engram_pct')}%</div><div class="s">+{h.get('improvement_pts')} points</div></div>
  <div class="fig"><div class="k">Naive accumulation</div><div class="v">{h.get('naive_pct')}%</div><div class="s">keeps every lesson</div></div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- watch it learn ----------
if show and show.get("examples"):
    section("01", "Watch it learn")
    st.markdown(
        f'<div class="sub">Held-out questions the base model got wrong, and the same questions after the '
        f"agent taught itself the database's hidden rules. On this run it went from solving "
        f"<b>{show['base_pass']} of {show['n_eval']}</b> held-out tasks to <b>{show['engram_pass']} of "
        f"{show['n_eval']}</b> &mdash; {show['n_flips']} flipped from wrong to right. The SQL below is the "
        "agent's actual output, unedited.</div>",
        unsafe_allow_html=True,
    )
    for i, ex in enumerate(show["examples"], start=1):
        lesson = ex["lessons"][0] if ex["lessons"] else "a rule learned from an earlier failure"
        pretty_type = ex["task_type"].replace("_", " ")
        st.markdown(
            f"""
<div class="demo">
  <div class="figtag">fig. {i:02d} &middot; <b>{escape(pretty_type)}</b></div>
  <div class="q">&ldquo;{escape(ex['question'])}&rdquo;</div>
  <div class="ba">
    <div class="side">
      <div class="lab">base model &middot; wrong</div>
      <div class="sql">{escape(ex['base_sql'])}</div>
    </div>
    <div class="side good">
      <div class="lab">engram &middot; correct</div>
      <div class="sql">{escape(ex['engram_sql'])}</div>
    </div>
  </div>
  <div class="learned"><b>lesson it wrote</b>{escape(lesson)}</div>
</div>
""",
            unsafe_allow_html=True,
        )

# ---------- curve ----------
section("02", "Held-out improvement")
st.markdown(
    f'<div class="key">'
    f'<span><span class="sw dash"></span>no memory</span>'
    f'<span><span class="sw" style="background:{RUST}"></span>naive accumulation</span>'
    f'<span><span class="sw" style="background:{BLUE}"></span>engram, curated</span>'
    f"</div>",
    unsafe_allow_html=True,
)
rows = []
for cond in ("no_memory", "naive", "engram"):
    if cond in agg:
        for step, m in zip(agg[cond]["steps"], agg[cond]["mean"], strict=True):
            rows.append({"tasks seen": step, "condition": cond, "success": round(100 * m, 1)})
curve = pd.DataFrame(rows)

base_axis_x = alt.X("tasks seen:Q", title="training tasks seen", axis=alt.Axis(grid=False, tickCount=6))
base_axis_y = alt.Y("success:Q", title="held-out success (%)", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(grid=False))

layers = []
control = curve[curve["condition"] == "no_memory"]
if not control.empty:
    layers.append(
        alt.Chart(control)
        .mark_line(strokeDash=[4, 4], strokeWidth=2, color=GRAY)
        .encode(x=base_axis_x, y=base_axis_y, tooltip=["condition", "tasks seen", "success"])
    )
hued = curve[curve["condition"] != "no_memory"]
if not hued.empty:
    hue_order = [c for c in ("naive", "engram") if c in agg]
    hue_range = [COLORS[c] for c in hue_order]
    layers.append(
        alt.Chart(hued)
        .mark_line(point=alt.OverlayMarkDef(size=55, filled=True), strokeWidth=3)
        .encode(
            x=base_axis_x,
            y=base_axis_y,
            color=alt.Color("condition:N", scale=alt.Scale(domain=hue_order, range=hue_range), legend=None),
            tooltip=["condition", "tasks seen", "success"],
        )
    )
last = curve.sort_values("tasks seen").groupby("condition", as_index=False).last()
last["hex"] = last["condition"].map(COLORS)
labels = (
    alt.Chart(last)
    .mark_text(align="left", dx=8, fontSize=12, fontWeight="bold", font="IBM Plex Mono")
    .encode(x="tasks seen:Q", y="success:Q", text=alt.Text("success:Q", format=".0f"), color=alt.Color("hex:N", scale=None, legend=None))
)
chart = (
    alt.layer(*layers, labels)
    .properties(height=340, background="transparent")
    .configure_view(strokeOpacity=0)
    .configure_axis(labelColor=INK_SOFT, titleColor=INK_SOFT, domainColor=HAIR, tickColor=HAIR, labelFont="IBM Plex Mono", labelFontSize=11, titleFont="IBM Plex Mono", titleFontSize=11)
)
st.altair_chart(chart, use_container_width=True)
st.markdown(
    f'<div class="sub">Mean of {n_seeds} seeds. Each point is success on the held-out pool, never used '
    "for learning. Memory lifts the base model by about 41 points. Curated and naive memory tie on "
    "accuracy at this scale; the difference is memory size, below.</div>",
    unsafe_allow_html=True,
)

# ---------- loop ----------
section("03", "How the loop works")
loop = [
    ("i", "Attempt", "Retrieve relevant lessons, then write one SQL query from the schema alone."),
    ("ii", "Check", "A deterministic checker runs the query and compares result sets. Hard pass or fail."),
    ("iii", "Reflect", "On a failure, the model proposes a short lesson about a likely hidden rule."),
    ("iv", "Curate", "Dedup, score each lesson's utility from real outcomes, consolidate, prune."),
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
    section("04", "Curation keeps memory compact")
    st.markdown(
        f'<div class="key">'
        f'<span><span class="sw" style="background:{HAIR}"></span>raw lessons proposed</span>'
        f'<span><span class="sw" style="background:{RUST}"></span>naive memory</span>'
        f'<span><span class="sw" style="background:{BLUE}"></span>engram memory</span>'
        f"</div>",
        unsafe_allow_html=True,
    )
    comp = []
    for step, raw, mem in zip(
        agg["engram"]["steps"], agg["engram"]["raw_lessons_seen"], agg["engram"]["memory_size"], strict=True
    ):
        comp.append({"tasks seen": step, "series": "raw", "lessons": round(raw, 1)})
        comp.append({"tasks seen": step, "series": "engram", "lessons": round(mem, 1)})
    for step, mem in zip(agg["naive"]["steps"], agg["naive"]["memory_size"], strict=True):
        comp.append({"tasks seen": step, "series": "naive", "lessons": round(mem, 1)})
    cdf = pd.DataFrame(comp)
    corder, ccolor = ["raw", "naive", "engram"], [HAIR, RUST, BLUE]
    comp_chart = (
        alt.Chart(cdf)
        .mark_line(point=alt.OverlayMarkDef(size=45, filled=True), strokeWidth=2.5)
        .encode(
            x=alt.X("tasks seen:Q", title="training tasks seen", axis=alt.Axis(grid=False, tickCount=6)),
            y=alt.Y("lessons:Q", title="lessons in memory", axis=alt.Axis(grid=False)),
            color=alt.Color("series:N", scale=alt.Scale(domain=corder, range=ccolor), legend=None),
            tooltip=["series", "tasks seen", "lessons"],
        )
        .properties(height=300, background="transparent")
        .configure_view(strokeOpacity=0)
        .configure_axis(labelColor=INK_SOFT, titleColor=INK_SOFT, domainColor=HAIR, tickColor=HAIR, labelFont="IBM Plex Mono", labelFontSize=11, titleFont="IBM Plex Mono", titleFontSize=11)
    )
    st.altair_chart(comp_chart, use_container_width=True)
    st.markdown(
        '<div class="sub">Same accuracy, smaller memory. Dedup merges duplicate lessons, so curated memory '
        "stays compact while naive accumulation keeps every restatement. This is where curation pays off at "
        "this scale; the accuracy payoff grows with longer runs.</div>",
        unsafe_allow_html=True,
    )

# ---------- lessons ----------
section("05", "What the agent taught itself")
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
  <div class="meta">
    <span class="ubar"><span style="width:{width}%"></span></span>
    <span>utility {u:+.2f}</span>
    <span>{escape(scope)}</span>
    <span>retrieved {les['retrieved_count']} &middot; helped {les['helped_count']} &middot; hurt {les['hurt_count']}</span>
    <span>{escape(les['source'])}</span>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

# ---------- reproduce ----------
section("06", "Reproduce")
st.markdown(
    '<div class="term"><span class="p">$</span> ENGRAM_LLM_MODE=replay python scripts/run_experiment.py</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub" style="margin-top:1rem;">Every number and query on this page replays from committed '
    "recordings, so it reproduces with no API key. Method and source are in the repository README.</div>"
    '<div class="colophon">engram &middot; reproducible end to end, no fine-tuning</div>',
    unsafe_allow_html=True,
)
