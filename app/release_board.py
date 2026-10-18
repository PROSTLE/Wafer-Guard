"""
Wafer Guard Release Board: the audit as a dashboard for the release meeting.

It contains NO numbers of its own. Everything on screen is read from:
  * results/metrics.json, results/*.csv, results/figures/   (written by wafer_guard_fixed.ipynb)
  * artifacts/ + data/   (the saved model re-scores every wafer live - see app/inference.py)
  * original/bnypi_executed.ipynb   (the original notebook's own printed log)
  * AUDIT.md, README.md and .run/   (pipeline runs started from the board - see scripts/run_pipeline.py)

Run from the repo root:
    streamlit run app/release_board.py
"""
import html
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path[:0] = [str(APP), str(ROOT / "scripts")]
import inference  # noqa: E402
import run_pipeline  # noqa: E402

METRICS = ROOT / "results" / "metrics.json"
FIGS = ROOT / "results" / "figures"
ABLATION_RUNS = ROOT / "results" / "ablation_runs.csv"
AUDIT = ROOT / "AUDIT.md"
README = ROOT / "README.md"
ORIGINAL_RUN = ROOT / "original" / "bnypi_executed.ipynb"

st.set_page_config(page_title="Wafer Guard Release Board", page_icon="🔬", layout="wide",
                   initial_sidebar_state="expanded")

PAGES = [("Overview", ":material/home:"), ("Model Performance", ":material/monitoring:"),
         ("Data Analysis", ":material/bar_chart:"), ("Ablation Study", ":material/science:"),
         ("Walk-Forward Eval", ":material/timeline:"), ("Wafer Explorer", ":material/grid_view:"),
         ("Audit Findings", ":material/fact_check:"), ("Configuration", ":material/settings:"),
         ("Reproduce", ":material/replay:")]
ABOUT = ("About", ":material/info:")

ss = st.session_state
ss.setdefault("page", "Overview")
ss.setdefault("theme", "light")  # every visit opens in light mode; the top-bar toggle switches it

# ---------------------------------------------------------------- theme
# Neutral greys carry the interface; one accent marks the repaired model; red is reserved for defects / failures.
FONT = "IBM Plex Sans"
THEMES = {
    "light": dict(bg="#fafafa", surface="#ffffff", surface2="#f4f4f5", border="#e4e4e7", hover="#f0f0f2",
                  text="#18181b", text2="#52525b", muted="#8b8b93", grid="#f0f0f2", sidebar="#f6f6f7",
                  primary="#18181b", on_primary="#ffffff", accent="#2563eb",
                  good="#15803d", bad="#dc2626", warn="#b45309", info="#2563eb", violet="#64748b",
                  original="#c4c4cc", repaired="#2563eb", logreg="#71717a", normal="#94a3b8", defect="#dc2626",
                  z_low="#2563eb", z_mid="#f4f4f5", z_high="#dc2626"),
    "dark": dict(bg="#0b0b0c", surface="#121214", surface2="#1a1a1d", border="#27272a", hover="#1f1f23",
                 text="#f4f4f5", text2="#a1a1aa", muted="#71717a", grid="#1f1f23", sidebar="#0f0f11",
                 primary="#f4f4f5", on_primary="#18181b", accent="#60a5fa",
                 good="#4ade80", bad="#f87171", warn="#fbbf24", info="#60a5fa", violet="#94a3b8",
                 original="#52525b", repaired="#60a5fa", logreg="#a1a1aa", normal="#64748b", defect="#f87171",
                 z_low="#3b82f6", z_mid="#1a1a1d", z_high="#ef4444"),
}
T = THEMES[ss.theme]

CSS = """
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp table,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, div[role="dialog"] { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
header[data-testid="stHeader"] { background: transparent; pointer-events: none; }  /* let clicks reach the top bar */
header[data-testid="stHeader"] * { pointer-events: none; }
header[data-testid="stHeader"] button, header[data-testid="stHeader"] a { pointer-events: auto; }
[data-testid="stAppDeployButton"] { display: none; }
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 1rem; padding-bottom: 3rem; max-width: 1440px; }
.stMarkdown p, .stMarkdown li, [data-testid="stWidgetLabel"] p, .stCheckbox label p { color: var(--text); }
[data-testid="stWidgetLabel"] p { font-size: .8rem; color: var(--text2); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted); }
.stMarkdown code, .wg-mono, .wg-table td.num, .wg-table td.mono { font-family: 'IBM Plex Mono', ui-monospace, monospace; }
.stMarkdown code { background: var(--surface2); color: var(--text); border: none; font-size: .82em; }
.stMarkdown a { color: var(--accent); }
.stMarkdown table { color: var(--text); } .stMarkdown th, .stMarkdown td { border-color: var(--border) !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color: var(--text); font-weight: 600; }
hr { border-color: var(--border) !important; }

/* sidebar: brand top, navigation, About pinned to the bottom */
section[data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--border); }
[data-testid="stSidebarHeader"] { position: absolute; top: 0; right: 4px; height: auto; padding: 0; margin: 0; z-index: 3; }
[data-testid="stSidebarCollapseButton"] { margin-top: 22px; }
[data-testid="stSidebarUserContent"] { padding-top: 22px !important; padding-bottom: 20px !important;
  min-height: 100vh; box-sizing: border-box; display: flex; flex-direction: column; }
[data-testid="stSidebarUserContent"] > div { flex: 1; display: flex; flex-direction: column; }
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] { flex: 1; }
[data-testid="stSidebarUserContent"] div:has(> .st-key-side_bottom), .st-key-side_bottom { margin-top: auto; }
.st-key-side_bottom { border-top: 1px solid var(--border); padding-top: 10px; }
section[data-testid="stSidebar"] .stButton button {
  justify-content: flex-start; border: none; border-radius: 6px; padding: 6px 10px; min-height: 0;
  background: transparent; color: var(--text2); font-weight: 400; box-shadow: none; }
section[data-testid="stSidebar"] .stButton button p { color: inherit; font-size: .9rem; }
section[data-testid="stSidebar"] .stButton button > div, section[data-testid="stSidebar"] .stButton button > div > span {
  width: 100%; justify-content: flex-start; gap: 10px; }
section[data-testid="stSidebar"] .stButton button [data-testid="stIconMaterial"] { font-size: 1.1rem; color: var(--muted); }
section[data-testid="stSidebar"] .stButton button:hover { background: var(--hover); color: var(--text); }
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] {
  background: var(--hover) !important; border-color: transparent !important; color: var(--text) !important; font-weight: 500; }
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] p,
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] span { color: var(--text) !important; }
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] [data-testid="stIconMaterial"] { color: var(--text); }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 2px; }

/* buttons: one solid primary, everything else quiet */
.stButton button[data-testid="stBaseButton-secondary"], .stDownloadButton button {
  background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: 6px; box-shadow: none; }
.stButton button[data-testid="stBaseButton-secondary"]:hover, .stDownloadButton button:hover { background: var(--hover); color: var(--text); border-color: var(--border); }
button[data-testid="stBaseButton-primary"] {
  background: var(--primary) !important; border: 1px solid var(--primary) !important; color: var(--on-primary) !important;
  border-radius: 6px; font-weight: 500; box-shadow: none !important; }
button[data-testid="stBaseButton-primary"] p, button[data-testid="stBaseButton-primary"] span { color: var(--on-primary) !important; }
button[data-testid="stBaseButton-primary"]:hover { opacity: .88; }
.stButton button[data-testid="stBaseButton-tertiary"] { color: var(--text2); }
.stButton button[data-testid="stBaseButton-tertiary"]:hover { color: var(--text); }
.stButton button[data-testid="stBaseButton-tertiary"] p { color: inherit; }

/* inputs */
.stTextInput div, .stNumberInput div, .stSelectbox div, .stMultiSelect div { background-color: var(--surface) !important; border-color: var(--border) !important; }
.stTextInput input, .stNumberInput input, .stSelectbox input, .stMultiSelect input { color: var(--text) !important; background: transparent !important; }
.stTextInput input::placeholder { color: var(--muted) !important; }
.stSelectbox div, .stMultiSelect div, [data-testid="stTextInputIcon"], [data-testid="stTextInputIcon"] span { color: var(--text) !important; }
[data-testid="stTextInputIcon"] span { color: var(--muted) !important; }
[data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"] { border-radius: 6px !important; }
[data-testid$="VirtualDropdown"] { background: var(--surface) !important; border: 1px solid var(--border); }
[data-testid$="VirtualDropdown"] [role="option"], [data-testid$="VirtualDropdown"] [role="option"] * { color: var(--text) !important; background: transparent; }
[data-testid$="VirtualDropdown"] [role="option"]:hover, [data-testid$="VirtualDropdown"] [role="option"][aria-selected="true"] { background: var(--hover) !important; }
.stNumberInput button { background: var(--surface2) !important; color: var(--text) !important; border-color: var(--border) !important; }

/* segmented control */
[data-testid="stButtonGroup"] button { background: var(--surface); color: var(--text2); border-color: var(--border); }
[data-testid="stButtonGroup"] button p { color: inherit; font-size: .8rem; }
[data-testid="stButtonGroup"] button[aria-checked="true"] { background: var(--hover) !important; color: var(--text) !important; border-color: var(--border) !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] p { color: var(--text) !important; font-weight: 500; }

/* expanders, dialogs, images */
[data-testid="stExpander"] details { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color: var(--text); }
div[role="dialog"] { background: var(--surface); color: var(--text); border-radius: 10px; }
div[role="dialog"] h2, div[role="dialog"] p { color: var(--text); }
[data-testid="stImage"] img { border-radius: 6px; border: 1px solid var(--border); }
[data-testid="stImageCaption"], [data-testid="caption"] { color: var(--muted) !important; }

/* panels: every st.container(key="card_…") - hairline border, no shadow */
div[class*="st-key-card_"] { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 18px 20px 14px; }
.st-key-topbar { padding: 0 44px 12px 0; border-bottom: 1px solid var(--border); margin-bottom: 8px; }

/* html components */
.wg-ico { font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; line-height: 1;
  letter-spacing: normal; text-transform: none; white-space: nowrap; direction: ltr; font-feature-settings: 'liga';
  -webkit-font-smoothing: antialiased; font-size: 22px; display: inline-block; }
.wg-brand { display: flex; gap: 10px; align-items: center; padding: 2px 10px 18px; }
.wg-brand .wg-ico { font-size: 24px; color: var(--text); }
.wg-brand-name { font-weight: 600; font-size: 1rem; letter-spacing: .01em; color: var(--text); }

.wg-run { font-size: .8rem; color: var(--text2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.wg-run.bad { color: var(--bad); }

.wg-page-title { font-size: 1.5rem; font-weight: 600; color: var(--text); margin: 10px 0 2px; letter-spacing: -.01em; }
.wg-page-sub { color: var(--text2); margin-bottom: 14px; font-size: .9rem; }

.wg-hero { display: grid; grid-template-columns: minmax(260px, 1fr) 2fr; gap: 0; background: var(--surface);
  border: 1px solid var(--border); border-radius: 8px; margin-bottom: 4px; }
@media (max-width: 1100px) { .wg-hero { grid-template-columns: 1fr; } }
.wg-hero-main { padding: 22px 24px; border-right: 1px solid var(--border); }
.wg-label { font-size: .78rem; color: var(--text2); }
.wg-hero-value { font-size: 3rem; font-weight: 600; color: var(--text); line-height: 1.1; margin: 6px 0 8px; font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
.wg-hero-gate { font-size: .88rem; color: var(--text); }
.wg-hero-gate .fail { color: var(--bad); font-weight: 500; } .wg-hero-gate .pass { color: var(--good); font-weight: 500; }
.wg-hero-note { font-size: .8rem; color: var(--muted); margin-top: 8px; line-height: 1.45; }
.wg-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); }
.wg-stat { padding: 18px 22px; border-bottom: 1px solid var(--border); border-right: 1px solid var(--border); }
.wg-stat:nth-child(3n) { border-right: none; } .wg-stat:nth-last-child(-n+3) { border-bottom: none; }
.wg-stat-value { font-size: 1.45rem; font-weight: 500; color: var(--text); margin-top: 4px; font-variant-numeric: tabular-nums; }
.wg-stat-sub { font-size: .76rem; color: var(--muted); margin-top: 2px; }

.wg-card-title { font-weight: 600; font-size: .95rem; color: var(--text); display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.wg-card-sub { font-weight: 400; font-size: .8rem; color: var(--muted); }
.wg-note { color: var(--muted); font-size: .78rem; line-height: 1.5; }
.wg-note b { color: var(--text2); font-weight: 500; }
.wg-legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: .78rem; color: var(--text2); }
.wg-legend span { display: inline-flex; align-items: center; gap: 6px; }
.wg-legend i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }

.wg-table-wrap { overflow-x: auto; }
.wg-table { width: 100%; border-collapse: collapse; font-size: .8rem; color: var(--text); }
.wg-table th { text-align: left; font-weight: 500; color: var(--muted); padding: 8px 8px; border-bottom: 1px solid var(--border); white-space: nowrap; }
.wg-table td { padding: 7px 8px; border-bottom: 1px solid var(--border); white-space: nowrap; font-variant-numeric: tabular-nums; }
.wg-table td.num { text-align: right; font-size: .78rem; }
.wg-table tr:last-child td { border-bottom: none; }
.wg-table tr.hl td { background: var(--hover); }
.c-tp, .c-tn, .c-fp, .c-fn { font-family: 'IBM Plex Mono', monospace; font-size: .76rem; }
.c-tp, .c-tn { color: var(--muted); } .c-fp { color: var(--warn); } .c-fn { color: var(--bad); font-weight: 500; }
.c-def { color: var(--bad); } .c-pass { color: var(--text2); }
.wg-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; }

.wg-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); border: 1px solid var(--border);
  border-radius: 8px; background: var(--surface); overflow: hidden; }
.wg-tile { padding: 12px 16px; border-right: 1px solid var(--border); }
.wg-tile:last-child { border-right: none; }
.wg-tile-label { font-size: .76rem; color: var(--text2); }
.wg-tile-value { font-size: 1.25rem; font-weight: 500; color: var(--text); margin-top: 3px; font-variant-numeric: tabular-nums; }
.wg-tile-sub { font-size: .74rem; color: var(--muted); margin-top: 2px; }

.wg-cm { display: grid; grid-template-columns: auto 1fr 1fr; gap: 1px; font-size: .8rem; background: var(--border); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.wg-cm div { padding: 14px 10px; text-align: center; background: var(--surface); }
.wg-cm .h { color: var(--text2); background: var(--surface2); padding: 8px 10px; }
.wg-cm .v { font-size: 1.5rem; font-weight: 500; color: var(--text); font-variant-numeric: tabular-nums; }
.wg-cm .v small { display: block; font-size: .7rem; font-weight: 400; color: var(--muted); }
.wg-cm .v.err { color: var(--bad); }

.wg-evidence { margin: 0; padding-left: 18px; color: var(--text); font-size: .86rem; line-height: 1.65; }
.wg-evidence b { color: var(--text); font-weight: 600; }
.wg-evidence code { background: var(--surface2); border-radius: 4px; padding: 0 4px; font-size: .78rem; font-family: 'IBM Plex Mono', monospace; }
.wg-sev { font-size: .76rem; font-weight: 400; color: var(--muted); margin-left: 4px; }
.wg-hit { display: flex; gap: 10px; align-items: baseline; }
.wg-hit-kind { font-size: .74rem; color: var(--muted); min-width: 72px; }
.wg-hit-title { font-weight: 500; color: var(--text); font-size: .88rem; } .wg-hit-detail { color: var(--muted); font-size: .8rem; }
.wg-log { background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px; font-family: 'IBM Plex Mono', ui-monospace, monospace;
  font-size: .74rem; color: var(--text2); white-space: pre-wrap; max-height: 360px; overflow-y: auto; margin: 0; }
.wg-zbar { width: 10px; height: 210px; border-radius: 3px; margin: 6px auto; }
.wg-zlabels { font-size: .72rem; color: var(--text2); text-align: center; }
.wg-check { display: grid; grid-template-columns: 22px 1fr; gap: 2px 8px; align-items: baseline; padding: 10px 0; border-bottom: 1px solid var(--border); }
.wg-check:last-child { border-bottom: none; }
.wg-check .mark { font-family: 'IBM Plex Mono', monospace; font-size: .85rem; }
.wg-check .mark.done { color: var(--good); } .wg-check .mark.warn { color: var(--warn); } .wg-check .mark.todo { color: var(--muted); }
.wg-check .mark.nogo { color: var(--bad); } .wg-check .mark.go { color: var(--good); } .wg-check .mark.run { color: var(--accent); }
.wg-check .t { color: var(--text); font-size: .88rem; font-weight: 500; } .wg-check .d { grid-column: 2; color: var(--muted); font-size: .78rem; }
"""


def icon(name: str) -> str:
    """A Material Symbols glyph (the icon font Streamlit already ships); st.html strips inline <svg>."""
    return f'<span class="wg-ico" translate="no">{name}</span>'


def inject_css() -> None:
    root = ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in T.items())
    st.html("<style>@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600"
            "&family=IBM+Plex+Mono:wght@400;500&display=swap');"
            f":root{{{root}}}{CSS}</style>")


# ---------------------------------------------------------------- small helpers
def esc(x) -> str:
    return html.escape(str(x))


def mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


def stamp(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M") if t else "–"


def go(page: str, **state) -> None:
    """Navigate (used as an on_click callback); `state` pre-sets widgets on the target page."""
    ss.page = page
    for k, v in state.items():
        if v is None:
            ss.pop(k, None)
        else:
            ss[k] = v


def card_title(title: str, sub: str = "") -> None:
    st.html(f'<div class="wg-card-title">{title}<span class="wg-card-sub">{sub}</span></div>')


def note(text: str) -> None:
    st.html(f'<div class="wg-note">{text}</div>')


def page_header(title: str, sub: str) -> None:
    st.html(f'<div class="wg-page-title">{esc(title)}</div><div class="wg-page-sub">{sub}</div>')


def tiles(items: list[tuple[str, str, str]]) -> str:
    return '<div class="wg-tiles">' + "".join(
        f'<div class="wg-tile"><div class="wg-tile-label">{esc(a)}</div><div class="wg-tile-value">{b}</div>'
        f'<div class="wg-tile-sub">{c}</div></div>' for a, b, c in items) + "</div>"


def table(df: pd.DataFrame, fmt: dict | None = None, highlight=None) -> str:
    """HTML table that follows the board's theme. `fmt` maps column -> format spec or callable returning HTML."""
    fmt = fmt or {}
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    rows = []
    for _, r in df.astype(object).iterrows():  # astype(object) keeps ints as ints
        cells = []
        for c in df.columns:
            v, f = r[c], fmt.get(c)
            numeric = isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_))
            if callable(f):
                s = f(v)
            elif v is None or (numeric and isinstance(v, float) and math.isnan(v)):
                s = "–"
            elif f and numeric:
                s = esc(format(v, f))
            else:
                s = esc(v)
            cells.append(f'<td class="{"num" if numeric else ""}">{s}</td>')
        cls = ' class="hl"' if highlight is not None and highlight(r) else ""
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return (f'<div class="wg-table-wrap"><table class="wg-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def legend(items: list[tuple[str, str]]) -> str:
    return '<div class="wg-legend">' + "".join(f'<span><i style="background:{c}"></i>{esc(l)}</span>'
                                               for l, c in items) + "</div>"


def chart(c, height: int = 260) -> None:
    """Render an Altair chart in the board's theme."""
    c = (c.properties(height=height, background="transparent")
         .configure(font=FONT)
         .configure_view(strokeWidth=0)
         .configure_axis(labelColor=T["text2"], titleColor=T["text2"], gridColor=T["grid"], domainColor=T["border"],
                         tickColor=T["border"], labelFontSize=11, titleFontSize=11, titleFontWeight=500)
         .configure_legend(labelColor=T["text2"], titleColor=T["text2"], labelFontSize=11, orient="top",
                           symbolType="square")
         .configure_title(color=T["text"]))
    st.altair_chart(c, theme=None, width="stretch")


def figure(name: str, caption: str) -> None:
    path = FIGS / f"{name}.png"
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        note(f"results/figures/{esc(name)}.png not found. Run the repaired notebook.")


def sci(v: float) -> str:
    return f"{v:.2e}"


def outcome(defect: int, flagged: bool) -> str:
    return {(1, True): "TP", (0, True): "FP", (1, False): "FN", (0, False): "TN"}[(int(defect), bool(flagged))]


def outcome_chip(o: str) -> str:
    return f'<span class="c-{o.lower()}">{o}</span>'


# ---------------------------------------------------------------- data (cached on file modification times)
@st.cache_data(show_spinner=False)
def read_metrics(mt: float) -> dict:
    return json.loads(METRICS.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Scoring every wafer with the saved model…")
def live_model(mts: tuple) -> dict:
    """Load artifacts/, re-score every wafer and check the result against metrics.json."""
    R = read_metrics(mts[0])
    try:
        b = inference.build(R)
    except Exception as exc:  # missing / incompatible artifacts must not take the whole board down
        return dict(error=f"{type(exc).__name__}: {exc}")
    b["verify"] = inference.verify(b["scored"], R)
    return b


@st.cache_data(show_spinner=False)
def original_log(mt: float) -> dict | None:
    """What the ORIGINAL notebook printed when we executed it (scripts/run_original.py)."""
    if not ORIGINAL_RUN.exists():
        return None
    nb = json.loads(ORIGINAL_RUN.read_text(encoding="utf-8"))
    text = "".join("".join(o.get("text", "")) for c in nb["cells"] for o in c.get("outputs", [])
                   if o.get("output_type") == "stream")
    ep = re.findall(r"epoch (\d+)/(\d+) \| total ([\d.]+) \| recon ([\d.]+) \| clf ([\d.]+)", text)
    loss = pd.DataFrame([dict(epoch=int(e), total=float(t), recon=float(r), clf=float(c)) for e, _, t, r, c in ep])
    m = re.search(r"F1 ([\d.]+) \| precision ([\d.]+) \| recall ([\d.]+) \| ROC-AUC ([\d.]+) \| AP ([\d.]+)", text)
    reported = dict(zip(["f1", "precision", "recall", "roc_auc", "ap"], map(float, m.groups()))) if m else None
    cm = re.search(r"TN=(\d+)\s+FP=(\d+)\s+FN=(\d+)\s+TP=(\d+)", text)
    confusion = dict(zip(["tn", "fp", "fn", "tp"], map(int, cm.groups()))) if cm else None
    return dict(loss=loss, reported=reported, confusion=confusion, text=text)


@st.cache_data(show_spinner=False)
def readme(mt: float) -> dict:
    text = README.read_text(encoding="utf-8") if README.exists() else ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = lines[0].lstrip("# ") if lines else ""
    name, _, subtitle = title.partition(": ")
    cmd = re.search(r"## Reproduce everything.*?```bash\n(.*?)```", text, re.S)
    return dict(text=text, name=name, subtitle=subtitle, commands=cmd.group(1).strip() if cmd else "")


@st.cache_data(show_spinner=False)
def read_csv(path: str, mt: float) -> pd.DataFrame | None:
    return pd.read_csv(path) if mt else None


@st.cache_data(show_spinner=False)
def audit_text(mt: float) -> str:
    return AUDIT.read_text(encoding="utf-8") if mt else ""
