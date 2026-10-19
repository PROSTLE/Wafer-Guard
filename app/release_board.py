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


# ---------------------------------------------------------------- pipeline runs
@st.cache_resource
def runs() -> dict:
    """The pipeline process started from this server (shared by every browser session)."""
    return {}


def read_run_status() -> dict | None:
    try:
        return json.loads(run_pipeline.STATUS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def active_run() -> bool:
    proc = runs().get("proc")
    return proc is not None and proc.poll() is None


def start_run(keys: list[str]) -> None:
    if active_run():
        return
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen([sys.executable, str(run_pipeline.__file__), *keys], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    runs().update(proc=proc, keys=keys, started=datetime.now().replace(microsecond=0))


def run_view() -> dict:
    """State of the latest pipeline run: running / done / failed / unknown / None (never run)."""
    s, reg = read_run_status(), runs()
    if active_run():
        if s is None or datetime.fromisoformat(s["started"]) < reg["started"]:
            return dict(state="running", idx=1, n=len(reg["keys"]), label="starting", status=None)
        cur = next((i for i, x in enumerate(s["steps"]) if x["state"] == "running"), 0)
        return dict(state="running", idx=cur + 1, n=len(s["steps"]), label=s["steps"][cur]["label"], status=s)
    if s is None:
        return dict(state=None, status=None)
    if s["state"] == "running":  # status file left by a run this server did not start (or one that was killed)
        return dict(state="unknown", status=s)
    return dict(state=s["state"], status=s)


@st.dialog("Run the pipeline", width="medium")
def run_dialog() -> None:
    st.markdown("Runs the project's own scripts in the background, in this order. Their outputs "
                "(`original/bnypi_executed.ipynb`, `results/`, `artifacts/`, `AUDIT.md`) are **overwritten**; "
                "the board refreshes itself when the run ends. The full run takes several minutes on a laptop CPU.")
    keys = [k for k, (label, script) in run_pipeline.STEPS.items()
            if st.checkbox(f"**{label}** · `scripts/{script}`", value=True, key=f"dlg_{k}")]
    if st.button("Start run", type="primary", icon=":material/play_arrow:", disabled=not keys or active_run()):
        start_run(keys)
        st.rerun()


# ---------------------------------------------------------------- load everything
inject_css()
RD = readme(mtime(README))

if not METRICS.exists():
    st.html(f'<div class="wg-page-title">{esc(RD["name"] or "Wafer Guard")}</div>')
    st.error("results/metrics.json not found. Run the pipeline (or `python scripts/run_notebook.py`) first.")
    if st.button("Run Pipeline", type="primary", icon=":material/play_arrow:", disabled=active_run()):
        run_dialog()
    v = run_view()
    if v["state"] == "running":
        st.info(f"Pipeline running · step {v['idx']}/{v['n']}: {v['label']}. Refresh the page when it finishes.")
    st.stop()

R = read_metrics(mtime(METRICS))
orig, rep, gate = R["original"], R["repaired_test"], R["release_gate"]
boot, seeds, wf = R["repaired_test_bootstrap"], R["repaired_test_seeds"], R["walk_forward"]
no_skill = R["baseline_no_skill_ap"]
LIVE = live_model((mtime(METRICS), mtime(inference.MODEL_PATH), mtime(inference.PREPROCESSOR_PATH),
                   mtime(inference.DATA)))
LIVE_OK = "error" not in LIVE
OLOG = original_log(mtime(ORIGINAL_RUN))


def require_live() -> bool:
    if not LIVE_OK:
        st.error(f"The saved model could not be loaded, so live wafer scores are unavailable. {LIVE['error']}")
    return LIVE_OK


def pipeline_steps() -> list[dict]:
    """The four stages of the audit, each checked against the files on disk."""
    m_metrics = mtime(METRICS)
    reported = OLOG["reported"] if OLOG else None
    ok1 = reported is not None and abs(reported["f1"] - orig["f1"]) <= 5e-5 + 1e-12  # log prints 4 decimals
    s1 = dict(title="Original notebook reproduced", state="done" if ok1 else ("warn" if OLOG else "todo"),
              detail=(f"logged F1 {reported['f1']:.4f}" if reported else "not run yet"), when=mtime(ORIGINAL_RUN))
    ok2 = LIVE_OK and LIVE["verify"]["ok"]
    s2 = dict(title="Repaired pipeline verified", state="done" if ok2 else "warn",
              detail="live re-score matches" if ok2 else "re-score mismatch", when=m_metrics)
    m_audit = mtime(AUDIT)
    s3 = dict(title="Audit report generated", state="done" if m_audit >= m_metrics else ("warn" if m_audit else "todo"),
              detail="AUDIT.md up to date" if m_audit >= m_metrics else
              ("older than metrics.json" if m_audit else "not generated"), when=m_audit)
    s4 = dict(title="Release decision", state="go" if gate["passed"] else "nogo",
              detail=f"{'GO' if gate['passed'] else 'NO-GO'} · F1 {gate['measured_f1']:.3f} vs {gate['gate_f1']:.2f}",
              when=m_metrics)
    steps = [s1, s2, s3, s4]
    v = run_view()
    if v["state"] == "running" and v["status"]:
        index = {"original": 0, "repaired": 1, "audit": 2}
        for st_ in v["status"]["steps"]:
            if st_["state"] == "running":
                steps[index[st_["key"]]].update(state="run", detail="running now…")
    return steps


def checklist_html(steps: list[dict]) -> str:
    """The audit stages as a plain checklist: a mark, the step and what was checked."""
    marks = {"done": "✓", "go": "✓", "warn": "!", "todo": "–", "nogo": "✕", "run": "…"}
    return "".join(f'<div class="wg-check"><span class="mark {s["state"]}">{marks[s["state"]]}</span>'
                   f'<span class="t">{s["title"]}</span><span class="d">{esc(s["detail"])}'
                   f'{" · " + stamp(s["when"]) if s.get("when") else ""}</span></div>' for s in steps)


# ---------------------------------------------------------------- sidebar
def sidebar() -> None:
    with st.sidebar:
        st.html(f'<div class="wg-brand">{icon("stacks")}<span class="wg-brand-name">{esc(RD["name"] or "Wafer Guard")}</span></div>')
        for name, glyph in PAGES:
            st.button(name, icon=glyph, key=f"nav_{name}", width="stretch", on_click=go, args=(name,),
                      type="primary" if ss.page == name else "tertiary")
        with st.container(key="side_bottom"):
            st.button(ABOUT[0], icon=ABOUT[1], key="nav_about", width="stretch", on_click=go, args=(ABOUT[0],),
                      type="primary" if ss.page == ABOUT[0] else "tertiary")


# ---------------------------------------------------------------- top bar
def run_status_text() -> str:
    """Only says something when there is something to act on: a run in progress, a failed run, a broken check."""
    v = run_view()
    if v["state"] == "running":
        return f'<div class="wg-run">Running step {v["idx"]} of {v["n"]}: {esc(v["label"])}…</div>'
    if v["state"] == "failed" and v["status"] and v["status"].get("finished"):
        failed = next((s for s in v["status"]["steps"] if s["state"] == "failed"), None)
        return f'<div class="wg-run bad">Last run failed{" at " + esc(failed["label"]) if failed else ""}. See Reproduce</div>'
    if not LIVE_OK:
        return '<div class="wg-run bad">Saved model could not be loaded</div>'
    if not LIVE["verify"]["ok"]:
        return '<div class="wg-run bad">Live re-score differs from metrics.json. See Reproduce</div>'
    return ""


def topbar() -> None:
    with st.container(key="topbar"):
        c_q, c_status, c_theme, c_run = st.columns([4.2, 3.6, 0.4, 1.3], vertical_alignment="center")
        c_q.text_input("Search", key="q", placeholder="Search wafers, sensors, experiments…",
                       label_visibility="collapsed", icon=":material/search:")
        running = active_run()

        @st.fragment(run_every=2 if running else None)
        def live_status():
            proc = runs().get("proc")
            if proc is not None and proc.poll() is not None and ss.get("run_seen") != proc.pid:
                ss.run_seen = proc.pid  # this session has not yet reloaded after the run ended
                st.rerun(scope="app")
            text = run_status_text()
            if text:
                st.html(text)

        with c_status:
            live_status()
        dark = ss.theme == "dark"
        c_theme.button("", icon=":material/light_mode:" if dark else ":material/dark_mode:", key="theme_toggle",
                       help="Light mode" if dark else "Dark mode", type="tertiary", on_click=go, args=(ss.page,),
                       kwargs=dict(theme="light" if dark else "dark"))
        if c_run.button("Running…" if running else "Run pipeline", type="primary", width="stretch",
                        disabled=running, key="run_top"):
            run_dialog()
        if ss.get("q", "").strip():
            search_results(ss.q)


def search_results(q: str) -> None:
    hits = search(q)
    with st.container(key="card_search"):
        h, x = st.columns([6, 1], vertical_alignment="center")
        with h:
            card_title(f"Search results for “{esc(q.strip())}”", f"{len(hits)} match{'es' if len(hits) != 1 else ''}")
        x.button("Clear", icon=":material/close:", type="tertiary", key="q_clear", on_click=lambda: ss.update(q=""))
        if not hits:
            ex = ""
            if LIVE_OK:
                last = LIVE["scored"].iloc[-1]
                ex = (f" (e.g. <b>{last['row']}</b>), a sensor (<b>{esc(LIVE['feature_names'][0])}</b>), "
                      f"a date (<b>{last['timestamp']:%Y-%m-%d}</b>)")
            note(f"Nothing matched. Try a wafer number{ex}, an experiment name, "
                 f"<b>fold {wf['folds'][-1]['fold']}</b>, a setting or a page name.")
        for i, (kind, title, detail, page, state) in enumerate(hits[:12]):
            a, b = st.columns([6, 1], vertical_alignment="center")
            a.html(f'<div class="wg-hit"><span class="wg-hit-kind">{esc(kind)}</span>'
                   f'<span class="wg-hit-title">{esc(title)}</span><span class="wg-hit-detail">{esc(detail)}</span></div>')
            b.button("Open", key=f"hit_{i}", type="tertiary", icon=":material/arrow_forward:", icon_position="right",
                     on_click=go, args=(page,), kwargs={**state, "q": ""})


EXPLORER_FILTERS = dict(ex_split=None, ex_actual=None, ex_decision=None, ex_psl=None, ex_sort=None, ex_page=None)


def search(q: str) -> list[tuple]:
    """Search pages, wafers (by CSV row or date), sensors, experiments, walk-forward folds, audit sections, settings."""
    ql = q.strip().lower()
    hits = []
    for name, _ in PAGES + [ABOUT]:
        if ql in name.lower():
            hits.append(("Page", name, "open page", name, {}))
    if LIVE_OK:
        sc, sensors = LIVE["scored"], [c for c in LIVE["raw"].columns if c.startswith("sensor_")]
        rows = set(sc["row"])
        m = re.fullmatch(r"(?:wafer\s*#?|#)?\s*(\d+)", ql)
        if m and int(m.group(1)) in rows:
            w = sc[sc["row"] == int(m.group(1))].iloc[0]
            hits.append(("Wafer", f"#{w['row']}", f"{w['timestamp']:%Y-%m-%d %H:%M} · {w['split']} · "
                         f"{'defect' if w['defect'] else 'pass'} · score {sci(w['score'])}",
                         "Wafer Explorer", dict(EXPLORER_FILTERS, ex_pick=int(w["row"]), ex_jump=True)))
        ms = re.fullmatch(r"(?:sensor[\s_]*|s)?(\d+)", ql)
        if ms and f"sensor_{int(ms.group(1))}" in sensors:
            name = f"sensor_{int(ms.group(1))}"
            kept = name in LIVE["feature_names"]
            hits.append(("Sensor", name, "used by the model" if kept else "dropped (constant in training)",
                         "Data Analysis", dict(sensor_pick=name)))
        if re.fullmatch(r"\d{4}-\d{2}(-\d{2})?( \d{2}(:\d{2})?)?", ql):
            match = sc[sc["timestamp"].dt.strftime("%Y-%m-%d %H:%M").str.startswith(ql)]
            for _, w in match.head(6).iterrows():
                hits.append(("Wafer", f"#{w['row']}", f"{w['timestamp']:%Y-%m-%d %H:%M} · {w['split']} · "
                             f"{'defect' if w['defect'] else 'pass'}", "Wafer Explorer",
                             dict(EXPLORER_FILTERS, ex_pick=int(w["row"]), ex_jump=True)))
            if len(match) > 6:
                hits.append(("Wafers", f"{len(match)} wafers on {q.strip()}", "showing the first 6", "Wafer Explorer", {}))
    for a in R["ablation"]:
        if ql in a["config"].lower():
            hits.append(("Experiment", a["config"], f"F1 {a['f1_mean']:.3f} ± {a['f1_std']:.3f} ({len(R['config']['eval_seeds'])} seeds)",
                         "Ablation Study", {}))
    for s in R["summary_table"]:
        if ql in s["model"].lower() or ql in s["test"].lower():
            hits.append(("Evaluation", s["model"], f"{s['test']} · F1 " + ("–" if s["f1"] is None else f"{s['f1']:.3f}"),
                         "Model Performance", {}))
    for f in wf["folds"]:
        if ql == "fold" or ql in (f"fold {f['fold']}", f"fold{f['fold']}"):
            hits.append(("Fold", f"Walk-forward fold {f['fold']}", f"{f['test_start']} → {f['test_end']} · F1 {f['f1']:.3f}",
                         "Walk-Forward Eval", {}))
    for line in audit_text(mtime(AUDIT)).splitlines():
        if line.startswith("#") and ql in line.lower() and len(ql) >= 3:
            hits.append(("Audit", line.lstrip("# ").strip(), "section of AUDIT.md", "Audit Findings", {}))
    flat = {**R["config"]["HP"], **{k: v for k, v in R["config"].items() if k != "HP"}}
    for k, v in flat.items():
        if len(ql) >= 3 and ql in k.lower():
            hits.append(("Setting", k, str(v), "Configuration", {}))
    return hits
