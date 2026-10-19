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
    title = next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), "")
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


# ---------------------------------------------------------------- shared charts
def perf_comparison_chart(height: int = 270) -> None:
    metrics = [("f1", "F1"), ("precision", "Precision"), ("recall", "Recall"), ("roc_auc", "ROC-AUC"), ("ap", "PR-AUC")]
    models = ["Original (leaky test)", "Repaired"]
    d = pd.DataFrame([dict(metric=label, model=m, value=src[k]) for k, label in metrics
                      for m, src in zip(models, [orig, rep])])
    x = alt.X("metric:N", sort=[l for _, l in metrics], title=None, axis=alt.Axis(labelAngle=0, labelFontSize=12))
    base = alt.Chart(d).encode(x=x, xOffset=alt.XOffset("model:N", sort=models))
    bars = base.mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2, size=30).encode(
        y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1.08]), title=None, axis=alt.Axis(values=[0, .2, .4, .6, .8, 1])),
        color=alt.Color("model:N", scale=alt.Scale(domain=models, range=[T["original"], T["repaired"]]), legend=None),
        tooltip=["model:N", "metric:N", alt.Tooltip("value:Q", format=".3f")])
    labels = base.mark_text(dy=-7, fontSize=11, color=T["text2"]).encode(y="value:Q", text=alt.Text("value:Q", format=".2f"))
    gate_rule = alt.Chart(pd.DataFrame({"y": [gate["gate_f1"]]})).mark_rule(strokeDash=[4, 4], color=T["muted"]).encode(
        y="y:Q", tooltip=[alt.Tooltip("y:Q", title="release gate (F1)")])
    chart(bars + labels + gate_rule, height)


def original_loss_chart(height: int = 230) -> bool:
    if not OLOG or OLOG["loss"].empty:
        note("original/bnypi_executed.ipynb has no training log. Run <b>Reproduce original notebook</b>.")
        return False
    d = OLOG["loss"].melt("epoch", ["recon", "clf"], var_name="loss", value_name="value")
    names = {"recon": "reconstruction", "clf": "classifier"}
    d["loss"] = d["loss"].map(names)
    c = alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=18), strokeWidth=1.6).encode(
        x=alt.X("epoch:Q", title="epoch"), y=alt.Y("value:Q", title="training loss"),
        color=alt.Color("loss:N", scale=alt.Scale(domain=list(names.values()), range=[T["accent"], T["logreg"]]),
                        legend=alt.Legend(title=None)),
        tooltip=["loss:N", "epoch:Q", alt.Tooltip("value:Q", format=".4f")])
    chart(c, height)
    return True


def score_hist(split: str, measure: str, height: int = 230) -> None:
    sc = LIVE["scored"]
    d = sc if split == "all" else sc[sc["split"] == split]
    s = d["score"].to_numpy()
    s = np.maximum(s, s[s > 0].min() if (s > 0).any() else 1e-12)  # float32 sigmoid can underflow to 0; log axis can't
    edges = np.logspace(math.floor(np.log10(s.min())), math.ceil(np.log10(s.max())), 41)
    rows = []
    for cls, name in [(0, "Pass"), (1, "Defect")]:
        counts, _ = np.histogram(s[d["defect"].to_numpy() == cls], edges)
        total = max(counts.sum(), 1)
        rows += [dict(x0=a, x1=b, count=int(c), share=c / total, cls=name) for a, b, c in zip(edges[:-1], edges[1:], counts)]
    h = pd.DataFrame(rows)
    h = h[h["count"] > 0]
    ytitle = "wafers" if measure == "count" else "share of class"
    bars = alt.Chart(h).mark_bar(opacity=0.7, binSpacing=0).encode(
        x=alt.X("x0:Q", scale=alt.Scale(type="log"), axis=alt.Axis(format=".0e", title="model score (log scale)")),
        x2="x1:Q", y=alt.Y(f"{measure}:Q", stack=None, title=ytitle, axis=alt.Axis(format="d" if measure == "count" else ".0%")),
        y2=alt.datum(0),
        color=alt.Color("cls:N", scale=alt.Scale(domain=["Pass", "Defect"], range=[T["normal"], T["defect"]]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("cls:N", title="actual"), alt.Tooltip("x0:Q", title="from", format=".2e"),
                 alt.Tooltip("x1:Q", title="to", format=".2e"), "count:Q", alt.Tooltip("share:Q", format=".1%")])
    thr = alt.Chart(pd.DataFrame({"t": [rep["threshold"]]}))
    rule = thr.mark_rule(strokeDash=[5, 4], color=T["text2"]).encode(x="t:Q", tooltip=[alt.Tooltip("t:Q", title="threshold", format=".2e")])
    label = thr.mark_text(align="left", dx=4, dy=-4, baseline="top", color=T["text2"], fontSize=10).encode(
        x="t:Q", y=alt.value(0), text=alt.value(f"threshold {rep['threshold']:.1e}"))
    chart(bars + rule + label, height)


def fingerprint(i: int, height: int = 230) -> None:
    """One wafer's model inputs: each continuous sensor's z-score, laid out in sensor order."""
    pre, names = LIVE["pre"], LIVE["feature_names"]
    n = len(pre.mean_)
    z = LIVE["X"][i, :n]
    cols = math.ceil(math.sqrt(n))
    raw = LIVE["raw"].iloc[i]
    d = pd.DataFrame(dict(sensor=names[:n], z=z, raw=[raw[s] for s in names[:n]], k=np.arange(n)))
    d["col"], d["row"] = d["k"] % cols, d["k"] // cols
    c = alt.Chart(d).mark_rect(cornerRadius=2).encode(
        x=alt.X("col:O", axis=None), y=alt.Y("row:O", axis=None),
        color=alt.Color("z:Q", scale=alt.Scale(domain=[-pre.clip_z, 0, pre.clip_z],
                                               range=[T["z_low"], T["z_mid"], T["z_high"]]), legend=None),
        tooltip=["sensor:N", alt.Tooltip("z:Q", title="z-score (model input)", format=".2f"),
                 alt.Tooltip("raw:Q", title="raw reading", format=".4g")])
    chart(c, height)


def z_legend(clip: float, height: int = 210) -> str:
    return (f'<div class="wg-zlabels">+{clip:g}σ</div><div class="wg-zbar" style="height:{height}px;'
            f'background:linear-gradient(to bottom,{T["z_high"]},{T["z_mid"]},{T["z_low"]})"></div>'
            f'<div class="wg-zlabels">−{clip:g}σ</div>')


def wafer_caption(w: pd.Series, kind: str) -> str:
    decision = "flagged" if w["flagged"] else "not flagged"
    return (f'<div style="text-align:center;font-size:.84rem;color:var(--text)">{kind} '
            f'<span class="wg-mono">#{w["row"]}</span>'
            f'<div class="wg-note">{w["timestamp"]:%Y-%m-%d %H:%M} · score <span class="wg-mono">{sci(w["score"])}</span><br>'
            f'{decision} · {outcome_chip(outcome(w["defect"], w["flagged"]))}</div></div>')


def wafers_table(d: pd.DataFrame, highlight_row=None) -> str:
    t = pd.DataFrame({
        "Wafer": d["row"].map(lambda r: f"#{r}"),
        "Timestamp": d["timestamp"].dt.strftime("%Y-%m-%d %H:%M"),
        "Split": d["split"],
        "PS-L": d["legacy"].map({True: "yes", False: ""}),
        "Score": d["score"],
        "Decision": d["flagged"].map({True: "flag", False: "pass"}),
        "Actual": d["defect"].map({1: "Defect", 0: "Pass"}),
        "Outcome": [outcome(a, b) for a, b in zip(d["defect"], d["flagged"])],
    })
    return table(t, {"Score": sci, "Outcome": outcome_chip,
                     "Actual": lambda v: f'<span class="{"c-def" if v == "Defect" else "c-pass"}">{v}</span>'},
                 highlight=(lambda r: r["Wafer"] == f"#{highlight_row}") if highlight_row is not None else None)


# ---------------------------------------------------------------- pages
def page_overview() -> None:
    page_header("Overview", esc(RD["subtitle"]))
    passed = gate["passed"]
    verdict = '<span class="pass">Meets</span>' if passed else '<span class="fail">Does not meet</span>'
    stats = [
        ("Original notebook F1", f"{orig['f1']:.3f}", f"leaky test, {orig['test_prevalence']:.0%} defects"),
        ("Precision", f"{rep['precision']:.3f}", f"{rep['tp']} of {rep['tp'] + rep['fp']} alarms real"),
        ("Recall", f"{rep['recall']:.3f}", f"{rep['tp']} of {rep['n_defects']} defects caught"),
        ("PR-AUC", f"{rep['ap']:.3f}", f"no-skill {no_skill:.3f}"),
        ("ROC-AUC", f"{rep['roc_auc']:.3f}", "0.5 = chance"),
        ("Test defect rate", f"{rep['prevalence']:.1%}", f"{rep['n_defects']} of {rep['n']} wafers"),
    ]
    cells = "".join(f'<div class="wg-stat"><div class="wg-label">{a}</div><div class="wg-stat-value">{b}</div>'
                    f'<div class="wg-stat-sub">{c}</div></div>' for a, b, c in stats)
    st.html(f"""<div class="wg-hero"><div class="wg-hero-main">
<div class="wg-label">Repaired model · test F1</div>
<div class="wg-hero-value">{rep["f1"]:.3f}</div>
<div class="wg-hero-gate">{verdict} the release gate of {gate["gate_f1"]:.2f}</div>
<div class="wg-hero-note">95% CI {boot["f1_ci95"][0]:.3f}–{boot["f1_ci95"][1]:.3f} · most recent {rep["n"]} wafers,
threshold frozen on validation</div></div><div class="wg-stats">{cells}</div></div>""")

    left, right = st.columns([1.12, 1], gap="medium")
    with left, st.container(key="card_perf"):
        a, b = st.columns([1, 1.2], vertical_alignment="center")
        with a:
            card_title("Original vs repaired", "test-set metrics")
        b.html(f'<div style="display:flex;justify-content:flex-end">'
               f'{legend([("Original (leaky test)", T["original"]), ("Repaired", T["repaired"])])}</div>')
        perf_comparison_chart(330)
        note(f"Original: random split, SMOTE-balanced test ({orig['test_size']} rows, {orig['test_prevalence']:.0%} defects). "
             f"Repaired: most recent {rep['n']} wafers ({rep['prevalence']:.1%} defects). Dashed line: release gate.")
    with right, st.container(key="card_fp"):
        a, b = st.columns([1.25, 1], vertical_alignment="center")
        with a:
            card_title("Sensor fingerprints", "test block")
        mode = b.segmented_control("Pick", ["Latest", "Top score"], default="Latest", key="fp_mode",
                                   label_visibility="collapsed",
                                   help="Which passing / defective test wafer to show: the most recent, or the one "
                                        "the model scores highest") or "Latest"
        if require_live():
            te = LIVE["scored"][LIVE["scored"]["split"] == "test"]
            key = "timestamp" if mode == "Latest" else "score"
            normal = te[te["defect"] == 0].sort_values(key).iloc[-1]
            defect = te[te["defect"] == 1].sort_values(key).iloc[-1]
            c1, c2, c3 = st.columns([1, 0.18, 1], vertical_alignment="center")
            with c1:
                fingerprint(normal.name, 225)
                st.html(wafer_caption(normal, "Passing wafer"))
            with c2:
                st.html(z_legend(LIVE["pre"].clip_z, 180))
            with c3:
                fingerprint(defect.name, 225)
                st.html(wafer_caption(defect, "Defective wafer"))
            note(f"Each cell is one of the {len(LIVE['pre'].mean_)} sensors the model reads, coloured by its z-score "
                 "against training statistics, in sensor order. SECOM records no on-wafer positions, so this is not "
                 "a spatial wafer map.")

    a, b, c = st.columns([0.95, 1, 1.3], gap="medium")
    with a, st.container(key="card_loss"):
        card_title("Training loss", "original notebook, as logged")
        if original_loss_chart(220):
            L = OLOG["loss"]
            note(f"Reconstruction loss {L['recon'].iloc[0]:.3f} → {L['recon'].iloc[-1]:.3f} (the broadcast bug rewards a "
                 f"constant output) while classifier loss falls to {L['clf'].iloc[-1]:.4f}. No validation loss was logged.")
    with b, st.container(key="card_dist"):
        card_title("Score distribution", f"test block, {rep['n']} wafers")
        if require_live():
            score_hist("test", "count", 220)
            note("Defect probability from the saved model. Wafers right of the dashed threshold are flagged.")
    with c, st.container(key="card_recent"):
        h, v = st.columns([1.6, 1], vertical_alignment="center")
        with h:
            card_title("Recent wafers")
        v.button("View all", icon=":material/arrow_forward:", icon_position="right", type="tertiary", key="view_all",
                 on_click=go, args=("Wafer Explorer",), kwargs=EXPLORER_FILTERS)
        if require_live():
            recent = LIVE["scored"].sort_values("timestamp", ascending=False).head(6)
            t = pd.DataFrame({"Wafer": recent["row"].map(lambda r: f"#{r}"),
                              "Time": recent["timestamp"].dt.strftime("%m-%d %H:%M"),
                              "Score": recent["score"],
                              "Prediction": recent["flagged"].map({True: "Defective", False: "Normal"}),
                              "Result": [outcome(a_, b_) for a_, b_ in zip(recent["defect"], recent["flagged"])]})
            st.html(table(t, {"Score": sci, "Result": outcome_chip,
                              "Prediction": lambda p: f'<span class="{"c-def" if p == "Defective" else "c-pass"}">{p}</span>'}))
            note(f"Newest wafers in the data ({recent['timestamp'].dt.year.iloc[0]}). Result: TP/TN correct, "
                 "FP false alarm, FN missed defect.")


def page_performance() -> None:
    page_header("Model Performance", f"Repaired model on the most recent {rep['n']} wafers, threshold chosen on the "
                f"validation block and frozen. No-skill PR-AUC = defect rate = {no_skill:.3f}.")
    st.html(tiles([
            ("F1", f"{rep['f1']:.3f}", f"95% CI {boot['f1_ci95'][0]:.3f}–{boot['f1_ci95'][1]:.3f}"),
            ("Precision", f"{rep['precision']:.3f}", f"{rep['tp']} / {rep['tp'] + rep['fp']} alarms real"),
            ("Recall", f"{rep['recall']:.3f}", f"{rep['tp']} / {rep['n_defects']} defects caught"),
            ("MCC", f"{rep['mcc']:.3f}", "Matthews correlation"),
            ("PR-AUC", f"{rep['ap']:.3f}", f"CI {boot['ap_ci95'][0]:.3f}–{boot['ap_ci95'][1]:.3f} · no-skill {no_skill:.3f}"),
            ("ROC-AUC", f"{rep['roc_auc']:.3f}", "0.5 = chance"),
        ]))

    left, right = st.columns([1.5, 1], gap="medium")
    with left, st.container(key="card_pr"):
        card_title("Precision–recall on the test block")
        curves = [pd.DataFrame({"recall": R["pr_curves"][k]["recall"], "precision": R["pr_curves"][k]["precision"],
                                "model": label})
                  for k, label in [("repaired", "repaired autoencoder"), ("logreg", "logistic regression")]]
        lines = alt.Chart(pd.concat(curves)).mark_line(interpolate="step-after", strokeWidth=2).encode(
            x=alt.X("recall:Q", scale=alt.Scale(domain=[0, 1]), title="recall (share of defects caught)"),
            y=alt.Y("precision:Q", scale=alt.Scale(domain=[0, 1]), title="precision (share of alarms that are real)"),
            color=alt.Color("model:N", scale=alt.Scale(domain=["repaired autoencoder", "logistic regression"],
                                                       range=[T["repaired"], T["logreg"]]), legend=alt.Legend(title=None)),
            tooltip=["model:N", alt.Tooltip("recall:Q", format=".3f"), alt.Tooltip("precision:Q", format=".3f")])
        base = alt.Chart(pd.DataFrame({"y": [no_skill]})).mark_rule(color=T["muted"], strokeDash=[5, 4]).encode(
            y="y:Q", tooltip=[alt.Tooltip("y:Q", title="no-skill precision", format=".3f")])
        chart(lines + base, 320)
        note(f"Dashed line: no-skill precision = test defect rate ({no_skill:.3f}). Logistic regression "
             f"(C = {R['baseline_logreg']['C']}) PR-AUC {R['baseline_logreg']['ap']:.3f} vs autoencoder {rep['ap']:.3f}.")
    with right, st.container(key="card_cm"):
        card_title("Confusion matrix", "test block")
        st.html(f'''<div class="wg-cm">
<div class="h"></div><div class="h">predicted pass</div><div class="h">predicted defect</div>
<div class="h" style="text-align:right">actual pass</div>
<div class="v">{rep["tn"]}<small>TN</small></div>
<div class="v err">{rep["fp"]}<small>FP · false alarms</small></div>
<div class="h" style="text-align:right">actual defect</div>
<div class="v err">{rep["fn"]}<small>FN · missed</small></div>
<div class="v">{rep["tp"]}<small>TP · caught</small></div></div>''')
        note(f"Threshold <b>{rep['threshold']:.2e}</b> was chosen on the validation block, then frozen. "
             f"Across {len(seeds['runs'])} seeds: F1 {seeds['mean']['f1']:.3f} ± {seeds['std']['f1']:.3f}, "
             f"PR-AUC {seeds['mean']['ap']:.3f} ± {seeds['std']['ap']:.3f}.")
        if LIVE_OK:
            v = LIVE["verify"]
            note(f"Re-scoring the saved model now gives TN {v['checks']['tn'][0]}, "
                 f"FP {v['checks']['fp'][0]}, FN {v['checks']['fn'][0]}, TP {v['checks']['tp'][0]}, "
                 f"{'identical to' if v['ok'] else 'different from'} metrics.json.")

    left, right = st.columns([1.3, 1], gap="medium")
    with left, st.container(key="card_hist"):
        card_title("Score distribution", "live model scores by actual class")
        b, c = st.columns([1.6, 1], vertical_alignment="center")
        split = b.segmented_control("Block", ["test", "val", "train", "all"], default="test", key="hist_split",
                                    label_visibility="collapsed") or "test"
        measure = c.segmented_control("Measure", ["count", "share"], default="count", key="hist_measure",
                                      label_visibility="collapsed") or "count"
        if require_live():
            score_hist(split, measure, 270)
            sel = LIVE["scored"] if split == "all" else LIVE["scored"][LIVE["scored"]["split"] == split]
            note("Train-block scores are in-sample (the model was fitted on those wafers, after SMOTE). "
                 f"“share” normalises each class so the {int(sel['defect'].sum())} defects are visible next to the "
                 f"{int((sel['defect'] == 0).sum())} passes.")
    with right, st.container(key="card_seeds"):
        card_title("Seed-to-seed variation", f"{len(seeds['runs'])} training seeds")
        sd = pd.DataFrame(seeds["runs"])
        st.html(table(sd, {"threshold": sci, **{k: ".3f" for k in ["precision", "recall", "f1", "ap", "roc_auc"]}},
                      highlight=lambda r: r["seed"] == R["config"]["seed"]))
        note(f"Highlighted: seed {R['config']['seed']}, the saved model. Mean ± std: " +
             ", ".join(f"{k} {seeds['mean'][k]:.3f} ± {seeds['std'][k]:.3f}" for k in ["precision", "recall", "f1", "ap"]))

    with st.container(key="card_summary"):
        card_title("All evaluations side by side")
        sm = pd.DataFrame(R["summary_table"])
        st.html(table(sm, {k: ".4f" for k in ["prevalence", "f1", "precision", "recall", "ap"]}))
    with st.container(key="card_training"):
        tr = R["repaired_training"]
        card_title("Training diagnostics", "repaired model, seed " + str(rep["seed"]))
        st.html(tiles([
            ("final recon loss", f"{tr['final_train_recon_loss']:.4f}", "per-element MSE, training"),
            ("final classifier loss", f"{tr['final_train_clf_loss']:.2e}", "BCE on SMOTE-balanced training"),
            ("test score median", sci(tr["test_score_median"]), f"max {tr['test_score_max']:.3f}"),
            ("val score median", sci(tr["val_score_median"]), f"max {tr['val_score_max']:.3f}"),
            ("test wafers ≥ 0.5", str(tr["test_wafers_scored_at_least_0_5"]),
             f"of which defects: {tr['test_defects_scored_at_least_0_5']}"),
            ("recon spread ratio", f"{rep['recon_spread_ratio']:.3f}", f"original model: {R['evidence_loss_bug']['original_model_recon_spread_ratio']:.3f}"),
        ]))
        note(f"Classifier loss on the SMOTE-balanced training set ends at {tr['final_train_clf_loss']:.2e}, while the median "
             f"test score is {sci(tr['test_score_median'])}; {tr['test_wafers_scored_at_least_0_5']} test wafers score ≥ 0.5, "
             f"{tr['test_defects_scored_at_least_0_5']} of them defects. The F1-best threshold from validation is "
             f"{sci(rep['threshold'])}.")


def page_data() -> None:
    d, leg, split = R["data"], R["legacy"], R["split"]
    page_header("Data Analysis", f"<code>data/secom_telemetry_modified.csv</code> · {d['time_start'][:10]} → {d['time_end'][:10]}")
    st.html(tiles([
        ("wafers", str(d["n_wafers"]), f"{d['n_pass']} pass · {d['n_fail']} defect"),
        ("defect rate", f"{d['prevalence']:.2%}", "natural prevalence"),
        ("sensors", str(d["n_sensors"]), f"{d['n_constant_sensors']} constant overall"),
        ("missing values", f"{d['missing_rate']:.2%}", "of all sensor cells"),
        ("PS-L legacy wafers", str(leg["n_legacy"]), f"{leg['share_legacy']:.1%} of wafers · {leg['fails_in_legacy']} failed"),
        ("model inputs", str(R["preprocessing"]["features_out"]), f"from {R['preprocessing']['features_in']} sensors"),
    ]))
    left, right = st.columns([1.6, 1], gap="medium")
    with left, st.container(key="card_drift"):
        a, b = st.columns([1.4, 1], vertical_alignment="center")
        with a:
            card_title("Defect rate over time", "rolling window, time order")
        window = b.slider("Window (wafers)", 25, 400, 150, 25, key="drift_window")
        if require_live():
            sc = LIVE["scored"]
            roll = pd.DataFrame({"timestamp": sc["timestamp"],
                                 "rate": sc["defect"].rolling(window, min_periods=window).mean()}).dropna()
            bands = pd.DataFrame([dict(split=k, start=pd.Timestamp(v["start"]), end=pd.Timestamp(v["end"]))
                                  for k, v in split.items()])
            band = alt.Chart(bands).mark_rect(opacity=0.12).encode(
                x="start:T", x2="end:T",
                color=alt.Color("split:N", scale=alt.Scale(domain=["train", "val", "test"],
                                                           range=[T["repaired"], T["warn"], T["defect"]]),
                                legend=alt.Legend(title="block")))
            line = alt.Chart(roll).mark_line(color=T["text"], strokeWidth=1.8).encode(
                x=alt.X("timestamp:T", title=None), y=alt.Y("rate:Q", title="share failing", axis=alt.Axis(format=".0%")),
                tooltip=[alt.Tooltip("timestamp:T", format="%Y-%m-%d %H:%M"), alt.Tooltip("rate:Q", format=".1%")])
            overall = alt.Chart(pd.DataFrame({"y": [d["prevalence"]]})).mark_rule(strokeDash=[5, 4], color=T["muted"]).encode(
                y="y:Q", tooltip=[alt.Tooltip("y:Q", title="overall", format=".2%")])
            chart(band + line + overall, 280)
            note(f"Each point = the defect rate of the previous {window} wafers; dashed = overall {d['prevalence']:.2%}. "
                 "The rate drifts, so evaluation must respect time order.")
    with right, st.container(key="card_monthly"):
        card_title("Monthly defect rate")
        mdf = pd.DataFrame({"month": list(d["monthly_defect_rate"]), "rate": list(d["monthly_defect_rate"].values())})
        c = alt.Chart(mdf).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=T["repaired"], size=38).encode(
            x=alt.X("month:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("rate:Q", title=None, axis=alt.Axis(format=".0%")),
            tooltip=["month:N", alt.Tooltip("rate:Q", format=".2%")])
        txt = c.mark_text(dy=-8, color=T["text2"], fontSize=11).encode(text=alt.Text("rate:Q", format=".1%"))
        chart(c + txt, 280)

    left, right = st.columns([1.3, 1], gap="medium")
    with left, st.container(key="card_split"):
        card_title("Chronological split", "train → validation → test")
        sp = pd.DataFrame(split).T.reset_index(names="block")
        st.html(table(sp, {"prevalence": ".2%"}))
        pr = R["preprocessing"]
        note(f"Preprocessing is fitted on training wafers only: {pr['sensors_dropped_constant_in_train']} sensors constant in "
             f"training are dropped ({pr['features_in']} → {pr['features_out']} features incl. "
             f"{pr['features_out'] - (pr['features_in'] - pr['sensors_dropped_constant_in_train'])} PS-L flags). Later wafers sit up "
             f"to {pr['test_max_abs_z_before_clip']:,.0f}σ from training statistics, so values are clipped at ±{pr['clip_z']:g}σ "
             f"({pr['test_cells_beyond_clip']} of {pr['test_cells_total']:,} test cells).")
    with right, st.container(key="card_legacy"):
        card_title("Legacy probe station PS-L", "sensor_1 firmware codes")
        cc = pd.DataFrame({"code": list(leg["code_counts"]), "wafers": list(leg["code_counts"].values())})
        c = alt.Chart(cc).mark_bar(color=T["logreg"], size=34, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("code:N", title="firmware code", axis=alt.Axis(labelAngle=0)), y=alt.Y("wafers:Q", title="wafers"),
            tooltip=["code:N", "wafers:Q"])
        chart(c + c.mark_text(dy=-8, color=T["text2"]).encode(text="wafers:Q"), 170)
        note(f"Real sensor_1 readings span {leg['sensor_1_normal_min']:.0f}–{leg['sensor_1_normal_max']:.0f}; codes appear on "
             f"PS-L wafers only ({leg['codes_outside_legacy']} outside). Defect rate PS-L {leg['defect_rate_legacy']:.1%} vs "
             f"others {leg['defect_rate_normal']:.1%}.")

    with st.container(key="card_sensor"):
        card_title("Sensor lookup", "raw readings by class")
        if require_live():
            raw, pre, names = LIVE["raw"], LIVE["pre"], LIVE["feature_names"]
            sensors = [c for c in raw.columns if c.startswith("sensor_")]
            if ss.get("sensor_pick") not in sensors:
                ss.sensor_pick = sensors[0]
            a, b = st.columns([1, 3], gap="medium")
            sensor = a.selectbox("Sensor", sensors, key="sensor_pick")
            is_leg = raw["sensor_0"].eq(0.0)
            mask = ~is_leg if sensor in ("sensor_0", "sensor_1") else pd.Series(True, index=raw.index)
            vals, y = raw.loc[mask, sensor], (raw.loc[mask, "label"] == 1).astype(int)
            kept = sensor in names
            j = names.index(sensor) if kept else None
            with a:
                st.html(tiles([
                    ("status", "model input" if kept else "dropped", "constant in training" if not kept else "z-scored, clipped"),
                    ("missing", f"{raw[sensor].isna().mean():.1%}", "of all wafers"),
                    ("train mean ± std", f"{pre.mean_[j]:.4g}" if kept else "–", f"± {pre.std_[j]:.4g}" if kept else ""),
                    ("mean pass / defect", f"{vals[y == 0].mean():.4g}", f"defect {vals[y == 1].mean():.4g}"),
                ]))
            with b:
                v = vals.dropna()
                if v.nunique() <= 1:
                    note(f"{sensor} takes a single value ({v.iloc[0] if len(v) else 'none'}), so there is nothing to plot.")
                else:
                    edges = np.histogram_bin_edges(v, bins=40)
                    rows = []
                    for cls, name in [(0, "Pass"), (1, "Defect")]:
                        cnt, _ = np.histogram(v[y.loc[v.index] == cls], edges)
                        rows += [dict(x0=e0, x1=e1, share=c_ / max(cnt.sum(), 1), count=int(c_), cls=name)
                                 for e0, e1, c_ in zip(edges[:-1], edges[1:], cnt)]
                    h = pd.DataFrame(rows)
                    c = alt.Chart(h[h["count"] > 0]).mark_bar(opacity=0.7, binSpacing=0).encode(
                        x=alt.X("x0:Q", title=f"{sensor} reading"), x2="x1:Q",
                        y=alt.Y("share:Q", stack=None, title="share of class", axis=alt.Axis(format=".0%")), y2=alt.datum(0),
                        color=alt.Color("cls:N", scale=alt.Scale(domain=["Pass", "Defect"], range=[T["normal"], T["defect"]]),
                                        legend=alt.Legend(title=None)),
                        tooltip=["cls:N", alt.Tooltip("x0:Q", format=".4g"), alt.Tooltip("x1:Q", format=".4g"), "count:Q"])
                    chart(c, 250)
                    if sensor in ("sensor_0", "sensor_1"):
                        note("PS-L wafers are excluded here: their value is a sentinel / firmware code, not a reading.")

    with st.expander("Notebook figures"):
        a, b = st.columns(2)
        with a:
            figure("01_legacy_sensor_1", f"{leg['n_legacy']} PS-L wafers report firmware codes instead of readings.")
        with b:
            figure("02_defect_rate_drift", "The defect rate drifts over time, so evaluation must respect time order.")


def page_ablation() -> None:
    page_header("Ablation Study", "Start from the repaired pipeline and re-introduce one original defect at a time "
                f"({len(R['config']['eval_seeds'])} training seeds each). The last row switches every defect back on.")
    abl = pd.DataFrame(R["ablation"])
    fixed = abl.iloc[0]                       # notebook §7: first row = repaired, last row = all defects together
    top = abl.iloc[1:-1].sort_values("f1_mean").iloc[-1]
    st.html(tiles([
        ("repaired F1", f"{fixed['f1_mean']:.3f}", f"± {fixed['f1_std']:.3f} over {len(R['config']['eval_seeds'])} seeds"),
        ("largest single-defect jump", f"{top['f1_mean']:.3f}", esc(top["config"])),
        ("all defects together", f"{abl.iloc[-1]['f1_mean']:.3f}", f"test prevalence {abl.iloc[-1]['test_prevalence']:.0%}"),
        ("release gate", f"{gate['gate_f1']:.2f}", "F1 required"),
    ]))
    with st.container(key="card_abl"):
        card_title("Re-introducing each defect into the repaired pipeline", "bar = mean test F1, whisker = ±1 std")
        abl["kind"] = ["repaired" if c == fixed["config"] else "defect re-introduced" for c in abl["config"]]
        abl["lo"], abl["hi"] = abl["f1_mean"] - abl["f1_std"], abl["f1_mean"] + abl["f1_std"]
        y = alt.Y("config:N", sort=abl["config"].tolist(), title=None, axis=alt.Axis(labelLimit=320))
        bars = alt.Chart(abl).mark_bar(size=20, cornerRadiusEnd=4).encode(
            x=alt.X("f1_mean:Q", title="test F1", scale=alt.Scale(domain=[0, 1.05])), y=y,
            color=alt.Color("kind:N", scale=alt.Scale(domain=["repaired", "defect re-introduced"],
                                                      range=[T["repaired"], T["original"]]), legend=alt.Legend(title=None)),
            tooltip=[alt.Tooltip("config:N", title="configuration"), alt.Tooltip("f1_mean:Q", title="F1 mean", format=".3f"),
                     alt.Tooltip("f1_std:Q", title="F1 std", format=".3f"), alt.Tooltip("ap_mean:Q", title="PR-AUC", format=".3f"),
                     alt.Tooltip("test_prevalence:Q", title="test defect rate", format=".1%"),
                     alt.Tooltip("recon_spread:Q", title="reconstruction spread", format=".3f")])
        whiskers = alt.Chart(abl).mark_rule(color=T["text2"]).encode(x="lo:Q", x2="hi:Q", y=y)
        labels = alt.Chart(abl).mark_text(align="left", dx=6, color=T["text2"], fontSize=11).encode(
            x="hi:Q", y=y, text=alt.Text("f1_mean:Q", format=".3f"))
        gate_rule = alt.Chart(pd.DataFrame({"x": [gate["gate_f1"]]})).mark_rule(color=T["muted"], strokeDash=[5, 4]).encode(x="x:Q")
        chart(bars + whiskers + labels + gate_rule, 340)
        st.html(table(abl[["config", "f1_mean", "f1_std", "ap_mean", "precision_mean", "recall_mean",
                           "test_prevalence", "recon_spread", "recon_mse"]],
                      {**{k: ".4f" for k in ["f1_mean", "f1_std", "ap_mean", "precision_mean", "recall_mean",
                                             "recon_spread", "recon_mse"]}, "test_prevalence": ".1%"}))
        bug = abl.set_index("config").loc["+ broadcast loss bug"]
        note(f"<b>recon_spread</b> = how much reconstructions vary between wafers relative to the inputs: "
             f"{fixed['recon_spread']:.3f} repaired vs {bug['recon_spread']:.3f} with the broadcast loss bug "
             f"(F1 {fixed['f1_mean']:.3f} vs {bug['f1_mean']:.3f}).")
    runs_df = read_csv(str(ABLATION_RUNS), mtime(ABLATION_RUNS))
    if runs_df is not None:
        with st.container(key="card_abl_runs"):
            card_title("Every run", f"{len(runs_df)} runs from results/ablation_runs.csv")
            dots = alt.Chart(runs_df).mark_circle(size=70, opacity=0.85).encode(
                x=alt.X("f1:Q", title="test F1", scale=alt.Scale(domain=[0, 1.05])),
                y=alt.Y("config:N", sort=abl["config"].tolist(), title=None, axis=alt.Axis(labelLimit=320)),
                color=alt.Color("seed:N", legend=alt.Legend(title="seed"),
                                scale=alt.Scale(range=[T["repaired"], T["original"], T["violet"], T["warn"], T["good"]])),
                tooltip=["config:N", "seed:N", alt.Tooltip("f1:Q", format=".3f"), alt.Tooltip("ap:Q", format=".3f"),
                         alt.Tooltip("precision:Q", format=".3f"), alt.Tooltip("recall:Q", format=".3f"),
                         alt.Tooltip("threshold:Q", format=".2e")])
            chart(dots, 320)


def page_walk_forward() -> None:
    page_header("Walk-Forward Evaluation", "Each fold trains on all wafers up to a point, tunes the threshold on the last "
                "quarter of that history and tests on the next block, so every prediction uses only the past.")
    p = wf["pooled"]
    st.html(tiles([
        ("pooled F1", f"{p['f1']:.3f}", f"{p['n']} wafers · {p['n_defects']} defects"),
        ("pooled precision", f"{p['precision']:.3f}", "share of alarms that are real"),
        ("pooled recall", f"{p['recall']:.3f}", "share of defects caught"),
        ("mean fold PR-AUC", f"{wf['mean_fold_ap']:.3f}", f"pooled prevalence {p['prevalence']:.1%}"),
    ]))
    folds = pd.DataFrame(wf["folds"])
    folds["fold_label"] = [f"Fold {r.fold} · {r.test_start[5:]}→{r.test_end[5:]}" for r in folds.itertuples()]
    with st.container(key="card_wf"):
        card_title("Per-fold metrics", "PR-AUC should beat the no-skill line (= fold defect rate)")
        m = folds.melt(["fold_label", "prevalence"], ["precision", "recall", "f1", "ap"], var_name="metric")
        order = folds["fold_label"].tolist()
        bars = alt.Chart(m).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("fold_label:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("metric:N", sort=["precision", "recall", "f1", "ap"]),
            y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("metric:N", sort=["precision", "recall", "f1", "ap"], legend=alt.Legend(title=None),
                            scale=alt.Scale(domain=["precision", "recall", "f1", "ap"],
                                            range=[T["repaired"], T["violet"], T["original"], T["warn"]])),
            tooltip=["fold_label:N", "metric:N", alt.Tooltip("value:Q", format=".3f")])
        prev = alt.Chart(folds).mark_tick(color=T["text"], thickness=2, size=70).encode(
            x=alt.X("fold_label:N", sort=order), y="prevalence:Q",
            tooltip=["fold_label:N", alt.Tooltip("prevalence:Q", title="defect rate (no-skill PR-AUC)", format=".3f")])
        chart(bars + prev, 320)
        note("Horizontal tick = fold defect rate, the PR-AUC a random ranking would get.")
    with st.container(key="card_wf_table"):
        card_title("Folds")
        st.html(table(folds.drop(columns=["fold_label"]),
                      {"prevalence": ".1%", "threshold": sci, **{k: ".3f" for k in ["precision", "recall", "f1", "ap", "roc_auc"]}}))


def page_explorer() -> None:
    page_header("Wafer Explorer", "Every wafer, re-scored live by the saved repaired model "
                f"(<code>artifacts/</code>). Threshold {rep['threshold']:.2e}, frozen from validation.")
    if not require_live():
        return
    sc = LIVE["scored"]
    with st.container(key="card_filters"):
        a, b, c, d, e = st.columns([1.6, 1, 1, 1, 1.3])
        splits = a.multiselect("Block", ["train", "val", "test"], default=["train", "val", "test"], key="ex_split")
        actual = b.selectbox("Actual", ["All", "Defect", "Pass"], key="ex_actual")
        decision = c.selectbox("Decision", ["All", "Flagged", "Not flagged"], key="ex_decision")
        psl = d.selectbox("PS-L", ["All", "PS-L only", "Exclude PS-L"], key="ex_psl")
        sort = e.selectbox("Sort", ["Newest first", "Oldest first", "Highest score", "Highest reconstruction error"],
                           key="ex_sort")
    f = sc[sc["split"].isin(splits)]
    if actual != "All":
        f = f[f["defect"] == (1 if actual == "Defect" else 0)]
    if decision != "All":
        f = f[f["flagged"] == (decision == "Flagged")]
    if psl != "All":
        f = f[f["legacy"] == (psl == "PS-L only")]
    by, asc = {"Newest first": ("timestamp", False), "Oldest first": ("timestamp", True),
               "Highest score": ("score", False), "Highest reconstruction error": ("recon_mse", False)}[sort]
    f = f.sort_values(by, ascending=asc)
    size = 15
    if ss.pop("ex_jump", False) and ss.get("ex_pick") in f["row"].tolist():  # arrived from search: show its page
        ss.ex_page = f["row"].tolist().index(ss.ex_pick) // size + 1

    left, right = st.columns([1.5, 1], gap="medium")
    with left, st.container(key="card_wafers"):
        pages = max(1, math.ceil(len(f) / size))
        if ss.get("ex_page", 1) > pages:
            ss.ex_page = 1
        h, p = st.columns([2.2, 1], vertical_alignment="bottom")
        with h:
            card_title("Wafers", f"{len(f)} match · {int(f['defect'].sum())} defects · {int(f['flagged'].sum())} flagged")
        page = p.number_input(f"Page (of {pages})", 1, pages, key="ex_page")
        chunk = f.iloc[(page - 1) * size: page * size]
        st.html(wafers_table(chunk, ss.get("ex_pick")))
        out = f.assign(outcome=[outcome(x, y) for x, y in zip(f["defect"], f["flagged"])])
        st.download_button("Download filtered wafers (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name="wafer_guard_scores.csv", mime="text/csv", icon=":material/download:")
        note("Wafer # = position in the CSV (0 = first data row). Train-block scores are in-sample.")
    with right, st.container(key="card_wafer"):
        options = f["row"].tolist()
        if not options:
            note("No wafers match the filters.")
            return
        if ss.get("ex_pick") not in options:
            ss.ex_pick = options[0]
        lookup = f.set_index("row")
        pick = st.selectbox("Wafer", options, key="ex_pick",
                            format_func=lambda r: f"#{r} · {lookup.at[r, 'timestamp']:%Y-%m-%d %H:%M} · "
                                                  f"{'defect' if lookup.at[r, 'defect'] else 'pass'}")
        i = int(sc.index[sc["row"] == pick][0])
        w = sc.loc[i]
        raw = LIVE["raw"].iloc[i]
        o = outcome(w["defect"], w["flagged"])
        test_recon = sc.loc[sc["split"] == "test", "recon_mse"].median()
        code = f" · code {raw['sensor_1']:.0f}" if w["legacy"] else ""
        st.html(tiles([
            ("score", sci(w["score"]), f"{w['score'] / rep['threshold']:.2g}× threshold"),
            ("decision", "flag" if w["flagged"] else "pass", f"actual: {'defect' if w['defect'] else 'pass'} {outcome_chip(o)}"),
            ("reconstruction MSE", f"{w['recon_mse']:.3f}", f"test median {test_recon:.3f}"),
            ("block", w["split"], f"{w['timestamp']:%Y-%m-%d %H:%M}"),
            ("station", "PS-L legacy" if w["legacy"] else "current", code.strip(" ·") or "real sensor_0/1 readings"),
        ]))
        a, b = st.columns([1, 0.16], vertical_alignment="center")
        with a:
            fingerprint(i, 250)
        b.html(z_legend(LIVE["pre"].clip_z, 200))
        n = len(LIVE["pre"].mean_)
        z = LIVE["X"][i, :n]
        names = LIVE["feature_names"][:n]
        top = np.argsort(-np.abs(z))[:8]
        t = pd.DataFrame({"sensor": [names[k] for k in top], "raw": [raw[names[k]] for k in top],
                          "train mean": LIVE["pre"].mean_[top], "train std": LIVE["pre"].std_[top], "z (input)": z[top]})
        st.html(table(t, {"raw": ".4g", "train mean": ".4g", "train std": ".4g", "z (input)": "+.2f"}))
        note("Sensors furthest from training statistics for this wafer (|z| capped at "
             f"{LIVE['pre'].clip_z:g}σ; missing readings enter the model as 0).")


def page_audit() -> None:
    page_header("Audit Findings", "Evidence for each defect, measured by the notebook (§3) and stored in metrics.json.")
    ev_l, ev_b, ev_c = R["evidence_leakage"], R["evidence_loss_bug"], R["evidence_legacy_blindness"]
    top = max(R["ablation"][1:-1], key=lambda a: a["f1_mean"])  # single defects only (§7 row order)
    a, b, c = st.columns(3, gap="medium")
    with a, st.container(key="card_ev1"):
        card_title("1 · Data leakage <span class='wg-sev'>cell 8</span>")
        st.html(f'''<ul class="wg-evidence">
<li>Test set after SMOTE: <b>{ev_l["test_prevalence_after_smote"]:.0%} defects</b></li>
<li>Synthetic share of test defects: <b>{ev_l["test_defects_share_synthetic"]:.0%}</b></li>
<li>Nearest-training-row distance: real defects <b>{ev_l["nn_distance_median"]["real_defect"]:.2f}</b> vs real passes
<b>{ev_l["nn_distance_median"]["real_pass"]:.2f}</b></li>
<li>Real test defects with their own SMOTE “child” in training: <b>{ev_l["real_test_defects_with_own_smote_child_in_train"]:.0%}</b>
(max gap {ev_l["max_segment_gap"]:.1e})</li>
<li>Real test wafers with future wafers in training: <b>{ev_l["real_test_wafers_with_future_training_rows"]:.0%}</b>
(median {ev_l["median_future_training_rows_per_test_wafer"]:.0f} each)</li>
<li>Scaler fitted on <b>{ev_l["scaler_fit_rows"]}</b> rows, test rows included</li></ul>''')
    with b, st.container(key="card_ev2"):
        card_title("2 · Broadcast loss <span class='wg-sev'>cells 10–14</span>")
        st.html(f'''<ul class="wg-evidence">
<li><code>{ev_b["recon_shape"]}</code> − <code>{ev_b["input_shape"]}</code> → <b><code>{ev_b["broadcast_shape"]}</code></b></li>
<li>Buggy loss of a flawless reconstruction: <b>{ev_b["buggy_loss_perfect_reconstruction"]:.3f}</b></li>
<li>Buggy loss of a constant “average wafer”: <b>{ev_b["buggy_loss_batch_mean_output"]:.3f}</b>, which training prefers</li>
<li>Correct loss: flawless <b>{ev_b["correct_loss_perfect_reconstruction"]:.3f}</b>, average wafer
<b>{ev_b["correct_loss_batch_mean_output"]:.3f}</b></li>
<li>Original reconstructions' spread vs inputs: <b>{ev_b["original_model_recon_spread_ratio"]:.3f}×</b>
(repaired {rep["recon_spread_ratio"]:.3f}×)</li></ul>''')
    with c, st.container(key="card_ev3"):
        card_title("3 · PS-L blindness <span class='wg-sev'>cell 8</span>")
        codes = ", ".join(f"{k}: {v:.2f}σ" for k, v in ev_c["z_of_codes_in_original"].items())
        st.html(f'''<ul class="wg-evidence">
<li><code>sensor_1</code> fill value: <b>{ev_c["imputer_mean_sensor_1_original"]:.1f}</b>
(real mean {ev_c["true_mean_sensor_1_normal_wafers"]:.1f})</li>
<li><code>sensor_0</code> σ inflated to <b>{ev_c["sensor_0_scaler_std_original"]:.1f}</b> (real {ev_c["sensor_0_std_real_readings"]:.1f})</li>
<li>Real readings squashed to <b>{ev_c["sensor_1_real_spread_in_original_z"]:.3f}σ</b> spread (<code>sensor_1</code>)</li>
<li>Sentinel 0.0 on <code>sensor_0</code> → <b>{ev_c["z_of_sentinel_sensor_0_in_original"]:.2f}σ</b></li>
<li>Codes → {codes}</li></ul>''')

    left, right = st.columns([1, 1], gap="medium")
    with left, st.container(key="card_lossfig"):
        card_title("The broadcast loss rewards the wrong answer")
        figure("03_loss_bug", "The broadcast loss prefers a constant output over a flawless reconstruction.")
    with right, st.container(key="card_risks"):
        card_title("Residual risks")
        li, leg, d = R["legacy_in_test"], R["legacy"], R["data"]
        st.html(f'''<ol class="wg-evidence">
<li><b>Small numbers:</b> the test block holds {rep["n_defects"]} defects; F1 95% CI is
{boot["f1_ci95"][0]:.3f}–{boot["f1_ci95"][1]:.3f}.</li>
<li><b>PS-L shortcut:</b> {leg["fails_in_legacy"]} of {leg["n_legacy"]} PS-L wafers failed in this data; in test the model
flagged {li["legacy_flagged"]} of {li["legacy_wafers"]}. {"Catching a real PS-L excursion is untested." if leg["fails_in_legacy"] == 0
else "Catching a real PS-L excursion rests on very few examples."}</li>
<li><b>Drift:</b> monthly defect rate: {", ".join(f"{k}: {v:.1%}" for k, v in d["monthly_defect_rate"].items())}.</li>
<li><b>Model value:</b> logistic regression PR-AUC {R["baseline_logreg"]["ap"]:.3f} vs autoencoder {rep["ap"]:.3f}.</li>
<li><b>Attribution:</b> re-introducing only “{esc(top["config"])}” lifts F1 from {R["ablation"][0]["f1_mean"]:.3f}
to {top["f1_mean"]:.3f}.</li>
</ol>''')
    text = audit_text(mtime(AUDIT))
    with st.expander(f"Full AUDIT.md (generated {stamp(mtime(AUDIT))})", expanded=False):
        if text:
            st.markdown(text)
        else:
            st.write("AUDIT.md not found. Run the **Render AUDIT.md** step.")


def page_config() -> None:
    cfg = R["config"]
    page_header("Configuration", "Settings recorded by the notebook run that produced metrics.json (§0 Setup).")
    left, right = st.columns(2, gap="medium")
    with left, st.container(key="card_hp"):
        card_title("Hyper-parameters", "unchanged from the original notebook")
        st.html(table(pd.DataFrame({"parameter": list(cfg["HP"]), "value": [str(v) for v in cfg["HP"].values()]})))
    with right, st.container(key="card_settings"):
        card_title("Repair settings & environment")
        rows = [("seed", cfg["seed"]), ("evaluation seeds", ", ".join(map(str, cfg["eval_seeds"]))),
                ("split fractions", ", ".join(f"{k} {v:.0%}" for k, v in cfg["split_fractions"].items()) +
                 f", test {1 - sum(cfg['split_fractions'].values()):.0%}"),
                ("clip z", f"±{cfg['clip_z']:g}σ"), ("PS-L codes", ", ".join(f"{c:g}" for c in cfg["legacy_codes"])),
                ("release gate F1", f"{gate['gate_f1']:.2f}"), ("decision threshold", sci(rep["threshold"])),
                ("torch", cfg["torch"]), ("device", cfg["device"]),
                ("board Python", sys.version.split()[0])]
        st.html(table(pd.DataFrame(rows, columns=["setting", "value"]).astype(str)))
    with st.container(key="card_files"):
        card_title("Files the board reads")
        paths = [inference.DATA, inference.MODEL_PATH, inference.PREPROCESSOR_PATH, METRICS, ABLATION_RUNS,
                 ROOT / "results" / "walk_forward.csv", *sorted(FIGS.glob("*.png")), AUDIT, README, ORIGINAL_RUN,
                 ROOT / "wafer_guard_fixed.ipynb"]
        rows = [dict(file=str(p.relative_to(ROOT)).replace("\\", "/"), exists="yes" if p.exists() else "missing",
                     size=f"{p.stat().st_size / 1024:,.1f} KB" if p.exists() else "–", modified=stamp(mtime(p))) for p in paths]
        st.html(table(pd.DataFrame(rows)))


def page_reproduce() -> None:
    page_header("Reproduce", "Every number on this board can be regenerated from the raw CSV with the project's scripts.")
    steps = pipeline_steps()
    with st.container(key="card_steps"):
        card_title("Audit pipeline status", "checked against the files on disk")
        st.html(checklist_html(steps))
    left, right = st.columns([1, 1.3], gap="medium")
    with left, st.container(key="card_run"):
        card_title("Run the pipeline", "background process")
        running = active_run()
        keys = [k for k, (label, script) in run_pipeline.STEPS.items()
                if st.checkbox(f"**{label}** · `scripts/{script}`", value=True, key=f"rp_{k}", disabled=running)]
        if st.button("Running…" if running else "Start run", type="primary", icon=":material/play_arrow:",
                     disabled=running or not keys, key="rp_start"):
            start_run(keys)
            st.rerun()
        note("Outputs are overwritten. The board reloads itself when the run ends. "
             "Equivalent commands (README):")
        st.html(f'<pre class="wg-log" style="white-space:pre;overflow-x:auto">{esc(RD["commands"] or "see README.md")}</pre>')
    with right, st.container(key="card_log"):
        @st.fragment(run_every=2 if active_run() else None)
        def run_log():
            v = run_view()
            s = v["status"]
            card_title("Latest run", {"running": "running…", "done": "finished", "failed": "failed",
                                      "unknown": "status unknown (not started by this server)", None: "never run"}[v["state"]])
            if s:
                t = pd.DataFrame([dict(step=x["label"], script=x["script"], state=x["state"],
                                       started=(x["started"] or "–").replace("T", " "),
                                       seconds=x["seconds"] if x["seconds"] is not None else "–",
                                       exit=x["returncode"] if x["returncode"] is not None else "–") for x in s["steps"]])
                st.html(table(t))
            log = run_pipeline.LOG.read_text(encoding="utf-8", errors="replace") if run_pipeline.LOG.exists() else ""
            tail = "\n".join(log.splitlines()[-80:])
            st.html(f'<pre class="wg-log">{esc(tail) or "No log yet."}</pre>')
        run_log()
    if LIVE_OK:
        with st.container(key="card_verify"):
            v = LIVE["verify"]
            card_title("Live verification", "saved model re-scored now vs results/metrics.json")
            t = pd.DataFrame([dict(metric=k, live=a, saved=b, match=("✓" if ((a == b) if isinstance(a, int) else abs(a - b) < 1e-6) else "✕"))
                              for k, (a, b) in v["checks"].items()], dtype=object)  # object: keep counts as ints
            st.html(table(t, {"live": lambda x: f"{x:.6f}" if isinstance(x, float) else str(x),
                              "saved": lambda x: f"{x:.6f}" if isinstance(x, float) else str(x)}))
            if OLOG and OLOG["reported"]:
                note(f"Original notebook log: F1 {OLOG['reported']['f1']:.4f}, precision {OLOG['reported']['precision']:.4f}, "
                     f"recall {OLOG['reported']['recall']:.4f}; metrics.json §2 reproduction: F1 {orig['f1']:.4f}, "
                     f"precision {orig['precision']:.4f}, recall {orig['recall']:.4f}.")


def mermaid(code: str) -> None:
    """Draw a README Mermaid diagram (st.markdown shows them as plain code)."""
    st.iframe(f"""<pre class="mermaid" style="margin:0">{esc(code)}</pre>
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs";
mermaid.initialize({{startOnLoad: true, theme: "{'dark' if ss.theme == 'dark' else 'neutral'}",
                     themeVariables: {{fontFamily: "IBM Plex Sans, sans-serif"}}}});
</script>""")


def page_about() -> None:
    """README.md as GitHub shows it: centred <p> blocks hold the logo and figures, Mermaid blocks hold diagrams."""
    with st.container(key="card_about"):
        if not RD["text"]:
            st.markdown("README.md not found.")
            return
        for part in re.split(r'(<p align="center">.*?</p>|```mermaid\n.*?```)', RD["text"], flags=re.S):
            if part.startswith("```mermaid"):
                mermaid(part[len("```mermaid\n"):-3])
            elif part.startswith("<p"):
                img = re.search(r'<img src="([^"]+)" alt="([^"]*)"', part)
                if img and (ROOT / img.group(1)).exists():
                    st.image(str(ROOT / img.group(1)), width=110 if img.group(1).endswith(".svg") else "stretch")
                elif not img:
                    note(re.sub(r"<[^>]+>", "", part).strip())
            elif part.strip():
                st.markdown(part)


# ---------------------------------------------------------------- render
sidebar()
topbar()
{
    "Overview": page_overview, "Model Performance": page_performance, "Data Analysis": page_data,
    "Ablation Study": page_ablation, "Walk-Forward Eval": page_walk_forward, "Wafer Explorer": page_explorer,
    "Audit Findings": page_audit, "Configuration": page_config, "Reproduce": page_reproduce, "About": page_about,
}.get(ss.page, page_overview)()
