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
