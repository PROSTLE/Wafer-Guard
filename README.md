# Wafer Guard: Audit & Repair of the Fab-7 Wafer Anomaly Detector

GDG on Campus VIT Vellore · Tech Wildcard Entry · **AI / Machine Learning** task.

A PyTorch notebook claimed **F1 ≥ 0.90** for detecting defective semiconductor wafers and was cleared for release.
This repository reproduces that claim, proves where it comes from, repairs the pipeline so that it respects time
and tensor shapes, and reports what the model really achieves at the natural defect rate.

| Deliverable | File |
|---|---|
| Repaired, executed notebook | [`wafer_guard_fixed.ipynb`](wafer_guard_fixed.ipynb) |
| Audit report (findings, metrics, go / no-go) | [`AUDIT.md`](AUDIT.md) |
| Release Board (interactive summary) | [`app/release_board.py`](app/release_board.py) |
| Every measured number | [`results/metrics.json`](results/metrics.json) |

## The three defects in one sentence each
1. **Data leakage:** SMOTE ran before the split (the test set became 50 % defects, most of them synthetic copies),
   the split was random on a time series, and the imputer and scaler saw the test rows.
2. **Corrupted loss gradients:** `[B, F] − [B, 1, F]` silently broadcast to `[B, B, F]`, so the autoencoder learned to
   output one "average wafer" for everyone.
3. **Invariant blindness:** legacy Probe Station PS-L wafers (`sensor_0 == 0.0`, `sensor_1 ∈ {100, 200, 300}`) were
   treated as real measurements, distorting imputation and scaling.

Numbers, evidence and fixes: see [`AUDIT.md`](AUDIT.md).

## No hand-typed results
`wafer_guard_fixed.ipynb` writes `results/metrics.json`. `AUDIT.md` is generated from that file by
`scripts/render_audit.py`, and the Release Board only reads it; it also re-scores every wafer live with the saved
model in `artifacts/` and checks the result against `metrics.json`. Nothing is hard-coded anywhere.
The evaluator-only file in the task's Drive folder (`secom_legacy_mask_EVALUATOR_ONLY.csv`) was **not** downloaded or used.

## Project layout
```
WaferGuard/
├── wafer_guard_fixed.ipynb      # the repaired pipeline (read this first)
├── AUDIT.md                     # generated audit report
├── data/secom_telemetry_modified.csv
├── original/
│   ├── bnypi.ipynb              # notebook as supplied
│   └── bnypi_executed.ipynb     # the same notebook, executed by us (evidence of the 0.99 claim)
├── results/
│   ├── metrics.json             # every number, produced by the notebook
│   ├── ablation_runs.csv        # bug-attribution runs (8 configurations × 5 seeds)
│   ├── walk_forward.csv
│   └── figures/*.png
├── artifacts/                   # repaired model weights + fitted preprocessor
├── app/
│   ├── release_board.py         # Streamlit dashboard
│   └── inference.py             # loads artifacts/ and re-scores wafers for the dashboard
└── scripts/
    ├── run_original.py          # executes the original notebook unchanged
    ├── run_notebook.py          # executes wafer_guard_fixed.ipynb top-to-bottom
    ├── render_audit.py          # metrics.json -> AUDIT.md
    └── run_pipeline.py          # runs the three steps above in order (the board's "Run Pipeline" button)
```

## Prerequisites
- Python 3.11+ (results were produced with Python 3.13 on CPU; no GPU needed)
- ~1 GB disk for PyTorch
