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
