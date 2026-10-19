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
