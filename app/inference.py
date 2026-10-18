"""
Load the repaired model saved by wafer_guard_fixed.ipynb (§9) and score wafers with it.

The classes below mirror the notebook's `WaferSAE` (§2) and `WaferPreprocessor.transform` (§4.2)
line for line; the fitted statistics and weights come from artifacts/, never from this file.
`verify()` re-scores the test block and compares the result with results/metrics.json, so any
drift between this module and the notebook shows up as a failed check on the Release Board.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "secom_telemetry_modified.csv"
MODEL_PATH = ROOT / "artifacts" / "wafer_sae_repaired.pt"
PREPROCESSOR_PATH = ROOT / "artifacts" / "preprocessor_repaired.joblib"


class WaferSAE(nn.Module):
    """Supervised autoencoder - identical to the notebook (§2)."""

    def __init__(self, d_in: int, d_hid: int, d_lat: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.LayerNorm(d_hid), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(d_hid, d_hid // 2), nn.LayerNorm(d_hid // 2), nn.GELU(),
            nn.Linear(d_hid // 2, d_lat),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_lat, d_hid // 2), nn.GELU(),
            nn.Linear(d_hid // 2, d_hid), nn.GELU(),
            nn.Linear(d_hid, d_in),
        )
        self.head = nn.Linear(d_lat, 1)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        z = self.encoder(x)
        return self.decoder(z), self.head(z).squeeze(-1), z


class _PreprocessorState:
    """Receives the attributes of the pickled notebook `WaferPreprocessor` (keep_, mean_, std_, ...)."""


def load_preprocessor():
    # The notebook pickled `__main__.WaferPreprocessor`; point that name at a plain attribute holder.
    main = sys.modules["__main__"]
    had = hasattr(main, "WaferPreprocessor")
    previous = getattr(main, "WaferPreprocessor", None)
    main.WaferPreprocessor = _PreprocessorState
    try:
        return joblib.load(PREPROCESSOR_PATH)
    finally:
        if had:
            main.WaferPreprocessor = previous
        else:
            del main.WaferPreprocessor


def load_data() -> pd.DataFrame:
    """The telemetry in time order, as the notebook sorts it (§1). `row` is the line's position in the CSV."""
    df = pd.read_csv(DATA)
    df.insert(0, "row", np.arange(len(df)))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("sensor_")]


def prepare(pre, frame: pd.DataFrame, legacy_codes) -> tuple[np.ndarray, np.ndarray]:
    """Notebook `WaferPreprocessor._prepare`."""
    X = frame[sensor_columns(frame)].to_numpy(dtype=np.float64).copy()
    if not pre.legacy_aware:
        return X, np.empty((len(frame), 0))
    is_legacy = frame["sensor_0"].eq(0.0).to_numpy()
    codes = frame["sensor_1"].to_numpy()
    one_hot = [(is_legacy & (codes == c)).astype(float) for c in legacy_codes]
    categorical = np.column_stack([is_legacy.astype(float), *one_hot])
    X[is_legacy, 0] = np.nan
    X[is_legacy, 1] = np.nan
    return X, categorical


def transform(pre, frame: pd.DataFrame, legacy_codes) -> np.ndarray:
    """Notebook `WaferPreprocessor.transform`: frozen training statistics, NaN -> 0, clip at ±clip_z."""
    X, categorical = prepare(pre, frame, legacy_codes)
    Z = (X[:, pre.keep_] - pre.mean_) / pre.std_
    Z = np.nan_to_num(Z, nan=0.0)
    Z = np.clip(Z, -pre.clip_z, pre.clip_z)
    return np.hstack([Z, categorical]).astype(np.float32)


def load_model(hp: dict, d_in: int) -> WaferSAE:
    model = WaferSAE(d_in, hp["hidden"], hp["latent"])
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


@torch.no_grad()
def score(model: WaferSAE, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Defect probability (notebook `predict_proba`) and per-wafer reconstruction MSE."""
    x = torch.from_numpy(X).float()
    recon, logits, _ = model(x)
    return torch.sigmoid(logits).numpy(), torch.mean((recon - x) ** 2, dim=1).numpy()


def causal_cut(timestamps: np.ndarray, fraction: float) -> int:
    """Notebook `causal_cut` (§4.1)."""
    i = int(round(fraction * len(timestamps)))
    while 0 < i < len(timestamps) and timestamps[i] == timestamps[i - 1]:
        i += 1
    return i


def split_labels(df: pd.DataFrame, fractions: dict) -> np.ndarray:
    """'train' / 'val' / 'test' for every wafer, with the notebook's chronological split."""
    ts = df["timestamp"].to_numpy()
    a = causal_cut(ts, fractions["train"])
    b = causal_cut(ts, fractions["train"] + fractions["val"])
    out = np.full(len(df), "test", dtype=object)
    out[:a], out[a:b] = "train", "val"
    return out


def build(R: dict) -> dict:
    """
    Score every wafer with the saved model.

    Returns the time-ordered raw telemetry (`raw`), the model inputs (`X`, `feature_names`), the fitted
    preprocessor (`pre`) and `scored`: one row per wafer with its split, live score and frozen-threshold decision.
    """
    cfg = R["config"]
    df = load_data()
    pre = load_preprocessor()
    X = transform(pre, df, cfg["legacy_codes"])
    model = load_model(cfg["HP"], X.shape[1])
    prob, recon = score(model, X)
    out = df[["row", "timestamp", "label"]].copy()
    out["defect"] = (df["label"].to_numpy() == 1).astype(int)
    out["split"] = split_labels(df, cfg["split_fractions"])
    out["legacy"] = df["sensor_0"].eq(0.0).to_numpy()
    out["score"] = prob
    out["recon_mse"] = recon
    out["flagged"] = prob >= R["repaired_test"]["threshold"]
    return dict(raw=df, X=X, feature_names=list(pre.feature_names_), pre=pre, scored=out)


def verify(scored: pd.DataFrame, R: dict) -> dict:
    """Re-compute the test-block metrics from live scores and compare them with metrics.json."""
    te = scored[scored["split"] == "test"]
    y, p = te["defect"].to_numpy(), te["score"].to_numpy()
    pred = te["flagged"].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    live = dict(n=int(len(y)), n_defects=int(y.sum()), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                f1=float(f1_score(y, pred, zero_division=0)),
                ap=float(average_precision_score(y, p)), roc_auc=float(roc_auc_score(y, p)))
    saved = R["repaired_test"]
    checks = {k: (live[k], saved[k]) for k in live}
    ok = all((a == b) if isinstance(a, int) else abs(a - b) < 1e-6 for a, b in checks.values())
    return dict(ok=ok, checks=checks)
