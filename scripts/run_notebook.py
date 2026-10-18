"""
Execute wafer_guard_fixed.ipynb top-to-bottom and save it WITH its outputs.

Running the notebook end-to-end is what produces every number in the project:
results/metrics.json, results/*.csv, results/figures/*.png and artifacts/*.

Usage (from the repo root):
    python scripts/run_notebook.py
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "wafer_guard_fixed.ipynb"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    start = time.time()
    NotebookClient(nb, timeout=3600, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, NOTEBOOK)
    print(f"Executed {NOTEBOOK.name} in {time.time() - start:.0f}s; outputs saved into the notebook.")


if __name__ == "__main__":
    main()
