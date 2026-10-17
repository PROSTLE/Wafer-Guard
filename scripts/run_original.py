"""
Run the ORIGINAL (buggy) notebook exactly as it was supplied, and save the
executed copy as evidence of what it reports.

Why this exists: the audit must start from evidence. Before claiming the 0.90
F1 is fake, we first reproduce it on our own machine.

The only change: the `!pip install ...` cell is skipped (dependencies are
installed from requirements.txt instead). No code line is edited.

Usage (from the repo root):
    python scripts/run_original.py
"""
import shutil
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "original" / "bnypi.ipynb"
RUN_DIR = ROOT / "original" / "_run"          # scratch folder the notebook runs in
OUT = ROOT / "original" / "bnypi_executed.ipynb"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")       # Windows consoles default to cp1252
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    # The original notebook reads the CSV from its working directory.
    shutil.copy(ROOT / "data" / "secom_telemetry_modified.csv", RUN_DIR)

    nb = nbformat.read(SRC, as_version=4)
    # Drop only the pip-install cell; everything else runs untouched.
    nb.cells = [c for c in nb.cells
                if not (c.cell_type == "code" and c.source.lstrip().startswith("!pip"))]

    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(RUN_DIR)}}).execute()
    nbformat.write(nb, OUT)

    # Print every text output so the reported numbers are visible in the console.
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                print(out["text"], end="")
    print(f"\nExecuted notebook saved to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
