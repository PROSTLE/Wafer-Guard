"""
Run the reproduction steps one after another and record progress for the Release Board.

The Release Board's "Run Pipeline" button starts this script in the background; it can also be run by hand.
Each step is one of the existing scripts, executed unchanged:
    original  -> scripts/run_original.py   (reproduce the original claim)
    repaired  -> scripts/run_notebook.py   (repaired pipeline: results/, artifacts/)
    audit     -> scripts/render_audit.py   (AUDIT.md from metrics.json)

Progress goes to .run/pipeline_status.json and the combined console output to .run/pipeline.log.

Usage (from the repo root):
    python scripts/run_pipeline.py                 # all steps
    python scripts/run_pipeline.py repaired audit  # a subset, in pipeline order
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / ".run"
STATUS = RUN_DIR / "pipeline_status.json"
LOG = RUN_DIR / "pipeline.log"

STEPS = {
    "original": ("Reproduce original notebook", "run_original.py"),
    "repaired": ("Run repaired notebook", "run_notebook.py"),
    "audit": ("Render AUDIT.md", "render_audit.py"),
}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_status(status: dict) -> None:
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    tmp.replace(STATUS)


def main() -> int:
    keys = [k for k in STEPS if k in sys.argv[1:]] or list(STEPS)
    unknown = set(sys.argv[1:]) - set(STEPS)
    if unknown:
        print(f"unknown step(s): {', '.join(sorted(unknown))}; choose from {', '.join(STEPS)}")
        return 2

    RUN_DIR.mkdir(exist_ok=True)
    status = dict(state="running", started=now(), finished=None,
                  steps=[dict(key=k, label=STEPS[k][0], script=STEPS[k][1], state="pending",
                              started=None, finished=None, seconds=None, returncode=None) for k in keys])
    write_status(status)

    with open(LOG, "w", encoding="utf-8") as log:
        for step in status["steps"]:
            step["state"], step["started"] = "running", now()
            write_status(status)
            log.write(f"\n===== {step['label']} ({step['script']}) · {step['started']}\n")
            log.flush()
            t0 = time.time()
            proc = subprocess.run([sys.executable, str(ROOT / "scripts" / step["script"])], cwd=ROOT,
                                  stdout=log, stderr=subprocess.STDOUT,
                                  env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            step.update(finished=now(), seconds=round(time.time() - t0, 1), returncode=proc.returncode,
                        state="done" if proc.returncode == 0 else "failed")
            write_status(status)
            if proc.returncode != 0:
                log.write(f"\n===== {step['script']} exited with code {proc.returncode}; later steps skipped\n")
                for later in status["steps"]:
                    if later["state"] == "pending":
                        later["state"] = "skipped"
                break

    status["state"] = "failed" if any(s["state"] == "failed" for s in status["steps"]) else "done"
    status["finished"] = now()
    write_status(status)
    return 0 if status["state"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
