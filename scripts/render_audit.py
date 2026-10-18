"""
Write AUDIT.md from results/metrics.json.

Every number in AUDIT.md is pulled from metrics.json (which the notebook writes),
so the audit can never quote a number the model did not produce.
Re-run after the notebook:   python scripts/render_audit.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))


def pct(x, d=1):
    return f"{100 * x:.{d}f} %"


def f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


data, leg, split = R["data"], R["legacy"], R["split"]
orig, rep, boot = R["original"], R["repaired_test"], R["repaired_test_bootstrap"]
seeds, wf, lr = R["repaired_test_seeds"], R["walk_forward"], R["baseline_logreg"]
ev_leak, ev_loss, ev_leg = R["evidence_leakage"], R["evidence_loss_bug"], R["evidence_legacy_blindness"]
prep, gate, train = R["preprocessing"], R["release_gate"], R["repaired_training"]
abl = {row["config"]: row for row in R["ablation"]}
fixed = abl["Repaired pipeline"]
verdict = "GO" if gate["passed"] else "NO-GO"

ablation_rows = "\n".join(
    f"| {name} | {f(r['f1_mean'])} ± {f(r['f1_std'])} | {f(r['ap_mean'])} | {pct(r['test_prevalence'])} "
    f"| {f(r['recon_spread'])} |"
    for name, r in abl.items()
)
fold_rows = "\n".join(
    f"| {r['fold']} | {r['test_start']} → {r['test_end']} | {r['n_defects']} / {r['n']} "
    f"| {f(r['precision'])} | {f(r['recall'])} | {f(r['f1'])} | {f(r['ap'])} |"
    for r in wf["folds"]
)
codes_z = ", ".join(f"{k} → {v:.2f}σ" for k, v in ev_leg["z_of_codes_in_original"].items())
n_seeds, n_folds = len(R["config"]["eval_seeds"]), len(wf["folds"])
blind = abl["+ PS-L legacy blindness"]
legacy_effect = ("within" if abs(blind["f1_mean"] - fixed["f1_mean"]) <= max(blind["f1_std"], fixed["f1_std"])
                 else "larger than")
if leg["fails_in_legacy"] == 0:
    legacy_risk = ("The data contains no failing PS-L wafer, so the model's ability to catch a real PS-L excursion "
                   "is untested.")
else:
    legacy_risk = (f"Only {leg['fails_in_legacy']} failing PS-L wafers exist, so the model's ability to catch a PS-L "
                   "excursion rests on very few examples.")

if gate["passed"]:
    go_no_go_text = (f"The repaired model meets the gate on an honest test (F1 {f(rep['f1'])} ≥ {gate['gate_f1']}). "
                     "Release with drift monitoring and the residual risks above tracked.")
else:
    go_no_go_text = (
        f"The original passed the {gate['gate_f1']} gate only on a leaky, balanced test (F1 {f(orig['f1'])}). "
        f"After repair the model ranks wafers {'better' if rep['ap'] > R['baseline_no_skill_ap'] else 'no better'} "
        f"than chance (PR-AUC {f(rep['ap'])} vs {f(R['baseline_no_skill_ap'])}) but its F1 of {f(rep['f1'])} is below "
        "the gate. Before any release: collect more labelled excursions (especially from PS-L), validate on a longer "
        "future window, and restate the gate in terms the line cares about (e.g. recall at a tolerable false-alarm rate).")
