"""
E1b (research completion, v1): tau=2 reference for E1. No sampling and no model: re-computes behavioural metrics
from the SAVED tau=2 Phase 1 / Phase 2 scanpaths (training/runs/external_gaze4asd_v2/phase1_vs_phase2/raw/
generated_scanpaths.json.gz) with E1's metric code (clipped coordinates), so the two temperatures are compared like for like.
The tau=2 run drew Phase 1 from an independent seed (g=2), so it has no common-random-number seed-null; only
the Phase 2 contrast and the match-to-real errors are comparable with E1.

  python scripts/research_completion/e1b_tau2_reference.py
"""
from __future__ import annotations
import gzip, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace("\\", "/")  # repository root
os.chdir(ROOT); sys.path[:0] = [ROOT, ROOT + "/tppgaze"]
import numpy as np
from training.external_gaze4asd_loader import StimulusBank, load_external_scanpaths
from training.eval_external_v2 import IMG_DIR, denorm, metric_row

OUT = "training/runs/research_completion/e1b_tau2_reference_v1"
SRC = "training/runs/external_gaze4asd_v2/phase1_vs_phase2/raw/generated_scanpaths.json.gz"
OLD = "training/runs/external_gaze4asd_v2/phase1_vs_phase2/raw/generated_and_real_summaries.npz"
G4A = "external_data/Gaze4ASD_processed_v2/fit_height_policy_a/gaze4asd_fit_height_policy_a.json"
METRICS = ["dur_mean", "n_fix", "fix_rate_per_s", "total_dur", "sacc_amp_norm_mean", "sacc_horizontal_frac", "dispersion", "revisit_rate", "sacc_dir_resultant"]
if os.path.exists(OUT + "/summaries_tau2_clipped.npz"):
    raise SystemExit("refusing to overwrite " + OUT)
os.makedirs(OUT, exist_ok=True)
bank = StimulusBank(IMG_DIR)
recs = load_external_scanpaths(G4A, bank)
imgs = sorted({r["image_id"] for r in recs}, key=lambda s: int(s.replace("top_image_", "").replace(".png", "")))
ix = {i: k for k, i in enumerate(imgs)}
arrs = {}
real = {g: {i: [] for i in imgs} for g in ("ASD", "TD")}
for r in recs:
    W, H = bank.dims(r["image_id"]); sp = np.asarray(r["scanpath"], float)
    real[r["group"]][r["image_id"]].append(metric_row("r", r["image_id"], "", denorm(sp, W, H), sp[:, 2], W, H))
for g in real:
    for m in METRICS:
        arrs[f"real_{g}_{m}"] = np.array([np.nanmean([x[m] for x in real[g][i]]) for i in imgs])
gen = json.load(gzip.open(SRC, "rt", encoding="utf-8"))
gen = gen["scanpaths"] if isinstance(gen, dict) else gen
cells = {}
for d in gen:
    sp = np.asarray(d["scanpath"], float)
    if not len(sp):
        continue
    W, H = bank.dims(d["image_id"])
    for clip in (True, False):
        xy = denorm(sp, W, H)
        if clip:
            xy = np.clip(xy, [0, 0], [W - 1, H - 1])
        cells.setdefault((clip, d["set"], int(d["rep"]), d["image_id"]), []).append(metric_row("g", d["image_id"], "", xy, sp[:, 2], W, H))
sets = sorted({k[1] for k in cells}); reps = sorted({k[2] for k in cells})
for clip in (True, False):
    for s in sets:
        for m in METRICS:
            a = np.full((len(reps), len(imgs)), np.nan)
            for (c, s_, rep, iid), rows in cells.items():
                if c == clip and s_ == s:
                    a[reps.index(rep), ix[iid]] = np.nanmean([r[m] for r in rows])
            arrs[f"{'clip' if clip else 'raw'}_{s}_{m}"] = a
# reproduction gate: unclipped recomputation must match the saved tau=2 summaries (saved scanpaths are rounded to 5 decimals)
Z = np.load(OLD)
gate = {f"{s}_{m}": float(np.nanmax(np.abs(arrs[f"raw_{s}_{m}"] - Z[f"{s}_{m}"]))) for s in sets for m in METRICS}
real_gate = {f"{g}_{m}": float(np.nanmax(np.abs(arrs[f"real_{g}_{m}"] - Z[f"real_{g}_{m}"]))) for g in real for m in METRICS}
np.savez_compressed(OUT + "/summaries_tau2_clipped.npz", **arrs)
json.dump(dict(source=SRC, sets=sets, replicates=len(reps), images=len(imgs), max_abs_diff_vs_saved_unclipped=gate, max_abs_diff_real_vs_saved=real_gate,
               off_image_fixation_fraction={s: None for s in sets}), open(OUT + "/checks.json", "w"), indent=1)
print("sets", sets, "reps", len(reps), "max gate", max(gate.values()), "max real gate", max(real_gate.values()))
print({k: round(v, 6) for k, v in gate.items() if v > 1e-4})
