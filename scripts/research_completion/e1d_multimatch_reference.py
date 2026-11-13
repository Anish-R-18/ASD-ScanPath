"""
E1d (research completion, v1): E1's MultiMatch conditioning analysis re-scored with the reference implementation
(multimatch-gaze 0.1.3, imported read-only from the scratch download), on exactly E1's scanpath pairs.
Gate: our implementation on the reconstructed pairs must reproduce E1's stored per-image values.

  python scripts/research_completion/e1d_multimatch_reference.py <path-to-multimatch_gaze-src>
"""
from __future__ import annotations
import csv, gzip, json, os, sys, zlib
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace("\\", "/")  # repository root
os.chdir(ROOT); sys.path[:0] = [ROOT, ROOT + "/tppgaze", sys.argv[1]]
import numpy as np
from multimatch_gaze import multimatch_gaze as ref
from training import metrics as M
from training.external_gaze4asd_loader import StimulusBank, load_external_scanpaths
from training.eval_external_v2 import IMG_DIR, denorm

E1 = "training/runs/research_completion/e1_phase_tau1_v2"
OUT = "training/runs/research_completion/e1d_multimatch_reference_v1"
G4A = "external_data/Gaze4ASD_processed_v2/fit_height_policy_a/gaze4asd_fit_height_policy_a.json"
SETS = {"P2_ASD": 0, "P2_TD": 1, "P1_s0": 0, "P1_s1": 1}  # seed g
MM = ["mm_shape", "mm_length", "mm_direction", "mm_position", "mm_duration"]
REF_ORDER = ["mm_shape", "mm_direction", "mm_length", "mm_position", "mm_duration"]
N_REP, MAX_PAIRS, N_BOOT = 10, 30, 10000
if os.path.exists(OUT + "/e1d_results.json"):
    raise SystemExit("refusing to overwrite " + OUT)
os.makedirs(OUT, exist_ok=True)
bank = StimulusBank(IMG_DIR)
recs = load_external_scanpaths(G4A, bank)
imgs = sorted({r["image_id"] for r in recs}, key=lambda s: int(s.replace("top_image_", "").replace(".png", "")))
real = {g: {i: [] for i in imgs} for g in ("ASD", "TD")}
for r in recs:
    W, H = bank.dims(r["image_id"]); sp = np.asarray(r["scanpath"], float)
    real[r["group"]][r["image_id"]].append(dict(xy=denorm(sp, W, H), dur=sp[:, 2]))
gen = {}
for s in SETS:
    for rep in range(N_REP):
        d = json.load(gzip.open(f"{E1}/generated/gaze4asd_{s}_seed{rep + 1:02d}.json.gz", "rt", encoding="utf-8"))
        for x in d["scanpaths"]:
            # float32, as E1 computed it in-run (the sampler returns float32; a float64 reload differed by up to 3.7e-7 in a first attempt)
            W, H = bank.dims(x["image_id"]); sp = np.asarray(x["scanpath"], np.float32)
            gen.setdefault((s, rep, x["image_id"]), []).append(dict(xy=np.clip(denorm(sp, W, H), [0, 0], [W - 1, H - 1]), dur=sp[:, 2]))

def reference(a, b, W, H):
    rec = lambda q: np.rec.fromarrays([q["xy"][:, 0], q["xy"][:, 1], q["dur"]], names="start_x,start_y,duration")
    r = ref.docomparison(rec(a), rec(b), screensize=[W, H], grouping=False)
    return None if np.any(np.isnan(r)) else dict(zip(REF_ORDER, map(float, r)))

stored = {(r["target"], r["set"], int(r["rep"]), r["image_id"]): r for r in csv.DictReader(open(f"{E1}/multimatch_per_image.csv", encoding="utf-8"))}
rows, gate = [], 0.0
for rep in range(N_REP):
    for iid in imgs:
        W, H = bank.dims(iid); diag = float(np.hypot(W, H))
        for tgt in ("ASD", "TD"):
            for s, sg in SETS.items():
                la, lb = real[tgt][iid], gen[(s, rep, iid)]
                pairs = [(i, j) for i in range(len(la)) for j in range(len(lb))]
                rng = np.random.default_rng([0, zlib.crc32(f"{tgt}|{rep}|{iid}|seed{sg}".encode())])
                ours, refv = {k: [] for k in MM}, {k: [] for k in MM}
                for q in rng.permutation(len(pairs))[:MAX_PAIRS]:
                    a, b = la[pairs[q][0]], lb[pairs[q][1]]
                    o = M.multimatch(a["xy"], a["dur"], b["xy"], b["dur"], diag)
                    if o:
                        for k in MM:
                            ours[k].append(o[k])
                    rr = reference(a, b, W, H)
                    if rr:
                        for k in MM:
                            refv[k].append(rr[k])
                om = {k: float(np.mean(v)) for k, v in ours.items()}
                st = stored[(tgt, s, rep, iid)]
                gate = max(gate, max(abs(om[k] - float(st[k])) for k in MM))
                rows.append(dict(target=tgt, set=s, rep=rep, image_id=iid, n_ref=len(refv[MM[0]]), **{k: float(np.mean(v)) if v else float("nan") for k, v in refv.items()}))
    print(f"replicate {rep + 1}/{N_REP}; gate so far {gate:.2e}", flush=True)
if gate > 1e-9:
    raise SystemExit(f"GATE FAILED: reconstructed pairs do not reproduce E1 ({gate})")
with open(OUT + "/multimatch_reference_per_image.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

rng = np.random.default_rng(0)
ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
def paired(d):
    d = np.asarray(d, float); n = len(d)
    b = d[rng.integers(0, n, (N_BOOT, n))].mean(1); f = (rng.choice([-1.0, 1.0], (N_BOOT, n)) * d).mean(1)
    return dict(mean=float(d.mean()), ci=ci(b), p=float((np.sum(np.abs(f) >= abs(d.mean()) - 1e-15) + 1) / (N_BOOT + 1)))
def cube(t, s):
    c = np.full((N_REP, len(imgs), len(MM)), np.nan)
    for r in rows:
        if r["target"] == t and r["set"] == s:
            c[r["rep"], imgs.index(r["image_id"])] = [r[k] for k in MM]
    return np.nanmean(c, 0)
res = []
for tgt, own2, oth2, own1, oth1 in (("ASD", "P2_ASD", "P2_TD", "P1_s0", "P1_s1"), ("TD", "P2_TD", "P2_ASD", "P1_s1", "P1_s0")):
    d2, d1 = cube(tgt, own2) - cube(tgt, oth2), cube(tgt, own1) - cube(tgt, oth1)
    for q, k in enumerate(MM):
        e = paired(d2[:, q] - d1[:, q]); res.append(dict(target=tgt, metric=k, phase2_own_minus_other=float(d2[:, q].mean()), phase1_seed_null=float(d1[:, q].mean()), conditioning_effect=e, p=e["p"]))
o = sorted(range(len(res)), key=lambda i: res[i]["p"]); run_ = 0.0
for r_, i in enumerate(o):
    run_ = max(run_, min(1.0, (len(res) - r_) * res[i]["p"])); res[i]["p_holm"] = run_
e1 = {(r["target"], r["metric"]): r for r in json.load(open(f"{E1}/e1_results.json"))["multimatch"]}
for r in res:
    x = e1[(r["target"], r["metric"])]["conditioning_effect"]
    r["ours"] = dict(mean=x["mean"], ci=x["ci"], p_holm=e1[(r["target"], r["metric"])]["p_holm"])
    r["same_sign_as_ours"] = bool(np.sign(r["conditioning_effect"]["mean"]) == np.sign(x["mean"]))
    print(f"{r['target']:3s} {r['metric'][3:]:9s} reference {r['conditioning_effect']['mean']:+.4f}{[round(v, 4) for v in r['conditioning_effect']['ci']]} holm={r['p_holm']:.4f} | ours {x['mean']:+.4f} holm={r['ours']['p_holm']:.4f} | same sign {r['same_sign_as_ours']}")
json.dump(dict(gate_max_abs_diff_ours_vs_stored=gate, pairs_excluded_by_reference=int(sum(MAX_PAIRS - r["n_ref"] for r in rows)), results=res), open(OUT + "/e1d_results.json", "w"), indent=1)
print("E1D_DONE")
