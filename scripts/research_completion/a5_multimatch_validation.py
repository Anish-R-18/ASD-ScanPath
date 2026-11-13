"""
A5 (research completion, v1): validation of our MultiMatch implementation (training/metrics.py, unchanged)
against the reference Python implementation multimatch-gaze 0.1.3 (MIT licence), which its authors test against
the original MATLAB toolbox. The reference source is imported read-only from the scratch download.

  python scripts/research_completion/a5_multimatch_validation.py <path-to-multimatch_gaze-src>
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

OUT = "training/runs/research_completion/a5_multimatch_validation_v1"
os.makedirs(OUT, exist_ok=False)
KEYS = ["mm_shape", "mm_direction", "mm_length", "mm_position", "mm_duration"]  # reference order
rng0 = np.random.default_rng(0)

def ours(xy1, d1, xy2, d2, diag, agg=np.mean):
    """Our implementation, optionally with a different aggregate (median) to isolate that difference."""
    if agg is np.mean:
        r = M.multimatch(xy1, d1, xy2, d2, diag)
        return None if r is None else [r[k] for k in KEYS]
    if len(xy1) < 2 or len(xy2) < 2:
        return None
    v1, v2 = np.diff(xy1, axis=0), np.diff(xy2, axis=0)
    n, m = len(v1), len(v2)
    C = np.linalg.norm(v1[:, None, :] - v2[None, :, :], axis=2)
    D = np.full((n, m), 1e18); D[0, 0] = C[0, 0]
    for i in range(n):
        for j in range(m):
            if i or j:
                D[i, j] = min(D[i - 1, j] if i else 1e18, D[i, j - 1] if j else 1e18, D[i - 1, j - 1] if i and j else 1e18) + C[i, j]
    i, j = n - 1, m - 1; a1, a2 = [], []
    while not (i == 0 and j == 0):
        a1.append(i); a2.append(j)
        opts = ([(D[i - 1, j], i - 1, j)] if i else []) + ([(D[i, j - 1], i, j - 1)] if j else []) + ([(D[i - 1, j - 1], i - 1, j - 1)] if i and j else [])
        _, i, j = min(opts)
    a1.append(0); a2.append(0); a1, a2 = a1[::-1], a2[::-1]
    V1, V2 = v1[a1], v2[a2]; l1, l2 = np.linalg.norm(V1, axis=1), np.linalg.norm(V2, axis=1)
    ang = np.arccos(np.clip(np.sum(V1 * V2, 1) / (l1 * l2 + 1e-9), -1, 1))
    return [1 - agg(np.linalg.norm(V1 - V2, axis=1)) / (2 * diag), 1 - agg(ang) / np.pi, 1 - agg(np.abs(l1 - l2)) / diag,
            1 - agg(np.linalg.norm(xy1[a1] - xy2[a2], axis=1)) / diag, 1 - agg(np.abs(d1[a1] - d2[a2]) / (np.maximum(d1[a1], d2[a2]) + 1e-9))]

def reference(xy1, d1, xy2, d2, W, H):
    rec = lambda xy, d: np.rec.fromarrays([xy[:, 0], xy[:, 1], d], names="start_x,start_y,duration")
    r = ref.docomparison(rec(xy1, d1), rec(xy2, d2), screensize=[W, H], grouping=False)
    return None if np.any(np.isnan(r)) else [float(x) for x in r]

out = {}
# 1. MATLAB test case shipped with the reference package
ld = lambda f: np.loadtxt(f"scripts/research_completion/external_reference/{f}", skiprows=1)
a, b = ld("segment_10_sub-19.tsv"), ld("segment_10_sub-01.tsv")
W, H = 720, 1280
matlab = [0.99082, 0.69902, 0.98927, 0.94767, 0.65563]
r_ref = reference(a[:, :2], a[:, 2], b[:, :2], b[:, 2], W, H)
r_med = ours(a[:, :2], a[:, 2], b[:, :2], b[:, 2], float(np.hypot(W, H)), np.median)
r_our = ours(a[:, :2], a[:, 2], b[:, :2], b[:, 2], float(np.hypot(W, H)))
out["matlab_test"] = dict(order=KEYS, matlab=matlab, reference=r_ref, ours_median=r_med, ours_mean=r_our, ref_max_abs_diff=float(np.max(np.abs(np.array(r_ref) - matlab))), ours_median_max_abs_diff=float(np.max(np.abs(np.array(r_med) - matlab))), ours_mean_max_abs_diff=float(np.max(np.abs(np.array(r_our) - matlab))))
print("MATLAB test:", json.dumps(out["matlab_test"]))
# 2. edge cases
ec = {}
p = np.array([[10., 10], [200, 50], [300, 300], [50, 400]]); d = np.array([.2, .3, .25, .4])
ec["identical"] = dict(reference=reference(p, d, p, d, 640, 480), ours=ours(p, d, p, d, 800.0))
p2 = np.array([[10., 10], [200, 50]]); d2 = np.array([.2, .3])
ec["two_fixations"] = dict(reference=reference(p2, d2, p, d, 640, 480), ours=ours(p2, d2, p, d, 800.0))
p3 = np.array([[10., 10], [10, 10], [300, 300]]); d3 = np.array([.2, .3, .25])
ec["zero_length_saccade"] = dict(reference=reference(p3, d3, p, d, 640, 480), ours=ours(p3, d3, p, d, 800.0))
ec["single_fixation"] = dict(reference=reference(p[:1], d[:1], p, d, 640, 480), ours=ours(p[:1], d[:1], p, d, 800.0))
out["edge_cases"] = ec
print("edge cases:", json.dumps(ec))
# 3. agreement on real and generated Gaze4ASD pairs (pixel coordinates, image as screen)
bank = StimulusBank(IMG_DIR)
recs = load_external_scanpaths("external_data/Gaze4ASD_processed_v2/fit_height_policy_a/gaze4asd_fit_height_policy_a.json", bank)
imgs = sorted({r["image_id"] for r in recs}, key=lambda s: int(s.replace("top_image_", "").replace(".png", "")))
real = {g: {i: [] for i in imgs} for g in ("ASD", "TD")}
for r in recs:
    Wd, Hd = bank.dims(r["image_id"]); s = np.asarray(r["scanpath"], float)
    real[r["group"]][r["image_id"]].append((denorm(s, Wd, Hd), s[:, 2]))
gen = {1.0: {}, 2.0: {}}
for rep in range(10):
    for s in json.load(gzip.open(f"training/runs/external_gaze4asd_v2/temperature_tau1/generated_scanpaths/seed_{rep + 1:02d}.json.gz", "rt"))["scanpaths"]:
        gen[1.0].setdefault((s["group"], rep, s["image_id"]), {})[s["sim"]] = np.asarray(s["scanpath"], float)
for s in json.load(gzip.open("training/runs/external_gaze4asd_v2/phase1_vs_phase2/raw/generated_scanpaths.json.gz", "rt")):
    if s["set"] in ("P2_ASD", "P2_TD"):
        gen[2.0].setdefault((s["set"][3:], s["rep"], s["image_id"]), {})[s["sim"]] = np.asarray(s["scanpath"], float)
def glist(t, g, rep, iid):
    Wd, Hd = bank.dims(iid)
    return [(np.clip(denorm(sp, Wd, Hd), [0, 0], [Wd - 1, Hd - 1]), sp[:, 2]) for sp in gen[t][(g, rep, iid)].values()]
agree = {k: [] for k in ("ours_mean", "ours_median", "reference")}
short = 0; tot = 0
for _ in range(3000):
    iid = imgs[rng0.integers(30)]; Wd, Hd = bank.dims(iid); diag = float(np.hypot(Wd, Hd)); g = ("ASD", "TD")[rng0.integers(2)]
    A = real[g][iid][rng0.integers(len(real[g][iid]))]
    if rng0.random() < .5:
        L = real[("ASD", "TD")[rng0.integers(2)]][iid]
    else:
        L = glist((1.0, 2.0)[rng0.integers(2)], g, int(rng0.integers(10)), iid)
    Bp = L[rng0.integers(len(L))]
    tot += 1
    rr = reference(A[0], A[1], Bp[0], Bp[1], Wd, Hd)
    if rr is None:
        short += 1; continue
    agree["reference"].append(rr); agree["ours_mean"].append(ours(A[0], A[1], Bp[0], Bp[1], diag)); agree["ours_median"].append(ours(A[0], A[1], Bp[0], Bp[1], diag, np.median))
R_, O_, Md = (np.array(agree[k]) for k in ("reference", "ours_mean", "ours_median"))
out["pair_agreement"] = dict(pairs=int(len(R_)), excluded_by_reference_min_length=short, of=tot, by_dimension={k: dict(ours_mean_minus_ref=float((O_[:, q] - R_[:, q]).mean()), ours_mean_r=float(np.corrcoef(O_[:, q], R_[:, q])[0, 1]), ours_mean_max_abs=float(np.abs(O_[:, q] - R_[:, q]).max()), ours_median_max_abs=float(np.abs(Md[:, q] - R_[:, q]).max()), ours_median_frac_within_1e6=float((np.abs(Md[:, q] - R_[:, q]) < 1e-6).mean())) for q, k in enumerate(KEYS)})
print("pair agreement:", json.dumps(out["pair_agreement"]))
n2 = sum(1 for t in gen for v in gen[t].values() for sp in v.values() if len(sp) == 2)
nall = sum(len(v) for t in gen for v in gen[t].values())
out["generated_two_fixation_scanpaths"] = dict(count=n2, of=nall)
# 4. paper conclusions re-run with the reference: tau=1 vs tau=2 similarity to own group, and own minus other
MAXP = 30
def mm_mean(tag, la, lb, Wd, Hd, same=False):
    pairs = [(i, j) for i in range(len(la)) for j in range(len(lb)) if not (same and j <= i)]
    rng = np.random.default_rng([0, zlib.crc32(tag.encode())])
    vr, vo = [], []
    for q in rng.permutation(len(pairs))[:MAXP]:
        A, Bq = la[pairs[q][0]], lb[pairs[q][1]]
        o = ours(A[0], A[1], Bq[0], Bq[1], float(np.hypot(Wd, Hd)))
        if o is not None:
            vo.append(o)
        rr = reference(A[0], A[1], Bq[0], Bq[1], Wd, Hd)
        if rr is not None:
            vr.append(rr)
    return np.nanmean(vr, 0) if vr else np.full(5, np.nan), np.nanmean(vo, 0) if vo else np.full(5, np.nan)
cube = {}
for t in (1.0, 2.0):
    for rep in range(10):
        for iid in imgs:
            Wd, Hd = bank.dims(iid)
            for g, oth in (("ASD", "TD"), ("TD", "ASD")):
                cube[("own", t, g, rep, iid)] = mm_mean(f"RvG|{g}|{rep}|{iid}", real[g][iid], glist(t, g, rep, iid), Wd, Hd)
                cube[("other", t, g, rep, iid)] = mm_mean(f"RvG_wrong|{g}|{rep}|{iid}", real[g][iid], glist(t, oth, rep, iid), Wd, Hd)
    print("tau", t, "done", flush=True)
rvr = {(g, iid): mm_mean(f"RvR|{g}|{iid}", real[g][iid], real[g][iid], *bank.dims(iid), same=True) for g in ("ASD", "TD") for iid in imgs}
stored = {(r["group"], r["metric"]): r for r in json.load(open("training/runs/external_gaze4asd_v2/temperature_tau1/statistical_results.json"))["multimatch"]}
def boot(d):
    b = d[rng0.integers(0, len(d), (10000, len(d)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
concl = []
for g in ("ASD", "TD"):
    for q, k in enumerate(KEYS):
        for impl, ix in (("reference", 0), ("ours", 1)):
            per = lambda kind, t: np.array([np.nanmean([cube[(kind, t, g, rep, iid)][ix][q] for rep in range(10)]) for iid in imgs])
            o1, o2, w1, w2 = per("own", 1.0), per("own", 2.0), per("other", 1.0), per("other", 2.0)
            rv = np.array([rvr[(g, iid)][ix][q] for iid in imgs])
            concl.append(dict(group=g, metric=k, impl=impl, real_vs_real=float(np.nanmean(rv)), tau1=float(np.nanmean(o1)), tau2=float(np.nanmean(o2)), tau1_minus_tau2=float(np.nanmean(o1 - o2)), ci=boot(o1 - o2),
                              dist_rvr_tau1=float(abs(np.nanmean(o1) - np.nanmean(rv))), dist_rvr_tau2=float(abs(np.nanmean(o2) - np.nanmean(rv))), own_minus_other_tau1=float(np.nanmean(o1 - w1)), own_minus_other_tau1_ci=boot(o1 - w1)))
        s = stored[(g, k)]
        concl[-1]["stored_ours_tau1_minus_tau2"] = s["mean"]
out["conclusions"] = concl
for c in concl:
    print(f"{c['group']:3s} {c['metric'][3:]:9s} {c['impl']:9s} RvR={c['real_vs_real']:.4f} t1={c['tau1']:.4f} t2={c['tau2']:.4f} t1-t2={c['tau1_minus_tau2']:+.4f}{[round(x, 4) for x in c['ci']]} |t1-RvR|={c['dist_rvr_tau1']:.4f} |t2-RvR|={c['dist_rvr_tau2']:.4f} own-other={c['own_minus_other_tau1']:+.4f}{[round(x, 4) for x in c['own_minus_other_tau1_ci']]}" + (f" stored={c['stored_ours_tau1_minus_tau2']:+.4f}" if "stored_ours_tau1_minus_tau2" in c else ""))
json.dump(out, open(OUT + "/a5_results.json", "w"), indent=1)
print("A5_DONE")
