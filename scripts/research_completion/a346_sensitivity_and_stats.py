"""
A3 + A4 + A6 (research completion, v1). No model is run; read-only analyses of saved data and results.
  A4  Policy B accounting by group.
  A3  Real ASD-TD effects and tau=1 vs tau=2 realism under each corrected mapping/policy.
  A6  Image-centred robustness check of the real participant-level effects.

  python scripts/research_completion/a346_sensitivity_and_stats.py
"""
from __future__ import annotations
import csv, json, os, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace("\\", "/")  # repository root
os.chdir(ROOT); sys.path[:0] = [ROOT]
import numpy as np

OUT = "training/runs/research_completion/a346_sensitivity_stats_v1"
os.makedirs(OUT, exist_ok=False)
V2 = "training/runs/external_gaze4asd_v2"
D = {"fit_height_policy_a": "external_data/Gaze4ASD_processed_v2/fit_height_policy_a/gaze4asd_fit_height_policy_a.json",
     "stretch_policy_a": "external_data/Gaze4ASD_processed_v2/stretch_policy_a/gaze4asd_stretch_policy_a.json",
     "fit_height_policy_b": "external_data/Gaze4ASD_processed_v2/fit_height_policy_b/gaze4asd_fit_height_policy_b.json"}
MS = ["dur_mean", "dur_median", "n_fix", "total_dur", "fix_rate_per_s", "sacc_amp_norm_mean", "dispersion", "revisit_rate", "sacc_dir_resultant", "sacc_horizontal_frac"]
rng = np.random.default_rng(0)
N = 10000
ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
out = {}

# ---------------- A4: Policy B accounting
data = {v: json.load(open(p)) for v, p in D.items()}
acc = {}
for v, recs in data.items():
    acc[v] = {g: dict(participants=len({r["participant"] for r in recs if r["group"] == g}), trials=sum(r["group"] == g for r in recs), fixations=sum(len(r["scanpath"]) for r in recs if r["group"] == g)) for g in ("ASD", "TD")}
A, B = acc["fit_height_policy_a"], acc["fit_height_policy_b"]
pa = {(r["participant"], r["image_id"]) for r in data["fit_height_policy_a"]}; pb = {(r["participant"], r["image_id"]) for r in data["fit_height_policy_b"]}
grp = {r["participant"]: r["group"] for r in data["fit_height_policy_a"]}
lost = collections.Counter(grp[p] for p, _ in pa - pb); gained = collections.Counter(grp[p] for p, _ in pb - pa)
per_part_loss = {g: [sum(1 for (p, i) in pa - pb if p == q) for q in {r["participant"] for r in data["fit_height_policy_a"] if r["group"] == g}] for g in ("ASD", "TD")}
edge = {}
for g in ("ASD", "TD"):
    xy = np.concatenate([np.array(r["scanpath"])[:, :2] for r in data["fit_height_policy_a"] if r["group"] == g])
    edge[g] = float((np.abs(xy) >= 1 - 1e-9).any(1).mean() * 100)
out["A4_policy_accounting"] = dict(counts=acc, trials_only_in_A_by_group=dict(lost), trials_only_in_B_by_group=dict(gained),
    pct_trials_removed={g: 100 * (A[g]["trials"] - B[g]["trials"]) / A[g]["trials"] for g in A}, pct_fixations_removed={g: 100 * (A[g]["fixations"] - B[g]["fixations"]) / A[g]["fixations"] for g in A},
    pct_fixations_on_image_edge_policy_a=edge, participants_losing_any_trial={g: int(sum(x > 0 for x in per_part_loss[g])) for g in per_part_loss}, max_trials_lost_by_one_participant={g: int(max(per_part_loss[g])) for g in per_part_loss})
print("A4:", json.dumps(out["A4_policy_accounting"]))
from scipy.stats import fisher_exact
fx = np.array([[A["ASD"]["fixations"] - B["ASD"]["fixations"], B["ASD"]["fixations"]], [A["TD"]["fixations"] - B["TD"]["fixations"], B["TD"]["fixations"]]])
out["A4_policy_accounting"]["note"] = "Fixation counts are not independent (nested in participants); the participant-level comparison below is the appropriate test."
off = {g: [] for g in ("ASD", "TD")}
byp = collections.defaultdict(lambda: [0, 0])
for r in data["fit_height_policy_a"]:
    xy = np.array(r["scanpath"])[:, :2]; byp[r["participant"]][0] += int((np.abs(xy) >= 1 - 1e-9).any(1).sum()); byp[r["participant"]][1] += len(xy)
for p, (a_, n_) in byp.items():
    off[grp[p]].append(a_ / n_)
oa, ot = np.array(off["ASD"]), np.array(off["TD"])
obs = oa.mean() - ot.mean(); pool = np.r_[oa, ot]
perm = rng.permuted(np.tile(pool, (N, 1)), axis=1)
p = (np.sum(np.abs(perm[:, :len(oa)].mean(1) - perm[:, len(oa):].mean(1)) >= abs(obs) - 1e-12) + 1) / (N + 1)
out["A4_policy_accounting"]["participant_edge_fraction"] = dict(ASD_mean=float(oa.mean()), TD_mean=float(ot.mean()), diff=float(obs), diff_ci=ci(oa[rng.integers(0, len(oa), (N, len(oa)))].mean(1) - ot[rng.integers(0, len(ot), (N, len(ot)))].mean(1)), permutation_p=float(p))
print("A4 participant edge fraction:", json.dumps(out["A4_policy_accounting"]["participant_edge_fraction"]))

# ---------------- A3: real effects across versions (saved statistics)
real = {}
for v in D:
    s = json.load(open(f"{V2}/statistics/{v}_statistics.json"))["contrasts"]["real_ASD_vs_TD_participant"]
    real[v] = {m: dict(g=s[m]["effect"], ci=s[m]["effect_ci"], p_holm=s[m]["p_holm"]) for m in MS}
out["A3_real_effects_by_version"] = real
for m in MS:
    print(f"A3 real {m:22s} " + "  ".join(f"{v[:14]} g={real[v][m]['g']:+.2f} holm={real[v][m]['p_holm']:.4f}" for v in D))
# tau=1 vs tau=2 realism, real values per version, generated values unchanged (mapping-independent)
Z = np.load(f"{V2}/temperature_tau1/raw/summaries.npz")
imgs = list(Z["images"])
def real_img(v, g, m):
    acc_ = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{V2}/{v}/per_scanpath_metrics.csv", encoding="utf-8")):
        if r["population"] == f"real_{g}" and r[m] not in ("", "nan"):
            acc_[r["image_id"]].append(float(r[m]))
    return np.array([np.mean(acc_[i]) for i in imgs])
temp = {}
for v in D:
    temp[v] = {}
    for g in ("ASD", "TD"):
        for m in ("sacc_amp_norm_mean", "revisit_rate", "dispersion"):
            R = real_img(v, g, m); a, b = Z[f"clip_tau1_{g}_{m}"].mean(0), Z[f"clip_tau2_{g}_{m}"].mean(0)
            d = np.abs(b - R) - np.abs(a - R)
            temp[v][f"{g}_{m}"] = dict(real=float(R.mean()), mae_tau1=float(np.abs(a - R).mean()), mae_tau2=float(np.abs(b - R).mean()), reduction_pct=float(100 * d.mean() / np.abs(b - R).mean()), ci=ci(d[rng.integers(0, len(d), (N, len(d)))].mean(1)), images_favour_tau1=int((d > 0).sum()))
    for m in ("sacc_amp_norm_mean", "revisit_rate", "dispersion"):
        Dr = real_img(v, "ASD", m) - real_img(v, "TD", m)
        temp[v][f"contrast_{m}"] = dict(real=float(Dr.mean()), share_tau1=float((Z[f"clip_tau1_ASD_{m}"] - Z[f"clip_tau1_TD_{m}"]).mean() / Dr.mean()), share_tau2=float((Z[f"clip_tau2_ASD_{m}"] - Z[f"clip_tau2_TD_{m}"]).mean() / Dr.mean()))
out["A3_temperature_realism_by_version"] = temp
for v in D:
    print("A3 temp", v, " | ".join(f"{k}: {x['reduction_pct']:+.1f}% {[round(c, 4) for c in x['ci']]}" for k, x in temp[v].items() if "contrast" not in k))
    print("   shares", {k: (round(x["share_tau1"], 2), round(x["share_tau2"], 2)) for k, x in temp[v].items() if "contrast" in k})

# ---------------- A6: image-centred robustness of real participant effects (primary version)
rows = [r for r in csv.DictReader(open(f"{V2}/fit_height_policy_a/per_scanpath_metrics.csv", encoding="utf-8")) if r["population"] in ("real_ASD", "real_TD")]
npart = collections.Counter(r["participant"] for r in rows)
gp = {r["participant"]: r["population"][5:] for r in rows}
out["A6_design"] = dict(trials_per_participant={g: dict(min=int(min(n for p, n in npart.items() if gp[p] == g)), median=float(np.median([n for p, n in npart.items() if gp[p] == g])), max=int(max(n for p, n in npart.items() if gp[p] == g))) for g in ("ASD", "TD")},
                        participants_per_image={g: dict(min=int(min(collections.Counter(r["image_id"] for r in rows if r["population"] == f"real_{g}").values())), max=int(max(collections.Counter(r["image_id"] for r in rows if r["population"] == f"real_{g}").values()))) for g in ("ASD", "TD")})
print("A6 design:", json.dumps(out["A6_design"]))
def hedges(a, b):
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / s * (1 - 3 / (4 * (len(a) + len(b)) - 9))
res6 = []
for m in MS:
    vals = [(r["participant"], r["image_id"], float(r[m])) for r in rows if r[m] not in ("", "nan")]
    im = collections.defaultdict(list)
    for p_, i, x in vals:
        im[i].append(x)
    imean = {i: np.mean(x) for i, x in im.items()}
    for mode in ("raw", "image_centred"):
        pm = collections.defaultdict(list)
        for p_, i, x in vals:
            pm[p_].append(x - (imean[i] if mode == "image_centred" else 0))
        a = np.array([np.mean(x) for p_, x in pm.items() if gp[p_] == "ASD"]); b = np.array([np.mean(x) for p_, x in pm.items() if gp[p_] == "TD"])
        pool = np.r_[a, b]; perm = rng.permuted(np.tile(pool, (N, 1)), axis=1)
        p = (np.sum(np.abs(perm[:, :len(a)].mean(1) - perm[:, len(a):].mean(1)) >= abs(a.mean() - b.mean()) - 1e-12) + 1) / (N + 1)
        gb = [hedges(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))]) for _ in range(2000)]
        res6.append(dict(metric=m, mode=mode, g=float(hedges(a, b)), g_ci=ci(gb), p=float(p)))
for mode in ("raw", "image_centred"):
    sub = [r for r in res6 if r["mode"] == mode]
    o = sorted(range(len(sub)), key=lambda i: sub[i]["p"]); run_ = 0.0
    for k, i in enumerate(o):
        run_ = max(run_, min(1.0, (len(sub) - k) * sub[i]["p"])); sub[i]["p_holm"] = run_
out["A6_image_centred"] = res6
for m in MS:
    r0, r1 = [r for r in res6 if r["metric"] == m]
    print(f"A6 {m:22s} raw g={r0['g']:+.2f} {[round(x, 2) for x in r0['g_ci']]} holm={r0['p_holm']:.4f} | centred g={r1['g']:+.2f} {[round(x, 2) for x in r1['g_ci']]} holm={r1['p_holm']:.4f}")
json.dump(out, open(OUT + "/a346_results.json", "w"), indent=1)
print("A346_DONE")
