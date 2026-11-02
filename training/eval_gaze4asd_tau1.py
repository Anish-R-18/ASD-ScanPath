"""
Gaze4ASD behavioural evaluation at temperature 1.0, compared with the saved temperature 2.0 replicates.
No training and no NLL evaluation. Checkpoints, datasets and earlier results are read-only. Temperature was
fixed from Saliency4ASD validation images and is not tuned here.

  python training/eval_gaze4asd_tau1.py run       gates, generation, metrics, MultiMatch
  python training/eval_gaze4asd_tau1.py analyze   statistics, tables, figures

Fixed before generation:
  Prespecified tests, from the validation report: per-image error for saccade amplitude and revisit rate,
  ASD and TD, with tau=1 expected to be closer; Holm across these four tests.
  Transfer criterion for each test: tau=1 closer, image-level 95% CI excludes zero, at least 8 of 10 seeds agree.
  Coordinates clipped to the image is the primary analysis, as on validation; unclipped is the sensitivity analysis.
  Every other comparison is exploratory.
"""
from __future__ import annotations
import csv, gzip, json, os, sys, time, zlib
sys.path[:0] = [".", "tppgaze"]
import numpy as np

V2 = "training/runs/external_gaze4asd_v2"
OUT = f"{V2}/temperature_tau1"
RAW = OUT + "/raw"
FIG = OUT + "/figures"
GEN = OUT + "/generated_scanpaths"
DATA = "external_data/Gaze4ASD_processed_v2/fit_height_policy_a/gaze4asd_fit_height_policy_a.json"
PHASE1 = "training/runs/baseline/best_model.pth"
PHASE2 = "training/runs/phase2_embedding_only/best_model_conditioned.pth"
TAU2_SCANPATHS = f"{V2}/phase1_vs_phase2/raw/generated_scanpaths.json.gz"
TAU2_SUMMARY = f"{V2}/phase1_vs_phase2/raw/generated_and_real_summaries.npz"
REAL_PARTICIPANT = f"{V2}/statistics/participant_level_real.csv"
VAL_STATS = "training/runs/temperature_sensitivity/statistical_comparison.json"
TAUS = (1.0, 2.0)
GROUPS = ("ASD", "TD")
MODES = ("clip", "raw")
N_REP, N_SIM, T_END, MAX_PAIRS, N_BOOT = 10, 20, 3.0, 30, 10000
METRICS = {"dur_mean": "Fixation duration, mean (s)", "dur_median": "Fixation duration, median (s)", "n_fix": "Fixation count", "n_fix_window": "Fixation count, real time window", "fix_rate_per_s": "Fixation rate (per s)", "total_dur": "Total fixation time (s)",
           "sacc_amp_norm_mean": "Saccade amplitude (/diagonal)", "sacc_horizontal_frac": "Horizontal saccade fraction", "dispersion": "Spatial dispersion (/diagonal)", "revisit_rate": "Revisit rate", "sacc_dir_resultant": "Direction concentration"}
DISTS = {"D_dur": "Fixation durations", "D_nfix": "Fixation counts", "D_amp": "Saccade amplitudes", "D_disp": "Spatial dispersion", "D_rev": "Revisit rates", "D_pos": "Fixation positions", "D_dir": "Saccade directions (total variation)"}
WINDOWED = ("n_fix", "dispersion", "sacc_amp_norm_mean", "revisit_rate", "sacc_dir_resultant", "sacc_horizontal_frac")
PRESPECIFIED = ("sacc_amp_norm_mean", "revisit_rate")
MM = ["mm_shape", "mm_length", "mm_direction", "mm_position", "mm_duration"]
DIR_BINS = np.linspace(-np.pi, np.pi, 13)
tn = lambda t: f"tau{int(t)}"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(f"{OUT}/run_log.txt", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def write_rows(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r[k]) if isinstance(r.get(k), (list, dict)) else r.get(k, "")) for k in keys})


# =========================================================================== run
def run():
    import torch
    from omegaconf import OmegaConf
    from scipy.stats import wasserstein_distance as W1
    from training import metrics as M, model_utils as mu
    from training.conditioned import ConditionedTPPGaze
    from training.generate_cond import sample_conditioned, to_scanpaths
    from training.external_gaze4asd_loader import StimulusBank, load_external_scanpaths, summarize_records, GROUP_TO_ID
    from training.eval_external_v2 import IMG_DIR, BASE_CFG, denorm, metric_row, sha256

    if os.path.exists(OUT):
        raise SystemExit(f"Refusing to overwrite: {OUT}")
    os.makedirs(RAW); os.makedirs(FIG); os.makedirs(GEN)
    device = torch.device("cuda")
    t0 = time.time()
    h2 = sha256(PHASE2)
    log(f"Phase 2 {PHASE2} sha256={h2}")
    stored = json.load(open("training/runs/phase2_embedding_only/summary_full.json"))
    mean, std = stored["mean_log_inter_time"], stored["std_log_inter_time"]
    log(f"normalization from Saliency4ASD training summary: mean={mean} std={std}")
    assert GROUP_TO_ID == {"ASD": 0, "TD": 1}

    cfg = OmegaConf.load(BASE_CFG)
    base, _ = mu.build_model(BASE_CFG, PHASE1, device, mean, std, used_cfg_out="/tmp/tau1_ext.yaml")
    model = ConditionedTPPGaze(base, int(cfg.metadata.size + cfg.context.size)).to(device)
    model.load_state_dict(torch.load(PHASE2, map_location=device), strict=True)
    mu._fix_unregistered_device(model.base, device)
    model.eval()
    for q in model.parameters():
        q.requires_grad = False
    log("checkpoint loaded (strict); evaluation mode; all parameters frozen")

    bank = StimulusBank(IMG_DIR)
    recs = load_external_scanpaths(DATA, bank)
    info = summarize_records(recs, bank)
    assert (info["participants_ASD"], info["participants_TD"], info["trials_ASD"], info["trials_TD"], info["stimuli"]) == (33, 133, 771, 3605, 30)
    images = sorted({r["image_id"] for r in recs}, key=lambda s: int(s.replace("top_image_", "").replace(".png", "")))
    Z2 = np.load(TAU2_SUMMARY)
    assert list(Z2["images"]) == images
    log(f"data verified: 33 ASD, 133 TD participants; 771, 3605 trials; {len(images)} stimuli; image order matches the tau=2 run")

    # ---------------- saved tau=2 scanpaths
    saved = {}
    with gzip.open(TAU2_SCANPATHS, "rt", encoding="utf-8") as f:
        for s in json.load(f):
            if s["set"] in ("P2_ASD", "P2_TD"):
                saved.setdefault((s["set"][3:], s["rep"], s["image_id"]), {})[s["sim"]] = np.asarray(s["scanpath"], float)

    def sample(rep, k, iid, g, t):
        gid = GROUP_TO_ID[g]
        seed = 100000 * (rep + 1) + 1000 * (k + 1) + gid
        torch.manual_seed(seed)
        it, mk, ms = sample_conditioned(model, bank.tensor(iid), gid, t_end=T_END, n_sim=N_SIM, temperature=t)
        return seed, {sim: np.round(sp, 5) for sim, sp in enumerate(to_scanpaths(it, mk, ms)) if len(sp)}, it.shape[1]

    # gate: the seed rule really reproduces saved tau=2 sets. No saved result is replaced.
    worst = 0.0
    for rep, k, g in ((0, 0, "ASD"), (9, 15, "TD")):
        _, again, _ = sample(rep, k, images[k], g, 2.0)
        ref = saved[(g, rep, images[k])]
        assert set(again) == set(ref), "saved tau=2 set has different scanpaths"
        worst = max(worst, max(float(np.abs(again[s] - ref[s]).max()) for s in ref))
    log(f"seed gate: two saved tau=2 sets regenerated, max abs diff={worst:.2e}")
    if worst > 1e-4:
        raise SystemExit("GATE FAILED: saved tau=2 sets not reproduced, so the seeds are not paired")

    # ---------------- generate tau=1
    gen = {1.0: {}, 2.0: saved}
    first_max, longest, n_missing = 0.0, 0, 0
    for rep in range(N_REP):
        store = []
        for k, iid in enumerate(images):
            for g in GROUPS:
                seed, paths, L = sample(rep, k, iid, g, 1.0)
                longest = max(longest, L)
                gen[1.0][(g, rep, iid)] = paths
                ref = saved[(g, rep, iid)]
                if set(paths) != set(ref):
                    n_missing += len(set(paths) ^ set(ref))
                for s in set(paths) & set(ref):
                    first_max = max(first_max, abs(float(paths[s][0, 2] - ref[s][0, 2])))
                store += [dict(image_id=iid, group=g, sim=s, seed=seed, scanpath=sp.tolist()) for s, sp in paths.items()]
        with gzip.open(f"{GEN}/seed_{rep + 1:02d}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(dict(temperature=1.0, replicate=rep + 1, seed_rule="100000*(rep+1)+1000*(image_rank+1)+group_id", scanpaths=store), f)
        log(f"generated replicate {rep + 1}/{N_REP} ({time.time() - t0:.0f}s); first-duration max diff vs saved tau=2={first_max:.2e}; sets differing in scanpath membership={n_missing}; longest={longest}")
    if first_max > 2e-5 or n_missing:
        raise SystemExit("GATE FAILED: first durations differ from the saved tau=2 run")

    # ---------------- real
    def angles(xy):
        v = np.diff(xy, axis=0)
        return np.arctan2(v[:, 1], v[:, 0])
    def samples(paths, iid):
        Wd, Hd = bank.dims(iid)
        diag = float(np.hypot(Wd, Hd))
        rows = [metric_row("x", iid, "", p["xy"], p["dur"], Wd, Hd) for p in paths]
        two = [p for p in paths if len(p["xy"]) >= 2]
        ang = np.concatenate([angles(p["xy"]) for p in two]) if two else np.zeros(0)
        return dict(rows=rows, dur=np.concatenate([p["dur"] for p in paths]), nfix=np.array([len(p["xy"]) for p in paths], float),
                    amp=np.concatenate([np.linalg.norm(np.diff(p["xy"], axis=0), axis=1) / diag for p in two]) if two else np.zeros(0),
                    disp=np.array([x["dispersion"] for x in rows]), rev=np.array([x["revisit_rate"] for x in rows]),
                    x=np.concatenate([p["xy"][:, 0] for p in paths]) / diag, y=np.concatenate([p["xy"][:, 1] for p in paths]) / diag,
                    hist=np.histogram(ang, bins=DIR_BINS)[0].astype(float), cs=np.array([np.cos(ang).sum(), np.sin(ang).sum()]))
    def dist_set(a, b):
        ha, hb = a["hist"], b["hist"]
        return dict(D_dur=W1(a["dur"], b["dur"]), D_nfix=W1(a["nfix"], b["nfix"]), D_amp=W1(a["amp"], b["amp"]), D_disp=W1(a["disp"], b["disp"]), D_rev=W1(a["rev"], b["rev"]), D_pos=0.5 * (W1(a["x"], b["x"]) + W1(a["y"], b["y"])), D_dir=0.5 * np.abs(ha / ha.sum() - hb / hb.sum()).sum())
    def merge(sets):
        out = {}
        for k in sets[0]:
            if k == "rows":
                out[k] = [r for s in sets for r in s[k]]
            elif k in ("hist", "cs"):
                out[k] = sum(s[k] for s in sets)
            else:
                out[k] = np.concatenate([s[k] for s in sets])
        return out
    real = {g: {i: [] for i in images} for g in GROUPS}
    for r in recs:
        Wd, Hd = bank.dims(r["image_id"])
        sp = np.asarray(r["scanpath"], float)
        real[r["group"]][r["image_id"]].append(dict(xy=denorm(sp, Wd, Hd), dur=sp[:, 2]))
    real_s = {g: {i: samples(real[g][i], i) for i in images} for g in GROUPS}
    window = {g: {i: float(np.mean([p["dur"].sum() for p in real[g][i]])) for i in images} for g in GROUPS}
    arrs = {}
    for g in GROUPS:
        for m in METRICS:
            src = "n_fix" if m == "n_fix_window" else m
            arrs[f"real_{g}_{m}"] = np.array([np.nanmean([r[src] for r in real_s[g][i]["rows"]]) for i in images])
        arrs[f"real_{g}_dur_pooled_median"] = np.array([np.median(real_s[g][i]["dur"]) for i in images])
        arrs[f"real_{g}_cs"] = sum(real_s[g][i]["cs"] for i in images)

    # ---------------- generated, one code path for both temperatures
    pre = lambda mode, t, g: f"{mode}_{tn(t)}_{g}"
    for mode in MODES:
        for t in TAUS:
            for g in GROUPS:
                for m in list(METRICS) + ["dur_pooled_median"] + list(DISTS):
                    arrs[f"{pre(mode, t, g)}_{m}"] = np.full((N_REP, len(images)), np.nan)
                for d in DISTS:
                    arrs[f"{pre(mode, t, g)}_pooled_{d}"] = np.full(len(images), np.nan)
                    arrs[f"{pre(mode, t, g)}_global_{d}"] = np.full(N_REP, np.nan)
                arrs[f"{pre(mode, t, g)}_cs"] = np.zeros((N_REP, 2))
    for t in TAUS:
        for g in GROUPS:
            for m in WINDOWED:
                arrs[f"win_{tn(t)}_{g}_{m}"] = np.full((N_REP, len(images)), np.nan)
    off = {tn(t): {g: np.zeros((N_REP, 2)) for g in GROUPS} for t in TAUS}
    mm_rows = []
    def mm_mean(tag, la, lb, same, diag):
        pairs = [(i, j) for i in range(len(la)) for j in range(len(lb)) if not (same and j <= i)]
        rng = np.random.default_rng([0, zlib.crc32(tag.encode())])
        v = {k: [] for k in MM}
        for q in rng.permutation(len(pairs))[:MAX_PAIRS]:
            a, b = la[pairs[q][0]], lb[pairs[q][1]]
            r = M.multimatch(a["xy"], a["dur"], b["xy"], b["dur"], diag)
            if r:
                for k in MM:
                    v[k].append(r[k])
        return {k: float(np.mean(v[k])) if v[k] else float("nan") for k in MM}, len(v[MM[0]])
    for iid in images:
        diag = float(np.hypot(*bank.dims(iid)))
        for g in GROUPS:
            v, n = mm_mean(f"RvR|{g}|{iid}", real[g][iid], real[g][iid], True, diag)
            mm_rows.append(dict(kind="RvR", target=g, tau="", rep=-1, image_id=iid, n_pairs=n, **v))

    pooled = {mode: {t: {g: {i: [] for i in images} for g in GROUPS} for t in TAUS} for mode in MODES}
    for rep in range(N_REP):
        sets = {mode: {t: {g: [] for g in GROUPS} for t in TAUS} for mode in MODES}
        for k, iid in enumerate(images):
            Wd, Hd = bank.dims(iid)
            diag = float(np.hypot(Wd, Hd))
            cur = {}
            for t in TAUS:
                for g in GROUPS:
                    paths = list(gen[t][(g, rep, iid)].values())
                    L_raw, L_clip = [], []
                    for sp in paths:
                        off[tn(t)][g][rep] += [float((np.abs(sp[:, :2]) > 1).any(1).sum()), len(sp)]
                        xy = denorm(sp, Wd, Hd)
                        L_raw.append(dict(xy=xy, dur=sp[:, 2]))
                        L_clip.append(dict(xy=np.clip(xy, [0, 0], [Wd - 1, Hd - 1]), dur=sp[:, 2]))
                    cur[(t, g)] = L_clip
                    for mode, L in (("clip", L_clip), ("raw", L_raw)):
                        s = samples(L, iid)
                        p = pre(mode, t, g)
                        for m in METRICS:
                            if m == "n_fix_window":
                                arrs[f"{p}_{m}"][rep, k] = np.mean([(np.r_[0.0, np.cumsum(x["dur"])[:-1]] < window[g][iid]).sum() for x in L])
                            else:
                                arrs[f"{p}_{m}"][rep, k] = np.nanmean([r[m] for r in s["rows"]])
                        arrs[f"{p}_dur_pooled_median"][rep, k] = np.median(s["dur"])
                        arrs[f"{p}_cs"][rep] += s["cs"]
                        for d, val in dist_set(s, real_s[g][iid]).items():
                            arrs[f"{p}_{d}"][rep, k] = val
                        pooled[mode][t][g][iid].append(s)
                        sets[mode][t][g].append(s)
                    rows_w = []
                    for x in L_clip:
                        on = np.r_[0.0, np.cumsum(x["dur"])[:-1]] < window[g][iid]
                        rows_w.append(metric_row("w", iid, "", x["xy"][on], x["dur"][on], Wd, Hd))
                    for m in WINDOWED:
                        arrs[f"win_{tn(t)}_{g}_{m}"][rep, k] = np.nanmean([r[m] for r in rows_w])
            for t in TAUS:
                for g, other in (("ASD", "TD"), ("TD", "ASD")):
                    v, n = mm_mean(f"RvG|{g}|{rep}|{iid}", real[g][iid], cur[(t, g)], False, diag)
                    mm_rows.append(dict(kind="RvG", target=g, tau=tn(t), rep=rep, image_id=iid, n_pairs=n, **v))
                    v, n = mm_mean(f"RvG_wrong|{g}|{rep}|{iid}", real[g][iid], cur[(t, other)], False, diag)
                    mm_rows.append(dict(kind="RvG_wrong", target=g, tau=tn(t), rep=rep, image_id=iid, n_pairs=n, **v))
        for mode in MODES:
            for t in TAUS:
                for g in GROUPS:
                    for d, val in dist_set(merge(sets[mode][t][g]), merge([real_s[g][i] for i in images])).items():
                        arrs[f"{pre(mode, t, g)}_global_{d}"][rep] = val
        log(f"metrics replicate {rep + 1}/{N_REP} ({time.time() - t0:.0f}s)")
    for mode in MODES:
        for t in TAUS:
            for g in GROUPS:
                for k, iid in enumerate(images):
                    for d, val in dist_set(merge(pooled[mode][t][g][iid]), real_s[g][iid]).items():
                        arrs[f"{pre(mode, t, g)}_pooled_{d}"][k] = val
    # regression: recomputed tau=2 metrics must match the saved summaries
    # Saved scanpaths hold 5 decimals, so revisit (radius) and horizontal fraction (|dx|>|dy|), which are
    # thresholded, can flip in an isolated scanpath. Six cells did; each matched exactly at full precision.
    THRESHOLDED = ("revisit_rate", "sacc_horizontal_frac")
    reg = {}
    for g in GROUPS:
        for m in ("dur_mean", "n_fix", "fix_rate_per_s", "total_dur", "sacc_amp_norm_mean", "dispersion", "revisit_rate", "sacc_dir_resultant", "sacc_horizontal_frac", "n_fix_window"):
            d = arrs[f"raw_tau2_{g}_{m}"] - Z2[f"P2_{g}_{m}"]
            reg[f"{g}_{m}"] = dict(grand_mean_diff=float(abs(d.mean())), max_cell_diff=float(np.abs(d).max()), cells_over_1e3=int((np.abs(d) > 1e-3).sum()))
    log("tau=2 metrics recomputed from saved scanpaths vs saved summaries: " + json.dumps(reg))
    flips = sum(x["cells_over_1e3"] for k, x in reg.items() if k.split("_", 1)[1] in THRESHOLDED)
    others = sum(x["cells_over_1e3"] for k, x in reg.items() if k.split("_", 1)[1] not in THRESHOLDED)
    log(f"threshold flips from rounding: {flips} of {4 * N_REP * len(images)} cells; other metrics over 1e-3: {others}")
    if max(x["grand_mean_diff"] for x in reg.values()) > 1e-4 or others or flips > 12:
        raise SystemExit("GATE FAILED: saved tau=2 metrics not reproduced")
    np.savez_compressed(f"{RAW}/summaries.npz", images=np.array(images), **arrs)
    write_rows(f"{RAW}/multimatch_per_image.csv", mm_rows)
    json.dump(dict(checkpoint_sha256=h2, normalization=dict(mean=mean, std=std), seed_gate_max_diff=worst, first_duration_max_diff=first_max, longest_sequence=longest, tau2_regression=reg, off_image={t: {g: dict(rate=float(v[:, 0].sum() / v[:, 1].sum()), per_seed=(v[:, 0] / v[:, 1]).tolist()) for g, v in x.items()} for t, x in off.items()}, empty_scanpaths={tn(t): int(N_REP * len(images) * 2 * N_SIM - sum(len(p) for p in gen[t].values())) for t in TAUS}, images=images), open(f"{RAW}/run_checks.json", "w"), indent=2)
    assert sha256(PHASE2) == h2
    log(f"checkpoint unchanged. total {time.time() - t0:.0f}s. RUN_DONE")


# ======================================================================= analyze
def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    rng = np.random.default_rng(0)
    Z = np.load(f"{RAW}/summaries.npz")
    checks = json.load(open(f"{RAW}/run_checks.json"))
    images = list(Z["images"])
    nI = len(images)
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    Bidx = rng.integers(0, nI, (N_BOOT, nI))

    def paired(d):
        d = np.asarray(d, float)
        s = d.std(ddof=1)
        flips = (rng.choice([-1.0, 1.0], (N_BOOT, nI)) * d).mean(1)
        return dict(mean=float(d.mean()), median=float(np.median(d)), ci=ci(d[Bidx].mean(1)), p=float((np.sum(np.abs(flips) >= abs(d.mean()) - 1e-15) + 1) / (N_BOOT + 1)), dz=float(d.mean() / s) if s > 0 else 0.0, n_pos=int((d > 0).sum()), n=nI)
    def holm(items, key="p", out="p_holm"):
        o = sorted(range(len(items)), key=lambda i: items[i][key]); r_ = 0.0
        for r, i in enumerate(o):
            r_ = max(r_, min(1.0, (len(items) - r) * items[i][key]))
            items[i][out] = r_
    seedstat = lambda x: dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), min=float(x.min()), max=float(x.max()), n_pos=int((x > 0).sum()), n=len(x))
    G = lambda mode, t, g, m: Z[f"{mode}_{tn(t)}_{g}_{m}"]
    ALL = {**METRICS, "dur_pooled_median": "Fixation duration, pooled per-image median (s)"}

    res = {}
    for mode in MODES:
        direct, realism, gvr, dist, gaps = [], [], [], [], []
        for g in GROUPS:
            for m, lab in ALL.items():
                a, b, R = G(mode, 1.0, g, m), G(mode, 2.0, g, m), Z[f"real_{g}_{m}"]
                split = 0.5 * ((a[:5].mean(0) - a[5:].mean(0)) + (b[:5].mean(0) - b[5:].mean(0)))
                sd = 0.5 * (a.mean(1).std(ddof=1) + b.mean(1).std(ddof=1))
                row = dict(metric=m, label=lab, group=g, real=float(R.mean()), tau1=float(a.mean()), tau2=float(b.mean()), tau1_seed_sd=float(a.mean(1).std(ddof=1)), tau2_seed_sd=float(b.mean(1).std(ddof=1)), noise_split_half_mean=float(split.mean()), effect_to_seed_sd=float(abs((b.mean(0) - a.mean(0)).mean()) / sd) if sd > 0 else None)
                row.update(paired(b.mean(0) - a.mean(0))); row["seeds"] = seedstat(b.mean(1) - a.mean(1))
                direct.append(row)
                for t, x in ((1.0, a), (2.0, b)):
                    r2 = dict(metric=m, label=lab, group=g, tau=tn(t), real_mean=float(R.mean()), real_median=float(np.median(R)), gen_mean=float(x.mean()), gen_median=float(np.median(x.mean(0))), seed_sd=float(x.mean(1).std(ddof=1)))
                    r2.update(paired(x.mean(0) - R)); r2["seeds"] = seedstat(x.mean(1) - R.mean())
                    gvr.append(r2)
                e1, e2 = np.abs(a.mean(0) - R), np.abs(b.mean(0) - R)
                row2 = dict(metric=m, label=lab, group=g, real=float(R.mean()), tau1=float(a.mean()), tau2=float(b.mean()), bias_tau1=float((a.mean(0) - R).mean()), bias_tau2=float((b.mean(0) - R).mean()), mae_tau1=float(e1.mean()), mae_tau2=float(e2.mean()), rel_pct_of_tau2=float(100 * (e2 - e1).mean() / e2.mean()),
                            abs_bias_gain=float(abs((b.mean(0) - R).mean()) - abs((a.mean(0) - R).mean())), abs_bias_gain_ci=ci(np.abs((b.mean(0) - R)[Bidx].mean(1)) - np.abs((a.mean(0) - R)[Bidx].mean(1))))
                row2.update(paired(e2 - e1)); row2["seeds"] = seedstat(np.abs(b - R).mean(1) - np.abs(a - R).mean(1))
                realism.append(row2)
            for d, lab in DISTS.items():
                a, b = Z[f"{mode}_tau1_{g}_pooled_{d}"], Z[f"{mode}_tau2_{g}_pooled_{d}"]
                row = dict(metric=d, label=lab, group=g, tau1=float(a.mean()), tau2=float(b.mean()), rel_pct_of_tau2=float(100 * (b - a).mean() / b.mean()))
                row.update(paired(b - a)); row["seeds"] = seedstat(G(mode, 2.0, g, d).mean(1) - G(mode, 1.0, g, d).mean(1))
                row["global_tau1"], row["global_tau2"] = seedstat(Z[f"{mode}_tau1_{g}_global_{d}"]), seedstat(Z[f"{mode}_tau2_{g}_global_{d}"])
                dist.append(row)
        for m, lab in METRICS.items():
            Dr = Z[f"real_ASD_{m}"] - Z[f"real_TD_{m}"]
            D1, D2 = G(mode, 1.0, "ASD", m) - G(mode, 1.0, "TD", m), G(mode, 2.0, "ASD", m) - G(mode, 2.0, "TD", m)
            p1, p2 = D1.mean(0), D2.mean(0)
            r1, r2 = D1.mean(1), D2.mean(1)
            same = lambda x: int((np.sign(x) == np.sign(Dr.mean())).sum())
            row = dict(metric=m, label=lab, real=float(Dr.mean()), real_ci=ci(Dr[Bidx].mean(1)), real_image=paired(Dr), tau1=float(p1.mean()), tau1_ci=ci(p1[Bidx].mean(1)), tau1_image=paired(p1), tau2=float(p2.mean()), tau2_ci=ci(p2[Bidx].mean(1)), tau2_image=paired(p2), ratio_tau1=float(p1.mean() / Dr.mean()), ratio_tau2=float(p2.mean() / Dr.mean()),
                       tau1_seeds_real_sign=same(r1), tau2_seeds_real_sign=same(r2), tau1_seeds=seedstat(r1), tau2_seeds=seedstat(r2), gap_tau1=float(abs((Dr - p1).mean())), gap_tau2=float(abs((Dr - p2).mean())), gap_gain=float(abs((Dr - p2).mean()) - abs((Dr - p1).mean())), gap_gain_ci=ci(np.abs((Dr - p2)[Bidx].mean(1)) - np.abs((Dr - p1)[Bidx].mean(1))),
                       r_tau1=float(np.corrcoef(Dr, p1)[0, 1]), r_tau2=float(np.corrcoef(Dr, p2)[0, 1]), tau2_minus_tau1=paired(p2 - p1), tau2_minus_tau1_seeds=seedstat(r2 - r1))
            gaps.append(row)
        for fam in (direct, realism, dist):
            holm(fam)
        for t in ("tau1", "tau2"):
            sub = [r for r in gvr if r["tau"] == t]
            holm(sub)
        res[mode] = dict(direct=direct, realism=realism, generated_vs_real=gvr, distances=dist, group_differences=gaps)
    pres = [r for r in res["clip"]["realism"] if r["metric"] in PRESPECIFIED]
    tests = [dict(metric=r["metric"], group=r["group"], p=r["p"]) for r in pres]
    holm(tests)
    for r, t in zip(pres, tests):
        r["p_holm_prespecified"] = t["p_holm"]
        r["meets_transfer_criterion"] = bool(r["mean"] > 0 and r["ci"][0] > 0 and r["seeds"]["n_pos"] >= 8)

    # window-matched, clipped
    win_rows, win_gaps = [], []
    for g in GROUPS:
        for m in WINDOWED:
            a, b, R = Z[f"win_tau1_{g}_{m}"], Z[f"win_tau2_{g}_{m}"], Z[f"real_{g}_{m}"]
            e1, e2 = np.abs(a.mean(0) - R), np.abs(b.mean(0) - R)
            row = dict(metric=m, label=METRICS[m], group=g, real=float(R.mean()), tau1=float(a.mean()), tau2=float(b.mean()), bias_tau1=float((a.mean(0) - R).mean()), bias_tau2=float((b.mean(0) - R).mean()), mae_tau1=float(e1.mean()), mae_tau2=float(e2.mean()), abs_bias_gain=float(abs((b.mean(0) - R).mean()) - abs((a.mean(0) - R).mean())), abs_bias_gain_ci=ci(np.abs((b.mean(0) - R)[Bidx].mean(1)) - np.abs((a.mean(0) - R)[Bidx].mean(1))))
            row.update(paired(e2 - e1)); row["seeds"] = seedstat(np.abs(b - R).mean(1) - np.abs(a - R).mean(1))
            win_rows.append(row)
    holm(win_rows)
    for m in WINDOWED:
        Dr = Z[f"real_ASD_{m}"] - Z[f"real_TD_{m}"]
        p1, p2 = Z[f"win_tau1_ASD_{m}"] - Z[f"win_tau1_TD_{m}"], Z[f"win_tau2_ASD_{m}"] - Z[f"win_tau2_TD_{m}"]
        win_gaps.append(dict(metric=m, real=float(Dr.mean()), tau1=float(p1.mean()), tau2=float(p2.mean()), tau1_ci=ci(p1.mean(0)[Bidx].mean(1)), tau2_ci=ci(p2.mean(0)[Bidx].mean(1)), tau1_seeds_real_sign=int((np.sign(p1.mean(1)) == np.sign(Dr.mean())).sum()), tau2_seeds_real_sign=int((np.sign(p2.mean(1)) == np.sign(Dr.mean())).sum())))

    # MultiMatch, clipped
    Mm = list(csv.DictReader(open(f"{RAW}/multimatch_per_image.csv", encoding="utf-8")))
    iix = {i: k for k, i in enumerate(images)}
    def cube(kind, g, tau):
        c = np.full((N_REP, nI, len(MM)), np.nan)
        for r in Mm:
            if r["kind"] == kind and r["target"] == g and r["tau"] == tau:
                c[int(r["rep"]), iix[r["image_id"]]] = [float(r[k]) if r[k] not in ("", "nan") else np.nan for k in MM]
        return c
    mmres = []
    for g in GROUPS:
        rv = np.full((nI, len(MM)), np.nan)
        for r in Mm:
            if r["kind"] == "RvR" and r["target"] == g:
                rv[iix[r["image_id"]]] = [float(r[k]) for k in MM]
        c1, c2, w1, w2 = cube("RvG", g, "tau1"), cube("RvG", g, "tau2"), cube("RvG_wrong", g, "tau1"), cube("RvG_wrong", g, "tau2")
        for q, k in enumerate(MM):
            x1, x2 = np.nanmean(c1[:, :, q], 0), np.nanmean(c2[:, :, q], 0)
            row = dict(metric=k, group=g, real_vs_real=float(np.nanmean(rv[:, q])), real_vs_real_median=float(np.nanmedian(rv[:, q])), tau1=float(x1.mean()), tau1_median=float(np.median(x1)), tau2=float(x2.mean()), tau2_median=float(np.median(x2)), tau1_seed_sd=float(np.nanmean(c1[:, :, q], 1).std(ddof=1)), tau2_seed_sd=float(np.nanmean(c2[:, :, q], 1).std(ddof=1)),
                       distance_from_real_vs_real_tau1=float(abs(x1.mean() - np.nanmean(rv[:, q]))), distance_from_real_vs_real_tau2=float(abs(x2.mean() - np.nanmean(rv[:, q]))))
            row.update(paired(x1 - x2)); row["seeds"] = seedstat(np.nanmean(c1[:, :, q], 1) - np.nanmean(c2[:, :, q], 1))
            row["own_minus_other_tau1"] = paired(x1 - np.nanmean(w1[:, :, q], 0))
            row["own_minus_other_tau2"] = paired(x2 - np.nanmean(w2[:, :, q], 0))
            mmres.append(row)
    holm(mmres)

    # transfer from validation
    val = json.load(open(VAL_STATS))
    vr = {(r["group"], r["metric"]): r for r in val["clip"]["realism"]}
    vg = {r["metric"]: r for r in val["clip"]["group_differences"]}
    transfer = []
    for r in res["clip"]["realism"]:
        v = vr.get((r["group"], r["metric"]))
        if v:
            transfer.append(dict(metric=r["metric"], label=r["label"], group=r["group"], validation_gain=v["mean"], validation_ci=v["ci"], validation_rel_pct=v["rel_pct_of_tau2"], validation_seeds_for_tau1=v["seeds"]["n_pos"], external_gain=r["mean"], external_ci=r["ci"], external_rel_pct=r["rel_pct_of_tau2"], external_seeds_for_tau1=r["seeds"]["n_pos"], validation_bias_tau1=v["bias_tau1"], validation_bias_tau2=v["bias_tau2"], external_bias_tau1=r["bias_tau1"], external_bias_tau2=r["bias_tau2"], same_direction=bool(np.sign(v["mean"]) == np.sign(r["mean"]))))
    transfer_gaps = [dict(metric=d["metric"], validation_ratio_tau1=vg[d["metric"]]["ratio_tau1"], validation_ratio_tau2=vg[d["metric"]]["ratio_tau2"], external_ratio_tau1=d["ratio_tau1"], external_ratio_tau2=d["ratio_tau2"], validation_gap_gain=vg[d["metric"]]["gap_gain_favouring_tau1"], external_gap_gain=d["gap_gain"]) for d in res["clip"]["group_differences"] if d["metric"] in vg]

    # real participant-level values are read from the saved v2 statistics, not recomputed
    participant = [r for r in csv.DictReader(open(REAL_PARTICIPANT, encoding="utf-8")) if r.get("version", "fit_height_policy_a") == "fit_height_policy_a"]
    circ = {}
    for g in GROUPS:
        cs = Z[f"real_{g}_cs"]
        circ[f"real_{g}"] = float(np.degrees(np.arctan2(cs[1], cs[0])))
        for t in TAUS:
            cs = Z[f"clip_{tn(t)}_{g}_cs"]
            circ[f"{tn(t)}_{g}"] = float(np.degrees(np.arctan2(cs.sum(0)[1], cs.sum(0)[0])))

    # ---------------- outputs
    rows = []
    for t in TAUS:
        for mode in MODES:
            for g in GROUPS:
                for k, iid in enumerate(images):
                    for rep in range(N_REP):
                        rows.append(dict(tau=t, coordinates=mode, group=g, replicate=rep + 1, image_id=iid, **{m: float(G(mode, t, g, m)[rep, k]) for m in ALL}))
    write_rows(f"{OUT}/generated_metrics.csv", rows)
    comp = []
    dd = {(r["metric"], r["group"]): r for r in res["clip"]["direct"]}
    rr = {(r["metric"], r["group"]): r for r in res["clip"]["realism"]}
    for m, lab in ALL.items():
        a, t = dd[(m, "ASD")], dd[(m, "TD")]
        ra, rt = rr[(m, "ASD")], rr[(m, "TD")]
        for stat, f in (("mean", np.mean), ("median", np.median)):
            comp.append({"Metric": lab, "Statistic": stat, "Real ASD": float(f(Z[f"real_ASD_{m}"])), "Real TD": float(f(Z[f"real_TD_{m}"])), "Generated ASD tau=1": float(f(G("clip", 1.0, "ASD", m).mean(0))), "Generated TD tau=1": float(f(G("clip", 1.0, "TD", m).mean(0))), "Generated ASD tau=2": float(f(G("clip", 2.0, "ASD", m).mean(0))), "Generated TD tau=2": float(f(G("clip", 2.0, "TD", m).mean(0))),
                         **({"ASD MAE tau=1": ra["mae_tau1"], "ASD MAE tau=2": ra["mae_tau2"], "TD MAE tau=1": rt["mae_tau1"], "TD MAE tau=2": rt["mae_tau2"], "ASD gain favouring tau=1": ra["mean"], "ASD gain CI": ra["ci"], "ASD Holm p": ra["p_holm"], "TD gain favouring tau=1": rt["mean"], "TD gain CI": rt["ci"], "TD Holm p": rt["p_holm"]} if stat == "mean" else {})})
    for r in [x for x in mmres if x["group"] == "ASD"]:
        t = next(x for x in mmres if x["group"] == "TD" and x["metric"] == r["metric"])
        comp.append({"Metric": r["metric"], "Statistic": "mean similarity to the same real group (real columns: real vs real)", "Real ASD": r["real_vs_real"], "Real TD": t["real_vs_real"], "Generated ASD tau=1": r["tau1"], "Generated TD tau=1": t["tau1"], "Generated ASD tau=2": r["tau2"], "Generated TD tau=2": t["tau2"], "ASD gain favouring tau=1": r["mean"], "ASD gain CI": r["ci"], "ASD Holm p": r["p_holm"], "TD gain favouring tau=1": t["mean"], "TD gain CI": t["ci"], "TD Holm p": t["p_holm"]})
    write_rows(f"{OUT}/tau1_vs_tau2_comparison.csv", comp)
    json.dump(dict(note="Clipped coordinates are primary. Differences are generated minus real.", generated_vs_real=res["clip"]["generated_vs_real"], generated_vs_real_unclipped=res["raw"]["generated_vs_real"], real_asd_vs_td_participant_level_saved=participant, asd_vs_td=res["clip"]["group_differences"], multimatch=mmres, circular_mean_direction_degrees_descriptive=circ, off_image=checks["off_image"]), open(f"{OUT}/real_vs_generated.json", "w"), indent=2)
    json.dump(dict(scheme="One value per image pooled over 10 seeds; paired; percentile bootstrap and sign-flip permutation, 10,000 each; Holm within each family. Gains are positive when tau=1 is closer.", prespecified=pres, direct_temperature_effect=res["clip"]["direct"], realism=res["clip"]["realism"], distances=res["clip"]["distances"], group_differences=res["clip"]["group_differences"], multimatch=mmres, window_matched=dict(realism=win_rows, group_differences=win_gaps), unclipped=res["raw"], transfer_from_validation=dict(realism=transfer, group_differences=transfer_gaps)), open(f"{OUT}/statistical_results.json", "w"), indent=2)

    # ---------------- figures
    ASD, TD, INK, INK2, MUTED, GRID, AXIS, SURF = "#eb6834", "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 10, "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .8, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "savefig.dpi": 200, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True})
    ms = ["sacc_amp_norm_mean", "dispersion", "revisit_rate", "dur_mean", "n_fix", "n_fix_window", "fix_rate_per_s", "total_dur", "sacc_horizontal_frac", "sacc_dir_resultant"]
    fig, axs = plt.subplots(2, 5, figsize=(12, 4.8))
    for ax, m in zip(axs.ravel(), ms):
        for j, (g, c) in enumerate((("ASD", ASD), ("TD", TD))):
            ax.plot([j - .3, j + .3], [Z[f"real_{g}_{m}"].mean()] * 2, color=c, lw=2.2, solid_capstyle="round")
            for o, t, face in ((-.15, 1.0, c), (.15, 2.0, SURF)):
                v = G("clip", t, g, m).mean(1)
                ax.plot([j + o] * 2, [v.min(), v.max()], color=c, lw=1); ax.plot(j + o, v.mean(), "o", mfc=face, mec=c, ms=6, mew=1.3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(GROUPS, color=INK2); ax.set_xlim(-.6, 1.6); ax.set_title(METRICS[m], loc="left", fontsize=8.2); ax.grid(axis="x", visible=False)
    fig.legend(handles=[Line2D([], [], color=INK2, lw=2.2, label="Real Gaze4ASD mean (bar)"), Line2D([], [], marker="o", color=INK2, ls="", label="Generated, tau = 1"), Line2D([], [], marker="o", mfc=SURF, mec=INK2, ls="", label="Generated, tau = 2"), Line2D([], [], color=INK2, lw=1, label="Range across 10 seeds")], frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(.05, -.05))
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_metrics_tau1_vs_tau2.png", bbox_inches="tight"); plt.close(fig)
    fig, axs = plt.subplots(2, 5, figsize=(12, 3.8))
    gd = {d["metric"]: d for d in res["clip"]["group_differences"]}
    for ax, m in zip(axs.ravel(), ms):
        d = gd[m]
        ax.plot(d["real_ci"], [2, 2], color=INK, lw=1.4); ax.plot(d["real"], 2, "o", color=INK, ms=6, mec=SURF)
        ax.plot(d["tau1_ci"], [1, 1], color=INK2, lw=1.4); ax.plot(d["tau1"], 1, "o", color=INK2, ms=6, mec=SURF)
        ax.plot(d["tau2_ci"], [0, 0], color=INK2, lw=1.4); ax.plot(d["tau2"], 0, "o", mfc=SURF, mec=INK2, ms=6, mew=1.2)
        ax.axvline(0, color=AXIS, lw=1); ax.set_yticks([2, 1, 0]); ax.set_yticklabels(["Real", "tau = 1", "tau = 2"], color=INK2, fontsize=7.5); ax.set_ylim(-.6, 2.6); ax.grid(axis="y", visible=False); ax.set_title(METRICS[m], loc="left", fontsize=8); ax.tick_params(axis="x", labelsize=7); ax.xaxis.set_major_locator(MaxNLocator(4))
    fig.suptitle("ASD minus TD on Gaze4ASD, with 95% image bootstrap", x=.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_group_differences.png", bbox_inches="tight"); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(9, 2.9), sharey=True)
    y = np.arange(len(MM))[::-1]
    for ax, g, c in ((axs[0], "ASD", ASD), (axs[1], "TD", TD)):
        for i, r in enumerate([r for r in mmres if r["group"] == g]):
            ax.plot(r["real_vs_real"], y[i] + .2, "D", color=MUTED, ms=5); ax.plot(r["tau1"], y[i], "o", color=c, ms=6, mec=SURF); ax.plot(r["tau2"], y[i] - .2, "o", mfc=SURF, mec=c, ms=6, mew=1.3)
        ax.set_title(f"Similarity to real {g} (1 = identical)", loc="left"); ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y); axs[0].set_yticklabels([m.replace("mm_", "").capitalize() for m in MM], color=INK2)
    fig.legend(handles=[Line2D([], [], marker="D", color=MUTED, ls="", label="Real vs real"), Line2D([], [], marker="o", color=INK2, ls="", label="tau = 1"), Line2D([], [], marker="o", mfc=SURF, mec=INK2, ls="", label="tau = 2")], frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(.1, -.1))
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_multimatch.png", bbox_inches="tight"); plt.close(fig)
    fig, axs = plt.subplots(2, 4, figsize=(11, 4.2))
    for ax, m in zip(axs.ravel(), ["sacc_amp_norm_mean", "dispersion", "revisit_rate", "dur_mean", "n_fix", "fix_rate_per_s", "sacc_dir_resultant", "sacc_horizontal_frac"]):
        for j, (g, c) in enumerate((("ASD", ASD), ("TD", TD))):
            R = Z[f"real_{g}_{m}"]
            d = np.abs(G("clip", 2.0, g, m) - R).mean(1) - np.abs(G("clip", 1.0, g, m) - R).mean(1)
            ax.plot(np.arange(1, 11) + (j - .5) * .25, d, "o", color=c, ms=4.5, mec=SURF, mew=.6)
        ax.axhline(0, color=AXIS, lw=1); ax.set_title(METRICS[m], loc="left", fontsize=8.2); ax.set_xticks([1, 5, 10]); ax.grid(axis="x", visible=False); ax.yaxis.set_major_locator(MaxNLocator(4))
    fig.legend(handles=[Line2D([], [], marker="o", color=ASD, ls="", label="ASD"), Line2D([], [], marker="o", color=TD, ls="", label="TD")], frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(.05, -.05))
    fig.suptitle("Per-image error: tau = 2 minus tau = 1, for each of the 10 seeds. Above zero, tau = 1 is closer to real.", x=.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_seed_stability.png", bbox_inches="tight"); plt.close(fig)
    order = ["sacc_amp_norm_mean", "revisit_rate", "dispersion", "dur_mean", "n_fix", "n_fix_window", "fix_rate_per_s", "total_dur", "sacc_horizontal_frac", "sacc_dir_resultant"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    tr = {(t["group"], t["metric"]): t for t in transfer}
    yy = np.arange(len(order))[::-1]
    for ax, g, c in ((axs[0], "ASD", ASD), (axs[1], "TD", TD)):
        for i, m in enumerate(order):
            t = tr[(g, m)]
            ax.plot([t["validation_rel_pct"], t["external_rel_pct"]], [yy[i], yy[i]], color=AXIS, lw=1)
            ax.plot(t["validation_rel_pct"], yy[i], "o", mfc=SURF, mec=c, ms=6, mew=1.3); ax.plot(t["external_rel_pct"], yy[i], "o", color=c, ms=6, mec=SURF)
        ax.axvline(0, color=AXIS, lw=1); ax.set_title(f"{g}: change in per-image error", loc="left"); ax.set_xlabel("% of tau = 2 error; above 0, tau = 1 is closer"); ax.grid(axis="y", visible=False)
    axs[0].set_yticks(yy); axs[0].set_yticklabels([METRICS[m] for m in order], color=INK2, fontsize=8)
    fig.legend(handles=[Line2D([], [], marker="o", mfc=SURF, mec=INK2, ls="", label="Saliency4ASD validation"), Line2D([], [], marker="o", color=INK2, ls="", label="Gaze4ASD")], frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(.25, -.09))
    fig.tight_layout(); fig.savefig(f"{FIG}/fig5_transfer_from_validation.png", bbox_inches="tight"); plt.close(fig)

    # ---------------- digest
    print("checks", json.dumps({k: v for k, v in checks.items() if k not in ("images", "normalization")})[:1100])
    print("participant-level columns", list(participant[0].keys()) if participant else None)
    for r in participant:
        print({k: r[k] for k in list(r)[:12]})
    print("circular means", {k: round(v, 1) for k, v in circ.items()})
    print("\n--- PRESPECIFIED (+ favours tau1)")
    for r in pres:
        print(f"{r['group']:4s}{r['metric']:22s} real={r['real']:.4f} t1={r['tau1']:.4f} t2={r['tau2']:.4f} bias {r['bias_tau1']:+.4f}/{r['bias_tau2']:+.4f} MAE {r['mae_tau1']:.4f}/{r['mae_tau2']:.4f} gain={r['mean']:+.4f}{[round(x, 4) for x in r['ci']]} ({r['rel_pct_of_tau2']:+.1f}%) p={r['p']:.4f} holm4={r['p_holm_prespecified']:.4f} I+{r['n_pos']}/30 seeds+{r['seeds']['n_pos']}/10 meets={r['meets_transfer_criterion']}")
    for mode in MODES:
        print(f"\n########## {mode}")
        print("--- direct, tau2 - tau1")
        for r in res[mode]["direct"]:
            print(f"{r['group']:4s}{r['metric']:22s} real={r['real']:.4f} t1={r['tau1']:.4f}(sd{r['tau1_seed_sd']:.4f}) t2={r['tau2']:.4f}(sd{r['tau2_seed_sd']:.4f}) d={r['mean']:+.4f}{[round(x, 4) for x in r['ci']]} holm={r['p_holm']:.4f} I+{r['n_pos']}/30 seeds+{r['seeds']['n_pos']}/10 noise={r['noise_split_half_mean']:+.4f} eff/sd={r['effect_to_seed_sd']:.1f}")
        print("--- realism (+ favours tau1)")
        for r in res[mode]["realism"]:
            print(f"{r['group']:4s}{r['metric']:22s} bias {r['bias_tau1']:+.4f}/{r['bias_tau2']:+.4f} absbias {r['abs_bias_gain']:+.4f}{[round(x, 4) for x in r['abs_bias_gain_ci']]} MAE {r['mae_tau1']:.4f}/{r['mae_tau2']:.4f} gain={r['mean']:+.4f}{[round(x, 4) for x in r['ci']]} ({r['rel_pct_of_tau2']:+.1f}%) holm={r['p_holm']:.4f} I+{r['n_pos']}/30 seeds+{r['seeds']['n_pos']}/10")
        print("--- distances (+ favours tau1)")
        for r in res[mode]["distances"]:
            print(f"{r['group']:4s}{r['metric']:8s} t1={r['tau1']:.4f} t2={r['tau2']:.4f} gain={r['mean']:+.4f}{[round(x, 4) for x in r['ci']]} ({r['rel_pct_of_tau2']:+.1f}%) holm={r['p_holm']:.4f} seeds+{r['seeds']['n_pos']}/10 global {r['global_tau1']['mean']:.4f}/{r['global_tau2']['mean']:.4f}")
        print("--- group differences")
        for d in res[mode]["group_differences"]:
            print(f"{d['metric']:22s} real={d['real']:+.4f}{[round(x, 4) for x in d['real_ci']]} t1={d['tau1']:+.4f}{[round(x, 4) for x in d['tau1_ci']]} t2={d['tau2']:+.4f}{[round(x, 4) for x in d['tau2_ci']]} ratio {d['ratio_tau1']:+.2f}/{d['ratio_tau2']:+.2f} sign {d['tau1_seeds_real_sign']}/{d['tau2_seeds_real_sign']} gap {d['gap_tau1']:.4f}/{d['gap_tau2']:.4f} gain={d['gap_gain']:+.4f}{[round(x, 4) for x in d['gap_gain_ci']]} r {d['r_tau1']:.2f}/{d['r_tau2']:.2f} t2-t1 {d['tau2_minus_tau1']['mean']:+.4f}{[round(x, 4) for x in d['tau2_minus_tau1']['ci']]} seeds+{d['tau2_minus_tau1_seeds']['n_pos']}/10")
    print("\n--- window matched (+ favours tau1)")
    for r in win_rows:
        print(f"{r['group']:4s}{r['metric']:22s} real={r['real']:.4f} t1={r['tau1']:.4f} t2={r['tau2']:.4f} bias {r['bias_tau1']:+.4f}/{r['bias_tau2']:+.4f} absbias {r['abs_bias_gain']:+.4f}{[round(x, 4) for x in r['abs_bias_gain_ci']]} gain={r['mean']:+.4f}{[round(x, 4) for x in r['ci']]} holm={r['p_holm']:.4f} seeds+{r['seeds']['n_pos']}/10")
    for d in win_gaps:
        print(f"gap {d['metric']:22s} real={d['real']:+.4f} t1={d['tau1']:+.4f}{[round(x, 4) for x in d['tau1_ci']]} t2={d['tau2']:+.4f}{[round(x, 4) for x in d['tau2_ci']]} sign {d['tau1_seeds_real_sign']}/{d['tau2_seeds_real_sign']}")
    print("\n--- multimatch (+ favours tau1)")
    for r in mmres:
        print(f"{r['group']:4s}{r['metric'][3:]:10s} RvR={r['real_vs_real']:.4f} t1={r['tau1']:.4f} t2={r['tau2']:.4f} gain={r['mean']:+.4f}{[round(x, 4) for x in r['ci']]} holm={r['p_holm']:.4f} I+{r['n_pos']}/30 seeds+{r['seeds']['n_pos']}/10 | own-other t1 {r['own_minus_other_tau1']['mean']:+.4f}{[round(x, 4) for x in r['own_minus_other_tau1']['ci']]} t2 {r['own_minus_other_tau2']['mean']:+.4f}{[round(x, 4) for x in r['own_minus_other_tau2']['ci']]}")
    print("\n--- transfer")
    for t in transfer:
        print(f"{t['group']:4s}{t['metric']:22s} val {t['validation_gain']:+.4f} ({t['validation_rel_pct']:+.1f}%) ext {t['external_gain']:+.4f} ({t['external_rel_pct']:+.1f}%) same={t['same_direction']}")
    print("ANALYZE_DONE")


if __name__ == "__main__":
    {"run": run, "analyze": analyze}[sys.argv[1]]()
