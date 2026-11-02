"""
Controlled Phase 1 vs Phase 2 comparison on corrected Gaze4ASD (fit-to-height, Policy A).
No training. Both checkpoints are read-only; any zero or modified embedding exists only in memory.

  python training/compare_phase1_phase2.py run       evaluate, generate, compute MultiMatch
  python training/compare_phase1_phase2.py verify    label and loss checks, Saliency4ASD test as a reference
  python training/compare_phase1_phase2.py analyze   statistics, tables, figures

Improvement is always oriented so that positive favours Phase 2.
All statistics are exploratory. The mapping is strongly supported but partly inferred.
"""
from __future__ import annotations
import csv, gzip, json, os, sys, time, zlib
sys.path[:0] = [".", "tppgaze"]
import numpy as np

OUT = "training/runs/external_gaze4asd_v2/phase1_vs_phase2"
RAW = OUT + "/raw"
FIG = OUT + "/comparison_figures"
DATA = "external_data/Gaze4ASD_processed_v2/fit_height_policy_a/gaze4asd_fit_height_policy_a.json"
PHASE1 = "training/runs/baseline/best_model.pth"
PHASE2 = "training/runs/phase2_embedding_only/best_model_conditioned.pth"
PRIOR = "training/runs/external_gaze4asd_v2/fit_height_policy_a/per_trial_nll.csv"
PRIOR_REP = "training/runs/external_gaze4asd_v2/comparison/generation_replicates.json"
BATCH, N_REP, N_SIM, TEMP, T_END, MAX_PAIRS, N_BOOT = 128, 10, 20, 2.0, 3.0, 30, 10000
METRICS = {"dur_mean": "Fixation duration (s)", "n_fix": "Fixation count", "n_fix_window": "Fixation count, real time window", "total_dur": "Total fixation time (s)", "fix_rate_per_s": "Fixation rate (per s)",
           "sacc_amp_norm_mean": "Saccade amplitude (/diagonal)", "sacc_horizontal_frac": "Horizontal saccade fraction", "dispersion": "Spatial dispersion (/diagonal)", "revisit_rate": "Revisit rate", "sacc_dir_resultant": "Direction concentration"}
MM = ["mm_shape", "mm_length", "mm_direction", "mm_position", "mm_duration"]
DIR_BINS = np.linspace(-np.pi, np.pi, 13)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(f"{OUT}/evaluation_log.txt", "a", encoding="utf-8") as f:
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
    from training.dataset import load_scanpaths
    from training import metrics as M, model_utils as mu
    from training.conditioned import ConditionedTPPGaze
    from training.generate_cond import sample_conditioned, to_scanpaths
    from training.external_gaze4asd_loader import StimulusBank, load_external_scanpaths, summarize_records, collate_records
    from training.eval_external_v2 import IMG_DIR, BASE_CFG, denorm, metric_row, sha256

    if os.path.exists(OUT):
        raise SystemExit(f"Refusing to overwrite: {OUT}")
    os.makedirs(RAW); os.makedirs(FIG)
    device = torch.device("cuda")
    t0 = time.time()
    h1, h2 = sha256(PHASE1), sha256(PHASE2)
    log(f"Phase 1 {PHASE1} sha256={h1}")
    log(f"Phase 2 {PHASE2} sha256={h2}")

    split = json.load(open("training/runs/baseline/split.json"))
    sal = load_scanpaths("processed_data/asd_scanpaths.json", "processed_data/td_scanpaths.json")
    mean, std = mu.compute_log_inter_time_stats([s for s in sal if s["image_id"] in set(map(int, split["train"]))])
    stored = json.load(open("training/runs/phase2_embedding_only/summary_full.json"))
    assert abs(mean - stored["mean_log_inter_time"]) < 1e-9 and abs(std - stored["std_log_inter_time"]) < 1e-9
    cdim = int(OmegaConf.load(BASE_CFG).metadata.size + OmegaConf.load(BASE_CFG).context.size)

    base1, _ = mu.build_model(BASE_CFG, PHASE1, device, mean, std, used_cfg_out="/tmp/p1p2_a.yaml")
    base1.eval()
    p1 = ConditionedTPPGaze(base1, cdim).to(device).eval()  # in-memory zero embedding, so the sampler code is shared
    base2, _ = mu.build_model(BASE_CFG, PHASE1, device, mean, std, used_cfg_out="/tmp/p1p2_b.yaml")
    p2 = ConditionedTPPGaze(base2, cdim).to(device)
    p2.load_state_dict(torch.load(PHASE2, map_location=device), strict=True)
    mu._fix_unregistered_device(p2.base, device)
    p2.eval()
    for m in (p1, p2):
        for q in m.parameters():
            q.requires_grad = False
    log("both checkpoints loaded (strict)")

    sd1, sd2 = base1.state_dict(), p2.base.state_dict()
    diff = [k for k in sd1 if not torch.equal(sd1[k], sd2[k])]
    W = p2.group_emb.weight.detach()
    weight_check = dict(n_tensors=len(sd1), tensors_differing=len(diff), differing=diff[:10], asd_norm=float(W[0].norm()), td_norm=float(W[1].norm()), cosine=float(torch.dot(W[0], W[1]) / (W[0].norm() * W[1].norm())),
                        shared_norm=float(W.mean(0).norm()), half_difference_norm=float(((W[0] - W[1]) / 2).norm()))
    log("base weight identity: " + json.dumps(weight_check))

    bank = StimulusBank(IMG_DIR)
    recs = load_external_scanpaths(DATA, bank)
    info = summarize_records(recs, bank)
    assert (info["participants_ASD"], info["participants_TD"], info["trials_ASD"], info["trials_TD"], info["stimuli"]) == (33, 133, 771, 3605, 30)
    log("data: " + json.dumps({k: v for k, v in info.items() if k not in ("image_sizes",)}, default=list))
    images = sorted({r["image_id"] for r in recs}, key=lambda s: int(s.replace("top_image_", "").replace(".png", "")))

    cache = {}
    @torch.no_grad()
    def meta(model, ids):
        key = id(model)
        for i in set(ids):
            if (key, i) not in cache:
                cache[(key, i)] = model.base.get_metadata(bank.tensor(i).unsqueeze(0).to(device))
        return torch.cat([cache[(key, i)] for i in ids], 0)

    @torch.no_grad()
    def score(model, emb, order=None):
        """Same batches and same code for every variant. `emb` maps group ids to a context shift, or is None."""
        idx = list(range(len(recs))) if order is None else list(order)
        out = [None] * len(recs)
        for st in range(0, len(idx), BATCH):
            ii = idx[st:st + BATCH]
            b = [recs[i] for i in ii]
            c = collate_records(b, device)
            f = model.base.get_features(c["inter_times"], c["marks"])
            ctx = model.base.merge.forward([model.base.get_context(f, True), meta(model, [r["image_id"] for r in b])])
            if emb is not None:
                ctx = ctx + emb(c["group_ids"]).unsqueeze(1)
            it = torch.where(c["mask"].bool(), c["inter_times"], torch.ones_like(c["inter_times"]))
            lpt = model.base.inter_time_dist.forward(ctx, None).log_prob(it) * c["mask"]
            lpp = model.base.marks_dist.forward(ctx, None).log_prob(c["marks"]) * c["mask"]
            for j, i in enumerate(ii):
                out[i] = (int(c["mask"][j].sum()), float(lpt[j].sum()), float(lpp[j].sum()))
        return out

    var = {"phase1": score(p1, None), "phase2_true": score(p2, lambda g: p2.group_emb(g)), "phase2_swapped": score(p2, lambda g: p2.group_emb(1 - g)),
           "phase2_shared_mean": score(p2, lambda g: W.mean(0, keepdim=True).expand(len(g), -1)), "phase2_base_no_embedding": score(p2, None)}
    # gates
    prior = list(csv.DictReader(open(PRIOR, encoding="utf-8")))
    assert len(prior) == len(recs)
    dmax = max(max(abs(float(p["sum_lp_duration"]) - v[1]), abs(float(p["sum_lp_position"]) - v[2])) for p, v in zip(prior, var["phase2_true"]))
    d0 = max(abs(a[1] - b[1]) + abs(a[2] - b[2]) for a, b in zip(var["phase1"], var["phase2_base_no_embedding"]))
    log(f"gate: Phase 2 trial log-likelihoods vs earlier run, max abs diff={dmax:.6f}; Phase 1 vs Phase 2 without embedding, max abs diff={d0:.6f}")
    if dmax > 1e-3 or d0 > 1e-3 or diff:
        raise SystemExit("GATE FAILED")
    with open(f"{RAW}/per_trial_nll_all_variants.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "participant", "image_id", "n_events"] + [f"{v}_{k}" for v in var for k in ("lp_duration", "lp_position")])
        for i, r in enumerate(recs):
            w.writerow([r["group"], r["participant"], r["image_id"], var["phase1"][i][0]] + [x for v in var for x in var[v][i][1:]])
    shuf = []
    for sd in (1, 2, 3):
        o = np.random.default_rng(sd).permutation(len(recs))
        a, b = score(p1, None, o), score(p2, lambda g: p2.group_emb(g), o)
        n = sum(x[0] for x in a)
        shuf.append(dict(seed=sd, phase1_nll=-sum(x[1] + x[2] for x in a) / n, phase2_nll=-sum(x[1] + x[2] for x in b) / n))
    json.dump(dict(weight_check=weight_check, gate_max_diff_prior=dmax, gate_max_diff_base=d0, shuffled_batches=shuf, seeds=dict(phase2="100000*(rep+1)+1000*(rank+1)+gid", phase1="100000*(rep+1)+1000*(rank+1)+2")), open(f"{RAW}/run_checks.json", "w"), indent=2)
    log("shuffled batches: " + json.dumps(shuf))

    # ---------------- real summaries
    def angles(xy):
        v = np.diff(xy, axis=0)
        return np.arctan2(v[:, 1], v[:, 0])
    real = {"ASD": {}, "TD": {}}
    for r in recs:
        W_, H_ = bank.dims(r["image_id"])
        sp = np.asarray(r["scanpath"])
        real[r["group"]].setdefault(r["image_id"], []).append(dict(xy=denorm(sp, W_, H_), dur=sp[:, 2]))
    mkeys = [k for k in METRICS if k != "n_fix_window"]
    arrs = {}
    win = {g: np.zeros(len(images)) for g in real}
    for g in real:
        arrs[f"real_{g}_hist"] = np.zeros((len(images), 12))
        for m in METRICS:
            arrs[f"real_{g}_{m}"] = np.zeros(len(images))
        for k, iid in enumerate(images):
            W_, H_ = bank.dims(iid)
            rows = [metric_row("r", iid, "", x["xy"], x["dur"], W_, H_) for x in real[g][iid]]
            for m in mkeys:
                arrs[f"real_{g}_{m}"][k] = np.nanmean([r[m] for r in rows])
            arrs[f"real_{g}_n_fix_window"][k] = arrs[f"real_{g}_n_fix"][k]
            win[g][k] = np.mean([x["dur"].sum() for x in real[g][iid]])
            arrs[f"real_{g}_hist"][k] = np.histogram(np.concatenate([angles(x["xy"]) for x in real[g][iid]]), bins=DIR_BINS)[0]

    # ---------------- generation: same sampler, settings and code path for all sets
    sets = {"P1": (p1, 0, 2), "P2_ASD": (p2, 0, 0), "P2_TD": (p2, 1, 1)}
    for s in sets:
        arrs[f"{s}_hist"] = np.zeros((N_REP, len(images), 12))
        for m in mkeys:
            arrs[f"{s}_{m}"] = np.zeros((N_REP, len(images)))
    arrs["P1_n_fix_window_ASD"] = np.zeros((N_REP, len(images))); arrs["P1_n_fix_window_TD"] = np.zeros((N_REP, len(images)))
    arrs["P2_ASD_n_fix_window"] = np.zeros((N_REP, len(images))); arrs["P2_TD_n_fix_window"] = np.zeros((N_REP, len(images)))
    mm_rows, gen_store = [], []
    targets = [("RvG", "ASD", "P1"), ("RvG", "TD", "P1"), ("RvG", "ASD", "P2_ASD"), ("RvG", "TD", "P2_TD"), ("RvG_wrong", "ASD", "P2_TD"), ("RvG_wrong", "TD", "P2_ASD")]

    def mm_mean(tag, la, lb, same, diag):
        pairs = [(i, j) for i in range(len(la)) for j in range(len(lb)) if not (same and j <= i)]
        rng = np.random.default_rng([0, zlib.crc32(tag.encode())])
        acc = {k: [] for k in MM}
        for q in rng.permutation(len(pairs))[:MAX_PAIRS]:
            a, b = la[pairs[q][0]], lb[pairs[q][1]]
            r = M.multimatch(a["xy"], a["dur"], b["xy"], b["dur"], diag)
            if r:
                for k in MM:
                    acc[k].append(r[k])
        return {k: float(np.mean(acc[k])) for k in MM}, len(acc[MM[0]])

    for k, iid in enumerate(images):
        W_, H_ = bank.dims(iid)
        diag = float(np.hypot(W_, H_))
        for g in real:
            v, n = mm_mean(f"RvR|{g}|{iid}", real[g][iid], real[g][iid], True, diag)
            mm_rows.append(dict(kind="RvR", target=g, model="real", rep=-1, image_id=iid, n_pairs=n, **v))
    for rep in range(N_REP):
        for k, iid in enumerate(images):
            W_, H_ = bank.dims(iid)
            diag = float(np.hypot(W_, H_))
            cur = {}
            for s, (model, gid, seedg) in sets.items():
                torch.manual_seed(100000 * (rep + 1) + 1000 * (k + 1) + seedg)
                it, mk, ms = sample_conditioned(model, bank.tensor(iid), gid, t_end=T_END, n_sim=N_SIM, temperature=TEMP)
                L = []
                rows = []
                ang = []
                for sim, spn in enumerate(to_scanpaths(it, mk, ms)):
                    if len(spn) < 1:
                        continue
                    xy = denorm(spn, W_, H_)
                    L.append(dict(xy=xy, dur=spn[:, 2]))
                    rows.append(metric_row("g", iid, "", xy, spn[:, 2], W_, H_))
                    if len(xy) >= 2:
                        ang.append(angles(xy))
                    gen_store.append(dict(set=s, rep=rep, image_id=iid, sim=sim, scanpath=spn.round(5).tolist()))
                cur[s] = L
                for m in mkeys:
                    arrs[f"{s}_{m}"][rep, k] = np.nanmean([r[m] for r in rows])
                arrs[f"{s}_hist"][rep, k] = np.histogram(np.concatenate(ang), bins=DIR_BINS)[0]
                on = [np.r_[0.0, np.cumsum(x["dur"])[:-1]] for x in L]
                if s == "P1":
                    for g in real:
                        arrs[f"P1_n_fix_window_{g}"][rep, k] = np.mean([(o < win[g][k]).sum() for o in on])
                else:
                    arrs[f"{s}_n_fix_window"][rep, k] = np.mean([(o < win[s.split('_')[1]][k]).sum() for o in on])
            for kind, g, s in targets:
                v, n = mm_mean(f"{kind}|{g}|{s}|{rep}|{iid}", real[g][iid], cur[s], False, diag)
                mm_rows.append(dict(kind=kind, target=g, model=s, rep=rep, image_id=iid, n_pairs=n, **v))
        log(f"replicate {rep + 1}/{N_REP} done ({time.time() - t0:.0f}s)")

    np.savez_compressed(f"{RAW}/generated_and_real_summaries.npz", images=np.array(images), **arrs)
    write_rows(f"{RAW}/multimatch_per_image.csv", mm_rows)
    with gzip.open(f"{RAW}/generated_scanpaths.json.gz", "wt", encoding="utf-8") as f:
        json.dump(gen_store, f)
    # Phase 2 replicates must match the earlier run, which used the same seeds
    prior_rep = {r["metric"]: r for r in json.load(open(PRIOR_REP))}
    chk = {m: dict(new=float((arrs[f"P2_ASD_{m}"] - arrs[f"P2_TD_{m}"]).mean()), prior=prior_rep[m]["pooled_diff"]) for m in prior_rep if m in mkeys}
    log("Phase 2 replicate reproduction: max abs diff=" + str(max(abs(x["new"] - x["prior"]) for x in chk.values())))
    assert sha256(PHASE1) == h1 and sha256(PHASE2) == h2
    log(f"checkpoints unchanged. total {time.time() - t0:.0f}s. RUN_DONE")


# ======================================================================= analyze
def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    rng = np.random.default_rng(0)
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    # ---------------- NLL
    T = list(csv.DictReader(open(f"{RAW}/per_trial_nll_all_variants.csv", encoding="utf-8")))
    parts = sorted({r["participant"] for r in T}); imgs = sorted({r["image_id"] for r in T}, key=lambda s: int(s.split("_")[-1].split(".")[0]))
    pix = {p: i for i, p in enumerate(parts)}; iix = {p: i for i, p in enumerate(imgs)}
    group_of = {r["participant"]: r["group"] for r in T}
    pg = np.array([group_of[p] == "ASD" for p in parts])
    n = np.array([int(r["n_events"]) for r in T], float)
    tp = np.array([pix[r["participant"]] for r in T]); ti = np.array([iix[r["image_id"]] for r in T]); isasd = np.array([r["group"] == "ASD" for r in T])
    def col(v, k):
        return np.array([float(r[f"{v}_lp_{k}"]) for r in T])
    variants = ["phase1", "phase2_true", "phase2_swapped", "phase2_shared_mean"]
    LP = {v: {"duration": col(v, "duration"), "position": col(v, "position")} for v in variants}
    LP2 = {v: {**LP[v], "joint": LP[v]["duration"] + LP[v]["position"]} for v in variants}
    def nll(v, kind, sel):
        return float(-LP2[v][kind][sel].sum() / n[sel].sum())

    def cluster_tests(a, b, kind, sel):
        """NLL(a) minus NLL(b): positive favours b. Participant bootstrap stratified by group, image bootstrap, crossed bootstrap."""
        d = LP2[b][kind] - LP2[a][kind]
        C = np.zeros((len(parts), len(imgs), 2))
        np.add.at(C, (tp[sel], ti[sel], 0), n[sel]); np.add.at(C, (tp[sel], ti[sel], 1), d[sel])
        P = C.sum(1); I = C.sum(0)
        obs = float(P[:, 1].sum() / P[:, 0].sum())
        idx_a = np.where(pg & (P[:, 0] > 0))[0]; idx_t = np.where(~pg & (P[:, 0] > 0))[0]
        parr = []
        if len(idx_a):
            parr.append(P[idx_a[rng.integers(0, len(idx_a), (N_BOOT, len(idx_a)))]].sum(1))
        if len(idx_t):
            parr.append(P[idx_t[rng.integers(0, len(idx_t), (N_BOOT, len(idx_t)))]].sum(1))
        ps = sum(parr)
        ib = I[rng.integers(0, len(imgs), (N_BOOT, len(imgs)))].sum(1)
        have = np.where(P[:, 0] > 0)[0]
        wp = np.zeros((N_BOOT, len(parts)))
        wi = rng.multinomial(len(imgs), np.ones(len(imgs)) / len(imgs), N_BOOT).astype(float)
        for arr in ([idx_a] if len(idx_a) else []) + ([idx_t] if len(idx_t) else []):
            wp[:, arr] = rng.multinomial(len(arr), np.ones(len(arr)) / len(arr), N_BOOT)
        cn = np.einsum("bi,bi->b", wp @ C[:, :, 0], wi)
        cd = np.einsum("bi,bi->b", wp @ C[:, :, 1], wi)
        flips = rng.choice([-1.0, 1.0], (N_BOOT, len(have)))
        pm = (flips * P[have, 1]).sum(1) / P[have, 0].sum()
        ip = (rng.choice([-1.0, 1.0], (N_BOOT, len(imgs))) * I[:, 1]).sum(1) / I[:, 0].sum()
        return dict(improvement=obs, participant_ci=ci(ps[:, 1] / ps[:, 0]), image_ci=ci(ib[:, 1] / ib[:, 0]), crossed_ci=ci(cd / cn),
                    participants_favouring=int((P[have, 1] > 0).sum()), participants=len(have), images_favouring=int((I[:, 1] > 0).sum()), images=len(imgs),
                    participant_signflip_p=float((np.sum(np.abs(pm) >= abs(obs) - 1e-15) + 1) / (N_BOOT + 1)), image_signflip_p=float((np.sum(np.abs(ip) >= abs(obs) - 1e-15) + 1) / (N_BOOT + 1)))

    sels = {"overall": np.ones(len(T), bool), "ASD": isasd, "TD": ~isasd}
    rows, boot = [], {}
    for g, sel in sels.items():
        boot[g] = {}
        for k in ("joint", "duration", "position"):
            t = cluster_tests("phase1", "phase2_true", k, sel)
            a, b = nll("phase1", k, sel), nll("phase2_true", k, sel)
            boot[g][k] = t
            rows.append(dict(group=g, component=k, n_trials=int(sel.sum()), n_events=int(n[sel].sum()), phase1_nll=a, phase2_nll=b, relative_pct=100 * t["improvement"] / abs(a), **t))
    o = sorted(range(len(rows)), key=lambda i: rows[i]["participant_signflip_p"]); run_ = 0.0
    for r, i in enumerate(o):
        run_ = max(run_, min(1.0, (len(rows) - r) * rows[i]["participant_signflip_p"]))
        rows[i]["participant_p_holm"] = run_
    write_rows(f"{OUT}/group_nll_comparison.csv", rows)
    checks = json.load(open(f"{RAW}/run_checks.json"))
    # attribution: is the gain group information or a shared shift?
    attr = {}
    for g, sel in sels.items():
        attr[g] = {k: dict(phase1=nll("phase1", k, sel), shared_mean=nll("phase2_shared_mean", k, sel), true=nll("phase2_true", k, sel), swapped=nll("phase2_swapped", k, sel),
                           shared_vs_phase1=cluster_tests("phase1", "phase2_shared_mean", k, sel), true_vs_shared=cluster_tests("phase2_shared_mean", "phase2_true", k, sel), true_vs_swapped=cluster_tests("phase2_swapped", "phase2_true", k, sel)) for k in ("joint", "duration", "position")}
    json.dump(boot, open(f"{OUT}/participant_bootstrap.json", "w"), indent=2)
    json.dump(dict(data=DATA, batch_size=BATCH, trials=len(T), events=int(n.sum()), by_group={r["group"] + "_" + r["component"]: {k: r[k] for k in ("phase1_nll", "phase2_nll", "improvement", "relative_pct", "participant_ci", "image_ci", "crossed_ci", "participant_p_holm")} for r in rows}, attribution=attr, checks=checks), open(f"{OUT}/overall_nll_comparison.json", "w"), indent=2)

    # ---------------- behaviour
    Z = np.load(f"{RAW}/generated_and_real_summaries.npz")
    def gen(model, g, m):
        if m == "n_fix_window":
            return Z[f"P1_n_fix_window_{g}"] if model == "P1" else Z[f"P2_{g}_n_fix_window"]
        return Z[f"P1_{m}"] if model == "P1" else Z[f"P2_{g}_{m}"]
    def signflip(d):
        pm = (rng.choice([-1.0, 1.0], (N_BOOT, len(d))) * d).mean(1)
        return float((np.sum(np.abs(pm) >= abs(d.mean()) - 1e-15) + 1) / (N_BOOT + 1))
    def compare(e1_pool, e2_pool, e1_rep, e2_rep, extra):
        d = e1_pool - e2_pool
        b = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
        sd = e1_rep.mean(1) - e2_rep.mean(1)
        return dict(**extra, phase1_mean_abs_error=float(e1_pool.mean()), phase2_mean_abs_error=float(e2_pool.mean()), improvement=float(d.mean()), relative_pct=float(100 * d.mean() / e1_pool.mean()), image_ci=ci(b), image_p=signflip(d),
                    images_favouring_phase2=int((d > 0).sum()), seed_mean=float(sd.mean()), seed_sd=float(sd.std(ddof=1)), seed_min=float(sd.min()), seed_max=float(sd.max()), seeds_favouring_phase2=int((sd > 0).sum()), seeds=len(sd))
    gc = []
    for g in ("ASD", "TD"):
        for m, lab in METRICS.items():
            R = Z[f"real_{g}_{m}"]; g1, g2 = gen("P1", g, m), gen("P2", g, m)
            gc.append(compare(np.abs(g1.mean(0) - R), np.abs(g2.mean(0) - R), np.abs(g1 - R), np.abs(g2 - R), dict(metric=m, label=lab, group=g, real_mean=float(R.mean()), phase1_gen_mean=float(g1.mean()), phase2_gen_mean=float(g2.mean()), phase1_signed_bias=float((g1.mean(0) - R).mean()), phase2_signed_bias=float((g2.mean(0) - R).mean()))))
        Hr = Z[f"real_{g}_hist"]; Hr = Hr / Hr.sum(1, keepdims=True)
        def tv(H):
            return 0.5 * np.abs(H / H.sum(-1, keepdims=True) - Hr).sum(-1)
        h1, h2 = Z["P1_hist"], Z[f"P2_{g}_hist"]
        gc.append(compare(tv(h1.sum(0)), tv(h2.sum(0)), tv(h1), tv(h2), dict(metric="sacc_direction_tv", label="Saccade direction distribution (total variation)", group=g, real_mean=0.0, phase1_gen_mean=float(tv(h1.sum(0)).mean()), phase2_gen_mean=float(tv(h2.sum(0)).mean()), phase1_signed_bias=None, phase2_signed_bias=None)))
    o = sorted(range(len(gc)), key=lambda i: gc[i]["image_p"]); run_ = 0.0
    for r, i in enumerate(o):
        run_ = max(run_, min(1.0, (len(gc) - r) * gc[i]["image_p"]))
        gc[i]["image_p_holm"] = run_
    write_rows(f"{OUT}/generation_comparison.csv", gc)
    # group differences
    diffs = []
    for m, lab in METRICS.items():
        Dr = Z[f"real_ASD_{m}"] - Z[f"real_TD_{m}"]
        P2 = gen("P2", "ASD", m) - gen("P2", "TD", m)
        P1a, P1t = gen("P1", "ASD", m), gen("P1", "TD", m)
        null_rep = P1a[:5] - P1t[5:]
        null_pool = P1a[:5].mean(0) - P1t[5:].mean(0)
        d2 = P2.mean(0)
        idx = rng.integers(0, len(Dr), (N_BOOT, len(Dr)))
        gap_gain = np.abs((Dr - null_pool)[idx].mean(1)) - np.abs((Dr - d2)[idx].mean(1))
        rs = P2.mean(1); nr = null_rep.mean(1)
        diffs.append(dict(metric=m, label=lab, real_diff=float(Dr.mean()), real_diff_ci=ci(Dr[idx].mean(1)), phase2_diff=float(d2.mean()), phase2_diff_ci=ci(d2[idx].mean(1)), phase2_seed_min=float(rs.min()), phase2_seed_max=float(rs.max()),
                          phase1_null_diff=float(null_pool.mean()), phase1_null_seed_min=float(nr.min()), phase1_null_seed_max=float(nr.max()), same_direction=bool(np.sign(d2.mean()) == np.sign(Dr.mean())), phase2_to_real_ratio=float(d2.mean() / Dr.mean()),
                          seeds_same_sign_as_real=int((np.sign(rs) == np.sign(Dr.mean())).sum()), phase2_outside_phase1_null_range=bool(rs.min() > nr.max() or rs.max() < nr.min()), group_gap_error_phase1=float(abs((Dr - null_pool).mean())), group_gap_error_phase2=float(abs((Dr - d2).mean())), gap_improvement=float(abs((Dr - null_pool).mean()) - abs((Dr - d2).mean())), gap_improvement_ci=ci(gap_gain), per_image_r_real_vs_phase2=float(np.corrcoef(Dr, d2)[0, 1]), per_image_r_real_vs_phase1_null=float(np.corrcoef(Dr, null_pool)[0, 1])))
    json.dump(dict(match_to_real=gc, group_differences=diffs, notes="Mean absolute error between per-image generated and real means; positive improvement favours Phase 2. Phase 1 has no group input: its null difference compares independent seed sets (reps 1-5 vs 6-10)."), open(f"{OUT}/behavioral_comparison.json", "w"), indent=2)
    # ---------------- MultiMatch
    Mm = list(csv.DictReader(open(f"{RAW}/multimatch_per_image.csv", encoding="utf-8")))
    def cube(kind, target, model):
        c = np.full((N_REP, len(imgs), len(MM)), np.nan)
        for r in Mm:
            if r["kind"] == kind and r["target"] == target and r["model"] == model:
                c[int(r["rep"]), iix[r["image_id"]]] = [float(r[k]) for k in MM]
        return c
    def refr(target):
        c = np.full((len(imgs), len(MM)), np.nan)
        for r in Mm:
            if r["kind"] == "RvR" and r["target"] == target:
                c[iix[r["image_id"]]] = [float(r[k]) for k in MM]
        return c
    mmres = []
    for g, m2 in (("ASD", "P2_ASD"), ("TD", "P2_TD")):
        c1, c2 = cube("RvG", g, "P1"), cube("RvG", g, m2)
        w = cube("RvG_wrong", g, "P2_TD" if g == "ASD" else "P2_ASD"); rv = refr(g)
        for q, k in enumerate(MM):
            d = c2[:, :, q].mean(0) - c1[:, :, q].mean(0)
            b = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
            sd = c2[:, :, q].mean(1) - c1[:, :, q].mean(1)
            dw = c2[:, :, q].mean(0) - w[:, :, q].mean(0)
            mmres.append(dict(metric=k, group=g, real_vs_real=float(rv[:, q].mean()), phase1=float(c1[:, :, q].mean()), phase2=float(c2[:, :, q].mean()), phase2_minus_phase1=float(d.mean()), image_ci=ci(b), image_p=signflip(d), images_favouring_phase2=int((d > 0).sum()), seed_min=float(sd.min()), seed_max=float(sd.max()), seeds_favouring_phase2=int((sd > 0).sum()),
                              own_condition_minus_other_condition=float(dw.mean()), own_vs_other_ci=ci(dw[rng.integers(0, len(dw), (N_BOOT, len(dw)))].mean(1)), own_vs_other_p=signflip(dw)))
    for key in ("image_p", "own_vs_other_p"):
        o = sorted(range(len(mmres)), key=lambda i: mmres[i][key]); run_ = 0.0
        for r, i in enumerate(o):
            run_ = max(run_, min(1.0, (len(mmres) - r) * mmres[i][key]))
            mmres[i][key + "_holm"] = run_
    json.dump(mmres, open(f"{OUT}/multimatch_comparison.json", "w"), indent=2)

    # ---------------- does the embedding carry group-specific information?
    ia, it_ = np.where(pg)[0], np.where(~pg)[0]
    Sa, St, Si = rng.integers(0, len(ia), (N_BOOT, len(ia))), rng.integers(0, len(it_), (N_BOOT, len(it_))), rng.integers(0, len(imgs), (N_BOOT, len(imgs)))
    gi = (~isasd).astype(int)
    def per_unit(a, b, kind):
        d = LP2[b][kind] - LP2[a][kind]
        P = np.zeros((len(parts), 2)); np.add.at(P, (tp, 0), n); np.add.at(P, (tp, 1), d)
        I = np.zeros((2, len(imgs), 2)); np.add.at(I, (gi, ti, 0), n); np.add.at(I, (gi, ti, 1), d)
        return P, I
    def by_group_boot(P, I):
        x, y = P[ia[Sa]].sum(1), P[it_[St]].sum(1); u, v = I[0][Si].sum(1), I[1][Si].sum(1)
        return x[:, 1] / x[:, 0], y[:, 1] / y[:, 0], u[:, 1] / u[:, 0], v[:, 1] / v[:, 0]
    spec = dict(equal_group_weight_phase2_minus_phase1={}, right_label_minus_wrong_label_averaged_over_groups={}, fixed_embedding_vs_phase1={})
    for k in ("joint", "duration", "position"):
        P, I = per_unit("phase1", "phase2_true", k); pa, pt, ga, gt = by_group_boot(P, I)
        spec["equal_group_weight_phase2_minus_phase1"][k] = dict(improvement=0.5 * (P[ia, 1].sum() / P[ia, 0].sum() + P[it_, 1].sum() / P[it_, 0].sum()), participant_ci=ci(0.5 * (pa + pt)), image_ci=ci(0.5 * (ga + gt)))
        P, I = per_unit("phase2_swapped", "phase2_true", k); pa, pt, ga, gt = by_group_boot(P, I)
        ra, rt = P[ia, 1].sum() / P[ia, 0].sum(), P[it_, 1].sum() / P[it_, 0].sum()
        per_img = I[0, :, 1] / I[0, :, 0] + I[1, :, 1] / I[1, :, 0]
        spec["right_label_minus_wrong_label_averaged_over_groups"][k] = dict(asd_right_minus_wrong=ra, td_right_minus_wrong=rt, value=0.5 * (ra + rt), participant_ci=ci(0.5 * (pa + pt)), image_ci=ci(0.5 * (ga + gt)), images_favouring_right_label=int((per_img > 0).sum()))
        spec["fixed_embedding_vs_phase1"][k] = dict(asd_embedding_on_asd=cluster_tests("phase1", "phase2_true", k, isasd), asd_embedding_on_td=cluster_tests("phase1", "phase2_swapped", k, ~isasd), td_embedding_on_asd=cluster_tests("phase1", "phase2_swapped", k, isasd), td_embedding_on_td=cluster_tests("phase1", "phase2_true", k, ~isasd))
    merged = json.load(open(f"{OUT}/overall_nll_comparison.json")); merged["embedding_specificity"] = spec
    if os.path.exists(f"{RAW}/verification_checks.json"):
        merged["verification"] = json.load(open(f"{RAW}/verification_checks.json"))
    json.dump(merged, open(f"{OUT}/overall_nll_comparison.json", "w"), indent=2)

    # ---------------- figures
    ASD, TD, INK, INK2, MUTED, GRID, AXIS, SURF = "#eb6834", "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 10, "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .8, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "savefig.dpi": 200, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True})
    # 1. NLL improvement
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.7), sharey=True)
    colr = {"overall": INK, "ASD": ASD, "TD": TD}; ys = {"overall": 2, "ASD": 1, "TD": 0}
    for ax, k in zip(axs, ("joint", "duration", "position")):
        for g in sels:
            r = boot[g][k]
            ax.plot(r["crossed_ci"], [ys[g] - .14] * 2, color=colr[g], lw=1.1, alpha=.55); ax.plot(r["participant_ci"], [ys[g] + .14] * 2, color=colr[g], lw=1.8, solid_capstyle="round")
            ax.plot(r["improvement"], ys[g] + .14, "o", color=colr[g], ms=6, mec=SURF, mew=1)
        ax.axvline(0, color=AXIS, lw=1); ax.set_title(f"{k.capitalize()} NLL", loc="left"); ax.grid(axis="y", visible=False); ax.set_xlabel("Nats per fixation (positive favours Phase 2)"); ax.xaxis.set_major_locator(MaxNLocator(5))
    axs[0].set_yticks([2, 1, 0]); axs[0].set_yticklabels(["Overall", "ASD", "TD"], color=INK2); axs[0].set_ylim(-.6, 2.6)
    fig.suptitle("Upper line: 95% participant bootstrap. Lower, lighter line: participants and images resampled together.", x=.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_nll_improvement.png", bbox_inches="tight"); plt.close(fig)
    # 2. behavioural match
    ms = list(METRICS)
    fig, axs = plt.subplots(2, 5, figsize=(12, 4.8))
    for ax, m in zip(axs.ravel(), ms):
        for j, (g, c) in enumerate((("ASD", ASD), ("TD", TD))):
            R = Z[f"real_{g}_{m}"].mean(); a, b = gen("P1", g, m).mean(1), gen("P2", g, m).mean(1)
            ax.plot([j - .3, j + .3], [R, R], color=c, lw=2.2, solid_capstyle="round")
            ax.plot([j - .15] * 2, [a.min(), a.max()], color=INK2, lw=1); ax.plot(j - .15, a.mean(), "s", mfc=SURF, mec=INK2, ms=5.5, mew=1.2)
            ax.plot([j + .15] * 2, [b.min(), b.max()], color=c, lw=1); ax.plot(j + .15, b.mean(), "o", color=c, ms=6, mec=SURF, mew=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["ASD", "TD"], color=INK2); ax.set_xlim(-.6, 1.6); ax.set_title(METRICS[m], loc="left", fontsize=8.2); ax.grid(axis="x", visible=False)
    fig.legend(handles=[Line2D([], [], color=INK2, lw=2.2, label="Real group mean (bar)"), Line2D([], [], marker="s", mfc=SURF, mec=INK2, ls="", label="Phase 1 generic"), Line2D([], [], marker="o", color=INK2, ls="", label="Phase 2 own condition")], frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(.05, -.05))
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_generated_vs_real.png", bbox_inches="tight"); plt.close(fig)
    # 3. ASD-TD differences, one small multiple per metric because units differ
    fig, axs = plt.subplots(2, 5, figsize=(12, 3.8))
    for ax, d in zip(axs.ravel(), diffs):
        ax.plot(d["real_diff_ci"], [2, 2], color=INK, lw=1.4); ax.plot(d["real_diff"], 2, "o", color=INK, ms=6, mec=SURF)
        ax.plot(d["phase2_diff_ci"], [1, 1], color=INK2, lw=1.4); ax.plot(d["phase2_diff"], 1, "o", mfc=SURF, mec=INK2, ms=6, mew=1.2)
        ax.plot([d["phase1_null_seed_min"], d["phase1_null_seed_max"]], [0, 0], color=MUTED, lw=1.4); ax.plot(d["phase1_null_diff"], 0, "s", mfc=SURF, mec=MUTED, ms=5, mew=1.2)
        ax.axvline(0, color=AXIS, lw=1); ax.set_yticks([2, 1, 0]); ax.set_yticklabels(["Real", "Phase 2", "Phase 1 null"], color=INK2, fontsize=7.5); ax.set_ylim(-.6, 2.6); ax.grid(axis="y", visible=False); ax.set_title(d["label"], loc="left", fontsize=8); ax.tick_params(axis="x", labelsize=7); ax.xaxis.set_major_locator(MaxNLocator(4))
    fig.suptitle("ASD minus TD. Real and Phase 2: 95% image bootstrap. Phase 1: range across independent seed splits, expected near zero\n(except the real-window count, where Phase 1 is truncated at each group's own real fixation time).", x=.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_group_differences.png", bbox_inches="tight"); plt.close(fig)
    # 4. MultiMatch
    fig, axs = plt.subplots(1, 2, figsize=(9, 2.9), sharey=True)
    y = np.arange(len(MM))[::-1]
    for ax, g, c in ((axs[0], "ASD", ASD), (axs[1], "TD", TD)):
        rr = [r for r in mmres if r["group"] == g]
        for i, r in enumerate(rr):
            ax.plot(r["real_vs_real"], y[i] + .2, "D", color=MUTED, ms=5); ax.plot(r["phase1"], y[i], "s", mfc=SURF, mec=INK2, ms=6, mew=1.2); ax.plot(r["phase2"], y[i] - .2, "o", color=c, ms=6, mec=SURF, mew=1)
        ax.set_title(f"Similarity to real {g} (1 = identical)", loc="left"); ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y); axs[0].set_yticklabels([m.replace("mm_", "").capitalize() for m in MM], color=INK2)
    fig.legend(handles=[Line2D([], [], marker="D", color=MUTED, ls="", label="Real vs real"), Line2D([], [], marker="s", mfc=SURF, mec=INK2, ls="", label="Phase 1"), Line2D([], [], marker="o", color=INK2, ls="", label="Phase 2")], frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(.1, -.1))
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_multimatch.png", bbox_inches="tight"); plt.close(fig)
    # 5. each embedding applied to each real group
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.5), sharey=True)
    for ax, k in zip(axs, ("joint", "duration", "position")):
        f = spec["fixed_embedding_vs_phase1"][k]
        for y_, key, c, mk in ((3, "asd_embedding_on_asd", ASD, "o"), (2, "td_embedding_on_asd", TD, "s"), (1, "asd_embedding_on_td", ASD, "o"), (0, "td_embedding_on_td", TD, "s")):
            ax.plot(f[key]["participant_ci"], [y_, y_], color=c, lw=1.8, solid_capstyle="round"); ax.plot(f[key]["improvement"], y_, mk, color=c, ms=6, mec=SURF, mew=1)
        ax.axvline(0, color=AXIS, lw=1); ax.axhline(1.5, color=GRID, lw=.8); ax.set_title(f"{k.capitalize()} NLL", loc="left"); ax.grid(axis="y", visible=False); ax.xaxis.set_major_locator(MaxNLocator(5)); ax.set_xlabel("Gain over Phase 1, nats per fixation")
    axs[0].set_yticks([3, 2, 1, 0]); axs[0].set_yticklabels(["Real ASD", "Real ASD", "Real TD", "Real TD"], color=INK2); axs[0].set_ylim(-.6, 3.6)
    fig.legend(handles=[Line2D([], [], marker="o", color=ASD, ls="", label="ASD embedding applied"), Line2D([], [], marker="s", color=TD, ls="", label="TD embedding applied")], frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(.08, -.12))
    fig.tight_layout(); fig.savefig(f"{FIG}/fig5_embedding_by_group.png", bbox_inches="tight"); plt.close(fig)

    # ---------------- digest for review
    print("=== checks ===", json.dumps(checks["weight_check"]), checks["gate_max_diff_prior"], checks["gate_max_diff_base"], json.dumps(checks["shuffled_batches"]))
    print("\n=== NLL ===")
    for r in rows:
        print(f"{r['group']:8s}{r['component']:9s} P1={r['phase1_nll']:+.4f} P2={r['phase2_nll']:+.4f} imp={r['improvement']:+.4f} rel={r['relative_pct']:+.2f}% part{[round(x,4) for x in r['participant_ci']]} img{[round(x,4) for x in r['image_ci']]} cross{[round(x,4) for x in r['crossed_ci']]} P {r['participants_favouring']}/{r['participants']} I {r['images_favouring']}/{r['images']} p={r['participant_signflip_p']:.4f} holm={r['participant_p_holm']:.4f} ip={r['image_signflip_p']:.4f}")
    print("\n=== attribution ===")
    for g in attr:
        for k in ("joint", "position", "duration"):
            a = attr[g][k]
            print(f"{g:8s}{k:9s} P1={a['phase1']:+.4f} shared={a['shared_mean']:+.4f} true={a['true']:+.4f} swapped={a['swapped']:+.4f} | shared-P1 {a['shared_vs_phase1']['improvement']:+.4f}{[round(x,4) for x in a['shared_vs_phase1']['participant_ci']]} | true-shared {a['true_vs_shared']['improvement']:+.4f}{[round(x,4) for x in a['true_vs_shared']['participant_ci']]} cross{[round(x,4) for x in a['true_vs_shared']['crossed_ci']]} P {a['true_vs_shared']['participants_favouring']}/{a['true_vs_shared']['participants']} I {a['true_vs_shared']['images_favouring']}/30 | true-swapped {a['true_vs_swapped']['improvement']:+.4f}{[round(x,4) for x in a['true_vs_swapped']['participant_ci']]}")
    print("\n=== match to real (MAE; + favours P2) ===")
    for r in gc:
        print(f"{r['group']:4s}{r['metric']:22s} real={r['real_mean']:.3f} P1={r['phase1_gen_mean']:.3f} P2={r['phase2_gen_mean']:.3f} MAE {r['phase1_mean_abs_error']:.4f}->{r['phase2_mean_abs_error']:.4f} imp={r['improvement']:+.4f} ({r['relative_pct']:+.1f}%) {[round(x,4) for x in r['image_ci']]} I {r['images_favouring_phase2']}/30 p={r['image_p']:.4f} holm={r['image_p_holm']:.3f} seeds {r['seeds_favouring_phase2']}/10 [{r['seed_min']:+.4f},{r['seed_max']:+.4f}]")
    print("\n=== group differences ===")
    for d in diffs:
        print(f"{d['metric']:22s} real={d['real_diff']:+.4f}{[round(x,4) for x in d['real_diff_ci']]} P2={d['phase2_diff']:+.4f}{[round(x,4) for x in d['phase2_diff_ci']]} seeds[{d['phase2_seed_min']:+.4f},{d['phase2_seed_max']:+.4f}] same={d['same_direction']} ratio={d['phase2_to_real_ratio']:+.2f} seedsign={d['seeds_same_sign_as_real']}/10 | null={d['phase1_null_diff']:+.4f}[{d['phase1_null_seed_min']:+.4f},{d['phase1_null_seed_max']:+.4f}] outside={d['phase2_outside_phase1_null_range']} | gap {d['group_gap_error_phase1']:.4f}->{d['group_gap_error_phase2']:.4f} gain={d['gap_improvement']:+.4f}{[round(x,4) for x in d['gap_improvement_ci']]} r={d['per_image_r_real_vs_phase2']:.2f} nullr={d['per_image_r_real_vs_phase1_null']:.2f}")
    print("\n=== multimatch ===")
    for r in mmres:
        print(f"{r['group']:4s}{r['metric'][3:]:10s} RvR={r['real_vs_real']:.3f} P1={r['phase1']:.4f} P2={r['phase2']:.4f} d={r['phase2_minus_phase1']:+.4f}{[round(x,4) for x in r['image_ci']]} holm={r['image_p_holm']:.3f} I {r['images_favouring_phase2']}/30 seeds {r['seeds_favouring_phase2']}/10 | own-other {r['own_condition_minus_other_condition']:+.4f}{[round(x,4) for x in r['own_vs_other_ci']]} holm={r['own_vs_other_p_holm']:.3f}")
    print("\n=== embedding specificity ===")
    for k in ("joint", "duration", "position"):
        print(k, "equal-group P2-P1", json.dumps(spec["equal_group_weight_phase2_minus_phase1"][k]))
        print(k, "right-wrong", json.dumps(spec["right_label_minus_wrong_label_averaged_over_groups"][k]))
        for key, t in spec["fixed_embedding_vs_phase1"][k].items():
            print(f"  {key:24s} {t['improvement']:+.4f} part{[round(x, 4) for x in t['participant_ci']]} img{[round(x, 4) for x in t['image_ci']]} cross{[round(x, 4) for x in t['crossed_ci']]} P {t['participants_favouring']}/{t['participants']} I {t['images_favouring']}/30")
    print("ANALYZE_DONE")


# ======================================================================== verify
def verify():
    """Independent check of a surprising result: labels, loss path, and Saliency4ASD test as a known reference."""
    import torch
    from omegaconf import OmegaConf
    from training.dataset import load_scanpaths
    from training import model_utils as mu
    from training.conditioned import ConditionedTPPGaze
    from training.external_gaze4asd_loader import StimulusBank, load_external_scanpaths, collate_records
    from training.eval_external_v2 import IMG_DIR, BASE_CFG

    device = torch.device("cuda")
    split = json.load(open("training/runs/baseline/split.json"))
    sal = load_scanpaths("processed_data/asd_scanpaths.json", "processed_data/td_scanpaths.json")
    mean, std = mu.compute_log_inter_time_stats([s for s in sal if s["image_id"] in set(map(int, split["train"]))])
    cdim = int(OmegaConf.load(BASE_CFG).metadata.size + OmegaConf.load(BASE_CFG).context.size)
    base1, _ = mu.build_model(BASE_CFG, PHASE1, device, mean, std, used_cfg_out="/tmp/p1p2_c.yaml")
    base1.eval()
    base2, _ = mu.build_model(BASE_CFG, PHASE1, device, mean, std, used_cfg_out="/tmp/p1p2_d.yaml")
    p2 = ConditionedTPPGaze(base2, cdim).to(device)
    p2.load_state_dict(torch.load(PHASE2, map_location=device), strict=True)
    mu._fix_unregistered_device(p2.base, device)
    p2.eval()
    W = p2.group_emb.weight.detach()

    @torch.no_grad()
    def by_group(recs, bank, name_of):
        """Project loss function, unmodified; groups are scored separately only in the summary."""
        cache, tot = {}, {}
        models = {"phase1": (base1, lambda g: None), "phase2_true": (p2.base, lambda g: W[g]), "phase2_swapped": (p2.base, lambda g: W[1 - g]), "phase2_shared_mean": (p2.base, lambda g: W.mean(0, keepdim=True).expand(len(g), -1))}
        for st in range(0, len(recs), BATCH):
            b = recs[st:st + BATCH]
            c = collate_records(b, device)
            for i in {name_of(r) for r in b}:
                if i not in cache:
                    cache[i] = base1.get_metadata(bank.tensor(i).unsqueeze(0).to(device))
            meta = torch.cat([cache[name_of(r)] for r in b], 0)
            for v, (m, emb) in models.items():
                feats = m.get_features(c["inter_times"], c["marks"])
                ctx = m.merge.forward([m.get_context(feats, True), meta])
                cond = emb(c["group_ids"])
                if cond is not None:
                    ctx = ctx + cond.unsqueeze(1)
                it = torch.where(c["mask"].bool(), c["inter_times"], torch.ones_like(c["inter_times"]))
                lp = (m.inter_time_dist.forward(ctx, None).log_prob(it) + m.marks_dist.forward(ctx, None).log_prob(c["marks"])) * c["mask"]
                for gid, g in enumerate(("ASD", "TD")):
                    s = c["group_ids"] == gid
                    x = tot.setdefault(v, {}).setdefault(g, [0.0, 0.0])
                    x[0] += float(lp[s].sum()); x[1] += float(c["mask"][s].sum())
                if st == 0:
                    whole = mu.tppgaze_nll(m, c["inter_times"], c["marks"], c["mask"], meta, cond=cond)
                    assert abs(float(whole["nll"]) + float(lp.sum() / c["mask"].sum())) < 1e-4, "project loss disagrees"
        res = {}
        for v, g in tot.items():
            res[v] = {k: -a / b for k, (a, b) in g.items()}
            res[v]["overall"] = -sum(a for a, _ in g.values()) / sum(b for _, b in g.values())
        return res

    bank = StimulusBank(IMG_DIR)
    recs = load_external_scanpaths(DATA, bank)
    labels_ok = all(r["source_file"].startswith(f"{r['group']}_ScanPath/{r['participant']}/") for r in recs)
    ext = by_group(recs, bank, lambda r: r["image_id"])
    sal_bank = StimulusBank(OmegaConf.load("training/config_phase2_embonly.yaml").images_dir)
    test_ids = set(map(int, split["test"]))
    sal_res = by_group([s for s in sal if s["image_id"] in test_ids], sal_bank, lambda r: f"{r['image_id']}.png")
    out = dict(gaze4asd_labels_match_raw_folders=labels_ok, group_ids_in_training=dict(ASD=0, TD=1), gaze4asd=ext, saliency4asd_test_reference_only=sal_res, known_saliency_phase1=-0.4957, known_saliency_phase2=-0.5030)
    json.dump(out, open(f"{RAW}/verification_checks.json", "w"), indent=2)
    print(json.dumps(out, indent=1))
    print("VERIFY_DONE")


if __name__ == "__main__":
    {"run": run, "analyze": analyze, "verify": verify}[sys.argv[1]]()
