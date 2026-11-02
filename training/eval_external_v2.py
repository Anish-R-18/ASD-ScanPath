"""
Corrected Gaze4ASD external evaluation of the FROZEN Phase-2 model. No training.

Runs the three v2 datasets plus the two old datasets as references, all through the
same code and against the same generated scanpaths, then writes raw per-trial and
per-scanpath results. Statistics and figures are in stats_external_v2.py.

Fit-to-height + Policy A is primary because of geometric evidence, not model scores.
The mapping is strongly supported but partly inferred, not experimentally confirmed.
"""
from __future__ import annotations
import csv, hashlib, json, os, sys, time, zlib
sys.path[:0] = [".", "tppgaze"]
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter

from training.dataset import load_scanpaths
from training import metrics as M
from training import model_utils as mu
from training.conditioned import ConditionedTPPGaze
from training.generate_cond import sample_conditioned, to_scanpaths
from training.external_gaze4asd_loader import StimulusBank, load_external_scanpaths, summarize_records, collate_records, GROUP_TO_ID

OUT = "training/runs/external_gaze4asd_v2"
IMG_DIR = "external_data/Gaze4ASD/Image Stimuli"
V2 = "external_data/Gaze4ASD_processed_v2"
VERSIONS = {
    "fit_height_policy_a": f"{V2}/fit_height_policy_a/gaze4asd_fit_height_policy_a.json",
    "stretch_policy_a": f"{V2}/stretch_policy_a/gaze4asd_stretch_policy_a.json",
    "fit_height_policy_b": f"{V2}/fit_height_policy_b/gaze4asd_fit_height_policy_b.json",
    "old_native_policy_a": "external_data/Gaze4ASD_processed/gaze4asd_policy_a.json",
    "old_native_policy_b": "external_data/Gaze4ASD_processed/gaze4asd_policy_b.json",
}
PHASE1 = "training/runs/baseline/best_model.pth"
PHASE2 = "training/runs/phase2_embedding_only/best_model_conditioned.pth"
BASE_CFG = "tppgaze/data/config.yaml"
N_SIM, TEMP, T_END, BATCH, MAX_PAIRS, SEED = 20, 2.0, 3.0, 128, 30, 0
REG_TOL = 0.002
MM_KEYS = ["mm_shape", "mm_length", "mm_direction", "mm_position", "mm_duration"]


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(f"{OUT}/evaluation_log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------------- NLL
@torch.no_grad()
def nll_rows(cmodel, records, meta_fn, device, order=None):
    """Per-trial log-likelihood sums. `order` changes only batch composition."""
    idx = list(range(len(records))) if order is None else list(order)
    rows = [None] * len(records)
    for st in range(0, len(idx), BATCH):
        ii = idx[st:st + BATCH]
        b = [records[i] for i in ii]
        c = collate_records(b, device)
        meta = meta_fn([r["image_id"] for r in b])
        feats = cmodel.base.get_features(c["inter_times"], c["marks"])
        ctx = cmodel.base.merge.forward([cmodel.base.get_context(feats, True), meta]) + cmodel.group_emb(c["group_ids"]).unsqueeze(1)
        it = torch.where(c["mask"].bool(), c["inter_times"], torch.ones_like(c["inter_times"]))
        lpt = cmodel.base.inter_time_dist.forward(ctx, None).log_prob(it) * c["mask"]
        lpp = cmodel.base.marks_dist.forward(ctx, None).log_prob(c["marks"]) * c["mask"]
        edge = (c["marks"].abs().max(dim=-1).values >= 1.0).float() * c["mask"]
        for j, i in enumerate(ii):
            r = records[i]
            rows[i] = dict(group=r["group"], participant=r.get("participant", ""), image_id=r["image_id"], n_events=int(c["mask"][j].sum()),
                           sum_lp_duration=float(lpt[j].sum()), sum_lp_position=float(lpp[j].sum()),
                           n_edge=int(edge[j].sum()), sum_lp_position_edge=float((lpp[j] * edge[j]).sum()))
    return rows


def agg(rows, group=None):
    s = [r for r in rows if group is None or r["group"] == group]
    n = sum(r["n_events"] for r in s)
    d, p = sum(r["sum_lp_duration"] for r in s), sum(r["sum_lp_position"] for r in s)
    ne, pe = sum(r["n_edge"] for r in s), sum(r["sum_lp_position_edge"] for r in s)
    return dict(n_trials=len(s), n_events=n, nll=-(d + p) / n, nll_duration=-d / n, nll_position=-p / n, edge_events=ne,
                edge_pct=100 * ne / n, nll_position_edge=(-pe / ne if ne else None), nll_position_nonedge=(-(p - pe) / (n - ne) if n > ne else None))


def loo_position_nll(records):
    """Position NLL of a centre-bias reference: gaze density from all OTHER stimuli."""
    bins = np.linspace(-1, 1, 61)
    H = {}
    for r in records:
        a = np.asarray(r["scanpath"])
        H.setdefault(r["image_id"], np.zeros((60, 60)))
        H[r["image_id"]] += np.histogram2d(a[:, 0], a[:, 1], bins=[bins, bins])[0]
    total = sum(H.values())
    area = (2 / 60) ** 2
    per, N, S = {}, 0, 0.0
    for k, h in H.items():
        d = gaussian_filter(total - h, 1.5, mode="nearest") + 1e-6
        d = d / (d.sum() * area)
        s = float((h * np.log(d)).sum())
        per[k] = {"n": int(h.sum()), "sum_log": s}
        N += h.sum(); S += s
    return {"nll_position": -S / N, "per_image": per}


# --------------------------------------------------------------- behaviour
def denorm(spn, W, H):
    return np.stack([(spn[:, 0] + 1) / 2 * (W - 1), (spn[:, 1] + 1) / 2 * (H - 1)], axis=1)


def metric_row(pop, image_id, participant, xy, dur, W, H):
    m = M.scanpath_metrics(xy, dur, float(np.hypot(W, H)))
    m.pop("sacc_dir_mean", None)  # a linear mean of angles is not meaningful; replaced below
    m.pop("sacc_dir_std", None)
    m["fix_rate_per_s"] = m["n_fix"] / m["total_dur"] if m["total_dur"] > 0 else np.nan
    if len(xy) >= 2:
        v = np.diff(xy, axis=0)
        a = np.arctan2(v[:, 1], v[:, 0])
        C, S = np.cos(a).mean(), np.sin(a).mean()
        m["sacc_dir_circ_mean"] = float(np.arctan2(S, C))
        m["sacc_dir_resultant"] = float(np.hypot(C, S))
        m["sacc_horizontal_frac"] = float(np.mean(np.abs(v[:, 0]) > np.abs(v[:, 1])))
    else:
        m["sacc_dir_circ_mean"] = m["sacc_dir_resultant"] = m["sacc_horizontal_frac"] = np.nan
    return dict(population=pop, image_id=image_id, participant=participant, **m)


def multimatch_tables(version, by_img, images, bank):
    types = {"RvR_ASD": ("real_ASD", "real_ASD", True), "RvR_TD": ("real_TD", "real_TD", True), "RvR_ASD_vs_TD": ("real_ASD", "real_TD", False),
             "RvG_ASD": ("real_ASD", "gen_ASD", False), "RvG_TD": ("real_TD", "gen_TD", False),
             "RvG_ASD_mismatch": ("real_ASD", "gen_TD", False), "RvG_TD_mismatch": ("real_TD", "gen_ASD", False),
             "GvG_ASD": ("gen_ASD", "gen_ASD", True), "GvG_TD": ("gen_TD", "gen_TD", True), "GenASD_vs_GenTD": ("gen_ASD", "gen_TD", False)}
    rows = []
    for name, (pa, pb, same) in types.items():
        for iid in images:
            la, lb = by_img[pa].get(iid, []), by_img[pb].get(iid, [])
            pairs = [(i, j) for i in range(len(la)) for j in range(len(lb)) if not (same and j <= i)]
            if not pairs:
                continue
            rng = np.random.default_rng([SEED, zlib.crc32(f"{version}|{name}|{iid}".encode())])
            pick = rng.permutation(len(pairs))[:MAX_PAIRS]
            W, H = bank.dims(iid)
            diag = float(np.hypot(W, H))
            acc = {k: [] for k in MM_KEYS}
            for q in pick:
                a, b = la[pairs[q][0]], lb[pairs[q][1]]
                mm = M.multimatch(a["xy"], a["dur"], b["xy"], b["dur"], diag)
                if mm:
                    for k in MM_KEYS:
                        acc[k].append(mm[k])
            if acc[MM_KEYS[0]]:
                rows.append(dict(type=name, image_id=iid, n_pairs=len(acc[MM_KEYS[0]]), **{k: float(np.mean(acc[k])) for k in MM_KEYS}))
    return rows


# ------------------------------------------------------------------------- main
def main():
    if os.path.exists(OUT):
        raise SystemExit(f"Refusing to overwrite existing results: {OUT}")
    for d in ["", "/generated", "/comparison", "/statistics", "/figures"] + [f"/{v}" if not v.startswith("old") else f"/comparison/{v}" for v in VERSIONS]:
        os.makedirs(OUT + d, exist_ok=True)
    t0 = time.time()
    np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA required"
    log(f"device={device} gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    ck0 = {p: sha256(p) for p in (PHASE1, PHASE2)}
    for p, h in ck0.items():
        log(f"checkpoint {p} sha256={h}")

    # ---- inputs
    bank = StimulusBank(IMG_DIR)
    data = {v: load_external_scanpaths(p, bank) for v, p in VERSIONS.items()}
    inputs = {v: dict(path=p, sha256=sha256(p), **summarize_records(data[v], bank)) for v, p in VERSIONS.items()}
    for v, s in inputs.items():
        log(f"input {v}: " + json.dumps({k: x for k, x in s.items() if k not in ("path", "sha256")}))
        assert s["stimuli"] == 30 and s["stimuli_resolved_to_files"] == 30 and s["participants_in_both_groups"] == 0
    json.dump(inputs, open(f"{OUT}/comparison/input_verification.json", "w"), indent=2, default=list)

    # ---- frozen model: Saliency4ASD train-split normalization, as used in training
    split = json.load(open("training/runs/baseline/split.json"))
    sal = load_scanpaths("processed_data/asd_scanpaths.json", "processed_data/td_scanpaths.json")
    train_ids, test_ids = set(map(int, split["train"])), set(map(int, split["test"]))
    mean, std = mu.compute_log_inter_time_stats([s for s in sal if s["image_id"] in train_ids])
    stored = json.load(open("training/runs/phase2_embedding_only/summary_full.json"))
    assert abs(mean - stored["mean_log_inter_time"]) < 1e-9 and abs(std - stored["std_log_inter_time"]) < 1e-9
    log(f"normalization mean={mean:.6f} std={std:.6f} (matches summary_full.json)")
    base, _ = mu.build_model(BASE_CFG, PHASE1, device, mean, std, used_cfg_out="/tmp/eval_v2_cfg.yaml")
    cdim = int(OmegaConf.load(BASE_CFG).metadata.size + OmegaConf.load(BASE_CFG).context.size)
    cmodel = ConditionedTPPGaze(base, cdim).to(device)
    cmodel.load_state_dict(torch.load(PHASE2, map_location=device), strict=True)
    mu._fix_unregistered_device(cmodel.base, device)
    cmodel.eval()
    for p in cmodel.parameters():
        p.requires_grad = False

    meta_cache = {}
    @torch.no_grad()
    def meta_for(ids):
        for i in set(ids):
            if i not in meta_cache:
                meta_cache[i] = cmodel.base.get_metadata(bank.tensor(i).unsqueeze(0).to(device))
        return torch.cat([meta_cache[i] for i in ids], 0)

    # ---- regression gate: new code must reproduce known results
    reg = {}
    sal_bank = StimulusBank(OmegaConf.load("training/config_phase2_embonly.yaml").images_dir)
    sal_cache = {}
    @torch.no_grad()
    def sal_meta(ids):
        for i in set(ids):
            if i not in sal_cache:
                sal_cache[i] = cmodel.base.get_metadata(sal_bank.tensor(f"{i}.png").unsqueeze(0).to(device))
        return torch.cat([sal_cache[i] for i in ids], 0)
    sal_test = [s for s in sal if s["image_id"] in test_ids]
    sal_rows = nll_rows(cmodel, sal_test, sal_meta, device)
    s_agg = agg(sal_rows)
    old_sal = json.load(open("training/runs/phase2_embedding_only/test_eval_phase2.json"))["phase2"]["overall"]
    old_ext = json.load(open("training/runs/external_gaze4asd/external_nll.json"))
    reg["saliency4asd_test"] = dict(new=s_agg, old_nll=old_sal["nll"], diff=s_agg["nll"] - old_sal["nll"])
    full_sal = {g or "overall": agg(sal_rows, g) for g in (None, "ASD", "TD")}
    json.dump(full_sal, open(f"{OUT}/comparison/saliency4asd_test_reproduced.json", "w"), indent=2)
    for v, pol in (("old_native_policy_a", "A"), ("old_native_policy_b", "B")):
        a = agg(nll_rows(cmodel, data[v], meta_for, device))
        reg[v] = dict(new=a, old_nll=old_ext[pol]["overall"]["nll"], diff=a["nll"] - old_ext[pol]["overall"]["nll"])
    for k, x in reg.items():
        log(f"regression {k}: new={x['new']['nll']:.5f} old={x['old_nll']:.5f} diff={x['diff']:+.6f}")
    json.dump(reg, open(f"{OUT}/comparison/regression_checks.json", "w"), indent=2)
    if max(abs(x["diff"]) for x in reg.values()) > REG_TOL:
        raise SystemExit("REGRESSION GATE FAILED")
    log("regression gate passed")

    # ---- one generated set per image and condition, shared by all versions
    images = sorted({r["image_id"] for r in data["fit_height_policy_a"]}, key=lambda s: int(s.replace("top_image_", "").replace(".png", "")))
    gen, by_gen = [], {"gen_ASD": {}, "gen_TD": {}}
    n_empty = n_short = n_oob = n_gen_fix = 0
    for k, iid in enumerate(images):
        W, H = bank.dims(iid)
        for g, gid in GROUP_TO_ID.items():
            torch.manual_seed(1000 * (k + 1) + gid)
            it, mk, ms = sample_conditioned(cmodel, bank.tensor(iid), gid, t_end=T_END, n_sim=N_SIM, temperature=TEMP)
            for s, spn in enumerate(to_scanpaths(it, mk, ms)):
                if len(spn) < 1:
                    n_empty += 1
                    continue
                n_short += int(len(spn) < 3); n_gen_fix += len(spn); n_oob += int((np.abs(spn[:, :2]) > 1).any(axis=1).sum())
                gen.append(dict(image_id=iid, condition=g, sim=s, scanpath=spn.round(6).tolist()))
                by_gen[f"gen_{g}"].setdefault(iid, []).append(dict(xy=denorm(spn, W, H), dur=spn[:, 2], norm=spn))
    torch.manual_seed(1000 * 1 + 0)
    it, mk, ms = sample_conditioned(cmodel, bank.tensor(images[0]), 0, t_end=T_END, n_sim=N_SIM, temperature=TEMP)
    again = to_scanpaths(it, mk, ms)
    first = [x for x in gen if x["image_id"] == images[0] and x["condition"] == "ASD"]
    repro = all(len(a) == len(b["scanpath"]) and np.allclose(a[:, :], np.array(b["scanpath"]), atol=1e-5) for a, b in zip(again, first))
    gen_info = dict(n_sim=N_SIM, temperature=TEMP, t_end=T_END, seed_rule="torch.manual_seed(1000*(stimulus_rank+1)+group_id)", generated=len(gen), empty_dropped=n_empty,
                    below_3_fixations=n_short, fixations=n_gen_fix, fixations_outside_image_pct=100 * n_oob / n_gen_fix, reproducible_rerun_first_stimulus=bool(repro))
    log("generated: " + json.dumps(gen_info))
    json.dump(gen, open(f"{OUT}/generated/generated_scanpaths.json", "w"))
    json.dump(gen_info, open(f"{OUT}/generated/generation_info.json", "w"), indent=2)

    # ---- each version
    all_nll = {}
    for v, recs in data.items():
        d = f"{OUT}/{v}" if not v.startswith("old") else f"{OUT}/comparison/{v}"
        rows = nll_rows(cmodel, recs, meta_for, device)
        write_csv(f"{d}/per_trial_nll.csv", rows)
        res = {g or "overall": agg(rows, g) for g in (None, "ASD", "TD")}
        res["participants"] = {g: len({r["participant"] for r in recs if r["group"] == g}) for g in GROUP_TO_ID}
        res["stimuli"] = len({r["image_id"] for r in recs})
        res["references"] = dict(uniform_position_nll=float(np.log(4.0)))
        loo = loo_position_nll(recs)
        res["references"]["leave_one_image_out_centre_bias_position_nll"] = loo["nll_position"]
        json.dump(loo["per_image"], open(f"{d}/centre_bias_reference_per_image.json", "w"), indent=1)
        shuf = []
        for sd in (1, 2, 3):
            o = np.random.default_rng(sd).permutation(len(recs))
            a = agg(nll_rows(cmodel, recs, meta_for, device, order=o))
            shuf.append({k: a[k] for k in ("nll", "nll_duration", "nll_position")})
        res["shuffled_batches"] = shuf
        json.dump(res, open(f"{d}/nll.json", "w"), indent=2)
        all_nll[v] = res
        log(f"{v}: NLL overall={res['overall']['nll']:.4f} dur={res['overall']['nll_duration']:.4f} pos={res['overall']['nll_position']:.4f} | ASD {res['ASD']['nll']:.4f} TD {res['TD']['nll']:.4f} | shuffled nll " + " ".join(f"{s['nll']:.4f}" for s in shuf))

        by_img = {"real_ASD": {}, "real_TD": {}, **by_gen}
        out = []
        for r in recs:
            W, H = bank.dims(r["image_id"])
            sp = np.asarray(r["scanpath"])
            rec = dict(xy=denorm(sp, W, H), dur=sp[:, 2], participant=r["participant"])
            by_img[f"real_{r['group']}"].setdefault(r["image_id"], []).append(rec)
            out.append(metric_row(f"real_{r['group']}", r["image_id"], r["participant"], rec["xy"], rec["dur"], W, H))
        window = {}
        for key in ("ASD", "TD"):
            for iid, L in by_img[f"real_{key}"].items():
                window[(key, iid)] = float(np.mean([x["dur"].sum() for x in L]))
        for key in ("ASD", "TD"):
            for iid, L in by_gen[f"gen_{key}"].items():
                W, H = bank.dims(iid)
                for x in L:
                    m = metric_row(f"gen_{key}", iid, "", x["xy"], x["dur"], W, H)
                    on = np.r_[0.0, np.cumsum(x["dur"])[:-1]]
                    m["n_fix_in_real_window"] = int((on < window[(key, iid)]).sum())
                    out.append(m)
        for m in out:
            m.setdefault("n_fix_in_real_window", m["n_fix"])
        keys = ["population", "image_id", "participant"] + [k for k in out[0] if k not in ("population", "image_id", "participant")]
        write_csv(f"{d}/per_scanpath_metrics.csv", [{k: m[k] for k in keys} for m in out])
        np.savez_compressed(f"{d}/fixation_durations.npz", **{p: np.concatenate([x["dur"] for L in by_img[p].values() for x in L]) for p in by_img})
        write_csv(f"{d}/multimatch_per_image.csv", multimatch_tables(v, by_img, images, bank))
        log(f"{v}: metrics and MultiMatch done ({time.time() - t0:.0f}s)")

    json.dump(all_nll, open(f"{OUT}/comparison/nll_all_versions.json", "w"), indent=2)
    for p, h in ck0.items():
        assert sha256(p) == h, f"checkpoint changed: {p}"
    log(f"checkpoints unchanged. peak GPU MiB={torch.cuda.max_memory_allocated() / 2**20:.0f}. total {time.time() - t0:.0f}s. EVAL_DONE")


if __name__ == "__main__":
    main()
