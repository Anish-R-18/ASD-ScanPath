"""
Monte Carlo stability of the generated ASD-vs-TD differences. No training.

Generated scanpaths do not depend on the coordinate mapping, so any change between
seeds is sampling noise. Repeats the primary configuration (temperature 2.0, 20 samples
per condition per image) with independent seeds, then pools all samples per image.
"""
from __future__ import annotations
import csv, json, sys
sys.path[:0] = [".", "tppgaze"]
import numpy as np
import torch
from omegaconf import OmegaConf

from training.dataset import load_scanpaths
from training import model_utils as mu
from training.conditioned import ConditionedTPPGaze
from training.generate_cond import sample_conditioned, to_scanpaths
from training.external_gaze4asd_loader import StimulusBank, load_external_scanpaths, GROUP_TO_ID
from training.eval_external_v2 import OUT, IMG_DIR, VERSIONS, PHASE1, PHASE2, BASE_CFG, N_SIM, TEMP, T_END, denorm, metric_row, sha256

N_REP, N_PERM = 10, 10000
METRICS = ["dur_mean", "n_fix", "total_dur", "fix_rate_per_s", "sacc_amp_norm_mean", "dispersion", "revisit_rate", "sacc_dir_resultant", "sacc_horizontal_frac"]


def test(d, rng):
    pm = (rng.choice([-1.0, 1.0], (N_PERM, len(d))) * d).mean(1)
    p = (np.sum(np.abs(pm) >= abs(d.mean()) - 1e-12) + 1) / (N_PERM + 1)
    b = d[rng.integers(0, len(d), (N_PERM, len(d)))]
    return dict(diff=float(d.mean()), diff_ci=[float(np.percentile(b.mean(1), 2.5)), float(np.percentile(b.mean(1), 97.5))], dz=float(d.mean() / d.std(ddof=1)), p=float(p))


def main():
    device = torch.device("cuda")
    h0 = sha256(PHASE2)
    split = json.load(open("training/runs/baseline/split.json"))
    sal = load_scanpaths("processed_data/asd_scanpaths.json", "processed_data/td_scanpaths.json")
    mean, std = mu.compute_log_inter_time_stats([s for s in sal if s["image_id"] in set(map(int, split["train"]))])
    base, _ = mu.build_model(BASE_CFG, PHASE1, device, mean, std, used_cfg_out="/tmp/rep_v2_cfg.yaml")
    cdim = int(OmegaConf.load(BASE_CFG).metadata.size + OmegaConf.load(BASE_CFG).context.size)
    cmodel = ConditionedTPPGaze(base, cdim).to(device)
    cmodel.load_state_dict(torch.load(PHASE2, map_location=device), strict=True)
    mu._fix_unregistered_device(cmodel.base, device)
    cmodel.eval()

    bank = StimulusBank(IMG_DIR)
    images = sorted({r["image_id"] for r in load_external_scanpaths(VERSIONS["fit_height_policy_a"], bank)}, key=lambda s: int(s.replace("top_image_", "").replace(".png", "")))
    # replicate index, condition, image, metric -> mean over that replicate's samples
    A = {g: {m: np.zeros((N_REP, len(images))) for m in METRICS} for g in GROUP_TO_ID}
    for rep in range(N_REP):
        for k, iid in enumerate(images):
            W, H = bank.dims(iid)
            for g, gid in GROUP_TO_ID.items():
                torch.manual_seed(100000 * (rep + 1) + 1000 * (k + 1) + gid)
                it, mk, ms = sample_conditioned(cmodel, bank.tensor(iid), gid, t_end=T_END, n_sim=N_SIM, temperature=TEMP)
                rows = [metric_row("g", iid, "", denorm(s, W, H), s[:, 2], W, H) for s in to_scanpaths(it, mk, ms) if len(s)]
                for m in METRICS:
                    A[g][m][rep, k] = np.nanmean([r[m] for r in rows])
        print("replicate", rep + 1, flush=True)

    rng = np.random.default_rng(0)
    prim = json.load(open(f"{OUT}/statistics/fit_height_policy_a_statistics.json"))["contrasts"]
    old = json.load(open(f"{OUT}/statistics/old_native_policy_a_statistics.json"))["contrasts"]
    real_img = {}
    with open(f"{OUT}/fit_height_policy_a/per_scanpath_metrics.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["population"].startswith("real"):
                for m in METRICS:
                    try:
                        real_img.setdefault((r["population"], r["image_id"], m), []).append(float(r[m]))
                    except ValueError:
                        pass
    out = []
    for m in METRICS:
        d = A["ASD"][m] - A["TD"][m]
        reps = [test(d[i], rng) for i in range(N_REP)]
        pooled = test(d.mean(0), rng)
        dr = np.array([np.nanmean(real_img[("real_ASD", i, m)]) - np.nanmean(real_img[("real_TD", i, m)]) for i in images])
        real_g = prim["real_ASD_vs_TD_participant"][m]
        out.append(dict(metric=m, real_diff_image=float(dr.mean()), real_participant_g=real_g["effect"], real_p_holm=real_g["p_holm"], pooled_diff=pooled["diff"], pooled_diff_ci=pooled["diff_ci"], pooled_dz_200_per_cond=pooled["dz"], pooled_p=pooled["p"],
                        same_direction_pooled=bool(np.sign(pooled["diff"]) == np.sign(dr.mean())), gen_to_real_ratio=float(pooled["diff"] / dr.mean()), per_image_correlation_real_vs_pooled_gen=float(np.corrcoef(dr, d.mean(0))[0, 1]),
                        replicate_dz_min=min(x["dz"] for x in reps), replicate_dz_max=max(x["dz"] for x in reps), replicate_dz_mean=float(np.mean([x["dz"] for x in reps])),
                        replicates_same_sign_as_real=int(sum(bool(np.sign(x["diff"]) == np.sign(dr.mean())) for x in reps)), replicates_p_below_05=int(sum(x["p"] < .05 for x in reps)), n_replicates=N_REP,
                        primary_seed_dz=prim["gen_ASD_vs_TD_image_paired"][m]["effect"], primary_seed_p=prim["gen_ASD_vs_TD_image_paired"][m]["p"]))
    o = sorted(range(len(out)), key=lambda i: out[i]["pooled_p"])
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(out) - r) * out[i]["pooled_p"]))
        out[i]["pooled_p_holm"] = run
    json.dump(out, open(f"{OUT}/comparison/generation_replicates.json", "w"), indent=2)
    with open(f"{OUT}/comparison/generation_replicates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    assert sha256(PHASE2) == h0
    for r in out:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    print("REPLICATES_DONE")


if __name__ == "__main__":
    main()
