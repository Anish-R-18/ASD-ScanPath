"""
Phase-2 behavioral analysis (NO retraining).
Generates ASD- and TD-conditioned scanpaths for the 45 held-out TEST images,
compares Gen-ASD vs Gen-TD and Real vs Generated, computes gaze-dynamics +
social-AOI metrics + MultiMatch similarity, and writes CSV/JSON/tables/plots.
"""
from __future__ import annotations
import sys, os, json, csv, random
sys.path.insert(0, "."); sys.path.insert(0, "tppgaze")
import numpy as np
import torch
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import wasserstein_distance

from training.dataset import load_scanpaths, ImageBank
from training import model_utils as mu
from training.conditioned import ConditionedTPPGaze
from training.generate_cond import sample_conditioned, to_scanpaths
from training import metrics as M

# ---- config ----
BASE_CFG = "tppgaze/data/config.yaml"
P2 = "training/runs/phase2_embedding_only/best_model_conditioned.pth"
CFG = OmegaConf.load("training/config_phase2_embonly.yaml")
AOI_CSV = "ImageAnnotations.csv"
OUT = "training/runs/phase2_embedding_only/analysis"
N_SIM = 20
TEMP = 2.0
T_END = 3.0
SEED = 0
MAX_PAIRS = 30
os.makedirs(OUT, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "CUDA required"

# ---- data + model ----
split = json.load(open("training/runs/baseline/split.json"))
test_ids = sorted(map(int, split["test"])); train_ids = set(map(int, split["train"]))
samples = load_scanpaths("processed_data/asd_scanpaths.json", "processed_data/td_scanpaths.json")
_raw = (json.load(open("processed_data/asd_scanpaths.json"))
        + json.load(open("processed_data/td_scanpaths.json")))
sizes = {r["image_id"]: (r["image_width"], r["image_height"]) for r in _raw}
mean, std = mu.compute_log_inter_time_stats([s for s in samples if s["image_id"] in train_ids])

# base is a valid TPPGaze (weights overwritten below by the conditioned checkpoint)
base, _ = mu.build_model(BASE_CFG, "training/runs/baseline/best_model.pth", device,
                         mean, std, used_cfg_out="/tmp/a1.yaml")
cond_dim = int(OmegaConf.load(BASE_CFG).metadata.size + OmegaConf.load(BASE_CFG).context.size)
cmodel = ConditionedTPPGaze(base, cond_dim).to(device)
cmodel.load_state_dict(torch.load(P2, map_location=device), strict=True)
mu._fix_unregistered_device(cmodel.base, device)
cmodel.eval()
bank = ImageBank(CFG.images_dir, ext=CFG.image_ext)
aoi = M.load_aoi(AOI_CSV)


def denorm(spn, W, H):
    x = (spn[:, 0] + 1) / 2 * (W - 1); y = (spn[:, 1] + 1) / 2 * (H - 1)
    return np.stack([x, y], axis=1)


# ---- collect the 4 populations: list of dict(image_id, xy_pix, dur) ----
pops = {"real_ASD": [], "real_TD": [], "gen_ASD": [], "gen_TD": []}
by_img = {"real_ASD": {}, "real_TD": {}, "gen_ASD": {}, "gen_TD": {}}

# real
for s in samples:
    if s["image_id"] not in set(test_ids):
        continue
    W, H = sizes[s["image_id"]]
    sp = np.asarray(s["scanpath"], float)
    rec = {"image_id": s["image_id"], "xy": denorm(sp, W, H), "dur": sp[:, 2]}
    key = "real_ASD" if s["group"] == "ASD" else "real_TD"
    pops[key].append(rec); by_img[key].setdefault(s["image_id"], []).append(rec)

# generated
for iid in test_ids:
    W, H = sizes[iid]
    img = bank.get(iid)
    for gid, gname in [(0, "gen_ASD"), (1, "gen_TD")]:
        it, mk, ms = sample_conditioned(cmodel, img, gid, t_end=T_END, n_sim=N_SIM, temperature=TEMP)
        for spn in to_scanpaths(it, mk, ms):
            if len(spn) < 1:
                continue
            rec = {"image_id": iid, "xy": denorm(spn, W, H), "dur": spn[:, 2]}
            pops[gname].append(rec); by_img[gname].setdefault(iid, []).append(rec)

print("population sizes:", {k: len(v) for k, v in pops.items()})

# ---- per-scanpath metrics -> CSV ----
rows = []
for pop, recs in pops.items():
    for r in recs:
        W, H = sizes[r["image_id"]]; diag = float(np.hypot(W, H))
        m = M.scanpath_metrics(r["xy"], r["dur"], diag)
        a = M.aoi_metrics(r["xy"], r["dur"], aoi.get(r["image_id"], []))
        rows.append({"population": pop, "image_id": r["image_id"], **m, **a})
metric_keys = [k for k in rows[0].keys() if k not in ("population", "image_id")]
with open(os.path.join(OUT, "per_scanpath_metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["population", "image_id"] + metric_keys)
    w.writeheader(); w.writerows(rows)


def arr(pop, key):
    return np.array([r[key] for r in rows if r["population"] == pop and np.isfinite(r[key])], float)


# ---- aggregate summary ----
agg = {}
for pop in pops:
    agg[pop] = {"n_scanpaths": len(pops[pop])}
    for k in metric_keys:
        a = arr(pop, k)
        agg[pop][k] = {"mean": float(np.mean(a)) if a.size else None,
                       "median": float(np.median(a)) if a.size else None,
                       "std": float(np.std(a)) if a.size else None}
json.dump(agg, open(os.path.join(OUT, "aggregate_summary.json"), "w"), indent=2)

# ---- distribution distances (Wasserstein) ----
dist_keys = ["n_fix", "dur_mean", "total_dur", "sacc_amp_norm_mean", "dispersion",
             "revisit_rate", "face_dwell_frac", "social_dwell_frac"]
comparisons = [("gen_ASD", "gen_TD"), ("real_ASD", "gen_ASD"), ("real_TD", "gen_TD"),
               ("real_ASD", "real_TD")]
wdist = {}
for a, b in comparisons:
    wdist[f"{a}__vs__{b}"] = {}
    for k in dist_keys:
        xa, xb = arr(a, k), arr(b, k)
        wdist[f"{a}__vs__{b}"][k] = float(wasserstein_distance(xa, xb)) if xa.size and xb.size else None
json.dump(wdist, open(os.path.join(OUT, "distribution_distances.json"), "w"), indent=2)

# ---- MultiMatch similarity ----
def sample_pairs(listA, listB, same=False, cap=MAX_PAIRS):
    pairs = []
    for i in range(len(listA)):
        for j in range(len(listB)):
            if same and j <= i:
                continue
            pairs.append((listA[i], listB[j]))
    random.shuffle(pairs)
    return pairs[:cap]

mm_types = {
    "RvR_ASD": ("real_ASD", "real_ASD", True),
    "RvR_TD": ("real_TD", "real_TD", True),
    "RvG_ASD": ("real_ASD", "gen_ASD", False),
    "RvG_TD": ("real_TD", "gen_TD", False),
    "GenASD_vs_GenTD": ("gen_ASD", "gen_TD", False),
}
mm_res = {}
for name, (pa, pb, same) in mm_types.items():
    acc = {k: [] for k in ["mm_shape", "mm_length", "mm_direction", "mm_position", "mm_duration"]}
    for iid in test_ids:
        la = by_img[pa].get(iid, []); lb = by_img[pb].get(iid, [])
        if not la or not lb:
            continue
        W, H = sizes[iid]; diag = float(np.hypot(W, H))
        for ra, rb in sample_pairs(la, lb, same=same):
            mm = M.multimatch(ra["xy"], ra["dur"], rb["xy"], rb["dur"], diag)
            if mm:
                for k in acc:
                    acc[k].append(mm[k])
    mm_res[name] = {k: (float(np.mean(v)) if v else None) for k, v in acc.items()}
    mm_res[name]["n_pairs"] = len(acc["mm_shape"])
json.dump(mm_res, open(os.path.join(OUT, "similarity_multimatch.json"), "w"), indent=2)

# ---- summary tables (text) ----
def fmt(pop, k):
    v = agg[pop][k]["mean"]; return f"{v:.3f}" if v is not None else "n/a"
tbl = ["PHASE-2 BEHAVIORAL ANALYSIS - AGGREGATE MEANS",
       f"populations: " + ", ".join(f"{k}={len(v)}" for k, v in pops.items()),
       "",
       f"{'metric':22s}{'real_ASD':>12}{'real_TD':>12}{'gen_ASD':>12}{'gen_TD':>12}"]
for k in metric_keys:
    tbl.append(f"{k:22s}{fmt('real_ASD',k):>12}{fmt('real_TD',k):>12}{fmt('gen_ASD',k):>12}{fmt('gen_TD',k):>12}")
tbl += ["", "MultiMatch (mean over pairs; 1=identical):",
        f"{'type':18s}{'shape':>9}{'length':>9}{'direct.':>9}{'position':>9}{'duration':>9}{'n_pairs':>9}"]
for name, d in mm_res.items():
    tbl.append(f"{name:18s}{d['mm_shape']:>9.3f}{d['mm_length']:>9.3f}{d['mm_direction']:>9.3f}"
               f"{d['mm_position']:>9.3f}{d['mm_duration']:>9.3f}{d['n_pairs']:>9d}")
open(os.path.join(OUT, "summary_tables.txt"), "w").write("\n".join(tbl) + "\n")
print("\n".join(tbl))

# ---- plots ----
def draw(ax, xy, dur, color):
    if len(xy) == 0:
        return
    ax.plot(xy[:, 0], xy[:, 1], "-", color=color, alpha=0.6, lw=1.5)
    ax.scatter(xy[:, 0], xy[:, 1], s=20 + 600 * dur, c=color, alpha=0.5, edgecolors="k", linewidths=0.5)

# (1) representative same-image ASD vs TD generated scanpaths
rep = test_ids[:4]
fig, axs = plt.subplots(len(rep), 2, figsize=(10, 4 * len(rep)))
for r, iid in enumerate(rep):
    im = Image.open(bank.path_for(iid)).convert("RGB")
    for c, (pop, col, title) in enumerate([("gen_ASD", "red", "ASD"), ("gen_TD", "blue", "TD")]):
        ax = axs[r, c]; ax.imshow(im); ax.set_title(f"img {iid} - gen {title}"); ax.axis("off")
        recs = by_img[pop].get(iid, [])
        if recs:
            draw(ax, recs[0]["xy"], recs[0]["dur"], col)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_representative_scanpaths.png"), dpi=110); plt.close(fig)

# (2) aggregate ASD vs TD (generated) + (3) real vs generated
def hist_panel(fname, keys, groups, labels, colors, title):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for ax, k in zip(axs.ravel(), keys):
        for g, lab, col in zip(groups, labels, colors):
            a = arr(g, k)
            if a.size:
                ax.hist(a, bins=25, alpha=0.5, label=lab, color=col, density=True)
        ax.set_title(k); ax.legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=110); plt.close(fig)

panel_keys = ["n_fix", "dur_mean", "sacc_amp_norm_mean", "face_dwell_frac"]
hist_panel("fig_asd_vs_td_generated.png", panel_keys,
           ["gen_ASD", "gen_TD"], ["gen ASD", "gen TD"], ["red", "blue"],
           "Generated ASD vs TD")
hist_panel("fig_real_vs_generated_ASD.png", panel_keys,
           ["real_ASD", "gen_ASD"], ["real ASD", "gen ASD"], ["darkred", "orange"],
           "Real vs Generated (ASD)")
hist_panel("fig_real_vs_generated_TD.png", panel_keys,
           ["real_TD", "gen_TD"], ["real TD", "gen TD"], ["darkblue", "cyan"],
           "Real vs Generated (TD)")

print("\nARTIFACTS:", sorted(os.listdir(OUT)))
print("ANALYSIS_DONE")
