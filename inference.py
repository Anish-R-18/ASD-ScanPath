"""
Standalone inference + visualization for the Phase-2 ASD/TD-conditioned TPP-Gaze model.

Does NOT retrain or modify the checkpoint / datasets / training pipeline. It reuses
the existing conditioned sampler (training/generate_cond.sample_conditioned, which
mirrors the official TPPGaze.sample()) and the official image preprocessing.

Usage:
  python inference.py --image path/to/image.png --condition ASD
  python inference.py --image path/to/image.png --condition TD
  python inference.py --image path/to/image.png --condition both   # side-by-side

Outputs (default outputs/):
  <cond>_scanpath.png   visualization over the image
  <cond>_scanpath.json  raw prediction [(x, y, duration), ...]
  <stem>_asd_vs_td.png  side-by-side (when --condition both)
"""
from __future__ import annotations
import argparse, json, os, sys, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
from PIL import Image

_ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (_ROOT, os.path.join(_ROOT, "tppgaze")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from omegaconf import OmegaConf
from tppgaze.tppgaze import TPPGaze
from tppgaze.utils import get_preprocess
from training.conditioned import ConditionedTPPGaze
from training import model_utils as mu
from training.generate_cond import sample_conditioned, to_scanpaths

GID = {"ASD": 0, "TD": 1}
# Saliency4ASD train-split log-inter-time stats the model was trained with (fallback).
DEFAULT_MEAN, DEFAULT_STD = -1.4365682299606377, 0.6936114719391417


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(_ROOT, p)


def load_stats(stats_json):
    try:
        d = json.load(open(resolve(stats_json)))
        return float(d["mean_log_inter_time"]), float(d["std_log_inter_time"])
    except Exception:
        return DEFAULT_MEAN, DEFAULT_STD


def load_model(ckpt, base_cfg, mean, std, device):
    cfg = OmegaConf.load(resolve(base_cfg))
    cfg.mean_log_inter_times = mean
    cfg.std_log_inter_times = std
    tmp = os.path.join(tempfile.gettempdir(), "tppgaze_infer_cfg.yaml")
    OmegaConf.save(cfg, tmp)
    base = TPPGaze(tmp, resolve(ckpt), device)          # builds modules (DenseNet via hub)
    base.to(device)
    cond_dim = int(cfg.metadata.size + cfg.context.size)
    cmodel = ConditionedTPPGaze(base, cond_dim).to(device)
    cmodel.load_state_dict(torch.load(resolve(ckpt), map_location=device), strict=True)
    mu._fix_unregistered_device(cmodel.base, device)    # move unregistered position_vec to device
    cmodel.eval()
    return cmodel


def predict(cmodel, image_path, condition, device, n_sim=1, temperature=2.0, t_end=3.0, seed=0):
    """Returns (pil_image, W, H, scanpath_px (L,3), scanpath_norm (L,3))."""
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    tensor = get_preprocess()(im)                        # (3,512,512), official preprocessing
    torch.manual_seed(seed)
    it, mk, ms = sample_conditioned(cmodel, tensor, GID[condition],
                                    t_end=t_end, n_sim=max(1, n_sim), temperature=temperature)
    sps = to_scanpaths(it, mk, ms)                        # list of (L,3) [x_norm,y_norm,dur_s]
    spn = max(sps, key=len) if len(sps) > 1 else sps[0]   # pick the richest sample
    # map normalized [-1,1] back onto the DISPLAYED image resolution
    xpx = (spn[:, 0] + 1) / 2 * (W - 1)
    ypx = (spn[:, 1] + 1) / 2 * (H - 1)
    sppx = np.stack([xpx, ypx, spn[:, 2]], axis=1)
    return im, W, H, sppx, spn


def draw(ax, im, sppx, condition):
    W, H = im.size
    ax.imshow(im)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    x, y, dur = sppx[:, 0], sppx[:, 1], sppx[:, 2]
    L = len(x)
    # arrows between consecutive fixations
    for i in range(1, L):
        ax.add_patch(FancyArrowPatch((x[i - 1], y[i - 1]), (x[i], y[i]),
                     arrowstyle="-|>", mutation_scale=14, color="yellow",
                     lw=1.6, alpha=0.85, shrinkA=6, shrinkB=6, zorder=3))
    # circles sized by duration (semi-transparent so the image stays visible)
    diag = float(np.hypot(W, H))
    sizes = (0.012 * diag + 0.09 * diag * dur) ** 2 * 0.35   # area ~ grows with duration
    colors = ["#39b54a" if i == 0 else ("#e3342f" if i == L - 1 else "#f6a623") for i in range(L)]
    ax.scatter(x, y, s=sizes, c=colors, alpha=0.45, edgecolors="black", linewidths=1.2, zorder=4)
    for i in range(L):
        ax.text(x[i], y[i], str(i + 1), color="white", fontsize=10, fontweight="bold",
                ha="center", va="center", zorder=6)
        off = 0.018 * diag
        ax.text(x[i] + off, y[i] - off, f"{dur[i]:.2f}s", color="white", fontsize=8,
                ha="left", va="bottom", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.55))
    ax.set_title(f"Condition: {condition}", fontsize=15, fontweight="bold", pad=10)
    # legend
    handles = [
        Patch(facecolor="#39b54a", edgecolor="k", label="1 = first fixation (order)"),
        Patch(facecolor="#e3342f", edgecolor="k", label="last fixation"),
        Patch(facecolor="#f6a623", edgecolor="k", label="circle size = duration"),
        FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", color="yellow", label="arrow = gaze movement"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, fontsize=8, frameon=True, framealpha=0.85)


def save_single(im, sppx, condition, out_png):
    fig, ax = plt.subplots(figsize=(10, 8))
    draw(ax, im, sppx, condition)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_json(out_json, condition, image_path, W, H, sppx, spn, temperature, seed):
    json.dump({
        "condition": condition, "image": os.path.abspath(image_path),
        "image_width": W, "image_height": H, "temperature": temperature, "seed": seed,
        "n_fixations": int(len(sppx)),
        "scanpath_pixels": [[round(float(a), 3), round(float(b), 3), round(float(c), 4)] for a, b, c in sppx],
        "scanpath_normalized": [[round(float(a), 6), round(float(b), 6), round(float(c), 6)] for a, b, c in spn],
    }, open(out_json, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser(description="ASD/TD-conditioned TPP-Gaze inference + visualization")
    ap.add_argument("--image", required=True)
    ap.add_argument("--condition", choices=["ASD", "TD", "both"], default="both")
    ap.add_argument("--out-dir", default=os.path.join(_ROOT, "test", "outputs"),
                    help="parent output dir; a per-image subfolder (named after the image stem) is created inside it")
    ap.add_argument("--checkpoint", default="training/runs/phase2_embedding_only/best_model_conditioned.pth")
    ap.add_argument("--base-cfg", default="tppgaze/data/config.yaml")
    ap.add_argument("--stats-json", default="training/runs/phase2_embedding_only/summary_full.json")
    ap.add_argument("--n-sim", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="auto|cpu|cuda")
    args = ap.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device in (None, "auto") else args.device)
    # per-image output folder (named after the image stem) inside the parent --out-dir
    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_dir = os.path.join(args.out_dir, stem)
    os.makedirs(out_dir, exist_ok=True)
    mean, std = load_stats(args.stats_json)
    print(f"[inference] device={device}  stats(mean/std log-inter)={mean:.4f}/{std:.4f}")
    cmodel = load_model(args.checkpoint, args.base_cfg, mean, std, device)

    conditions = ["ASD", "TD"] if args.condition == "both" else [args.condition]
    results = {}
    for cond in conditions:
        im, W, H, sppx, spn = predict(cmodel, args.image, cond, device,
                                      n_sim=args.n_sim, temperature=args.temperature, seed=args.seed)
        png = os.path.join(out_dir, f"{cond.lower()}_scanpath.png")
        js = os.path.join(out_dir, f"{cond.lower()}_scanpath.json")
        save_single(im, sppx, cond, png)
        save_json(js, cond, args.image, W, H, sppx, spn, args.temperature, args.seed)
        results[cond] = (im, sppx, png, js)
        print(f"[inference] {cond}: {len(sppx)} fixations  total={sppx[:,2].sum():.2f}s  -> {png}  {js}")

    if args.condition == "both":
        side = os.path.join(out_dir, f"{stem}_asd_vs_td.png")
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))
        for ax, cond in zip(axs, ["ASD", "TD"]):
            draw(ax, results[cond][0], results[cond][1], cond)
        fig.suptitle(f"ASD vs TD generated scanpaths — {os.path.basename(args.image)}",
                     fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(side, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"[inference] side-by-side -> {side}")


if __name__ == "__main__":
    main()
