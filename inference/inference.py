"""
Standalone inference + visualization for the Phase-2 ASD/TD-conditioned TPP-Gaze model.

Does NOT retrain or modify the checkpoint / datasets / training pipeline. It reuses
the existing conditioned sampler (training/generate_cond.sample_conditioned, which
mirrors the official TPPGaze.sample()) and the official image preprocessing.

Usage:
  python inference/inference.py --image path/to/image.png --condition ASD
  python inference/inference.py --image path/to/image.png --condition TD
  python inference/inference.py --image path/to/image.png --condition both   # side-by-side
  python inference/inference.py --image path/to/image.png --temperature 2.0  # official TPP-Gaze config value
  python inference/inference.py --image path/to/image.png --compare-phase1   # adds the unconditioned Phase 1 model
  python inference/inference.py --image path/to/image.png --n-samples 4      # 4 seeds per condition, as a grid

Temperature defaults to 1.0, the behavioural-generation setting selected on Saliency4ASD
validation images. It changes sampling only: it is not a model or likelihood improvement.
It raises each spatial mixture sigma to the power tau; durations are not tempered.

Every condition (and Phase 1) is sampled with the same seed, so the panels share their
random numbers and differ only through the model (common random numbers).

Outputs (default inference/outputs/<image_stem>/; Git ignores this folder):
  <cond>_scanpath.png   visualization over the image
  <cond>_scanpath.json  scanpath_pixels: [x, y, duration], x and y clipped to the image as in
                        the official generate_predictions; scanpath_normalized: as sampled;
                        events: one record per fixation; model, checkpoint hash and settings
  <stem>_asd_vs_td.png  side-by-side (when --condition both)
  <stem>_phase1_vs_asd_vs_td.png   (when --compare-phase1)
  <cond>_samples.json, <stem>_samples_grid.png   (when --n-samples > 1)
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
from PIL import Image

# Repository root (this file lives in <root>/inference/). Checkpoint and config paths are resolved against it.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
# Selected on Saliency4ASD validation images for behavioural generation; does not affect NLL.
DEFAULT_TEMPERATURE = 1.0
PHASE1_CKPT = "training/runs/baseline/best_model.pth"
PHASE2_CKPT = "training/runs/phase2_embedding_only/best_model_conditioned.pth"
# Phase 1 has no group embedding; it is shown under this label.
PHASE1 = "Phase 1"
NOTE = "Model-generated research sample. Not a diagnostic or clinical tool."


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(_ROOT, p)


def sha256(path):
    h = hashlib.sha256()
    with open(resolve(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_stats(stats_json):
    try:
        d = json.load(open(resolve(stats_json)))
        return float(d["mean_log_inter_time"]), float(d["std_log_inter_time"])
    except Exception:
        print(f"[inference] WARNING: could not read {stats_json}; using the built-in training stats")
        return DEFAULT_MEAN, DEFAULT_STD


def load_model(ckpt, base_cfg, mean, std, device, phase1=False):
    """Phase 2 (default): base + learned group embedding. phase1=True: the Phase 1 checkpoint
    in the same wrapper with an all-zero embedding, which is exactly the unconditioned model
    (the wrapper is zero-initialised), so both phases run through the same sampler code."""
    cfg = OmegaConf.load(resolve(base_cfg))
    cfg.mean_log_inter_times = mean
    cfg.std_log_inter_times = std
    tmp = os.path.join(tempfile.gettempdir(), f"tppgaze_infer_cfg_{os.getpid()}.yaml")
    OmegaConf.save(cfg, tmp)
    base = TPPGaze(tmp, resolve(ckpt), device)          # builds modules (DenseNet via hub)
    base.to(device)
    cond_dim = int(cfg.metadata.size + cfg.context.size)
    if phase1:
        base.load_state_dict(torch.load(resolve(ckpt), map_location=device), strict=True)
        cmodel = ConditionedTPPGaze(base, cond_dim).to(device)
        assert float(cmodel.group_emb.weight.abs().sum()) == 0.0
    else:
        cmodel = ConditionedTPPGaze(base, cond_dim).to(device)
        cmodel.load_state_dict(torch.load(resolve(ckpt), map_location=device), strict=True)
    mu._fix_unregistered_device(cmodel.base, device)    # move unregistered position_vec to device
    cmodel.eval()
    return cmodel


def same_base(m1, m2):
    """True when the two wrapped models have bit-identical base weights."""
    a, b = m1.base.state_dict(), m2.base.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def to_pixels(spn, W, H):
    """Map normalized [-1,1] onto the displayed image, clipped to it; (L,3) [x_px, y_px, dur_s]."""
    xpx = np.clip((spn[:, 0] + 1) / 2 * (W - 1), 0, W - 1)
    ypx = np.clip((spn[:, 1] + 1) / 2 * (H - 1), 0, H - 1)
    return np.stack([xpx, ypx, spn[:, 2]], axis=1)


def sample(cmodel, tensor, condition, seed, n_sim=1, temperature=DEFAULT_TEMPERATURE, t_end=3.0):
    """All n_sim sampled scanpaths (list of (L,3) [x_norm, y_norm, dur_s]) for one seed.
    Note: the base model's transformer mixes across the batch (batch_first=False), so a
    sample drawn with n_sim > 1 depends on the other samples in its batch."""
    torch.manual_seed(seed)
    it, mk, ms = sample_conditioned(cmodel, tensor, GID.get(condition, 0),
                                    t_end=t_end, n_sim=max(1, n_sim), temperature=temperature)
    return to_scanpaths(it, mk, ms)


def predict(cmodel, image_path, condition, device, n_sim=1, temperature=DEFAULT_TEMPERATURE, t_end=3.0, seed=0):
    """Returns (pil_image, W, H, scanpath_px (L,3), scanpath_norm (L,3)).
    With n_sim > 1 the first sample is returned (not the longest, which would bias the display)."""
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    tensor = get_preprocess()(im)                        # (3,512,512), official preprocessing
    spn = sample(cmodel, tensor, condition, seed, n_sim, temperature, t_end)[0]
    return im, W, H, to_pixels(spn, W, H), spn


def events(spn, W, H):
    """One record per fixation: order, onset, duration, pixel and normalized position."""
    px = to_pixels(spn, W, H)
    onsets = np.concatenate([[0.0], np.cumsum(spn[:, 2])[:-1]]) if len(spn) else np.zeros(0)
    return [dict(index=i + 1, onset_s=round(float(onsets[i]), 4), duration_s=round(float(spn[i, 2]), 4),
                 x_px=round(float(px[i, 0]), 3), y_px=round(float(px[i, 1]), 3),
                 x_norm=round(float(spn[i, 0]), 6), y_norm=round(float(spn[i, 1]), 6),
                 clipped=bool(abs(spn[i, 0]) > 1 or abs(spn[i, 1]) > 1)) for i in range(len(spn))]


def draw(ax, im, sppx, condition, show_image=True, subtitle=None):
    """Scanpath over the image (or over a blank canvas of the same size when show_image=False)."""
    W, H = im.size
    if show_image:
        ax.imshow(im)
    else:
        ax.add_patch(plt.Rectangle((-0.5, -0.5), W, H, fc="#eeeeee", ec="#555555", lw=1.0, zorder=0))
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5); ax.set_aspect("equal"); ax.axis("off")
    x, y, dur = sppx[:, 0], sppx[:, 1], sppx[:, 2]
    L = len(x)
    diag = float(np.hypot(W, H))
    outline = [pe.withStroke(linewidth=3.2, foreground="black")]
    # arrows between consecutive fixations (yellow with a dark outline, visible on light and dark images)
    for i in range(1, L):
        ax.add_patch(FancyArrowPatch((x[i - 1], y[i - 1]), (x[i], y[i]),
                     arrowstyle="-|>", mutation_scale=14, color="yellow",
                     lw=1.6, alpha=0.95, shrinkA=6, shrinkB=6, zorder=3, path_effects=outline))
    # circles: radius in image pixels, so the size is independent of the image resolution;
    # area grows with duration
    colors = ["#39b54a" if i == 0 else ("#e3342f" if i == L - 1 else "#f6a623") for i in range(L)]
    for i in range(L):
        ax.add_patch(Circle((x[i], y[i]), diag * (0.010 + 0.045 * dur[i]), fc=colors[i], ec="black",
                            lw=1.2, alpha=0.5, zorder=4))
        ax.text(x[i], y[i], str(i + 1), color="white", fontsize=10, fontweight="bold",
                ha="center", va="center", zorder=6, path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
    ax.set_title(condition if subtitle is None else f"{condition}\n{subtitle}", fontsize=13, fontweight="bold", pad=8)
    # durations listed under the panel instead of on the image, where they overlap when fixations cluster
    items = [f"{i + 1}: {d:.2f}" for i, d in enumerate(dur)]
    lines = ["  ".join(items[k:k + 7]) for k in range(0, len(items), 7)] or ["(no fixations)"]
    ax.text(0.5, -0.02, f"{L} fixations, {dur.sum():.2f} s. Durations (s):\n" + "\n".join(lines),
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, family="monospace")


def legend_handles():
    return [
        Patch(facecolor="#39b54a", edgecolor="k", label="first fixation (numbers give the order)"),
        Patch(facecolor="#e3342f", edgecolor="k", label="last fixation"),
        Patch(facecolor="#f6a623", edgecolor="k", label="circle area grows with duration"),
        Line2D([0], [0], color="yellow", lw=2, path_effects=[pe.withStroke(linewidth=4, foreground="black")],
               label="arrow = movement to the next fixation"),
    ]


def save_panels(panels, im, out_png, title, footer, show_image=True, ncols=None):
    """panels: list of (label, scanpath_px, subtitle). One figure, shared legend and footer."""
    n = len(panels)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    W, H = im.size
    pw = 6.0 if ncols > 1 else 9.0
    # panels placed explicitly in inches (title above, image, duration list below), so aspect-locked
    # images leave no large gaps whatever the image shape
    img_h, gap = pw * H / W, 0.25
    title_h = 0.75 if any(p[2] for p in panels) else 0.45
    cap_h = 0.25 + 0.17 * max(1, max(int(np.ceil(len(p[1]) / 7)) for p in panels))
    row_h = title_h + img_h + cap_h
    top, bottom = (0.5 if title else 0.05), 0.75
    fig_w, fig_h = ncols * pw + (ncols - 1) * gap, nrows * row_h + top + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    for k, (label, sppx, sub) in enumerate(panels):
        r, c = divmod(k, ncols)
        y0 = fig_h - top - r * row_h - title_h - img_h
        ax = fig.add_axes([c * (pw + gap) / fig_w, y0 / fig_h, pw / fig_w, img_h / fig_h])
        draw(ax, im, sppx, label, show_image, sub)
    if title:
        fig.suptitle(title, fontsize=14, y=1 - 0.15 / fig_h, va="top")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=2 if ncols == 1 else 4, fontsize=8.5,
               frameon=True, bbox_to_anchor=(0.5, 0.25 / fig_h))
    fig.text(0.5, 0.1 / fig_h, footer, ha="center", va="bottom", fontsize=8.5, color="#333333", wrap=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_single(im, sppx, condition, out_png, footer="", show_image=True):
    save_panels([(condition, sppx, None)], im, out_png, "", footer or NOTE, show_image)


def run_info(args, condition, ckpt_hash):
    ckpt = args.phase1_checkpoint if condition == PHASE1 else args.checkpoint
    return dict(phase="phase1_unconditioned" if condition == PHASE1 else "phase2_conditioned",
                checkpoint=ckpt.replace("\\", "/"), checkpoint_sha256=ckpt_hash,
                torch=torch.__version__, numpy=np.__version__, device=str(args.device_resolved))


def save_json(out_json, condition, image_path, W, H, sppx, spn, temperature, seed, extra=None):
    d = {
        "condition": condition, "image": os.path.abspath(image_path),
        "image_width": W, "image_height": H, "temperature": temperature, "seed": seed,
        "n_fixations": int(len(sppx)),
        "scanpath_pixels": [[round(float(a), 3), round(float(b), 3), round(float(c), 4)] for a, b, c in sppx],
        "scanpath_normalized": [[round(float(a), 6), round(float(b), 6), round(float(c), 6)] for a, b, c in spn],
    }
    if extra is not None:
        d.update(extra)
    json.dump(d, open(out_json, "w"), indent=2)


def footer_text(args, seeds=None):
    s = f"seeds {seeds[0]}-{seeds[-1]} (one per column; one row per condition); panels in a column" if seeds \
        else f"seed = {args.seed}; panels"
    return (f"Generated samples. tau = {args.temperature:g}, viewing window {args.t_end:g} s, {s} share their "
            f"random numbers. {NOTE}")


def main():
    ap = argparse.ArgumentParser(description="ASD/TD-conditioned TPP-Gaze inference + visualization")
    ap.add_argument("--image", required=True)
    ap.add_argument("--condition", choices=["ASD", "TD", "both"], default="both")
    ap.add_argument("--out-dir", default=os.path.join(_ROOT, "inference", "outputs"),
                    help="parent output dir; a per-image subfolder (named after the image stem) is created inside it")
    ap.add_argument("--checkpoint", default=PHASE2_CKPT)
    ap.add_argument("--phase1-checkpoint", default=PHASE1_CKPT)
    ap.add_argument("--compare-phase1", action="store_true",
                    help="also sample the unconditioned Phase 1 model with the same seed and settings")
    ap.add_argument("--base-cfg", default="tppgaze/data/config.yaml")
    ap.add_argument("--stats-json", default="training/runs/phase2_embedding_only/summary_full.json")
    ap.add_argument("--n-sim", type=int, default=1,
                    help="batch size per sampling call (default 1); the first sample is shown")
    ap.add_argument("--n-samples", type=int, default=1,
                    help="number of independent samples per condition, seeds seed..seed+K-1 (default 1)")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                    help="sampling temperature tau (default: %(default)s, selected for behavioural generation; "
                         "not a likelihood change). tau=2.0 is the official TPP-Gaze config value and narrows spatial spread")
    ap.add_argument("--t-end", type=float, default=3.0, help="viewing window in seconds (default: %(default)s, as in training)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-image", action="store_true",
                    help="draw scanpaths on a blank canvas of the image size (for sharing without the stimulus)")
    ap.add_argument("--device", default=None, help="auto|cpu|cuda")
    args = ap.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device in (None, "auto") else args.device)
    args.device_resolved = device
    if not os.path.isfile(args.image):
        raise SystemExit(f"[inference] image not found: {args.image}")
    # per-image output folder (named after the image stem) inside the parent --out-dir
    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_dir = os.path.join(args.out_dir, stem)
    os.makedirs(out_dir, exist_ok=True)
    mean, std = load_stats(args.stats_json)
    print(f"[inference] device={device}  stats(mean/std log-inter)={mean:.4f}/{std:.4f}")
    models = {"phase2": load_model(args.checkpoint, args.base_cfg, mean, std, device)}
    hashes = {"phase2": sha256(args.checkpoint)}
    if args.compare_phase1:
        models["phase1"] = load_model(args.phase1_checkpoint, args.base_cfg, mean, std, device, phase1=True)
        hashes["phase1"] = sha256(args.phase1_checkpoint)
        print(f"[inference] Phase 1 loaded; base weights identical to Phase 2's base: {same_base(models['phase1'], models['phase2'])}")
    for k, v in hashes.items():
        print(f"[inference] {k} checkpoint sha256 {v[:16]}...")

    conditions = ["ASD", "TD"] if args.condition == "both" else [args.condition]
    if args.compare_phase1:
        conditions = [PHASE1] + conditions
    im = Image.open(args.image).convert("RGB")
    W, H = im.size
    tensor = get_preprocess()(im)                        # (3,512,512), official preprocessing
    settings = dict(t_end_s=args.t_end, n_sim=max(1, args.n_sim), note=NOTE)
    show = not args.no_image
    label = {PHASE1: "Phase 1 (unconditioned)", "ASD": "Phase 2, ASD condition", "TD": "Phase 2, TD condition"}
    fname = {PHASE1: "phase1", "ASD": "asd", "TD": "td"}

    if args.n_samples > 1:
        seeds = list(range(args.seed, args.seed + args.n_samples))
        panels = []
        for cond in conditions:
            model = models["phase1" if cond == PHASE1 else "phase2"]
            recs = []
            for s in seeds:
                spn = sample(model, tensor, cond, s, args.n_sim, args.temperature, args.t_end)[0]
                recs.append(dict(seed=s, n_fixations=len(spn), events=events(spn, W, H)))
                panels.append((label[cond], to_pixels(spn, W, H), f"seed {s}"))
            js = os.path.join(out_dir, f"{fname[cond]}_samples.json")
            json.dump(dict(condition=cond, image=os.path.abspath(args.image), image_width=W, image_height=H,
                           temperature=args.temperature, seeds=seeds, model=run_info(args, cond, hashes["phase1" if cond == PHASE1 else "phase2"]),
                           **settings, samples=recs), open(js, "w"), indent=2)
            print(f"[inference] {cond}: {len(seeds)} samples, fixations {[r['n_fixations'] for r in recs]} -> {js}")
        grid = os.path.join(out_dir, f"{stem}_samples_grid.png")
        save_panels(panels, im, grid, f"{len(seeds)} independent samples per condition: {os.path.basename(args.image)}",
                    footer_text(args, seeds), show, ncols=len(seeds))
        print(f"[inference] grid -> {grid}")
        return

    results = {}
    for cond in conditions:
        key = "phase1" if cond == PHASE1 else "phase2"
        spn = sample(models[key], tensor, cond, args.seed, args.n_sim, args.temperature, args.t_end)[0]
        sppx = to_pixels(spn, W, H)
        png = os.path.join(out_dir, f"{fname[cond]}_scanpath.png")
        js = os.path.join(out_dir, f"{fname[cond]}_scanpath.json")
        save_single(im, sppx, label[cond], png, footer_text(args), show)
        ev = events(spn, W, H)
        save_json(js, cond, args.image, W, H, sppx, spn, args.temperature, args.seed,
                  dict(model=run_info(args, cond, hashes[key]), **settings,
                       n_clipped=int(sum(e["clipped"] for e in ev)), events=ev))
        results[cond] = sppx
        if len(sppx) == 0:
            print(f"[inference] WARNING: {cond} sample has no fixation inside the {args.t_end:g} s window")
        print(f"[inference] {cond}: {len(sppx)} fixations  total={sppx[:,2].sum():.2f}s  -> {png}  {js}")

    if len(conditions) > 1:
        tag = "phase1_vs_asd_vs_td" if args.compare_phase1 and args.condition == "both" else \
              "_vs_".join(fname[c] for c in conditions)
        side = os.path.join(out_dir, f"{stem}_{tag}.png")
        save_panels([(label[c], results[c], None) for c in conditions], im, side,
                    f"Generated scanpaths: {os.path.basename(args.image)}", footer_text(args), show)
        print(f"[inference] side-by-side -> {side}")


if __name__ == "__main__":
    main()
