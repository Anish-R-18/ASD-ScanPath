"""
TPP-Gaze fine-tuning pipeline -- Phase 1 baseline (combined ASD + TD).

Reuses the ORIGINAL model under ../tppgaze unchanged. Implements the paper's
joint NLL (Eq. 10: duration NLL + position NLL), AdamW, early stopping, best-
checkpoint saving, and train/val logging. Works on CUDA when available, else CPU.

Subcommands
-----------
  smoke   : load a few samples -> forward -> loss -> backward; verify grads,
            masks, and finiteness (NaN/Inf). No optimisation.
  short   : a few epochs on a small image subset to confirm the loop runs and
            loss can decrease. (Use for the local CPU sanity check.)
  train   : full fine-tuning (intended for the RTX 5060 GPU box).

Examples
--------
  python training/train_tpp.py smoke --config training/config.yaml
  python training/train_tpp.py short --config training/config.yaml --epochs 3 --limit-images 12 --batch-size 16
  python training/train_tpp.py train --config training/config.yaml
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Make both the project root and the cloned repo importable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for p in (_ROOT, os.path.join(_ROOT, "tppgaze")):
    if p not in sys.path:
        sys.path.insert(0, p)

from training.dataset import (  # noqa: E402
    ScanpathDataset,
    ImageBank,
    collate_fn,
    load_scanpaths,
)
from training.splits import (  # noqa: E402
    make_image_split,
    assign_samples,
    summarise,
    save_split,
)
from training import model_utils as mu  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def resolve(root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root, path)


def load_cfg(config_path: str):
    cfg = OmegaConf.load(config_path)
    for k in ("model_cfg_path", "checkpoint_path", "asd_json", "td_json", "images_dir", "out_dir"):
        cfg[k] = resolve(_ROOT, cfg[k])
    return cfg


def make_loader(samples, batch_size, shuffle, seed, num_workers=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ScanpathDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=g,
        drop_last=False,
    )


def run_batch(model, batch, image_bank, device, backbone_cache):
    inter = batch["inter_times"].to(device)
    marks = batch["marks"].to(device)
    mask = batch["mask"].to(device)
    metadata = mu.compute_metadata(
        model, batch["image_ids"], image_bank, device, backbone_cache=backbone_cache
    )
    return mu.tppgaze_nll(model, inter, marks, mask, metadata)


@torch.no_grad()
def evaluate(model, loader, image_bank, device, backbone_cache, limit_batches=None):
    model.eval()
    tot_nll = tot_dur = tot_pos = tot_ev = 0.0
    for bi, batch in enumerate(loader):
        if limit_batches is not None and bi >= limit_batches:
            break
        out = run_batch(model, batch, image_bank, device, backbone_cache)
        ev = float(out["n_events"])
        tot_nll += float(out["nll"]) * ev
        tot_dur += float(out["nll_duration"]) * ev
        tot_pos += float(out["nll_position"]) * ev
        tot_ev += ev
    tot_ev = max(tot_ev, 1.0)
    return {"nll": tot_nll / tot_ev, "nll_duration": tot_dur / tot_ev, "nll_position": tot_pos / tot_ev}


# --------------------------------------------------------------------------- #
# data preparation shared by all subcommands
# --------------------------------------------------------------------------- #
def prepare(cfg, device, limit_images=None):
    samples = load_scanpaths(cfg.asd_json, cfg.td_json)
    split = make_image_split(samples, frac=tuple(cfg.split), seed=cfg.seed)
    assigned = assign_samples(samples, split)

    if limit_images is not None:
        keep_tr = set(split["train"][:limit_images])
        keep_va = set(split["val"][:max(1, limit_images // 4)])
        assigned["train"] = [s for s in assigned["train"] if s["image_id"] in keep_tr]
        assigned["val"] = [s for s in assigned["val"] if s["image_id"] in keep_va]

    # Train-split-only log-inter-time stats (recomputed; NOT the pretrained ones).
    mean_lit, std_lit = mu.compute_log_inter_time_stats(assigned["train"])

    os.makedirs(cfg.out_dir, exist_ok=True)
    used_cfg_path = os.path.join(cfg.out_dir, "model_config_used.yaml")
    model, used_cfg = mu.build_model(
        cfg.model_cfg_path, cfg.checkpoint_path, device,
        mean_log_inter_time=mean_lit, std_log_inter_time=std_lit,
        used_cfg_out=used_cfg_path,
    )
    n_frozen, n_train = (0, 0)
    if cfg.freeze_backbone:
        n_frozen, n_train = mu.freeze_backbone(model)

    image_bank = ImageBank(cfg.images_dir, ext=cfg.image_ext)

    backbone_cache = None
    if cfg.get("cache_backbone", True):
        used_ids = sorted({s["image_id"] for s in assigned["train"] + assigned["val"]})
        backbone_cache = mu.build_backbone_cache(model, image_bank, used_ids, device)

    info = {
        "split": split,
        "assigned": assigned,
        "mean_log_inter_time": mean_lit,
        "std_log_inter_time": std_lit,
        "pretrained_mean_log_inter_time": float(OmegaConf.load(cfg.model_cfg_path).mean_log_inter_times),
        "pretrained_std_log_inter_time": float(OmegaConf.load(cfg.model_cfg_path).std_log_inter_times),
        "n_frozen_tensors": n_frozen,
        "n_trainable_tensors": n_train,
    }
    return model, image_bank, backbone_cache, info


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #
def cmd_smoke(cfg, args):
    device = pick_device(cfg.device if args.device is None else args.device)
    set_global_seed(cfg.seed)
    print(f"[smoke] device = {device}")
    model, image_bank, backbone_cache, info = prepare(cfg, device, limit_images=4)

    print(f"[smoke] recomputed train-split log-inter-time stats: "
          f"mean={info['mean_log_inter_time']:.6f} std={info['std_log_inter_time']:.6f} "
          f"(pretrained: mean={info['pretrained_mean_log_inter_time']:.6f} "
          f"std={info['pretrained_std_log_inter_time']:.6f})")
    print(f"[smoke] param tensors: frozen={info['n_frozen_tensors']} trainable={info['n_trainable_tensors']}")

    loader = make_loader(info["assigned"]["train"], batch_size=4, shuffle=True, seed=cfg.seed)
    batch = next(iter(loader))

    print(f"[smoke] batch shapes: marks={tuple(batch['marks'].shape)} "
          f"inter_times={tuple(batch['inter_times'].shape)} mask={tuple(batch['mask'].shape)} "
          f"lengths={batch['lengths'].tolist()} groups={batch['groups']}")

    # mask sanity: #ones per row must equal each scanpath length
    ok_mask = torch.equal(batch["mask"].sum(1).long(), batch["lengths"])
    print(f"[smoke] mask matches lengths: {ok_mask}")

    mu.set_train_mode(model, freeze_bb=cfg.freeze_backbone)
    inter = batch["inter_times"].to(device)
    marks = batch["marks"].to(device)
    mask = batch["mask"].to(device)
    metadata = mu.compute_metadata(model, batch["image_ids"], image_bank, device, backbone_cache)
    print(f"[smoke] metadata shape: {tuple(metadata.shape)}")

    out = mu.tppgaze_nll(model, inter, marks, mask, metadata)
    nll = out["nll"]
    print(f"[smoke] loss: nll={float(nll):.4f}  duration={float(out['nll_duration']):.4f}  "
          f"position={float(out['nll_position']):.4f}  n_events={int(out['n_events'])}")

    finite = torch.isfinite(nll).item()
    print(f"[smoke] loss finite (no NaN/Inf): {finite}")

    nll.backward()
    # gradient checks over TRAINABLE params only
    grads = [(n, p) for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
    n_with_grad = len(grads)
    n_trainable = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    total_norm = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for _, p in grads)))
    any_nonzero = any(float(p.grad.abs().sum()) > 0 for _, p in grads)
    all_finite_grad = all(torch.isfinite(p.grad).all().item() for _, p in grads)
    # confirm the FROZEN backbone received no gradient
    bb_grad = any(p.grad is not None for p in model.metadata.backbone.parameters())

    print(f"[smoke] trainable tensors={n_trainable}  with_grad={n_with_grad}  "
          f"grad_global_norm={total_norm:.4f}")
    print(f"[smoke] gradients non-zero: {any_nonzero}   all grads finite: {all_finite_grad}")
    print(f"[smoke] frozen backbone received grad: {bb_grad}  (expected False)")

    # a few representative trainable grads
    show = ["inter_time_dist.linear.weight", "marks_dist.linear.weight",
            "metadata.fc1.weight", "context.context_init"]
    for name, p in model.named_parameters():
        if name in show and p.grad is not None:
            print(f"[smoke]   grad[{name}] L2={float(p.grad.norm()):.4e}")

    passed = ok_mask and finite and any_nonzero and all_finite_grad and (not bb_grad)
    print(f"[smoke] RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def train_loop(cfg, args, device, model, image_bank, backbone_cache, info, epochs, tag):
    os.makedirs(cfg.out_dir, exist_ok=True)
    save_split(info["split"], os.path.join(cfg.out_dir, "split.json"))

    tr_loader = make_loader(info["assigned"]["train"], cfg.batch_size, True, cfg.seed, cfg.num_workers)
    va_loader = make_loader(info["assigned"]["val"], cfg.batch_size, False, cfg.seed, cfg.num_workers)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    log_path = os.path.join(cfg.out_dir, f"log_{tag}.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_nll", "train_dur", "train_pos",
             "val_nll", "val_dur", "val_pos", "lr", "secs", "best"]
        )

    best_val = float("inf")
    best_epoch = -1
    best_path = os.path.join(cfg.out_dir, "best_model.pth")
    epochs_no_improve = 0
    history = []

    # epoch 0 = pretrained-model baseline on our val set (no training yet)
    init_val = evaluate(model, va_loader, image_bank, device, backbone_cache, cfg.get("limit_val_batches"))
    print(f"[{tag}] initial (pretrained) val: nll={init_val['nll']:.4f} "
          f"dur={init_val['nll_duration']:.4f} pos={init_val['nll_position']:.4f}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        mu.set_train_mode(model, freeze_bb=cfg.freeze_backbone)
        tot_nll = tot_dur = tot_pos = tot_ev = 0.0
        for bi, batch in enumerate(tr_loader):
            if cfg.get("limit_train_batches") is not None and bi >= cfg.limit_train_batches:
                break
            opt.zero_grad(set_to_none=True)
            out = run_batch(model, batch, image_bank, device, backbone_cache)
            out["nll"].backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
            opt.step()
            ev = float(out["n_events"])
            tot_nll += float(out["nll"]) * ev
            tot_dur += float(out["nll_duration"]) * ev
            tot_pos += float(out["nll_position"]) * ev
            tot_ev += ev
        tot_ev = max(tot_ev, 1.0)
        tr = {"nll": tot_nll / tot_ev, "nll_duration": tot_dur / tot_ev, "nll_position": tot_pos / tot_ev}

        va = evaluate(model, va_loader, image_bank, device, backbone_cache, cfg.get("limit_val_batches"))
        secs = time.time() - t0

        improved = va["nll"] < best_val - 1e-6
        if improved:
            best_val, best_epoch, epochs_no_improve = va["nll"], epoch, 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1

        print(f"[{tag}] epoch {epoch:3d}/{epochs}  "
              f"train nll={tr['nll']:.4f} (dur={tr['nll_duration']:.4f} pos={tr['nll_position']:.4f})  "
              f"val nll={va['nll']:.4f} (dur={va['nll_duration']:.4f} pos={va['nll_position']:.4f})  "
              f"{secs:.1f}s{'  *best' if improved else ''}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, tr["nll"], tr["nll_duration"], tr["nll_position"],
                 va["nll"], va["nll_duration"], va["nll_position"], cfg.lr, round(secs, 2), int(improved)]
            )
        history.append({"epoch": epoch, "train": tr, "val": va})

        if epochs_no_improve >= cfg.patience:
            print(f"[{tag}] early stopping at epoch {epoch} (no val improvement for {cfg.patience} epochs)")
            break

    summary = {
        "tag": tag, "epochs_run": len(history),
        "initial_val": init_val,
        "best_val_nll": best_val, "best_epoch": best_epoch,
        "best_checkpoint": best_path,
        "first_epoch": history[0] if history else None,
        "last_epoch": history[-1] if history else None,
        "mean_log_inter_time": info["mean_log_inter_time"],
        "std_log_inter_time": info["std_log_inter_time"],
    }
    with open(os.path.join(cfg.out_dir, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def cmd_short(cfg, args):
    device = pick_device(cfg.device if args.device is None else args.device)
    set_global_seed(cfg.seed)
    if args.batch_size:
        cfg.batch_size = args.batch_size
    limit_images = args.limit_images if args.limit_images is not None else (cfg.get("limit_images") or 12)
    epochs = args.epochs or 3
    print(f"[short] device={device}  epochs={epochs}  limit_images={limit_images}  batch_size={cfg.batch_size}")
    model, image_bank, backbone_cache, info = prepare(cfg, device, limit_images=limit_images)
    print("[short] split summary:\n" + summarise(info["assigned"], info["split"]))
    print(f"[short] train scanpaths used={len(info['assigned']['train'])} "
          f"val scanpaths used={len(info['assigned']['val'])}")
    summary = train_loop(cfg, args, device, model, image_bank, backbone_cache, info, epochs, tag="short")
    f, l = summary["first_epoch"], summary["last_epoch"]
    print(f"\n[short] loss before vs after:")
    print(f"        initial(pretrained) val nll = {summary['initial_val']['nll']:.4f}")
    if f and l:
        print(f"        epoch 1 train nll          = {f['train']['nll']:.4f}")
        print(f"        epoch {summary['epochs_run']} train nll          = {l['train']['nll']:.4f}")
        print(f"        epoch 1 val   nll          = {f['val']['nll']:.4f}")
        print(f"        epoch {summary['epochs_run']} val   nll          = {l['val']['nll']:.4f}")
    print(f"        best val nll = {summary['best_val_nll']:.4f} @ epoch {summary['best_epoch']}")
    return 0


def cmd_train(cfg, args):
    device = pick_device(cfg.device if args.device is None else args.device)
    set_global_seed(cfg.seed)
    if args.batch_size:
        cfg.batch_size = args.batch_size
    epochs = args.epochs or cfg.max_epochs
    print(f"[train] device={device}  epochs(max)={epochs}  batch_size={cfg.batch_size}")
    model, image_bank, backbone_cache, info = prepare(cfg, device, limit_images=None)
    print("[train] split summary:\n" + summarise(info["assigned"], info["split"]))
    summary = train_loop(cfg, args, device, model, image_bank, backbone_cache, info, epochs, tag="full")
    print(f"[train] done. best val nll={summary['best_val_nll']:.4f} @ epoch {summary['best_epoch']}  "
          f"-> {summary['best_checkpoint']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="TPP-Gaze fine-tuning (Phase 1 baseline)")
    ap.add_argument("cmd", choices=["smoke", "short", "train"])
    ap.add_argument("--config", default=os.path.join(_THIS_DIR, "config.yaml"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit-images", type=int, default=None)
    ap.add_argument("--device", default=None, help="auto|cpu|cuda")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    return {"smoke": cmd_smoke, "short": cmd_short, "train": cmd_train}[args.cmd](cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
