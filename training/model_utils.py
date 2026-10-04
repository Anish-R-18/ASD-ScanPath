"""
Thin glue around the ORIGINAL TPP-Gaze model (../tppgaze).

Nothing here re-implements the network. We:
  * build the official `TPPGaze` module with train-split log-inter-time stats,
  * load the pretrained checkpoint,
  * freeze the DenseNet visual backbone,
  * compute the joint NLL of Eq. (10) of the paper by calling the model's own
    submodules (get_features / get_context / merge / inter_time_dist / marks_dist),
  * optionally cache the FROZEN backbone feature maps so the (trainable) metadata
    head can be fine-tuned without recomputing DenseNet every step. The cached
    path reproduces `metadata.forward` exactly (same layers, same order); it is
    only an efficiency shortcut for the frozen part.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from omegaconf import OmegaConf

from tppgaze.tppgaze import TPPGaze


# --------------------------------------------------------------------------- #
# Inter-time statistics (recomputed from the TRAIN split only)
# --------------------------------------------------------------------------- #
def compute_log_inter_time_stats(train_samples: List[Dict]) -> Tuple[float, float]:
    """mean / std of log(fixation_duration_seconds) over every event in train."""
    logs = []
    for s in train_samples:
        for fx in s["scanpath"]:
            logs.append(math.log(fx[2] + 1e-8))
    logs = np.asarray(logs, dtype=np.float64)
    return float(logs.mean()), float(logs.std())


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_model(
    model_cfg_path: str,
    checkpoint_path: str,
    device: torch.device,
    mean_log_inter_time: Optional[float] = None,
    std_log_inter_time: Optional[float] = None,
    used_cfg_out: Optional[str] = None,
):
    """Construct official TPPGaze, optionally override log-inter-time stats,
    load the pretrained weights. Returns (model, used_cfg)."""
    cfg = OmegaConf.load(model_cfg_path)
    if mean_log_inter_time is not None:
        cfg.mean_log_inter_times = float(mean_log_inter_time)
    if std_log_inter_time is not None:
        cfg.std_log_inter_times = float(std_log_inter_time)

    # TPPGaze.__init__ loads the cfg from a path, so persist the (possibly
    # overridden) cfg and hand that path to the constructor.
    if used_cfg_out is None:
        used_cfg_out = model_cfg_path + ".used.yaml"
    OmegaConf.save(cfg, used_cfg_out)

    model = TPPGaze(used_cfg_out, checkpoint_path, device)
    model.to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    _fix_unregistered_device(model, device)
    return model, cfg


def _fix_unregistered_device(model, device) -> None:
    """Move tensors that the original model stores as plain attributes (NOT
    registered buffers/parameters) onto `device`.

    The official Encoder keeps `self.position_vec` as a raw tensor
    (transformer_encoder.py), so `model.to(device)` silently leaves it on CPU and
    GPU runs crash with a device mismatch. We fix the placement at runtime instead
    of editing the original model file. No-op on CPU.
    """
    for mod in model.modules():
        registered = set(dict(mod.named_buffers(recurse=False))) | set(
            dict(mod.named_parameters(recurse=False))
        )
        for name, val in list(vars(mod).items()):
            if isinstance(val, torch.Tensor) and name not in registered:
                setattr(mod, name, val.to(device))


def freeze_backbone(model) -> Tuple[int, int]:
    """Freeze ONLY the DenseNet201 visual backbone. Returns (#frozen, #trainable)
    parameter tensors."""
    for p in model.metadata.backbone.parameters():
        p.requires_grad = False
    model.metadata.backbone.eval()  # keep frozen BN stats fixed
    n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    return n_frozen, n_train


def set_train_mode(model, freeze_bb: bool = True) -> None:
    model.train()
    if freeze_bb:
        model.metadata.backbone.eval()  # backbone BN must stay in eval


# --------------------------------------------------------------------------- #
# Metadata (image) features, with an optional frozen-backbone cache
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_backbone_cache(model, image_bank, image_ids: List[int], device) -> Dict[int, torch.Tensor]:
    """Precompute the FROZEN DenseNet backbone feature map per image (on CPU)."""
    cache: Dict[int, torch.Tensor] = {}
    model.metadata.backbone.eval()
    for iid in image_ids:
        img = image_bank.get(iid).unsqueeze(0).to(device)
        feats = model.metadata.backbone(img)  # (1, C, H, W), C=2048
        cache[iid] = feats.squeeze(0).cpu()
    return cache


def _metadata_head(model, backbone_feats: torch.Tensor, device) -> torch.Tensor:
    """Run the TRAINABLE metadata head on precomputed backbone features.

    Mirrors DenseNetDG3ish.forward() after `self.backbone(images)` exactly."""
    m = model.metadata
    x = m.coordconv(backbone_feats, device)
    x = m.conv_activation1(m.image_norm1(m.conv1(x)))
    x = m.conv_activation2(m.image_norm2(m.conv2(x)))
    x = m.conv_activation3(m.image_norm3(m.conv3(x)))
    x = m.flatten(x)
    x = m.fc1(x)
    x = m.activation(x)
    return x  # (B, 1, output_dim)


def compute_metadata(
    model,
    image_ids: List[int],
    image_bank,
    device,
    backbone_cache: Optional[Dict[int, torch.Tensor]] = None,
) -> torch.Tensor:
    """Return image/semantic metadata (B, 1, 256).

    If `backbone_cache` is given, only the trainable head runs (fast, grads flow
    to head). Otherwise the full official get_metadata() runs end to end."""
    if backbone_cache is not None:
        feats = torch.stack([backbone_cache[i] for i in image_ids], dim=0).to(device)
        return _metadata_head(model, feats, device)
    imgs = image_bank.batch(image_ids).to(device)
    return model.get_metadata(imgs)


# --------------------------------------------------------------------------- #
# Joint negative log-likelihood  -- paper Eq. (10)
#   L = - sum_n [ log p(tau_n | c_n) + log p(r_Fn | c_n) ]
# summed over real events (mask), reported as a per-event mean.
# --------------------------------------------------------------------------- #
def tppgaze_nll(
    model,
    inter_times: torch.Tensor,  # (B, L) fixation durations [s]
    marks: torch.Tensor,        # (B, L, 2) [x, y] in [-1, 1]
    mask: torch.Tensor,         # (B, L) 1 real / 0 pad
    metadata: torch.Tensor,     # (B, 1, 256)
    cond: Optional[torch.Tensor] = None,  # (B, C) Phase-2 group conditioning, added to context
) -> Dict[str, torch.Tensor]:
    # History over past events (causal in time); context[k] predicts event k.
    features = model.get_features(inter_times, marks)          # (B, L, 3)
    history = model.get_context(features, remove_last=True)    # (B, L, 256)
    context = model.merge.forward([history, metadata])         # (B, L, 512)

    # Phase-2: fuse the group condition into the fused history+image context,
    # broadcast across the sequence, right before the prediction heads. cond=None
    # (Phase-1) leaves the computation unchanged.
    if cond is not None:
        context = context + cond.unsqueeze(1)                  # (B, 1, C) -> (B, L, C)

    # Training uses the RAW distributions (temperature=None); temperature is a
    # sampling-time knob only.
    inter_dist = model.inter_time_dist.forward(context, None)
    marks_dist = model.marks_dist.forward(context, None)

    # inter_times must be > 0 for the log-normal density; padded steps are 0, so
    # substitute 1.0 there and mask them out afterwards.
    it_safe = torch.where(mask.bool(), inter_times, torch.ones_like(inter_times))
    logp_tau = inter_dist.log_prob(it_safe)                    # (B, L)
    logp_pos = marks_dist.log_prob(marks)                      # (B, L)

    n_events = mask.sum().clamp_min(1.0)
    nll_tau = -(logp_tau * mask).sum() / n_events
    nll_pos = -(logp_pos * mask).sum() / n_events
    nll = nll_tau + nll_pos
    return {
        "nll": nll,
        "nll_duration": nll_tau,
        "nll_position": nll_pos,
        "n_events": n_events.detach(),
    }
