"""
Dedicated loader for Gaze4ASD external evaluation.

Kept separate from training/dataset.py, whose load_scanpaths() requires numeric
image IDs, so the training pipeline and its data format stay untouched. Stimulus
IDs here are filenames such as top_image_1.png, each resolved to its image file.
Malformed data raises instead of being silently skipped.
"""
from __future__ import annotations
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from tppgaze.utils import get_preprocess, pad_sequence

GROUP_TO_ID = {"ASD": 0, "TD": 1}
MIN_FIX_S, T_END_S, MIN_LEN = 0.060, 3.0, 3
EPS = 1e-6


class StimulusBank:
    """Resolves a string stimulus ID to its image, native size, and model input tensor."""

    def __init__(self, images_dir: str):
        self.images_dir = images_dir
        self.preprocess = get_preprocess()
        self._tensors: Dict[str, torch.Tensor] = {}
        self._dims: Dict[str, tuple] = {}

    def path(self, image_id: str) -> str:
        p = os.path.join(self.images_dir, str(image_id))
        if not os.path.isfile(p):
            raise FileNotFoundError(f"No stimulus file for image_id={image_id!r}: {p}")
        return p

    def dims(self, image_id: str) -> tuple:
        if image_id not in self._dims:
            with Image.open(self.path(image_id)) as im:
                self._dims[image_id] = im.size
        return self._dims[image_id]

    def tensor(self, image_id: str) -> torch.Tensor:
        if image_id not in self._tensors:
            self._tensors[image_id] = self.preprocess(Image.open(self.path(image_id)).convert("RGB"))
        return self._tensors[image_id]


def load_external_scanpaths(path: str, bank: StimulusBank) -> List[Dict]:
    """Load one external JSON, keeping file order, and validate every record."""
    records = json.load(open(path, encoding="utf-8"))
    seen = set()
    for i, r in enumerate(records):
        where = f"{os.path.basename(path)} record {i}"
        for k in ("image_id", "group", "participant", "scanpath"):
            if k not in r:
                raise ValueError(f"{where}: missing {k}")
        if r["group"] not in GROUP_TO_ID:
            raise ValueError(f"{where}: bad group {r['group']!r}")
        key = (r["participant"], r["image_id"])
        if key in seen:
            raise ValueError(f"{where}: duplicate participant-image {key}")
        seen.add(key)
        W, H = bank.dims(r["image_id"])
        if "image_width" in r and (r["image_width"], r["image_height"]) != (W, H):
            raise ValueError(f"{where}: stored size differs from stimulus file")
        sp = np.asarray(r["scanpath"], dtype=float)
        if sp.ndim != 2 or sp.shape[1] != 3 or len(sp) < MIN_LEN or not np.isfinite(sp).all():
            raise ValueError(f"{where}: bad scanpath shape or values")
        if np.abs(sp[:, :2]).max() > 1.0 or sp[:, 2].min() < MIN_FIX_S - EPS or sp[:, 2].sum() > T_END_S + EPS:
            raise ValueError(f"{where}: coordinate or duration out of range")
    return records


def summarize_records(records: List[Dict], bank: StimulusBank) -> Dict:
    durs = np.concatenate([np.asarray(r["scanpath"])[:, 2] for r in records])
    xy = np.concatenate([np.asarray(r["scanpath"])[:, :2] for r in records])
    lens = np.array([len(r["scanpath"]) for r in records])
    ids = sorted({r["image_id"] for r in records})
    return {
        "participants_ASD": len({r["participant"] for r in records if r["group"] == "ASD"}),
        "participants_TD": len({r["participant"] for r in records if r["group"] == "TD"}),
        "participants_in_both_groups": len({r["participant"] for r in records if r["group"] == "ASD"} & {r["participant"] for r in records if r["group"] == "TD"}),
        "trials_ASD": sum(r["group"] == "ASD" for r in records),
        "trials_TD": sum(r["group"] == "TD" for r in records),
        "fixation_events": int(lens.sum()),
        "stimuli": len(ids),
        "stimuli_resolved_to_files": sum(os.path.isfile(bank.path(i)) for i in ids),
        "image_sizes": sorted({bank.dims(i) for i in ids}),
        "x_range": [float(xy[:, 0].min()), float(xy[:, 0].max())],
        "y_range": [float(xy[:, 1].min()), float(xy[:, 1].max())],
        "duration_range_s": [float(durs.min()), float(durs.max())],
        "scanpath_length_range": [int(lens.min()), int(lens.max())],
    }


def collate_records(records: List[Dict], device: Optional[torch.device] = None) -> Dict:
    """Pad one batch exactly as training/dataset.collate_fn does."""
    marks = pad_sequence([torch.tensor(np.asarray(r["scanpath"])[:, :2], dtype=torch.float32) for r in records])
    inter = pad_sequence([torch.tensor(np.asarray(r["scanpath"])[:, 2], dtype=torch.float32) for r in records])
    lens = torch.tensor([len(r["scanpath"]) for r in records], dtype=torch.long)
    mask = (torch.arange(marks.shape[1])[None, :] < lens[:, None]).float()
    gids = torch.tensor([GROUP_TO_ID[r["group"]] for r in records], dtype=torch.long)
    out = {"marks": marks, "inter_times": inter, "mask": mask, "group_ids": gids}
    return {k: v.to(device) for k, v in out.items()} if device is not None else out
