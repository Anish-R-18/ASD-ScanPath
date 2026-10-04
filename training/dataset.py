"""
Dataset / loader for TPP-Gaze fine-tuning on Saliency4ASD (Phase 1 baseline).

This module ONLY handles data. It does not touch the original TPP-Gaze model
code under ../tppgaze. Each processed scanpath is

    {image_id, group, scanpath: [[x, y, dur_s], ...]}

with x, y already normalised to [-1, 1] (model convention: x left=-1 right=+1,
y top=-1 bottom=+1) and dur_s the fixation duration in SECONDS, which is exactly
the inter-event time tau_n consumed by TPP-Gaze.

A sample exposes:
    marks       (L, 2)  float32   -> [x, y]
    inter_times (L,)    float32   -> fixation duration (seconds) == tau_n
    image_id    int
    group       str ("ASD" | "TD")

ASD and TD are combined for the baseline, but every sample keeps its `group`
field so later experiments can condition on / split by group.
"""
from __future__ import annotations
import json
import os
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset

# Reuse the ORIGINAL repo preprocessing (Resize(512,512) + ToTensor + ImageNet norm)
# and the ORIGINAL padding helper, so images are fed exactly as the pretrained
# model expects.
from tppgaze.utils import get_preprocess, pad_sequence
from PIL import Image

# Phase-2 group conditioning labels (participant IDs are unavailable, so we use group).
GROUP_TO_ID = {"ASD": 0, "TD": 1}
ID_TO_GROUP = {v: k for k, v in GROUP_TO_ID.items()}


def load_scanpaths(asd_json: str, td_json: str) -> List[Dict]:
    """Load + combine ASD and TD scanpaths, preserving the group field."""
    samples: List[Dict] = []
    for path in (asd_json, td_json):
        with open(path, "r") as f:
            for s in json.load(f):
                samples.append(
                    {
                        "image_id": int(s["image_id"]),
                        "group": s["group"],
                        "scanpath": s["scanpath"],
                    }
                )
    return samples


class ImageBank:
    """Loads + preprocesses each stimulus once and caches the (3,512,512) tensor.

    300 images -> ~0.9 GB if fully materialised; we cache lazily on CPU and move
    per-batch to the device in the trainer.
    """

    def __init__(self, images_dir: str, ext: str = ".png"):
        self.images_dir = images_dir
        self.ext = ext
        self._preprocess = get_preprocess()
        self._cache: Dict[int, torch.Tensor] = {}

    def path_for(self, image_id: int) -> str:
        return os.path.join(self.images_dir, f"{image_id}{self.ext}")

    def get(self, image_id: int) -> torch.Tensor:
        t = self._cache.get(image_id)
        if t is None:
            img = Image.open(self.path_for(image_id)).convert("RGB")
            t = self._preprocess(img)  # (3,512,512), float32
            self._cache[image_id] = t
        return t

    def batch(self, image_ids: List[int]) -> torch.Tensor:
        return torch.stack([self.get(i) for i in image_ids], dim=0)


class ScanpathDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        sp = torch.tensor(s["scanpath"], dtype=torch.float32)  # (L, 3): x, y, dur_s
        marks = sp[:, :2].contiguous()                         # (L, 2)
        inter_times = sp[:, 2].contiguous()                    # (L,)
        return {
            "marks": marks,
            "inter_times": inter_times,
            "length": sp.shape[0],
            "image_id": s["image_id"],
            "group": s["group"],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Right-pad variable-length scanpaths with 0 and build a float mask.

    marks       -> (B, Lmax, 2)  (pad value 0 == a valid coord, masked out in loss)
    inter_times -> (B, Lmax)     (pad value 0; loss substitutes 1.0 at padded steps)
    mask        -> (B, Lmax)     1.0 at real events, 0.0 at padding
    """
    marks = pad_sequence([b["marks"] for b in batch], padding_value=0.0)
    inter_times = pad_sequence([b["inter_times"] for b in batch], padding_value=0.0)
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    Lmax = marks.shape[1]
    mask = (torch.arange(Lmax)[None, :] < lengths[:, None]).float()  # (B, Lmax)
    # Group label for Phase-2 conditioning: ASD -> 0, TD -> 1 (Phase-1 code ignores it).
    group_ids = torch.tensor([GROUP_TO_ID[b["group"]] for b in batch], dtype=torch.long)
    return {
        "marks": marks,
        "inter_times": inter_times,
        "mask": mask,
        "lengths": lengths,
        "image_ids": [b["image_id"] for b in batch],
        "groups": [b["group"] for b in batch],
        "group_ids": group_ids,
    }
