"""
Reproducible IMAGE-LEVEL split (70 / 15 / 15) for TPP-Gaze fine-tuning.

Splitting on image_id (not on individual scanpaths) prevents leakage of a
stimulus across train/val/test. Because every one of the 300 Saliency4ASD
images is viewed by BOTH ASD and TD observers, an image-level split keeps both
groups represented in every split automatically; we still assert it.
"""
from __future__ import annotations
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np


def make_image_split(
    samples: List[Dict],
    frac=(0.70, 0.15, 0.15),
    seed: int = 0,
) -> Dict[str, List[int]]:
    """Return {'train': [image_ids], 'val': [...], 'test': [...]} ."""
    assert abs(sum(frac) - 1.0) < 1e-6, "fractions must sum to 1"
    image_ids = sorted({s["image_id"] for s in samples})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(image_ids)

    n = len(perm)
    n_train = int(round(frac[0] * n))
    n_val = int(round(frac[1] * n))
    train = sorted(int(i) for i in perm[:n_train])
    val = sorted(int(i) for i in perm[n_train : n_train + n_val])
    test = sorted(int(i) for i in perm[n_train + n_val :])
    return {"train": train, "val": val, "test": test}


def assign_samples(samples: List[Dict], split: Dict[str, List[int]]) -> Dict[str, List[Dict]]:
    id2split = {}
    for name, ids in split.items():
        for i in ids:
            id2split[i] = name
    out = {"train": [], "val": [], "test": []}
    for s in samples:
        out[id2split[s["image_id"]]].append(s)
    return out


def summarise(assigned: Dict[str, List[Dict]], split: Dict[str, List[int]]) -> str:
    lines = []
    for name in ("train", "val", "test"):
        ss = assigned[name]
        n_asd = sum(1 for s in ss if s["group"] == "ASD")
        n_td = sum(1 for s in ss if s["group"] == "TD")
        assert n_asd > 0 and n_td > 0, f"split {name} missing a group!"
        lines.append(
            f"  {name:5s}: images={len(split[name]):3d}  scanpaths={len(ss):5d}  "
            f"ASD={n_asd:5d}  TD={n_td:5d}"
        )
    return "\n".join(lines)


def save_split(split: Dict[str, List[int]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)
