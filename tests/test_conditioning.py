"""The Phase 2 mechanism: a zero-initialised 2 x 512 embedding added to the fused context.

Builds the official TPP-Gaze architecture with random head weights (no checkpoint needed).
TPP-Gaze loads DenseNet201 through torch.hub, so the first run needs internet access or a
populated torch hub cache; without it these tests are skipped.
"""
from __future__ import annotations
import os
import shutil
import tempfile
import unittest

import torch

from tests._common import ROOT
from training import model_utils as mu
from training.conditioned import ConditionedTPPGaze


def build_random_base():
    from tppgaze.tppgaze import TPPGaze
    tmp = tempfile.mkdtemp()
    cfg = os.path.join(tmp, "config.yaml")
    shutil.copyfile(ROOT / "tppgaze" / "data" / "config.yaml", cfg)
    try:
        return TPPGaze(cfg, None, torch.device("cpu")).eval()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


try:
    torch.manual_seed(0)
    BASE = build_random_base()
    SKIP = None
except Exception as e:  # no network and no torch hub cache for DenseNet201
    BASE, SKIP = None, f"TPP-Gaze could not be built ({type(e).__name__}); needs DenseNet201 via torch.hub"


@unittest.skipIf(BASE is None, SKIP or "")
class Conditioning(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        B, L = 4, 6
        self.inter = torch.rand(B, L) * 0.4 + 0.08
        self.marks = torch.rand(B, L, 2) * 2 - 1
        self.mask = torch.ones(B, L)
        self.meta = torch.randn(B, 1, 256)
        self.model = ConditionedTPPGaze(BASE, cond_dim=512).eval()

    def nll(self, group):
        g = torch.full((4,), group, dtype=torch.long)
        with torch.no_grad():
            return float(self.model.nll(self.inter, self.marks, self.mask, self.meta, g)["nll"])

    def test_embedding_is_2_by_512_and_starts_at_zero(self):
        self.assertEqual(tuple(self.model.group_emb.weight.shape), (2, 512))
        self.assertEqual(sum(p.numel() for p in self.model.group_emb.parameters()), 1024)
        self.assertEqual(float(self.model.group_emb.weight.abs().sum()), 0.0)

    def test_zero_embedding_equals_phase1(self):
        with torch.no_grad():
            base = float(mu.tppgaze_nll(BASE, self.inter, self.marks, self.mask, self.meta)["nll"])
        self.assertEqual(self.nll(0), base)   # ASD label
        self.assertEqual(self.nll(1), base)   # TD label

    def test_nonzero_embedding_changes_the_likelihood_by_group(self):
        with torch.no_grad():
            self.model.group_emb.weight.copy_(torch.randn(2, 512) * 0.1)
        self.assertNotAlmostEqual(self.nll(0), self.nll(1), places=6)

    def test_only_the_embedding_is_trainable_in_embedding_only_mode(self):
        for p in self.model.base.parameters():
            p.requires_grad = False
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(trainable, 1024)
        for p in self.model.base.parameters():
            p.requires_grad = True


if __name__ == "__main__":
    unittest.main()
