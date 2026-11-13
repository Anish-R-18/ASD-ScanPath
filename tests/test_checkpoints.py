"""Checks on the trained checkpoints. They are not distributed, so these tests are skipped unless the
files are present at the paths the code expects (see docs/handover_guide.md)."""
from __future__ import annotations
import hashlib
import unittest

import torch

from tests._common import ROOT

PHASE1 = ROOT / "training" / "runs" / "baseline" / "best_model.pth"
PHASE2 = ROOT / "training" / "runs" / "phase2_embedding_only" / "best_model_conditioned.pth"
# SHA-256 recorded when the results were produced (training/runs/research_completion/e1_phase_tau1_v2/environment.json)
EXPECTED_SHA = {PHASE1: "b613413027e775b98e449cbdbc9afa0253d6bd40f0cb2502230ed5bf1000b0bb",
                PHASE2: "35319673382378487d23c1871cab0cf05b71b1e7963e180ec024e5d80901a9a4"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@unittest.skipUnless(PHASE1.exists() and PHASE2.exists(), "checkpoints not present (not distributed)")
class Checkpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p1 = torch.load(PHASE1, map_location="cpu")
        raw = torch.load(PHASE2, map_location="cpu")
        cls.p2 = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw

    def test_hashes_match_the_reported_runs(self):
        for path, digest in EXPECTED_SHA.items():
            with self.subTest(checkpoint=path.name):
                self.assertEqual(sha256(path), digest)

    def test_phase2_adds_only_a_2x512_embedding_to_identical_base_weights(self):
        extra = [k for k in self.p2 if not k.startswith("base.")]
        self.assertEqual(extra, ["group_emb.weight"])
        self.assertEqual(tuple(self.p2["group_emb.weight"].shape), (2, 512))
        base2 = {k[len("base."):]: v for k, v in self.p2.items() if k.startswith("base.")}
        self.assertEqual(set(base2), set(self.p1))
        self.assertEqual(len(base2), 1290)
        same = all(torch.equal(self.p1[k], base2[k]) for k in self.p1)
        self.assertTrue(same, "Phase 2 base weights differ from Phase 1")

    def test_parameter_counts(self):
        n_base = sum(v.numel() for v in self.p1.values() if v.is_floating_point())
        self.assertEqual(n_base + 1024, sum(v.numel() for v in self.p2.values() if v.is_floating_point()))


if __name__ == "__main__":
    unittest.main()
