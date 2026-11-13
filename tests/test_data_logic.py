"""Preprocessing, splitting and behavioural metrics on small synthetic inputs (no dataset needed)."""
from __future__ import annotations
import json
import math
import unittest
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from tests._common import ROOT
from training.splits import make_image_split
from training.metrics import scanpath_metrics
from preprocessing.preprocess_dataset import process_segment
from preprocessing.preprocess_gaze4asd_v2 import to_normalized

CFG = SimpleNamespace(max_plausible_ms=3000.0, keep_first=False, min_fix_ms=60.0, t_end=3.0,
                      clamp_max_ms=None, min_len=3)


class Saliency4ASDPreprocessing(unittest.TestCase):
    def run_segment(self, rows, W=101, H=51):
        return process_segment(rows, W, H, CFG, defaultdict(int))

    def test_first_fixation_removed_short_removed_and_normalised(self):
        rows = [(0, 50, 25, 400), (1, 1, 1, 200), (2, 101, 51, 50), (3, 51, 26, 300), (4, 26, 13, 250)]
        sp, info = self.run_segment(rows)
        self.assertIsNone(info["reason"])
        # first fixation dropped, the 50 ms fixation dropped, three left
        self.assertEqual(len(sp), 3)
        self.assertEqual(sp[0], [-1.0, -1.0, 0.2])          # pixel (1, 1) maps to (-1, -1)
        self.assertAlmostEqual(sp[1][0], 0.0)                # pixel 51 of 101 is the centre
        self.assertAlmostEqual(sp[1][2], 0.3)                # durations in seconds

    def test_truncation_at_three_seconds(self):
        rows = [(0, 1, 1, 100)] + [(i, 10, 10, 900) for i in range(1, 5)]
        sp, info = self.run_segment(rows)
        self.assertTrue(info["truncated"])
        self.assertAlmostEqual(sum(d for _, _, d in sp), 3.0, places=6)

    def test_minimum_three_fixations(self):
        sp, info = self.run_segment([(0, 1, 1, 100), (1, 5, 5, 300), (2, 6, 6, 300)])
        self.assertIsNone(sp)
        self.assertEqual(info["reason"], "too_short")

    def test_implausible_duration_drops_segment(self):
        sp, info = self.run_segment([(0, 1, 1, 100), (1, 5, 5, 3500), (2, 6, 6, 300), (3, 7, 7, 300)])
        self.assertIsNone(sp)
        self.assertEqual(info["reason"], "impossible_duration")


class Gaze4ASDMapping(unittest.TestCase):
    def test_fit_to_height_maps_image_corners_and_centre(self):
        W, H = 1024, 768                     # displayed at 1440 / 768 = 1.875x, 1920 px wide, centred
        scale = 1440 / H; offset = (2560 - scale * W) / 2
        for sx, sy, ex, ey in ((offset, 0, -1, -1), (offset + scale * W, 1440, 1, 1), (1280, 720, 0, 0)):
            nx, ny = to_normalized(np.array(sx, float), np.array(sy, float), W, H, "fit_height")
            self.assertAlmostEqual(float(nx), ex, places=9)
            self.assertAlmostEqual(float(ny), ey, places=9)

    def test_stretch_maps_full_screen_to_image(self):
        nx, ny = to_normalized(np.array([0.0, 2560.0]), np.array([0.0, 1440.0]), 800, 600, "stretch")
        np.testing.assert_allclose(nx, [-1, 1]); np.testing.assert_allclose(ny, [-1, 1])


class Splits(unittest.TestCase):
    def test_image_split_is_70_15_15_disjoint_and_seeded(self):
        samples = [{"image_id": i} for i in range(1, 301)]
        a, b = make_image_split(samples, seed=0), make_image_split(samples, seed=0)
        self.assertEqual(a, b)
        self.assertEqual([len(a[k]) for k in ("train", "val", "test")], [210, 45, 45])
        self.assertFalse(set(a["train"]) & set(a["test"]) or set(a["val"]) & set(a["test"]))

    def test_matches_the_published_split(self):
        """The split used for every reported result is regenerated exactly from seed 0."""
        published = json.loads((ROOT / "results" / "in_domain" / "saliency4asd_image_split_seed0.json").read_text())
        ids = sorted(published["train"] + published["val"] + published["test"])
        regenerated = make_image_split([{"image_id": i} for i in ids], seed=0)
        self.assertEqual(regenerated, published)


class Metrics(unittest.TestCase):
    def test_known_values_on_a_square_path(self):
        xy = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], float)
        m = scanpath_metrics(xy, np.full(5, 0.2), diag=100.0)
        self.assertEqual(m["n_fix"], 5)
        self.assertAlmostEqual(m["total_dur"], 1.0)
        self.assertAlmostEqual(m["sacc_amp_mean"], 10.0)
        self.assertAlmostEqual(m["sacc_amp_norm_mean"], 0.1)
        self.assertAlmostEqual(m["revisit_rate"], 0.2)       # last fixation returns to the first
        # centroid (4, 4): distances 2 x |(4,4)| + 2 x |(6,4)| + |(6,6)|, averaged and divided by the diagonal
        expected = (2 * math.hypot(4, 4) + 2 * math.hypot(6, 4) + math.hypot(6, 6)) / 5 / 100
        self.assertAlmostEqual(m["dispersion"], expected, places=9)


if __name__ == "__main__":
    unittest.main()
