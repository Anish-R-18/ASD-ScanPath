# MultiMatch Implementation Validation

**Verdict.** Our implementation (`training/metrics.py::multimatch`, unchanged) follows the published method for the vector representation, alignment, the five dimensions and normalization.

It differs from the reference in two ways:
1. It summarizes each dimension with the **mean** of the aligned differences, where the reference uses the **median**.
2. It accepts scanpaths with **2 fixations**, where the reference requires at least 3.

With the median, our code reproduces the reference's MATLAB test case to 3e-6. The mean gives systematically lower scores. Rerunning the paper's MultiMatch comparisons with the reference changes no direction or significance of the temperature effects, but one secondary interpretation flips (Section 4). All eight significant E1 conditioning conclusions are unchanged under the reference (Section 4b).

The implementation was not silently replaced. Whether to switch to the median, or to the reference package, is a decision for the next phase.

Outputs: `training/runs/research_completion/a5_multimatch_validation_v1/a5_results.json`. Script: `scripts/research_completion/a5_multimatch_validation.py`.

## 1. Sources

- **Published method.** Jarodzka, Holmqvist and Nyström (ETRA 2010) and Dewhurst et al. (Behavior Research Methods 2012). We hold their metadata and the TPP-Gaze paper's description of the method. **We did not read either paper in full.**
- **Reference implementation.** `multimatch-gaze` 0.1.3 by Adina Wagner, MIT licence, downloaded from PyPI and read in full. Its test suite compares it with the original MATLAB toolbox.
- **MATLAB test inputs.** The two files from the reference repository's test suite are in `external_reference/`, with the licence.

## 2. Line-by-line comparison

| Element | Reference (`multimatch-gaze`) | Ours | Same? |
|---|---|---|---|
| Representation | Saccade vectors from consecutive fixations; saccade start = fixation *i* | Same (`np.diff`; start = `xy[i]`) | Yes |
| Alignment cost | Matrix of vector-difference lengths between all saccade pairs | Same | Yes |
| Path search | Dijkstra over right, down and diagonal moves, from (0,0) to (n−1, m−1) | Dynamic programming over the same moves | Yes, same optimal path; ties may break differently |
| Shape | ‖v₁ − v₂‖, normalized by 2 × diagonal | Same | Yes |
| Direction | Absolute angle difference, wrapped to [0, π], normalized by π | arccos of the normalized dot product, normalized by π | Yes, except zero-length saccades |
| Length | \|‖v₁‖ − ‖v₂‖\|, normalized by diagonal | Same | Yes |
| Position | Distance between aligned saccade start points, normalized by diagonal | Same | Yes |
| Duration | \|d₁ − d₂\| / max(d₁, d₂), for the fixation at each saccade start | Same, with a 1e-9 guard | Yes |
| **Aggregation** | **Median** over aligned pairs | **Mean** | **No** |
| **Minimum length** | **3 fixations** (else NaN) | **2 fixations** | **No** |
| Screen size | User-supplied screen dimensions | Image diagonal (coordinates in image pixels) | Equivalent: the image is the reference frame |
| Simplification | Optional (amplitude, direction and duration thresholds) | None | Matches the reference with simplification off |

**Zero-length saccades.** For a zero-length saccade, the reference uses atan2(0, 0) = 0, so the vector is treated as pointing right. Ours treats the angle via a guarded dot product, which yields π/2. Both choices are arbitrary.

## 3. Test results

**MATLAB test case** (`segment_10_sub-19` vs `sub-01`, no simplification). Columns: shape, direction, length, position, duration.

| Implementation | Values | Max error vs MATLAB |
|---|---|---|
| MATLAB toolbox (published in the reference test) | 0.99082, 0.69902, 0.98927, 0.94767, 0.65563 | — |
| Reference Python | identical to 5 decimals | 3.0e-6 |
| **Ours with median** | identical to 5 decimals | 3.0e-6 |
| Ours as used (mean) | 0.98687, 0.62619, 0.98310, 0.94660, 0.59782 | 0.073 |

**Edge cases.**

| Case | Reference | Ours |
|---|---|---|
| Identical scanpaths | 1, 1, 1, 1, 1 | 1, 0.99999994, 1, 1, 1 |
| Two-fixation scanpath | NaN (excluded) | Scored |
| Single fixation | NaN | Excluded (None) |
| Zero-length saccade | Differs (see Section 2) | Differs |

Two-fixation generated scanpaths are rare: 104 of 23,978. Real Gaze4ASD scanpaths all have at least 3 fixations.

**2,992 random real–real and real–generated Gaze4ASD pairs.** 8 of 3,000 were excluded by the reference minimum length.

| Dimension | Ours (mean) − reference | Correlation | Median variant identical to reference |
|---|---|---|---|
| Shape | −0.011 | 0.87 | 100% |
| Direction | −0.034 | 0.87 | 99.4% (rest: zero-length saccades) |
| Length | −0.022 | 0.83 | 100% |
| Position | −0.008 | 0.95 | 100% |
| Duration | +0.005 | 0.93 | 99.8% |

## 4. Do the paper's MultiMatch conclusions change?

The τ=1 vs τ=2 comparison on Gaze4ASD was rerun with the reference implementation, using identical scanpath pairs:

| Conclusion in the paper | Ours | Reference | Holds? |
|---|---|---|---|
| τ=2 scores higher on shape, length and position | Yes, all intervals exclude 0 | Yes, all intervals exclude 0 | Yes |
| τ=1 scores higher on direction | Yes | Yes | Yes |
| Duration differs little between temperatures | Yes | Yes | Yes |
| Real TD is closer to TD-conditioned samples on position, shape and length | Yes | Yes (position +0.0077 [0.0055, 0.0097]) | Yes |
| Real ASD is not closer to ASD-conditioned samples on position | Own minus other −0.0047 | −0.0036 [−0.0070, −0.0002] | Yes |
| ASD direction: τ=1 closer to the real-vs-real level | τ=1 0.0030 vs τ=2 0.0081 | τ=1 0.0060 vs τ=2 **0.0010** | **No, it flips** |
| TD position: τ=2 closer to real-vs-real | Yes | Yes | Yes |
| Duration own minus other, TD | −0.0037 [−0.0067, −0.0010] | −0.0023 [−0.0058, +0.0011] | Weaker; interval now includes 0 |

All scores are uniformly higher under the reference, by 0.01–0.04, including real-vs-real. Absolute MultiMatch values are therefore implementation-specific, and only within-implementation comparisons are meaningful.

## 4b. E1 conditioning analysis under the reference implementation (E1d)

E1's MultiMatch analysis compares own-vs-other condition similarity for Phase 2 against the Phase 1 seed-null, at τ=1. It was re-scored with `multimatch-gaze` on exactly E1's 72,000 scanpath pairs.

- **Script:** `scripts/research_completion/e1d_multimatch_reference.py`
- **Output:** `training/runs/research_completion/e1d_multimatch_reference_v1/`
- **Gate.** Our implementation on the reconstructed pairs reproduces E1's stored per-image values exactly (maximum difference 0.0).
  - A first attempt reloaded the generated coordinates as float64 and differed by up to 3.7e-7. E1 had computed them in float32, as the sampler returns them.
  - The script was corrected to reproduce float32, not the gate loosened.
- **Exclusions.** The reference excluded 407 of 72,000 pairs (0.57%) for its 3-fixation minimum.

| Real group | Dimension | Ours: effect (Holm p) | Reference: effect [95% CI] (Holm p) | Same conclusion? |
|---|---|---|---|---|
| ASD | Shape | −0.0016 (0.001) | −0.0016 [−0.0021, −0.0011] (0.001) | Yes |
| ASD | Length | −0.0021 (0.001) | −0.0015 [−0.0022, −0.0007] (0.003) | Yes |
| ASD | Direction | +0.0011 (0.41) | −0.0002 [−0.0036, +0.0033] (1.0) | Yes, n.s. in both (sign differs) |
| ASD | Position | −0.0062 (0.001) | −0.0069 [−0.0081, −0.0058] (0.001) | Yes |
| ASD | Duration | +0.0056 (0.001) | +0.0061 [+0.0035, +0.0087] (0.001) | Yes |
| TD | Shape | +0.0019 (0.001) | +0.0015 [+0.0010, +0.0020] (0.001) | Yes |
| TD | Length | +0.0026 (0.001) | +0.0018 [+0.0009, +0.0028] (0.003) | Yes |
| TD | Direction | −0.0014 (0.41) | −0.0010 [−0.0043, +0.0024] (1.0) | Yes, n.s. |
| TD | Position | +0.0074 (0.001) | +0.0084 [+0.0071, +0.0098] (0.001) | Yes |
| TD | Duration | −0.0050 (0.001) | −0.0044 [−0.0066, −0.0021] (0.003) | Yes |

**Every E1 MultiMatch conclusion is unchanged under the reference implementation.** Effect sizes differ by up to 0.0010, but none of the conclusions do.

## 5. Recommendations for the manuscript

1. State that MultiMatch was computed with our own implementation, using the **mean** of aligned differences, a 2-fixation minimum and no simplification, and that this differs from the reference toolbox's median.
2. Either recompute the reported MultiMatch numbers with the reference implementation, or report both. The script above already produces the reference values for the τ comparison.
3. Drop the "closer to real-vs-real" reading for ASD direction, because it depends on the implementation.
4. Cite the reference implementation if it is adopted. Its JOSS paper was not verified here.
