# Rehearsal Checklist: ASD-ScanPath Presentation

Use this with `SPEAKING_SCRIPT.md` (what to say) and `RESEARCH_QA.md` (questions). The target is 12–15 minutes including the demo.

## 1. Time allocation

| Slide | Topic | Time | You should be at |
|---|---|---|---|
| 1 | Title and opening | 0:35 | 0:35 |
| 2 | Motivation | 0:45 | 1:20 |
| 3 | Research questions | 0:40 | 2:00 |
| 4 | Related work | 0:45 | 2:45 |
| 5 | TPP-Gaze | 0:55 | 3:40 |
| 6 | Our conditioning | 1:10 | 4:50 |
| 7 | Datasets | 0:40 | 5:30 |
| 8 | Experimental design | 0:50 | 6:20 |
| 9 | In-domain likelihood | 0:55 | 7:15 |
| 10 | External likelihood | 1:00 | 8:15 |
| 11 | Generated contrasts | 0:55 | 9:10 |
| 12 | Per-group realism | 1:00 | 10:10 |
| 13 | Temperature | 0:50 | 11:00 |
| 14 | Demo slide | 1:00 | 12:00 |
| 14+ | Live demo (optional) | +0:45 | 12:45 |
| 15 | Limitations | 0:40 | 13:25 |
| 16 | Conclusions | 0:45 | 14:10 |

- **Totals.** 13:25 without the live demo and 14:10 with it, at 130 words per minute. Both are inside 12–15 minutes.
- **Checkpoints.**
  - At slide 9 you should be near 6:20. If you're past 7:00, start using the cut list (section 2).
  - At slide 14, if you're past 12:00, skip the live demo and just present the slide.
- **Speaker notes.** The PowerPoint notes give slightly different times. Use this table.

## 2. Cut list (if you run long)

Each cut is a sentence you can drop without losing the story. Together they save about 75 s.

| Slide | Drop this | Saves |
|---|---|---|
| 1 | "I'll show exactly where it helps and where it fails." | 4 s |
| 4 | The sentence about Saliency4ASD and gaze-based identification | 9 s |
| 5 | "Training minimizes negative log-likelihood…" (slide 8 explains it) | 6 s |
| 6 | "So the label can shift both where the next fixation goes and how long it lasts." | 7 s |
| 9 | The "For scale…" paragraph | 12 s |
| 10 | The post hoc check sentence ("A post hoc check also showed…") | 9 s |
| 12 | "For timing, TD moves toward real TD, but ASD moves away…" | 7 s |
| 13 | The "We chose tau equals 1 on validation images…" paragraph | 12 s |
| 14 | "In fact, the model's difference is positive on all 45 test images…" | 8 s |
| 14+ | The live demo | 45 s |

**Never cut:**
- "not a diagnostic tool" (slides 1 and 14);
- "the interval includes zero" (slide 9);
- "Phase 2 fits worse" (slide 10);
- the wrong-direction measures (slide 11);
- "neither group becomes more realistic" (slide 12).

## 3. Concepts to understand (don't memorize)

For each concept, test yourself: can you explain it in two sentences without notes, and then answer the follow-up? The full explanations are in `SPEAKING_SCRIPT.md` section 18.

| # | Concept | Self-test question | Follow-up to expect |
|---|---|---|---|
| 1 | Scanpath | What three things describe each fixation? | Why drop fixations shorter than 60 ms? (Preprocessing choice in §4.3; don't invent a reason) |
| 2 | Joint location and duration | Why not model location alone? | How are the two heads connected? (Shared context; separate heads given the context) |
| 3 | Temporal point process | What is the "event", the "inter-event time" and the "mark"? | Why intensity-free? (Log-normal mixture, Shchur et al. [16]) |
| 4 | Spatial GMM and temporal LGMM | Why 16 blobs and not one? Why a log scale for duration? | What does τ do to them? |
| 5 | Image features and history | What goes into the 512-d context? | What is the batch-layout issue? |
| 6 | The ASD/TD embedding | Where is it added, and how many parameters does it have? | Why not add it to the history encoder? |
| 7 | Freezing the base | What does freezing let us claim? | Would full fine-tuning be better? (Not tested) |
| 8 | Likelihood / NLL | What does a lower NLL mean? | Why can NLL be negative? (Densities can exceed 1) |
| 9 | Gain and intervals | What does an interval that includes zero mean? | Why cluster by image or child? |
| 10 | Temperature | What does τ change, and what does it never change? | Why does τ = 2 *narrow* the blobs? (All σ < 1) |
| 11 | Common random numbers | Why the same seed for all three models? | What is the seed-only null? What is d_z? |
| 12 | External evaluation | What does "Phase 2 fits worse externally" mean, and why can't we say why? | Is the ASD gain real? (Fragile, not ASD-specific) |
| 13 | Contrast vs realism | How can the contrast be right while no group improves? | Give the dispersion example: 0.113 / 0.112 → 0.101 / 0.108 |

The hardest to explain on the spot are numbers 8, 11 and 13. Practise those three out loud until they feel natural.

## 4. Important numbers

Learn the **bold** ones by heart. The others are for follow-up questions.

| What | Number | Meaning |
|---|---|---|
| Embedding size | **1,024** (2 × 512) | The only trained part in Phase 2 |
| Identical base tensors | 1,290 | Phase 1 = Phase 2 with a zero embedding |
| Base model parameters | 23,895,712 | 3,881,784 of them outside the frozen backbone |
| In-domain gain | **+0.0073** nats per fixation | 95% CI **[−0.0012, +0.0163]**: inconclusive |
| In-domain detail | 27 of 45 images; p = 0.1097 | Phase 1 −0.4957 → Phase 2 −0.5030 |
| Fine-tuning effect | −0.1665 → −0.4957 | Pretrained to Phase 1 (test NLL) |
| External overall | **−0.0172** | Phase 1 −0.5568 vs Phase 2 −0.5396: Phase 2 worse |
| External TD | −0.0229 | Worse in all three preprocessing versions |
| External ASD | +0.0104 | Participants only; +0.0027 under stretch |
| Primary shares of real contrast | **≈69%, 64%, 32%** | Dispersion, amplitude, revisit (τ = 1) |
| Same shares at τ = 2 | 46%, 33%, 15% | Same direction, smaller |
| Independent run, τ = 1 | 60%, 50%, 29% | Monte Carlo uncertainty |
| Wrong direction | **Count, total time, direction concentration** | ASD label: more fixations and more time |
| Absolute shift size | 1.1% and 0.8% of the diagonal | Dispersion and amplitude |
| Measures beyond seed noise | 8 of 9 | Holm p ≤ 0.0009 |
| Dispersion story | ASD 0.113 → 0.113; TD 0.112 → 0.101; real TD 0.108 | Conditions move apart |
| Fixations per scanpath | **≈10 generated vs ≈6 real** | 10.19 / 9.60 vs 6.25 / 6.41 |
| Count shift from conditioning | +0.66 | About one fifth of the shared excess |
| Off-image gaze | **≈2.7×** more for ASD | 7.35% vs 2.72% of fixations |
| Mapping evidence | r = 0.9999 (4,533 maps) | vs 0.91 stretch and 0.23 native |
| Real Gaze4ASD effects (g) | Revisit −1.04, dispersion +0.80, total time −0.71, amplitude +0.65, direction +0.55 | 33 ASD vs 133 TD children |
| Ages | 5–12 (Saliency4ASD); 3–7, means 5.04 / 4.54 (Gaze4ASD) | No individual ages |
| Dataset split | 210 / 45 / 45 images | By image |
| Gaze4ASD | 30 images; 33 ASD, 133 TD (README: 132) | External only |
| Demo, image 102 | Real +0.0133 vs generated +0.014 | Best match |
| Demo, image 116 | Real −0.0473 vs generated +0.0151 | Worst match (opposite sign) |

**Units.** NLL is in nats per fixation (one fixation is one event). Lower is better. "Gain" = Phase 1 − Phase 2, so positive favours Phase 2.

## 5. Questions to practise

Practise these out loud, answering in under 40 seconds each. The numbers refer to `RESEARCH_QA.md`.

**Most likely (practise first):**
1. Q3: What is the actual research contribution?
2. Q22: Why is the in-domain result inconclusive?
3. Q23: Why is external likelihood worse? (Say "we didn't test causes".)
4. Q11: Why freeze the base model?
5. Q33: Why doesn't per-group realism improve?
6. Q36: What does 69%, 64% and 32% mean?
7. Q20: Could age confound the results?
8. Q37: Can the model distinguish ASD and TD? (No.)
9. Q40: Is this a diagnostic tool? (No.)

**Likely from a technical examiner:**
10. Q7: Why temporal point processes?
11. Q25: Why identical seeds?
12. Q17 and Q18: the coordinate mapping, and why fit-to-height.
13. Q35: Does temperature improve the model?
14. Q34: Why does over-production matter?
15. Q14: What is the batch-layout issue?

**Prepare your own answers to:** what you did yourself, the hardest problem, and what you would change.

## 6. Demo fallback procedure

**Primary plan (live, about 45 s extra on slide 14).**
1. Before the talk, open a terminal in `gaze-ag/` and run the command once as a warm-up:
   ```bash
   python inference/inference.py --image TrainingData/TrainingData/Images/102.png --compare-phase1 --no-image --out-dir inference/demo/local_only/live
   ```
   It takes about 18 s. Check that `inference/demo/local_only/live/102/102_phase1_vs_asd_vs_td.png` opens and shows 13, 14 and 13 fixations.
2. Leave the command typed in the terminal, ready.
3. On slide 14, say the intro line, press Enter, and talk through the "while it runs" lines (script section 17).
4. Open the output and point out that it is identical to the top row of slide 14.

**Fallback (if anything goes wrong).**
- **Trigger:** any error, or no output after about 45 s.
- **Do:** stop immediately. Don't debug in front of the audience.
- **Say:**
  > The live run isn't cooperating, but the figure on this slide came from exactly this command. Our reproducibility check regenerated all 35 demo scanpaths identically.
- **Then:** continue presenting slide 14.
- **Backup file** (open it beforehand, minimized): `inference/demo/demo_results/s4a_best_102/phase_comparison_no_image.png`.

**If asked for more:**
- Variability: `inference/demo/demo_results/s4a_random_60/seeds_grid_no_image.png`.
- Scanpaths over an image: `inference/demo/demo_results/synthetic_scene/phase_comparison.png`, a synthetic, shareable drawing that is outside the training distribution.

**Do not project:**
- the photograph overlays in `inference/demo/local_only/s4a_*` or `real_vs_generated.png` (dataset photographs, some showing children; data terms unresolved);
- any image from `inference/inputs/`.

## 7. Final checklist

**The day before**
- [ ] Fill in the title-slide placeholders in PowerPoint: name, supervisor, affiliation, date.
- [ ] Two full timed run-throughs, one with the live demo. Note where you run long.
- [ ] One Q&A drill: someone (or you) asks the 15 questions in section 5 in random order.
- [ ] Run the demo command once and confirm the output path.
- [ ] Copy the PPTX and the PDF to a USB stick or cloud storage as a backup.

**30 minutes before**
- [ ] Laptop on the charger; notifications off; unrelated windows closed, especially any showing dataset photographs.
- [ ] PowerPoint in Presenter View; speaker notes visible only to you.
- [ ] Terminal open in `gaze-ag/`; warm-up demo run done; command typed and ready.
- [ ] Backup PNG open in an image viewer, minimized.
- [ ] Check on the projector that slide 14's panels and slide 12's table are readable.
- [ ] Clock or timer where you can see it.

**Right before you start**
- [ ] Know your first two sentences by heart.
- [ ] Remember four numbers: **+0.0073 (interval includes 0)**, **−0.0172**, **≈69 / 64 / 32%**, **≈10 vs ≈6 fixations**.
- [ ] Remember three "nevers":
  - never call a sample "an autistic child's scanpath";
  - never say "improves" for the in-domain result;
  - never guess why the external fit is worse.

## 8. Say this, not that

| Don't say | Say instead |
|---|---|
| "The model improves likelihood." | "A small gain whose interval includes zero, so inconclusive." |
| "The model captures ASD gaze." | "Generated spatial contrasts follow the real direction at reduced size; others go the wrong way." |
| "This is the ASD scanpath." | "This is one sample generated with the ASD label." |
| "τ = 1 improves the model." | "τ = 1 gives more realistic samples; likelihood is unaffected." |
| "External performance drops because of age." | "We didn't test why; age, tracker, display, stimuli and site all differ." |
| "The mapping is correct." | "The mapping is strongly supported but not confirmed." |
| "It could help diagnose autism." | "It has no clinical or diagnostic validity." |

## 9. Rehearsal plan

1. **Read-through with the slides.** Read the script aloud once, slide by slide, to connect words to visuals.
2. **Key points only.** Present using only the *Key points* bullets, in your own words. This is the most important round.
3. **Timed full run with the demo.** Compare your times with the table in section 1 and mark cuts if needed.
4. **Q&A drill.** Practise the 15 questions in section 5, giving the short answer first and the deeper one only if pushed.
5. **Final run** the day before, with no script.
