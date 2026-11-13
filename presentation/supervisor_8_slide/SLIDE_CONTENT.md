# Slide content: 8-slide supervisor presentation (shareable version)

Generated from `ASD_ScanPath_Supervisor_Presentation.pptx`. The spoken explanation for each slide is in `SPEAKING_SCRIPT.md` (and in the PowerPoint speaker notes); sources for figures and charts are in `ASSET_SOURCES.md`.

## Slide 1: What does a minimal ASD/TD label change in TPP-Gaze?

*Purpose:* Research gap, the three research questions, our two-phase approach and the short answer

- 1 · RESEARCH QUESTION AND APPROACH
- Gap
- TPP-Gaze models fixation location and duration jointly but has no group input. Separate ASD and TD models would differ in every weight, so the group effect could not be isolated. We test the smallest addition: one group label in a shared model.
- Research questions
- Q1
- In-domain likelihood
- Does the label lower NLL on held-out Saliency4ASD test images?
- Q2
- External likelihood
- Does any gain hold on Gaze4ASD: new children, images, tracker and display?
- Q3
- Generated behaviour
- Do ASD- and TD-labelled samples differ like the real groups, and does each label match its own group better?
- Our approach
- Phase 1: unconditioned baseline
- Fine-tune the released TPP-Gaze on pooled ASD + TD training data, with no group input.
- Phase 2: add the group label
- Freeze all Phase 1 weights; add a zero-initialized 2 × 512 embedding (1,024 parameters) and train only that.
- Compare Phase 2 with Phase 1
- Same scanpaths, same seeds: likelihood in-domain and on Gaze4ASD; generated behaviour vs real ASD–TD differences.
- Short answer. In-domain gain +0.0073 nats per fixation, inconclusive. External fit worse (−0.0172). Generated spatial contrasts partly right; timing contrasts wrong; neither group more realistic.
- [3] D'Amelio et al., TPP-Gaze, WACV 2025 · closest ASD scanpath work: Zhong et al., Pattern Recognition 2026 (abstract only; not compared) · manuscript §1, §2.4

## Slide 2: What we kept from TPP-Gaze, and what is new

*Purpose:* What we kept unchanged from TPP-Gaze vs what is new in our implementation, and why the design is minimal

- 2 · OUR CONTRIBUTION
- Kept unchanged from TPP-Gaze [3]
- Released code and pretrained transformer weights
- Image encoder: DenseNet201 + CoordConv (backbone frozen)
- History encoder: transformer + LSTM, including the released batch-layout behaviour (kept and disclosed)
- Heads: 16-component Gaussian mixture (location), 4-component log-normal mixture (duration)
- Joint NLL objective without a survival term; autoregressive sampler
- New in our implementation
- Conditioning wrapper: a 2 × 512 group embedding, zero-initialized, added to the fused context before both heads
- Phase 2 trains only the embedding: 1,024 of 23,896,736 parameters
- Conditioned scoring and sampling: the label is added at every step
- Our settings: duration normalization from our training split; τ = 1 chosen on validation (released config: τ = 2); minimum 3 fixations (TPP-Gaze paper: 4)
- Evaluation: clustered likelihood comparison and a shared-seed behaviour comparison (E1)
- Why this design. With a zero-initialized embedding and a frozen base, Phase 1 is exactly Phase 2 with a zero embedding; all 1,290 base tensors are identical in both checkpoints. Any Phase 2 − Phase 1 difference is caused by the 1,024 label parameters.
- Not claimed
- A new architecture
- An improvement over TPP-Gaze
- Any clinical or diagnostic use
- [3] D'Amelio et al., TPP-Gaze, WACV 2025 · manuscript §3.2, Table 1, §3.5, §4.4 · code: training/conditioned.py, training/generate_cond.py · run summaries: training/runs/*/summary_full.json

## Slide 3: Implementation: where the group embedding enters

*Purpose:* Implementation: where the group embedding enters, and that it is used in training, scoring and sampling

- 3 · MODEL IMPLEMENTATION
- ASD / TD label ℓ
- ASD = 0   TD = 1
- Stimulus image
- resized to 512 × 512
- Gaze history
- past (x, y, duration)
- Group embedding eℓ
- nn.Embedding(2, 512), zero-init
- Image encoder
- DenseNet201 + CoordConv → z (256-d)
- History encoder
- Transformer + LSTM → h (256-d)
- Fused context
- c = concat(h, z)
- 512-d
- Spatial head
- 16 Gaussian components
- → next (x, y)
- Temporal head
- 4 log-normal components
- → next duration
- Next fixation
- (x, y, d)
- c̃ = c + eℓ (every step)
- Sampling only: generated fixations join the history until 3 s are filled; scoring uses the real history
- Released TPP-Gaze (frozen in Phase 2)
- Our addition (only part trained in Phase 2)
- Unchanged modules
- Both encoders and both heads are released TPP-Gaze: fine-tuned in Phase 1, frozen in Phase 2.
- Our addition
- One 512-d vector per group, broadcast over the sequence and added to the fused context.
- Same in all code paths
- Training, likelihood scoring and the sampler all add the embedding at every step.
- Parameter count
- Base 23,895,712 + embedding 1,024; only the 1,024 train in Phase 2.
- Code: training/conditioned.py, train_tpp_cond.py, generate_cond.py · manuscript §3.3–3.6, Eq. 2, Figure 1 · counts: runs/phase2_embedding_only/summary_full.json

## Slide 4: Datasets, and the actual test images we used

*Purpose:* Data counts and preparation, plus test images 102 and 116 with generated and real scanpaths

- 4 · DATASETS AND EXPERIMENTAL STIMULI
- Saliency4ASD [4, 5]
- TRAINING + IN-DOMAIN TEST
- 300 images · split by image (seed 0): 210 / 45 / 45 → 4,229 / 897 / 903 scanpaths
- 2,705 ASD + 3,324 TD scanpaths · age 5–12 · no participant IDs
- Gaze4ASD [6]
- EXTERNAL TEST ONLY
- 30 stimuli, mostly faces · 33 ASD + 133 TD children (README: 132)
- 771 + 3,605 trials · age 3–7 · other tracker, display and site
- Real vs generated (statistics)
- Original photograph
- Phase 1
- Phase 2, ASD label
- Phase 2, TD label
- Image 102 · best-match example  ASD−TD dispersion: real +0.0133, generated +0.014 (mean of 10 seeds): same sign, similar size
- Photograph withheld
- Saliency4ASD sharing terms unresolved; shown only in the local restricted version
- *[Image: Blank-canvas generated scanpath for test image 102: Phase 1 (seed 0, tau 1)]*
- *[Image: Blank-canvas generated scanpath for test image 102: Phase 2 ASD label (seed 0, tau 1)]*
- *[Image: Blank-canvas generated scanpath for test image 102: Phase 2 TD label (seed 0, tau 1)]*
- Dispersion (÷ diagonal)
- Real: ASD 0.132 · TD 0.118
- Generated: ASD 0.150 · TD 0.138
- E1 replicate 1, 20 scanpaths per group
- Image 116 · worst-match example  ASD−TD dispersion: real −0.0473 (ASD less spread), generated +0.0151 (ASD more spread): wrong direction
- Photograph withheld
- Saliency4ASD sharing terms unresolved; shown only in the local restricted version
- *[Image: Blank-canvas generated scanpath for test image 116: Phase 1 (seed 0, tau 1)]*
- *[Image: Blank-canvas generated scanpath for test image 116: Phase 2 ASD label (seed 0, tau 1)]*
- *[Image: Blank-canvas generated scanpath for test image 116: Phase 2 TD label (seed 0, tau 1)]*
- Dispersion (÷ diagonal)
- Real: ASD 0.117 · TD 0.165
- Generated: ASD 0.162 · TD 0.109
- E1 replicate 1, 20 scanpaths per group
- Best and worst agreement among the 45 test images, by a rule fixed before any demo output: one strength and one limitation, not overall performance. Generated panels: one sample each, seed 0, τ = 1. Photographs and real-gaze plots: local restricted version only.
- [4] Duan et al., ACM MMSys 2019 · [5] Gutiérrez et al., SPIC 2021 · [6] Yang et al., ICME 2025 · counts: manuscript §4.1–4.2, Table 2 · examples: demo/demo_results

## Slide 5: Training: Phase 1 baseline, then the label only

*Purpose:* Phase 1 and Phase 2 training decisions, settings and the overfitting observation

- 5 · TRAINING PROCEDURE
- Preprocess
- Split at index resets; drop sequences with a fixation > 3 s; drop the 1st fixation and those < 60 ms; cut at 3 s; keep ≥ 3
- Split by image
- Seed 0: 210 / 45 / 45 images, so no test image is seen during training or selection
- Phase 1
- Fine-tune released TPP-Gaze on pooled ASD + TD; DenseNet201 frozen, features cached
- Set up Phase 2
- Load best Phase 1; add the zero 2 × 512 embedding; freeze every base weight
- Phase 2
- Train only the embedding; base in evaluation mode (no dropout)
- Select
- Best validation NLL checkpoint for each phase; τ also chosen on validation

|  | Phase 1: unconditioned baseline | Phase 2: group conditioning |
|---|---|---|
| Why | Adapt TPP-Gaze to these children's data before adding a label | Learn only the group difference; nothing else can change |
| Starts from | Released TPP-Gaze weights | Best Phase 1 checkpoint + zero embedding |
| Trained parameters | 3,881,784 (everything outside the frozen DenseNet backbone) | 1,024 (embedding only) |
| Optimizer | AdamW, lr 10⁻³, weight decay 0.1, batch 128, gradient clip 5 | AdamW, lr 10⁻⁴; otherwise the same |
| Stopping | 20 epochs without validation improvement | Same |
| Best validation NLL | −0.5845 at epoch 6; stopped at 26, when validation NLL had worsened to −0.384 | −0.5964 at epoch 8; stopped at 28 |

- NLL in nats per fixation. Duration normalization (log-duration mean −1.4366, SD 0.6936) uses the training split only. Test images and Gaze4ASD were never used for training or selection. An earlier Phase 2 configuration (lr 10⁻³) exists in the repository without local results; it is not reported.
- Sources: training/config.yaml, training/config_phase2_embonly.yaml, training/runs/baseline/split.json, training/runs/*/summary_full.json · manuscript §4.1, §4.3–4.4, §6.1

## Slide 6: Evaluation: likelihood and generated behaviour, kept separate

*Purpose:* Evaluation design: likelihood paths, three external preprocessing variants, E1 and the temperature run

- 6 · EVALUATION DESIGN
- A. Likelihood: Phase 2 vs Phase 1 on identical scanpaths
- IN-DOMAIN
- 45 held-out Saliency4ASD images, 903 scanpaths
- No participant IDs: image-cluster bootstrap and sign-flip test
- EXTERNAL
- Gaze4ASD: 30 images, 4,376 trials, never used for any choice
- Participant, image and crossed bootstraps
- Three external preprocessing variants, and why
- Fit-to-height + clip off-image gaze: primary, fixed before modelling
- Stretch + clip: the screen mapping is inferred (fit-to-height r = 0.9999 vs 0.91 for stretch)
- Fit-to-height + remove off-image gaze: off-image gaze differs by group (7.35% ASD vs 2.72% TD of fixations)
- B. Generated behaviour: experiment E1 (τ = 1)
- Phase 2
- Phase 1
- Seed stream A
- ASD label
- no label
- Seed stream B
- TD label
- no label
- 20 scanpaths per image and seed · 10 seeds · 30 Gaze4ASD + 45 Saliency4ASD test images
- Conditioning effect = (P2 ASD − P2 TD) − (P1 stream A − P1 stream B). The Phase 1 term is the seed-only null.
- Primary measures: dispersion, saccade amplitude, revisit rate; fixed before E1 from the earlier temperature run. Also count, duration, rate, total time, direction.
- Per-group realism: each label's per-image error against its own real group.
- Separate temperature run. Phase 2 sampled at τ = 1 and τ = 2 with paired seeds (10 seeds). τ only rescales the spatial component spread during sampling (σ becomes στ); NLL is always computed without τ. Statistics: Hedges' g, bootstrap intervals and sign-flip tests, Holm correction within families; all analyses exploratory.
- Sources: manuscript §3.8, §4.3, §5.1–5.5, §6.4 (off-image counts) · mapping evidence: manuscript §4.3 · E1: research_completion/ADDITIONAL_RESULTS.md

## Slide 7: Results: what improved, what did not, what is unclear

*Purpose:* Results: likelihood vs generated behaviour, with plain verdicts

- 7 · RESULTS AND OBSERVATIONS
- Likelihood: Phase 2 − Phase 1 joint NLL gain (nats per fixation, 95% CI)
- In-domain (Saliency4ASD test)
- +0.0073 [−0.0012, +0.0163]
- INCONCLUSIVE
- External, primary version
- −0.0172 [−0.0225, −0.0122]
- WORSE
- External, stretch mapping
- −0.0274 [−0.0322, −0.0226]
- WORSE
- External, off-image removed
- −0.0087 [−0.0130, −0.0042]
- WORSE
- −0.03
- −0.02
- −0.01
- 0.00
- +0.01
- +0.02
- **Chart:** Generated ASD−TD contrast as % of the real contrast (Gaze4ASD, E1, τ = 1)
  - Spatial dispersion: 69%, Saccade amplitude: 64%, Revisit rate: 32%
- 100% = real size. Approximate shares (an independent run gave 60 / 50 / 29%), not accuracy.

| Comparison | Result | Verdict |
|---|---|---|
| Likelihood · Phase 1 vs pretrained | In-domain test NLL −0.1665 → −0.4957 | Improved |
| Likelihood · Phase 2 vs Phase 1, in-domain | +0.0073, 95% CI [−0.0012, +0.0163]; 27 of 45 images favour Phase 2; sign-flip p = 0.1097 | Inconclusive |
| Likelihood · Phase 2 vs Phase 1, Gaze4ASD | Phase 1 better by 0.0172 (−0.5568 vs −0.5396); TD worse in all 3 versions; ASD +0.0104 (participant CI only) | Worse |
| Behaviour · label effect | 8 of 9 measures shift beyond the seed-only null; shifts ≈1% of the image diagonal | Present, small |
| Behaviour · spatial contrasts | Right direction: ≈69% (dispersion), 64% (amplitude), 32% (revisit) of real size | Partial |
| Behaviour · other contrasts | Fixation count, total fixation time and direction concentration go the wrong way | Wrong direction |
| Behaviour · per-group realism | No spatial measure closer to its real group; ≈10 generated vs ≈6 real fixations per scanpath | Not improved |
| Sampling · τ = 2 vs τ = 1 | Same directions, smaller contrasts (46 / 33 / 15%); likelihood unaffected | Caveat |

- Sources: manuscript §6.1, Tables 3–5 (external forest values: Table 5 CPU re-scoring; Table 4 GPU primary −0.0172), Tables 7–8, §6.6–6.8

## Slide 8: What we can conclude, and what comes next

*Purpose:* What we can conclude, what is unresolved, limitations and next experiments

- 8 · INTERPRETATION AND NEXT STEPS
- What we can conclude
- The label changes generated behaviour, but by ≈1% of the image diagonal; the generated ASD−TD dispersion difference is positive on all 45 test images
- Correct spatial contrasts come from the conditions moving apart (TD contracts past real TD), not per-group realism
- A 1,024-parameter label does not improve likelihood: inconclusive in-domain, worse externally
- Still unresolved
- Why the TD embedding fits both Gaze4ASD groups worse (not tested)
- How much fixation over-production (≈10 vs ≈6) drives the wrong timing contrasts
- Whether missing off-image gaze (≈2.7× more in ASD) contributes
- Limitations
- Gaze4ASD screen mapping inferred, not confirmed
- No participant-level age (3–7 vs 5–12 years; means 5.04 vs 4.54)
- Stimuli selected for group differences; 33 ASD children
- Exploratory statistics; dataset terms unresolved
- Next experiments
- Ask the Gaze4ASD authors for display geometry, the 132/133 count and ages
- Add stopping / viewing-time and off-image modelling; rerun E1 with the same null
- Richer conditioning vs the same Phase 1 control, with per-group criteria
- Preregistered replication on independent data
- Conclusion. We developed and evaluated an ASD/TD-conditioned extension of TPP-Gaze. It changes generated gaze behaviour and reproduces some spatial group-contrast directions, but does not consistently improve likelihood or individual-group realism.
- Group-level research model only: no classifier was evaluated and there is no clinical validation.
- Sources: manuscript §4.2 (ages), §6.3–6.7, §7–9 · demo/demo_results/manifest.json (45 of 45 images) · research_completion/UNRESOLVED_ISSUES.md

