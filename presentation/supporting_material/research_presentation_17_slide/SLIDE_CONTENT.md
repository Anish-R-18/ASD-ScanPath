# ASD-ScanPath research presentation: slide content

Generated from `ASD_ScanPath_Research_Presentation.pptx` (17 slides: 16 main slides and 1 backup). Every number on the slides comes from `paper/final/ASD_ScanPath_Final.md`; the demo selection values come from `demo/demo_results/manifest.json`. See `ASSET_SOURCES.md` for figures and charts.

## Narrative

1. **Why** (slides 2–4): gaze differs between autistic and typically developing children in where, in what order and for how long they look. TPP-Gaze models location and duration jointly. We ask what a minimal group label changes.
2. **How** (slides 5–8): TPP-Gaze as a marked point process; one zero-initialized 2 × 512 embedding added before both heads of the frozen model; Saliency4ASD for training and testing, Gaze4ASD only for external evaluation; likelihood and a common-random-numbers behavioural comparison.
3. **What we found** (slides 9–13): in-domain gain inconclusive; external fit worse; generated spatial contrasts right in direction at reduced size, temporal and direction-concentration contrasts wrong; neither group more realistic; temperature changes generated behaviour, not likelihood.
4. **What it looks like** (slide 14): same image and seed, Phase 1 vs Phase 2 ASD vs Phase 2 TD; a best-match and a worst-match image; one sample is not a child.
5. **What it means** (slides 15–16): limitations, conclusions and future work. Not a diagnostic tool.

## Timing

| Slide | Topic | Suggested time |
|---|---|---|
| 1 | Title | 30 s |
| 2 | Motivation and background | 45 s |
| 3 | Problem statement and objectives | 40 s |
| 4 | Related work and research gap | 45 s |
| 5 | TPP-Gaze background | 50 s |
| 6 | Proposed ASD/TD conditioning | 70 s |
| 7 | Datasets and preprocessing | 45 s |
| 8 | Experimental setup and metrics | 55 s |
| 9 | In-domain likelihood | 50 s |
| 10 | External Gaze4ASD evaluation | 65 s |
| 11 | Generated behavioural differences (E1) | 65 s |
| 12 | Per-group realism | 60 s |
| 13 | Sampling temperature and sensitivity | 50 s |
| 14 | Qualitative demo examples | 55 s |
| 15 | Limitations and threats to validity | 45 s |
| 16 | Conclusions and future work | 45 s |
| 17 | References (backup) | backup |

Total for the main slides: about 13 min 35 s, plus about 20 s if the optional live demo command is run on slide 14. Questions are extra.

## Slide 1: ASD/TD-Conditioned Gaze Scanpath Generation with Neural Temporal Point Processes

- **Section:** Title
- **Source:** Manuscript title and abstract
- **Visual:** Abstract scanpath motif (illustration, not data)

**On-slide content**

- RESEARCH PRESENTATION · SUPERVISOR REVIEW
- Likelihood, transfer and behavioural evaluation of a 1,024-parameter group embedding in TPP-Gaze
- [Presenter name]  ·  [Supervisor name]  ·  [Affiliation]  ·  [Date]
- Exploratory research study. The model describes group-level gaze statistics; it has no clinical or diagnostic validity.
- Illustration: a scanpath is an ordered sequence of fixations; circle size = duration

**Speaker notes**

> Opening. Suggested time: about 30 s.
>
> This talk presents an exploratory study: we added a tiny ASD/TD group embedding to a published scanpath model, TPP-Gaze, and asked whether it helps. The short answer is: mostly no, and the talk explains exactly where it helps, where it does not, and why that matters for evaluating conditional gaze models.
>
> Say early: this is a research model of group-level gaze statistics. It does not predict autism and is not a diagnostic or screening tool.
>
> Fill in presenter, supervisor, affiliation and date before presenting.

## Slide 2: Where, in what order and for how long we look

- **Section:** Motivation and background
- **Source:** §1, §2.3; refs [1]–[3]
- **Visual:** Three drawn cards (location, order, duration), illustrations only

**On-slide content**

- INTRODUCTION · MOTIVATION
- Gaze differs between autistic and typically developing children, in space and in time.
- Location
- Where each fixation lands on the image
- Order
- The sequence of fixations: the scanpath
- Duration
- How long each fixation lasts
- circle size ∝ duration
- Why model it
- Eye-tracking studies report ASD–TD differences in visual attention [1, 2]; effects vary across stimuli and regions of interest [1].
- These differences motivate computational models of gaze.
- What is usually missing
- Most scanpath models predict an ordered sequence of locations and pay less attention to timing.
- TPP-Gaze models fixation location and duration jointly, as a temporal point process [3].
- This study
- Give one shared TPP-Gaze model a minimal ASD/TD group label, then test whether it fits group-labelled gaze better and whether scanpaths generated under each label differ the way real groups differ.

**Speaker notes**

> Suggested time: about 45 s.
>
> Gaze has three components: where people look, in what order, and for how long. Eye-tracking studies report group differences between autistic and typically developing children [1, 2], but the effects depend on the stimulus and region [1].
>
> Most scanpath models predict the order of locations and treat timing as secondary. TPP-Gaze [3] is different: it predicts location and duration jointly.
>
> Our question: if we give that model only a minimal group label, does it describe group-labelled gaze better, and do its generated scanpaths differ like the real groups? The three small drawings are illustrations, not data.

## Slide 3: Research question and objectives

- **Section:** Problem statement and objectives
- **Source:** §1 (research question, contributions, what is not claimed)
- **Visual:** Four numbered research-question cards; 'not claimed' panel

**On-slide content**

- INTRODUCTION · RESEARCH QUESTIONS
- If a shared TPP-Gaze model gets only a small ASD/TD group embedding, what changes?
- In-domain fit
- Does the embedding describe group-labelled gaze better on held-out Saliency4ASD images? (likelihood)
- Transfer
- Does any benefit hold on a separate dataset, Gaze4ASD, never used for training or selection?
- Generated behaviour
- Do scanpaths generated under the ASD and TD labels differ the way real groups differ, and is each group more realistic?
- Sampling temperature
- Do the behavioural conclusions depend on the sampling temperature τ?
- What we do not claim
- The first ASD scanpath generator
- An improvement over TPP-Gaze
- Confirmed generalization
- Any biological mechanism
- Prediction or diagnosis of autism
- The model describes group-level gaze statistics only.

**Speaker notes**

> Suggested time: about 40 s.
>
> Four questions. One: in-domain likelihood on held-out Saliency4ASD images. Two: transfer to Gaze4ASD, which we only ever used for evaluation. Three: generated behaviour, both whether the ASD-minus-TD contrast has the real direction and whether each group's samples become more realistic. Four: whether the conclusions depend on sampling temperature.
>
> Also state up front what we do not claim: not the first ASD scanpath generator, not an improvement over TPP-Gaze, not confirmed generalization, no mechanism, and nothing diagnostic.

## Slide 4: Related work and research gap

- **Section:** Related work and research gap
- **Source:** §2.1–2.4
- **Visual:** Four literature cards and a gap statement

**On-slide content**

- INTRODUCTION · RELATED WORK
- We isolate what a minimal group label changes in a published duration-aware model.
- Saliency and scanpath models
- Saliency from image features [7]
- DeepGaze III: free-viewing scanpaths [8]
- SpFormer: transformer; duration in fixation features [9]
- Neural temporal point processes
- History-dependent event models [13, 14]
- Transformer Hawkes process [15]
- Intensity-free, log-normal mixture [16]
- TPP-Gaze applies this to gaze [3]
- ASD gaze data
- Meta-analysis: group differences vary by stimulus [1]
- Saliency4ASD: 300 images, ASD and control children [4, 5]
- Gaze-based ASD identification [6, 17]: not our aim
- ASD scanpath prediction
- Zhong et al. [18]: semantic-based saccadic scanpath prediction for ASD
- Described from the abstract only; full text not accessible to us
- Gap. Models already predict scanpaths with duration [3, 9] or target ASD specifically [18]. A simpler question remains open: how much does a minimal group label change a published joint location–duration model, and does the change hold on another dataset?
- Our design: keep TPP-Gaze, add one group embedding, compare against an unconditioned control, in-domain and externally, for likelihood and for generated behaviour. We claim no novelty or superiority over [18].

**Speaker notes**

> Suggested time: about 45 s.
>
> Four strands. Saliency and scanpath models: from image-feature saliency [7] to DeepGaze III [8] and SpFormer [9], which uses duration in its fixation features. Neural temporal point processes model irregular events; TPP-Gaze [3] treats fixation durations as inter-event times and locations as marks. ASD gaze: group differences are real but stimulus-dependent [1]; Saliency4ASD [4, 5] is our training data. Some work uses gaze to identify autism [6, 17]; ours does not. Closest is Zhong et al. [18]; we only had the abstract, so we make no claims about how it compares.
>
> The gap we address is narrow: the effect of a minimal group label on an existing joint model, with a matched unconditioned control and an external dataset.

## Slide 5: TPP-Gaze: a scanpath as a marked point process

- **Section:** TPP-Gaze background
- **Source:** §3.1–3.4, §3.6–3.7, Table 1
- **Visual:** Native illustration of marks and inter-event times (not data); component list

**On-slide content**

- METHOD · BACKGROUND
- Durations are the inter-event times; fixation locations are the marks of each event [3].
- Illustration (not data)
- Marks: locations xn ∈ [−1, 1]²
- d1
- d2
- d3
- d4
- d5
- 0 s
- 3 s viewing window
- Inter-event times: fixation durations
- At each step the model predicts the next event
- Image encoder  DenseNet201 + CoordConv → image representation z, 256-d [19, 20]
- History encoder  Transformer (4 layers, 4 heads) + LSTM → history state h, 256-d [15]
- Context  Concatenation of h and z: 512-d
- Duration head  Log-normal mixture, K = 4 components
- Location head  Gaussian mixture, J = 16 components, diagonal covariance
- Objective  Joint negative log-likelihood (NLL), nats per fixation; lower is better
- We use the released code and pretrained weights without modification [3]. Base model: 23,895,712 parameters, 3,881,784 outside the frozen image backbone.

**Speaker notes**

> Suggested time: about 50 s.
>
> TPP-Gaze treats a scanpath as a marked temporal point process. Each fixation is an event: its duration is the inter-event time, and its location is the mark. The drawing on the left is only an illustration.
>
> At each step the model encodes the image with DenseNet201 plus CoordConv into a 256-d vector, encodes the gaze history with a transformer and LSTM into another 256-d vector, concatenates them into a 512-d context, and predicts the next duration with a 4-component log-normal mixture and the next location with a 16-component Gaussian mixture. Training minimizes joint NLL in nats per fixation.
>
> Two uses matter later: likelihood evaluation scores observed scanpaths; generation samples one fixation at a time and feeds it back into the history.
>
> Disclosure if asked: the released transformer has a batch-layout issue (attention runs across scanpaths in a batch). We kept it because the pretrained weights were learned with it.

## Slide 6: Our modification: one group embedding

- **Section:** Proposed ASD/TD conditioning
- **Source:** §3.5, Figure 1, Figure 2, §4.4
- **Visual:** Native, editable architecture diagram redrawn from Figures 1–2; original model in teal, our addition in orange

**On-slide content**

- METHOD · PROPOSED CONDITIONING
- Everything except a 1,024-parameter embedding is the frozen Phase 1 model.
- ASD / TD label ℓ
- ASD = 0, TD = 1
- Stimulus image
- resized to 512 × 512
- Gaze history
- past locations and durations
- Group embedding eℓ
- nn.Embedding(2, 512)
- Image encoder
- DenseNet201 + CoordConv → z
- History encoder
- Transformer + LSTM → h
- Fused context
- concat(h, z)
- 512-d
- Spatial head
- Gaussian mixture, 16 comp.
- → location (x, y)
- Temporal head
- Log-normal mixture, 4 comp.
- → duration d
- Next fixation
- (x, y, d)
- added to the context before both heads
- Generation only: the sampled fixation joins the history; repeat until the 3 s window ends
- Original TPP-Gaze (frozen in Phase 2)
- Our addition (only trained part in Phase 2)
- 1,024
- trained parameters: a 2 × 512 embedding, one row per group
- initial value: at the start Phase 2 is numerically identical to Phase 1
- 1,290
- base tensors identical in both checkpoints: the embedding is the only difference

**Speaker notes**

> Suggested time: about 70 s. This is the key method slide.
>
> Teal boxes are the original TPP-Gaze; the orange parts are our only addition. The ASD/TD label selects one row of a 2 x 512 embedding table, and that vector is added to the 512-d fused context immediately before both prediction heads: c-tilde = c + e_l. So the label can shift both where the next fixation goes and how long it lasts.
>
> Training has two phases. Phase 1 fine-tunes TPP-Gaze on combined ASD and TD data, with no label. Phase 2 freezes everything and trains only the 1,024 embedding parameters. The embedding starts at zero, so Phase 2 begins identical to Phase 1, and we verified that all 1,290 base tensors are identical in the two checkpoints. Phase 1 is exactly Phase 2 with a zero embedding, so any difference between the models comes from these 1,024 numbers.
>
> The dashed loop is only used in generation.

## Slide 7: Datasets and preprocessing

- **Section:** Datasets and preprocessing
- **Source:** Table 2, §4.1–4.3
- **Visual:** Native table (Table 2) and preprocessing steps

**On-slide content**

- METHOD · DATA
- Saliency4ASD for training and testing; Gaze4ASD only for external evaluation.

|  | Saliency4ASD [4, 5] | Gaze4ASD [6] |
|---|---|---|
| Role | Training, validation, held-out test | External evaluation only |
| Images | 300 (210 / 45 / 45, split by image) | 30 selected stimuli |
| Scanpaths ASD / TD | 2,705 / 3,324 | 771 / 3,605 trials |
| Fixations ASD / TD | 18,475 / 25,504 | 4,830 / 23,161 |
| Participants ASD / TD | 14 / 14 per image (README); no IDs | 33 / 133 folders (README: 33 / 132) |
| Age (years) | ASD 5–12, mean 8; controls mean 8 | 3–7; means 5.04 ASD, 4.54 TD |
| Viewing, tracker | 3 s; Tobii T120, 1280 × 1024 | 3 s; Tobii Pro Fusion, 2560 × 1440 |

- No participant-level age in either release. Gaze4ASD stimuli are mostly smiling faces, selected for large ASD/TD differences.
- Preprocessing (both datasets)
- Split into scanpaths; drop the first fixation
- Remove fixations shorter than 60 ms
- Truncate at the 3 s window
- Keep scanpaths with at least 3 fixations
- Normalize coordinates to [−1, 1]
- Gaze4ASD only: screen gaze mapped to image coordinates (fit-to-height, assumed); off-image gaze clipped to the edge (Policy A) or removed (Policy B).

**Speaker notes**

> Suggested time: about 45 s.
>
> Saliency4ASD: 300 natural images, children with ASD and controls, split by image into 210 training, 45 validation and 45 test images. The release has no participant IDs, so in-domain uncertainty can only be estimated across images.
>
> Gaze4ASD: used only for external evaluation. 30 stimuli, mostly smiling faces, selected by its authors for large group differences. The README says 132 TD children but the release has 133 TD folders; we used the folders as released.
>
> Note the age gap: 5 to 12 years in Saliency4ASD, 3 to 7 in Gaze4ASD, and no participant-level age in either.
>
> Preprocessing is the same for both. For Gaze4ASD we must also map screen coordinates to the image; we come back to this on the external-results slide.

## Slide 8: Experimental design and evaluation metrics

- **Section:** Experimental setup and metrics
- **Source:** §4.4, §5.1, §5.3, §5.5
- **Visual:** Three-column design; native 2 × 2 diagram of the common-random-numbers design

**On-slide content**

- METHOD · EXPERIMENTS
- Matched models, matched scanpaths and matched random seeds isolate the embedding's effect.
- Training
- Phase 1
- Fine-tune TPP-Gaze on combined ASD + TD training data, no label; backbone frozen. AdamW [22], learning rate 10−3, batch 128.
- Phase 2
- Freeze the whole base; train only the embedding, learning rate 10−4.
- Selection
- Validation images chose both checkpoints and τ. Test images and Gaze4ASD were never used for training or selection.
- Likelihood
- Joint, duration and position NLL in nats per fixation (lower is better).
- Gain = Phase 1 NLL − Phase 2 NLL; positive favours Phase 2.
- Both models score the same scanpaths in the same batches.
- 95% intervals: paired cluster bootstrap (10,000) over images in-domain; over participants and images externally.
- Generated behaviour (E1)
- Common random numbers: each seed stream is shared, so models differ only through the embedding.
- Phase 2
- Phase 1
- Seed stream A
- Seed stream B
- ASD label
- TD label
- no label
- no label
- Effect = (ASD − TD contrast in Phase 2) − (A − B contrast in Phase 1, the seed-only null).
- 30 Gaze4ASD + 45 Saliency4ASD test images; 20 scanpaths × 10 seeds; τ = 1.
- Primary: dispersion, saccade amplitude, revisit rate (fixed before E1).
- Statistics: Hedges' g [23]; permutation and sign-flip tests; Holm correction [24] within prespecified families. All analyses are exploratory and were not preregistered.

**Speaker notes**

> Suggested time: about 55 s.
>
> Training: Phase 1 fine-tunes the model on combined data; Phase 2 freezes it and trains only the embedding. Validation images selected the checkpoints and the temperature; the test images and Gaze4ASD were never used for selection.
>
> Likelihood: we report NLL in nats per fixation, lower is better, and the gain as Phase 1 minus Phase 2, so positive favours conditioning. Intervals come from cluster bootstraps, never treating fixations as independent.
>
> Behaviour, experiment E1: common random numbers. For each seed stream we generate with the ASD label, with the TD label, and with Phase 1. The Phase 1 contrast between two seed streams is a seed-only null; the conditioning effect is the Phase 2 contrast minus that null. Dispersion, saccade amplitude and revisit rate were fixed as primary measures before E1, but they were chosen because an earlier run had reproduced them, so that choice is not independent.
>
> Everything is exploratory and not preregistered.

## Slide 9: In-domain likelihood: inconclusive gain

- **Section:** In-domain likelihood
- **Source:** §6.1, §6.2, Table 3
- **Visual:** Native forest plot of Table 3 gains with 95% intervals

**On-slide content**

- RESULTS · IN-DOMAIN (SALIENCY4ASD TEST)
- Phase 2 lowers held-out NLL slightly, but the 95% interval includes zero.
- +0.0073
- nats per fixation: joint NLL gain of Phase 2 over Phase 1 (1.48%)
- INCONCLUSIVE
- 95% image-level interval [−0.0012, +0.0163]
- 27 of 45 test images favour Phase 2
- Sign-flip test p = 0.1097
- For scale: fine-tuning (Phase 1) moved test NLL from −0.1665 (pretrained) to −0.4957; the embedding then moved it to −0.5030.
- Gain with 95% interval (positive favours Phase 2)
- Joint (primary)
- +0.0073 [−0.0012, +0.0163]
- Duration
- +0.0019 [+0.0002, +0.0035]
- Position
- +0.0055 [−0.0029, +0.0139]
- ASD subset
- +0.0086 [+0.0011, +0.0162]
- TD subset
- +0.0064 [−0.0058, +0.0189]
- −0.01
- 0.00
- +0.01
- +0.02
- nats per fixation
- interval excludes zero (unadjusted, exploratory)
- interval includes zero
- 45 test images, 903 scanpaths; paired image-level cluster bootstrap (images are the only available cluster).

**Speaker notes**

> Suggested time: about 50 s.
>
> On the 45 held-out Saliency4ASD test images, Phase 2 lowers joint NLL by 0.0073 nats per fixation, about 1.5%. But the 95% image-level interval runs from -0.0012 to +0.0163, so it includes zero; 27 of 45 images favour Phase 2 and the sign-flip test gives p = 0.11. We therefore call the gain inconclusive, not an improvement.
>
> The duration component and the ASD subset clear zero without correction; with several related comparisons we treat them as exploratory.
>
> For scale: fine-tuning itself (Phase 1) improved test NLL from -0.1665 to -0.4957. The embedding's contribution is small by comparison.

## Slide 10: External likelihood: Phase 2 fits worse

- **Section:** External Gaze4ASD evaluation
- **Source:** §4.3 (mapping evidence), §6.3, §6.4, Tables 4–5, Figure 3
- **Visual:** Manuscript Figure 3; verdict blocks

**On-slide content**

- RESULTS · EXTERNAL (GAZE4ASD)
- On Gaze4ASD the unconditioned model fits better, in every preprocessing version.
*[Image: Manuscript Figure 3: Phase 2 minus Phase 1 NLL gain with 95% intervals, in-domain and on Gaze4ASD under three preprocessing versions]*
- Manuscript Figure 3. Phase 2 − Phase 1 NLL gain, 95% intervals; positive favours Phase 2. Black: in-domain (image clusters). Colours: Gaze4ASD under three preprocessing versions (participant clusters).
- Coordinate mapping. Fit-to-height placement is strongly supported (authors' saliency maps match at median r = 0.9999 over 4,533 maps, vs 0.91 stretch, 0.23 native) but not confirmed from the original presentation code.
- WORSE
- Overall
- −0.0172
- Phase 1 −0.5568 vs Phase 2 −0.5396. Interval excludes zero by participants, images and both.
- WORSE
- TD trials
- −0.0229
- Worse under Phase 2 in all three preprocessing versions.
- FRAGILE
- ASD trials
- +0.0104
- Clears zero by participants only (17 of 30 images); +0.0027, interval includes zero, under the stretch mapping.
- Post hoc label swap: the ASD embedding also beats Phase 1 on TD trials (+0.0189), so the ASD gain is not ASD-specific.

**Speaker notes**

> Suggested time: about 65 s.
>
> Externally the answer is clearly negative. On identical Gaze4ASD trials and batches, Phase 1 fits better overall by 0.0172 nats per fixation, and the interval excludes zero whether we resample participants, images or both. TD trials are worse under conditioning.
>
> ASD trials improve slightly, +0.0104, but only at participant level, only 17 of 30 images favour Phase 2, and under the alternative stretch mapping the gain shrinks to +0.0027 with an interval including zero. A post hoc label swap shows the ASD embedding also helps TD trials, so this is not an ASD-specific effect.
>
> The figure shows that the overall and TD results hold under all three preprocessing versions.
>
> Mapping caveat: the release does not record where images were drawn on screen. Fit-to-height is strongly supported by the authors' own saliency maps but not confirmed.
>
> Why the TD fit worsens is untested; age, tracker, display, stimuli and site all differ between datasets.

## Slide 11: Generated contrasts: partly right, partly wrong

- **Section:** Generated behavioural differences (E1)
- **Source:** §5.3, §6.5, §6.6, Tables 6–7, Figure 5
- **Visual:** Manuscript Figure 5; share callouts

**On-slide content**

- RESULTS · GENERATED BEHAVIOUR (E1, τ = 1)
- The embedding shifts behaviour: spatial contrasts follow real data, temporal ones do not.
*[Image: Manuscript Figure 5: generated ASD minus TD contrast as a fraction of the real contrast on Gaze4ASD]*
- Manuscript Figure 5. Generated ASD−TD contrast ÷ real contrast (Gaze4ASD, 30 images). Markers: mean over 10 seeds; lines: single-seed range. 1 = real size; below 0 = wrong direction.
- Real Gaze4ASD (33 ASD vs 133 TD, Hedges' g): ASD children show larger saccades (+0.65), greater dispersion (+0.80), fewer revisits (−1.04), more concentrated directions (+0.55) and less total fixation time (−0.71).
- RIGHT DIRECTION, SMALLER
- share of the real contrast
- ≈69%
- Spatial dispersion
- ≈64%
- Saccade amplitude
- ≈32%
- Revisit rate
- WRONG DIRECTION
- Fixation count · total fixation time · direction concentration
- The ASD label gives more fixations and more total time than the TD label; real data show the opposite.
- 8 of 9 measures exceed the seed-only null (Holm p ≤ 0.0009), but the shifts are small: 1.1% (dispersion) and 0.8% (amplitude) of the image diagonal.

**Speaker notes**

> Suggested time: about 65 s.
>
> First, the real data at the bottom: autistic children in Gaze4ASD make larger saccades, spread their fixations more, revisit less, have more concentrated saccade directions and spend less total time fixating.
>
> E1 shows the embedding does change generated behaviour beyond seed noise: 8 of 9 measures shift beyond the Phase 1 seed-only null. The three primary spatial contrasts have the real direction, at roughly 69, 64 and 32 percent of the real size; these shares are approximate and vary by several points between independent runs (an earlier run gave 60, 50 and 29 percent).
>
> But fixation count, total time and direction concentration go the wrong way: the ASD label produces more fixations and more time. Note the real count difference itself is not significant.
>
> Keep size in perspective: highly significant under common random numbers, yet only about 1 percent of the image diagonal.

## Slide 12: Per-group realism does not improve

- **Section:** Per-group realism
- **Source:** §6.7, Table 8, MultiMatch paragraph
- **Visual:** Native dot plot (dispersion), native bar chart (fixation counts), native excerpt of Table 8

**On-slide content**

- RESULTS · PER-GROUP REALISM (E1, τ = 1)
- Correct contrasts arise because the conditions move apart, not toward their real groups.
- Spatial dispersion (÷ image diagonal), Gaze4ASD
- ASD condition
- TD condition
- Phase 1 = Phase 2 = 0.113
- Phase 1 0.112
- Phase 2 0.101
- real TD 0.108
- 0.100
- 0.105
- 0.110
- 0.115
- Hollow: Phase 1 · filled: Phase 2 · bar: real TD. TD contracts past real TD; ASD does not move.
**Chart: Fixations per scanpath (Gaze4ASD)**
- Real: ASD: 6.25, TD: 6.41
- Generated (Phase 2, τ = 1): ASD: 10.19, TD: 9.6
- Phase 2 vs Phase 1: error against the matching real group

| Measure | ASD: change in error | TD: change in error |
|---|---|---|
| Spatial dispersion | +0.7% (n.s.) | −6.5% (n.s.) |
| Saccade amplitude | −3.2% (n.s.) | −9.1% (n.s.) |
| Revisit rate | −11.8% (p = 0.0140) | +13.0% (n.s.) |
| Fixation count | −8.5% (p = 0.0018) | +10.0% (p = 0.0018) |
| Fixation rate | −28.1% (p = 0.0018) | +22.6% (p = 0.0018) |
| Mean fixation duration | −32.5% (p = 0.0018) | +28.2% (p = 0.0121) |

- Per-image absolute error, Phase 2 vs Phase 1. Positive = Phase 2 closer to the real group; negative = further away. Holm p over 18 tests; n.s. = Holm p > 0.05.
- No spatial measure improves for either group. Timing moves TD toward real TD and ASD away from real ASD.

**Speaker notes**

> Suggested time: about 60 s.
>
> A correct contrast does not mean either group became more realistic. Table on the right: compared with Phase 1, no spatial measure gets significantly closer to its real group for either label, and ASD revisit rate gets worse. In timing, Phase 2 moves TD toward real TD but ASD away from real ASD.
>
> Top left shows why the dispersion contrast is nevertheless correct: the ASD condition does not move at all (0.113 in both models), while the TD condition contracts from 0.112 to 0.101, past the real TD value of 0.108. The two conditions move apart; neither moves toward its own group.
>
> Bottom left: both labels over-produce fixations, about 10 per scanpath against about 6 real, because the sampler has no stopping model and fills the 3 s window. The conditioning shift in count (+0.66) is about a fifth of this shared excess.
>
> MultiMatch tells the same story (real ASD scanpaths are slightly closer to TD-conditioned samples), but all those effects are below 0.01 on a 0 to 1 scale.

## Slide 13: Sampling temperature: behaviour, not likelihood

- **Section:** Sampling temperature and sensitivity
- **Source:** §3.8, §6.8, Table 9, Figure 5 (τ = 2 shares)
- **Visual:** Two native bar charts (Table 9 reductions; contrast shares at τ = 1 and τ = 2)

**On-slide content**

- RESULTS · SAMPLING TEMPERATURE
- τ changes how far apart generated groups are, but never the likelihood.
- How τ works
- σj(τ) = σjτ
- Only the spatial component scales change; means, weights and durations are untouched.
- Every σ < 1, so τ = 2 narrows the components.
- τ = 1 reproduces the model's own distribution; chosen on validation images.
- NLL is computed without temperature.
**Chart: Per-image error reduction, τ = 1 vs τ = 2 (%)**
- Error reduction: Amplitude, ASD: 52.5, Amplitude, TD: 49.9, Revisit, ASD: 66.1, Revisit, TD: 56.5
**Chart: Generated ASD−TD contrast, % of real**
- τ = 1 (E1): Dispersion: 69, Amplitude: 64, Revisit rate: 32
- τ = 2: Dispersion: 46, Amplitude: 33, Revisit rate: 15
- All four prespecified tests favour τ = 1 in 10 of 10 seeds (Holm p ≤ 0.0005; exploratory). Every Phase 2 contrast keeps its direction at τ = 2, but shrinks. Only at τ = 2 did Phase 2 match real TD significantly worse on spatial measures.

**Speaker notes**

> Suggested time: about 50 s.
>
> Temperature only rescales the spatial mixture components: sigma to the power tau. Because every sigma is below 1, tau = 2 makes components narrower. Likelihood is always computed without temperature, so tau never affects NLL; it only affects generated behaviour.
>
> We chose tau = 1 on validation images; it is also the model's own distribution. On Gaze4ASD, tau = 1 has lower error on all four prespecified tests, in 10 of 10 seeds. Those tests were specified before generation but tau = 2 results had already been seen, so this is exploratory.
>
> Right chart: at tau = 2 every generated contrast keeps its direction but is smaller: 46, 33 and 15 percent instead of 69, 64 and 32. So the size of generated group differences depends on a sampling setting that likelihood cannot see. Always state tau when comparing generated scanpaths.

## Slide 14: Demo: same image, same seed, three models

- **Section:** Qualitative demo examples
- **Source:** demo/demo_results/manifest.json; demo/DEMO_GUIDE.md
- **Visual:** Blank-canvas demo strips for test images 102 (best match) and 116 (worst match)

**On-slide content**

- DEMONSTRATION · QUALITATIVE EXAMPLES
- Each panel is one stochastic sample, not a child and not a group profile.
- Phase 1 (unconditioned)
- Phase 2, ASD label
- Phase 2, TD label
- BEST MATCH · test image 102  real ASD−TD dispersion contrast +0.0133; generated +0.014
*[Image: Generated scanpaths for Saliency4ASD test image 102 on a blank canvas: Phase 1, Phase 2 ASD label, Phase 2 TD label]*
- WORST MATCH · test image 116  real contrast −0.0473; generated +0.0151 (opposite sign)
*[Image: Generated scanpaths for Saliency4ASD test image 116 on a blank canvas: Phase 1, Phase 2 ASD label, Phase 2 TD label]*
- How to read a panel
- first fixation
- last fixation
- circle area grows with duration
- numbers and arrows give the order
- One sample, one seed. All three panels share seed 0 and τ = 1, so they differ only through the embedding. Changing the seed changes a panel more than switching ASD and TD.
- Images chosen by a rule fixed in advance from saved E1 results. The generated contrast is positive on all 45 test images, while the real contrast ranges from −0.0473 to +0.0576.
- Stimulus photographs are not shown (data terms pending); scanpaths are drawn on a blank canvas of the image size.

**Speaker notes**

> Suggested time: about 55 s. Optionally run the live demo here (about 20 s):
>
> python inference.py --image TrainingData/TrainingData/Images/60.png --compare-phase1 --out-dir demo/local_only/live
>
> (Show the live output only on this machine; it overlays a dataset photograph.)
>
> Each row shows one Saliency4ASD test image: Phase 1, Phase 2 with the ASD label, and Phase 2 with the TD label, all with the same seed. Because they share random numbers, the panels look almost identical: the embedding is a small, systematic nudge, not a different behaviour.
>
> The two images were chosen by a rule fixed before generating any demo output. Image 102 is where the model's average ASD-minus-TD dispersion contrast best matches the real one; image 116 is the worst: the real contrast is negative, but the model's is positive. In fact the generated contrast is positive on all 45 test images, while the real one changes sign: the shift is roughly uniform rather than image-specific.
>
> Stress: one generated scanpath is one stochastic sample. It does not represent an individual child, and it is not diagnostic.

## Slide 15: Limitations and threats to validity

- **Section:** Limitations and threats to validity
- **Source:** §8, §6.4 (off-image gaze)
- **Visual:** 3 × 3 grid of limitation cards

**On-slide content**

- DISCUSSION · LIMITATIONS
- Several limitations cannot be resolved with the released data.
- Conditioning
- In-domain gain inconclusive; external fit worse; small ASD gain depends on the mapping.
- Generated behaviour
- Fixations over-produced; count, total time and direction contrasts wrong; no spatial realism gain.
- Coordinate mapping
- Gaze4ASD display geometry strongly supported but not confirmed by the authors.
- Off-image gaze
- ASD children look off-image ≈2.7× as often as TD; neither clipping nor removal is group-neutral.
- Age and site
- No participant-level age; 5–12 vs 3–7 years; tracker, display, stimuli and site differ.
- Data and sample
- Gaze4ASD stimuli selected for group differences, mostly faces; 33 ASD vs 133 TD.
- Statistics
- Exploratory, not preregistered; Holm within families only; validation chose checkpoints and τ.
- Implementation
- Released transformer's batch-layout issue kept; MultiMatch uses a mean-based variant.
- Clinical validity
- None established. This is not a diagnostic or screening tool.

**Speaker notes**

> Suggested time: about 45 s.
>
> The main threats: the conditioning results themselves are inconclusive or negative; the generator over-produces fixations and gets the temporal contrasts wrong.
>
> For Gaze4ASD, the screen-to-image mapping is strongly supported but unconfirmed, and off-image gaze is itself a group difference: autistic children looked outside the image about 2.7 times as often, so both of our filtering policies affect the groups differently.
>
> Age is a potential confound within Gaze4ASD and across datasets, and cannot be adjusted for because participant-level age is not released.
>
> The Gaze4ASD stimuli were selected for group differences and may have been selected using Saliency4ASD.
>
> All analyses are exploratory. The implementation keeps a disclosed batch-layout issue from the released code. And there is no clinical validity: this is not a diagnostic or screening tool.

## Slide 16: Conclusions and future work

- **Section:** Conclusions and future work
- **Source:** §9, §7
- **Visual:** Verdict list and future-work panel (dark slide)

**On-slide content**

- DISCUSSION · CONCLUSIONS
- INCONCLUSIVE
- In-domain likelihood: small gain (+0.0073 nats per fixation); interval includes zero.
- NEGATIVE
- External likelihood: worse overall (−0.5568 vs −0.5396), in every preprocessing version.
- PARTIAL
- Generated spatial contrasts follow the real direction at reduced size (≈69 / 64 / 32%).
- NEGATIVE
- Count, total time and direction contrasts wrong; neither group becomes more realistic.
- CAVEAT
- Temperature changes the size of generated differences, not the likelihood.
- Contribution. A controlled, transparently reported test of minimal group conditioning: likelihood in-domain and externally, a common-random-numbers behavioural comparison, and per-group realism reported alongside group contrasts.
- Future work
- Confirm the display mapping and participant count with the Gaze4ASD authors
- Obtain participant-level age to test age effects
- Model viewing time, stopping and off-image gaze
- Compare richer conditioning against the same control and per-group realism criteria
- Replicate on independent data with preregistered tests

**Speaker notes**

> Suggested time: about 45 s.
>
> To summarize: a 1,024-parameter group embedding gives a small, inconclusive in-domain gain and a worse external fit, robustly across preprocessing versions. Generated scanpaths reproduce the direction of the main real spatial differences at reduced size, get the temporal and direction-concentration differences wrong, and do not become more realistic for either group. Temperature changes the size of generated differences but not the likelihood.
>
> The contribution is methodological and honest: a controlled comparison with a matched unconditioned control, external testing, and the observation that a correct group contrast is weak evidence of group-specific realism.
>
> Next steps: confirm the mapping and participant count with the dataset authors, obtain age, model stopping and off-image gaze, and replicate with preregistered tests.
>
> Code is public at github.com/Anish-R-18/ASD-ScanPath; checkpoints are not redistributed.

## Slide 17: References

- **Section:** References (backup)
- **Source:** Manuscript reference list
- **Visual:** Two-column reference list, numbering as in the manuscript

**On-slide content**

- BACKUP
- [1] Frazier et al. A meta-analysis of gaze differences to social and nonsocial information between individuals with and without autism. J. Am. Acad. Child Adolesc. Psychiatry, 2017.
- [2] Guillon et al. Visual social attention in autism spectrum disorder: insights from eye tracking studies. Neurosci. Biobehav. Rev., 2014.
- [3] D'Amelio et al. TPP-Gaze: modelling gaze dynamics in space and time with neural temporal point processes. WACV, 2025.
- [4] Duan et al. A dataset of eye movements for the children with autism spectrum disorder. ACM MMSys, 2019.
- [5] Gutiérrez et al. Saliency4ASD: challenge, dataset and tools for visual attention modeling for ASD. Signal Process. Image Commun., 2021.
- [6] Yang et al. Gaze4ASD: a novel dataset and visual saliency map-based method for autism screening. IEEE ICME, 2025.
- [7] Itti, Koch, Niebur. A model of saliency-based visual attention for rapid scene analysis. IEEE TPAMI, 1998.
- [8] Kümmerer, Bethge, Wallis. DeepGaze III: modeling free-viewing human scanpaths with deep learning. J. Vision, 2022.
- [9] Zhong et al. SpFormer: spatio-temporal modeling for scanpaths with transformer. AAAI, 2024.
- [10] Jarodzka, Holmqvist, Nyström. A vector-based, multidimensional scanpath similarity measure. ETRA, 2010.
- [11] Dewhurst et al. Scanpath comparison in multiple dimensions with MultiMatch. Behav. Res. Methods, 2012.
- [12] Wagner, Halchenko, Hanke. multimatch-gaze: the MultiMatch algorithm for gaze path comparison in Python. JOSS, 2019.
- [13] Du et al. Recurrent marked temporal point processes: embedding event history to vector. ACM SIGKDD, 2016.
- [14] Shchur et al. Neural temporal point processes: a review. IJCAI, 2021.
- [15] Zuo et al. Transformer Hawkes process. ICML, 2020.
- [16] Shchur, Biloš, Günnemann. Intensity-free learning of temporal point processes. ICLR, 2020.
- [17] Liaqat et al. Predicting ASD diagnosis in children with synthetic and image-based eye gaze data. Signal Process. Image Commun., 2021.
- [18] Zhong et al. Semantic-based saccadic scanpath prediction for autism spectrum disorder. Pattern Recognition, 2026.
- [19] Huang et al. Densely connected convolutional networks. CVPR, 2017.
- [20] Liu et al. An intriguing failing of convolutional neural networks and the CoordConv solution. NeurIPS, 2018.
- [21] Judd et al. Learning to predict where humans look. ICCV, 2009.
- [22] Loshchilov, Hutter. Decoupled weight decay regularization. ICLR, 2019.
- [23] Hedges. Distribution theory for Glass's estimator of effect size and related estimators. J. Educ. Stat., 1981.
- [24] Holm. A simple sequentially rejective multiple test procedure. Scand. J. Stat., 1979.

**Speaker notes**

> Backup slide. Numbering follows the manuscript (paper/final/ASD_ScanPath_Final.md); full bibliographic details and DOIs are in the manuscript reference list and paper/final/references.bib.

