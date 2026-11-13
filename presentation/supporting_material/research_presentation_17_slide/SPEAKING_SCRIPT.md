# Speaking Script: ASD-ScanPath Research Presentation

This script goes with `ASD_ScanPath_Research_Presentation.pptx`: 16 main slides, plus slide 17 (references) as backup. Every number comes from the final manuscript (`paper/final/ASD_ScanPath_Final.md`), except the slide 14 demo values, which come from `inference/demo/demo_results/manifest.json`.

**How to use it**
- The text in the quote boxes is what you say. Practise it until you can say it in your own words; don't read it.
- **Key points** are what the audience must remember from each slide. If you forget the script, say those.
- **Transitions** link to the next slide. Say them while you click.
- Section 18 explains every technical idea in more depth. Read it before rehearsing: it is what lets you answer follow-up questions.
- Speaking pace: about 130 words per minute, which is calm and clear. Each suggested time matches its script's length at that pace.

## Timing plan

| Slide | Topic | Time | Cumulative |
|---|---|---|---|
| 1 | Title and opening | 0:35 | 0:35 |
| 2 | Motivation | 0:45 | 1:20 |
| 3 | Research questions | 0:40 | 2:00 |
| 4 | Related work and gap | 0:45 | 2:45 |
| 5 | TPP-Gaze background | 0:55 | 3:40 |
| 6 | Our conditioning | 1:10 | 4:50 |
| 7 | Datasets | 0:40 | 5:30 |
| 8 | Experimental design | 0:50 | 6:20 |
| 9 | In-domain likelihood | 0:55 | 7:15 |
| 10 | External likelihood | 1:00 | 8:15 |
| 11 | Generated contrasts | 0:55 | 9:10 |
| 12 | Per-group realism | 1:00 | 10:10 |
| 13 | Temperature | 0:50 | 11:00 |
| 14 | Demo slide | 1:00 | 12:00 |
| 14+ | Live demo (optional) | +0:45 | |
| 15 | Limitations | 0:40 | 12:40 |
| 16 | Conclusions | 0:45 | 13:25 |
| **Total** | without live demo | **13:25** | |
| **Total** | with live demo | **14:10** | |

These times assume a calm 130 words per minute. If you speak at 140 words per minute, the talk takes about one minute less. Either way it fits 12–15 minutes, including the demo. The PowerPoint speaker notes show slightly different times (13:35 in total, without the demo); **use this table.** If you are running long, use the cut list in `REHEARSAL_CHECKLIST.md`.

---

## Slide 1: ASD/TD-Conditioned Gaze Scanpath Generation with Neural Temporal Point Processes

**Suggested time:** 35 s (this is your opening)

> Good morning. I'm [your name], and today I'll present our work on ASD/TD-conditioned scanpath generation.
>
> We took a published model that predicts where people look in an image, and for how long. We gave it a small ASD-or-TD label, and tested whether that label helps.
>
> The short answer is: mostly, it does not. I'll show exactly where it helps and where it fails. And this is a research model, not a diagnostic tool.

**Key points to emphasize**
- One sentence on what you did, and an honest preview of the result.
- Say "not a diagnostic tool" early, so nobody misreads the rest.

**Transition:** "Let me start with why gaze is worth modelling at all."

---

## Slide 2: Where, in what order and for how long we look

**Suggested time:** 45 s

> When we look at a picture, our eyes jump and stop. Each stop is a *fixation*. Every fixation has a location, a place in the sequence, and a duration. The ordered sequence of fixations is called a *scanpath*.
>
> Eye-tracking studies report that autistic and typically developing children differ in visual attention, although the effects depend on the stimulus.
>
> Most scanpath models predict the sequence of locations and treat timing as secondary. TPP-Gaze predicts location and duration together.
>
> So our idea was: give one shared model a group label, and check whether it describes each group better.

**Key points to emphasize**
- A scanpath is location, order *and* duration.
- Group differences exist, but they depend on the stimulus.
- TPP-Gaze is the right starting point because it models timing too.

**Transition:** "That idea gives us four concrete research questions."

---

## Slide 3: Research question and objectives

**Suggested time:** 40 s

> We asked four questions.
>
> One: on held-out images from our training dataset, does the label help the model fit real gaze?
>
> Two: does that hold on a completely separate dataset?
>
> Three: do scanpaths *generated* with the ASD and TD labels differ the way real children differ? And is each group more realistic?
>
> Four: do these answers depend on the sampling temperature?
>
> On the right is what we do *not* claim: not the first ASD scanpath generator, not an improvement over TPP-Gaze, and nothing that predicts or diagnoses autism.

**Key points to emphasize**
- Two likelihood questions (in-domain and external) and two behaviour questions.
- Stating the non-claims early protects the whole talk.

**Transition:** "Here is where this sits in the existing literature."

---

## Slide 4: Related work and research gap

**Suggested time:** 45 s

> Our work connects four areas.
>
> Scanpath models like DeepGaze III and SpFormer predict where people look; SpFormer also uses duration. Temporal point processes model events at irregular times, and TPP-Gaze applies them to gaze. Saliency4ASD gave us our gaze data. Some work uses gaze to *identify* autism, but that is not our aim.
>
> The closest work is Zhong et al., on ASD scanpath prediction with semantic maps. We could only read its abstract, so we don't compare against it.
>
> Our gap is narrow: what does a minimal group label change in an existing model, and does that hold on a second dataset?

**Key points to emphasize**
- The gap is a *controlled test* of minimal conditioning. It is not a new architecture.
- Be open that you could only read the abstract of Zhong et al.

**Transition:** "To understand our change, I first need to explain how TPP-Gaze works."

---

## Slide 5: TPP-Gaze: a scanpath as a marked point process

**Suggested time:** 55 s

> A temporal point process models events that happen at irregular times. Here, each fixation is an event. Its duration is the time until the next event, and its location is attached to it as a "mark".
>
> At each step the model uses two inputs. The image goes through a DenseNet network and becomes 256 numbers. The gaze history goes through a transformer and an LSTM and becomes another 256. Together they form a 512-number context.
>
> From that context, two heads predict the next fixation: a log-normal mixture for its duration, and a mixture of 16 Gaussians for its location. Think of 16 soft blobs on the image, each with a weight.
>
> Training minimizes negative log-likelihood, which means it makes real fixations as probable as possible.

**Key points to emphasize**
- Duration = time between events; location = the event's mark.
- The model sees both the image and the history.
- Two probability heads: one for *where* (spatial mixture), one for *how long* (temporal mixture).

**Transition:** "Now I can show you exactly where our change goes."

---

## Slide 6: Our modification: one group embedding

**Suggested time:** 1 min 10 s

> The teal boxes are the original TPP-Gaze. The orange parts are our only change.
>
> The ASD or TD label selects one row from a small table, an *embedding* with 2 rows of 512 numbers, so 1,024 parameters in total. That row is simply *added* to the 512-number context, right before both heads. So the label can shift both where the next fixation goes and how long it lasts.
>
> We trained in two phases. In Phase 1, we fine-tuned TPP-Gaze on all children together, with no label. In Phase 2, we froze the whole model and trained *only* these 1,024 numbers.
>
> The embedding starts at zero, so Phase 2 begins exactly equal to Phase 1. We also checked that all 1,290 base weight tensors are identical in the two checkpoints. So any difference between Phase 1 and Phase 2 comes from the embedding alone. That is what makes the comparison clean.

**Key points to emphasize**
- 1,024 parameters, added to the context before both heads.
- Freezing plus zero initialization means that Phase 1 is exactly Phase 2 with a zero embedding.
- This is a deliberate design for a fair, isolated test, not a limitation we forgot about.

**Transition:** "Next, the data we trained and tested on."

---

## Slide 7: Datasets and preprocessing

**Suggested time:** 40 s

> We used two datasets.
>
> Saliency4ASD has 300 natural images viewed by children with ASD and controls. We split it *by image* into 210 training, 45 validation and 45 test images.
>
> Gaze4ASD is separate: 30 images, 33 ASD and 133 TD children. We used it only for testing, never for training or choosing anything.
>
> Notice the ages: 5 to 12 years versus 3 to 7, and neither release gives each child's age.
>
> For Gaze4ASD we also had to convert screen coordinates into image coordinates; I'll come back to that.

**Key points to emphasize**
- The split is by image, so test images are truly unseen.
- Gaze4ASD is external, used only for evaluation.
- The age difference matters later, as a limitation.

**Transition:** "With the data in place, here is how we designed the experiments."

---

## Slide 8: Experimental design and evaluation metrics

**Suggested time:** 50 s

> We evaluate in two ways.
>
> First, *likelihood*. Both models score the same held-out scanpaths, and we report negative log-likelihood per fixation, where lower is better. Our "gain" is Phase 1 minus Phase 2, so a positive gain favours conditioning. Intervals come from resampling whole images or children, because fixations from the same image or child are not independent.
>
> Second, *generated behaviour*, which is experiment E1. We use common random numbers: the ASD label, the TD label and Phase 1 all get the same seeds, so any difference comes from the model, not luck. Phase 1 on two seed streams gives a seed-only baseline, which we subtract.
>
> All analyses are exploratory.

**Key points to emphasize**
- Lower NLL is better, and a positive gain favours Phase 2.
- Clustered intervals, because fixations are not independent.
- Same seeds isolate the embedding's effect from random noise.

**Transition:** "So, first result: does conditioning help on held-out images?"

---

## Slide 9: In-domain likelihood: inconclusive gain

**Suggested time:** 55 s

> On the 45 held-out test images, Phase 2 lowers the NLL by 0.0073 nats per fixation, which is about 1.5 percent.
>
> But look at the interval: it goes from minus 0.0012 to plus 0.0163, so it includes zero. Only 27 of the 45 images favour Phase 2, and the sign-flip test gives a p-value of about 0.11. So we call this *inconclusive*, not an improvement.
>
> In the plot, the duration part and the ASD subset do clear zero, but only without correction for multiple comparisons, so we treat them as exploratory.
>
> For scale: fine-tuning itself moved the NLL from about minus 0.17 to minus 0.50. The embedding adds very little on top of that.

**Key points to emphasize**
- +0.0073, with an interval that includes zero: inconclusive.
- Don't present the sub-results as findings; they are exploratory.

**Transition:** "The external dataset gives a clearer answer, and it's negative."

---

## Slide 10: External likelihood: Phase 2 fits worse

**Suggested time:** 1 min

> On Gaze4ASD, the unconditioned Phase 1 model fits better overall, by 0.0172 nats per fixation. The interval excludes zero whether we resample children, images, or both. TD trials are clearly worse under conditioning.
>
> ASD trials are slightly better, plus 0.0104. But that holds only when we resample children, only 17 of 30 images agree, and under another coordinate mapping it shrinks to almost zero. A post hoc check also showed that the ASD embedding helps TD trials, so the gain isn't specific to autistic children.
>
> In the figure, the overall and TD results stay negative under all three preprocessing versions.
>
> One caveat: Gaze4ASD doesn't record where each image was drawn on screen. Our mapping matches the authors' own saliency maps almost perfectly, but it isn't confirmed.

**Key points to emphasize**
- Overall worse (−0.0172), TD worse, and both robust across preprocessing.
- The ASD gain is fragile and not ASD-specific.
- We did not test *why* it is worse; don't guess on stage.

**Transition:** "Likelihood is one view. Next: does the label change the scanpaths the model generates?"

---

## Slide 11: Generated contrasts: partly right, partly wrong

**Suggested time:** 55 s

> First, the real data, at the bottom. In Gaze4ASD, autistic children make larger saccades, meaning bigger jumps between fixations. They spread their fixations more, revisit places less, and spend less total time fixating.
>
> Does the embedding change generated behaviour? Yes. On 8 of 9 measures, the change is clearly bigger than the seed-only baseline.
>
> For the three spatial measures, the generated ASD-minus-TD difference points the right way. It reaches about 69, 64 and 32 percent of the real difference.
>
> But fixation count, total fixation time and direction concentration go the wrong way. The ASD label produces more fixations and more time, the opposite of the real data.
>
> And these shifts are small: about 1 percent of the image diagonal.

**Key points to emphasize**
- Real effect beyond seed noise, but small in absolute size.
- Three spatial contrasts are right at reduced size; three are in the wrong direction.
- Say "about": the shares vary by several points between runs.

**Transition:** "But a correct difference between groups doesn't mean each group became realistic, and that's the next slide."

---

## Slide 12: Per-group realism does not improve

**Suggested time:** 1 min

> Here we compare each label with its *own* real group. In the table, no spatial measure gets significantly closer for either group, and ASD revisit rate actually gets worse. For timing, TD moves toward real TD, but ASD moves *away* from real ASD.
>
> The dot plot shows why the dispersion difference is still correct. The ASD condition doesn't move: 0.113 in both models. The TD condition contracts from 0.112 to 0.101, past the real TD value of 0.108. So the two groups move apart, but not toward their real groups.
>
> And below: both labels produce about 10 fixations per scanpath, against about 6 in real data, because the sampler simply fills the 3-second window. That shared error is much bigger than the conditioning effect.

**Key points to emphasize**
- A right *contrast* is not the same as right *groups*.
- The ASD condition stays put; the TD condition contracts past real TD.
- Over-production of fixations (about 10 vs about 6) dominates.

**Transition:** "One sampling setting changes these numbers, and that's temperature."

---

## Slide 13: Sampling temperature: behaviour, not likelihood

**Suggested time:** 50 s

> Temperature, tau, is a sampling setting. It raises the width of each spatial blob, sigma, to the power tau. Every sigma is below 1, so tau equals 2 makes the blobs *narrower*. The likelihood never uses temperature.
>
> We chose tau equals 1 on validation images. On Gaze4ASD it was more realistic than tau equals 2 on all four pre-specified tests, in all 10 seeds.
>
> The right chart is the key point. At tau equals 2, every group difference keeps its direction but shrinks: 46, 33 and 15 percent instead of 69, 64 and 32. So the size of generated differences depends on a setting that likelihood can't see.

**Key points to emphasize**
- τ changes sampling, never likelihood, so it is not a model improvement.
- The direction of each difference is stable; its size is not.
- Always state τ when comparing generated scanpaths.

**Transition:** "Now let me show you what the generated scanpaths actually look like."

---

## Slide 14: Demo: same image, same seed, three models

**Suggested time:** 1 min (add about 45 s if you run the live demo; see section 17)

> Numbers give the order, green is the first fixation, red is the last, and bigger circles mean longer fixations.
>
> Each row is one test image: Phase 1, then Phase 2 with the ASD label, then Phase 2 with the TD label, all with the same seed. Because they share random numbers, they look almost the same. The embedding is a small nudge, not a different behaviour.
>
> Image 102 is where the model's average ASD-minus-TD dispersion difference best matches the real one. Image 116 is the worst: real ASD scanpaths there are *less* spread out, but the model says *more*. In fact, the model's difference is positive on all 45 test images, so it is roughly a uniform shift.
>
> And each panel is one random sample. It is not a child, and it is not a group profile.

**Key points to emphasize**
- Same seed means the panels differ only through the embedding.
- The best and worst images were chosen by a rule fixed in advance, not cherry-picked.
- One sample is not a person.

**Transition:** "Let me now summarize the limitations behind all of this."

---

## Slide 15: Limitations and threats to validity

**Suggested time:** 40 s

> These are the main limitations.
>
> The conditioning results are inconclusive or negative. Generated scanpaths have too many fixations and get the timing differences wrong.
>
> The Gaze4ASD coordinate mapping is strongly supported but not confirmed. Off-image gaze is itself a group difference: autistic children looked outside the image about 2.7 times as often, so neither clipping nor removing those fixations is neutral.
>
> Age could confound the group differences, and we can't adjust for it, because individual ages aren't released.
>
> All tests are exploratory. And there is no clinical validity.

**Key points to emphasize**
- Name the limitations yourself, before anyone asks.
- Age and the coordinate mapping are the two you will most likely be questioned on.

**Transition:** "So, what do we conclude?"

---

## Slide 16: Conclusions and future work

**Suggested time:** 45 s

> A 1,024-parameter group label gives a small, inconclusive gain in-domain and a worse fit externally. Generated scanpaths get the direction of the main spatial differences right, at reduced size. They get timing wrong, and neither group becomes more realistic. Temperature changes the size of these differences, not the likelihood.
>
> Our contribution is a careful, controlled test, and one lesson: a correct group difference is not enough; you must also check each group's realism.
>
> Next: confirm the mapping with the dataset authors, obtain age data, model stopping and off-image gaze, and replicate with pre-registered tests.

**Key points to emphasize**
- Repeat the verdicts exactly as on the slide: inconclusive, negative, partial, negative, caveat.
- End on the methodological lesson; it is your strongest, most defensible point.

**Transition:** go into the closing statement (section 19).

---

## Section 17: Live demo narration

Use this on slide 14 only if the laptop is set up (see `REHEARSAL_CHECKLIST.md`). The demo reproduces the **top row of slide 14** (image 102) live. It was tested on 2026-11-03: it took about 18 s, and all three scanpaths were identical to the pre-generated figure (maximum difference 0.0).

**Setup before the talk**
- Open a terminal in the project root (`gaze-ag/`).
- Type the command, but don't press Enter yet.
- Run it once beforehand as a warm-up.

**Before running (about 10 s)**

> Let me show that this is real and reproducible. I'll run the model now on image 102, with the same seed and settings as on this slide. It takes about 20 seconds, mostly loading the model.

**Command**

```bash
python inference/inference.py --image TrainingData/TrainingData/Images/102.png --compare-phase1 --no-image --out-dir inference/demo/local_only/live
```

`--no-image` draws on a blank canvas, as on the slides, so no dataset photograph is projected.

**While it runs (about 20 s; fill the wait)**

> Phase 1 and Phase 2 share the same frozen base; the program even checks that the weights are identical. All three panels use seed 0 and temperature 1, so they get the same random numbers. Any difference you see comes only from the 1,024-parameter embedding.

**Open the output:** `inference/demo/local_only/live/102/102_phase1_vs_asd_vs_td.png`

**What to point out (about 15 s)**
- **Reproducibility:**
  > This is exactly the top row of the slide: 13, 14 and 13 fixations, with the same positions and durations.
- **The encoding:** point to fixation 1 (green), which is in about the same place in all three panels, and fixation 10 (the large circle on the left), which is the longest fixation in all three (0.78, 0.75 and 0.83 s).
- **Phase 1 vs ASD:**
  > The first seven durations are identical to two decimals, and the first two positions match. Then the paths drift apart a little.
- **TD:**
  > The TD panel differs a bit more: durations change slightly from the first fixation, and later fixations go up to the right.
- **The extra ASD fixation:** the ASD sample has one more fixation (14 vs 13).
  > On average the ASD label does produce slightly more fixations, but one sample can't show that.

**Explaining image 102 (best match)**

> On this image the real ASD-minus-TD difference in spread is +0.0133, and the model's average difference over many samples is +0.014, almost the same. But that agreement is an *average* over 200 samples, which is 20 scanpaths times 10 seeds. You can't see it in one panel.

**Explaining image 116 (worst match; point to the bottom row of the slide)**

> On image 116 the real ASD scanpaths are *less* spread out than TD, −0.0473, but the model's difference is +0.0151, the opposite sign. This is why we call it a uniform shift: the model pushes ASD toward more spread on every test image, whatever the image.

**If the panels look almost identical**

> That's expected, and it's actually the point. Same seed, same frozen model; the only difference is 1,024 numbers. The group effect is real but small, about 1 percent of the image diagonal on average, and it only shows up over many samples and images, not in one picture.

**If the live demo fails** (an error message, or nothing after about 45 s)
- Stop at once. Do not debug in front of the audience.
- Say:
  > The live run isn't cooperating today, but the figure on this slide was produced by exactly this command and this seed. Our reproducibility check regenerated all 35 demo scanpaths identically.
- Then continue with slide 14 as normal.
- Backup file: `inference/demo/demo_results/s4a_best_102/phase_comparison_no_image.png`. The slide itself is the main fallback.

**Never say:** "this is how an autistic child looks at the image", "the ASD scanpath", or anything that links one sample to a person. Say instead: "a sample generated with the ASD label".

**If asked to see the scanpaths on a real image:** use the synthetic scene. It is the model's own test drawing, contains no dataset content, and can be shown freely.

```bash
python inference/inference.py --image inference/demo/inputs/synthetic_scene.png --compare-phase1 --out-dir inference/demo/local_only/live
```

Mention that it is outside the training distribution. Pre-generated version: `inference/demo/demo_results/synthetic_scene/phase_comparison.png`.

**If asked about variability:** open `inference/demo/demo_results/s4a_random_60/seeds_grid_no_image.png`.

> Four seeds per condition. Samples vary much more from seed to seed than ASD and TD differ at one seed.

---

## Section 18: Concept guide (understand these, don't memorize them)

Each concept has a simple version for the talk and a precise version for follow-up questions.

### 18.1 What a scanpath is
- **Simple.** The ordered list of places the eyes stopped at, and how long they stopped at each.
- **Precise.** A sequence of events (xₙ, dₙ), n = 1…N. Here xₙ ∈ [−1, 1]² is the normalized fixation location and dₙ is its duration in seconds, within a 3 s viewing window.

### 18.2 Why model location and duration jointly
- **Simple.** Where you look and how long you look are linked: a face may hold your gaze longer than a wall. A model that ignores timing misses half of the behaviour, and some group differences (total fixation time, count) are about timing.
- **Precise.** TPP-Gaze conditions both heads on the same context, built from the image and the full history of locations *and* durations. So timing affects future locations, and the reverse. Given the context, the two heads are separate distributions, and the joint NLL is the sum of their log-likelihoods.

### 18.3 What a temporal point process does
- **Simple.** It's a model for "what happens next, and when", for events at irregular times. Here every fixation is an event.
- **Precise.** A marked temporal point process defines the distribution of the next inter-event time and its mark, given the history. TPP-Gaze models the inter-event time (the fixation duration) directly with a log-normal mixture, following Shchur et al.'s intensity-free approach [16], instead of an intensity function. The location is the mark.

### 18.4 The spatial GMM and the temporal LGMM
- **Spatial GMM (simple).** 16 soft blobs on the image, each with a centre, a width and a weight. The next fixation is drawn from one of them. With several blobs, the model can keep several candidate targets at once, such as a face *or* a sign.
- **Spatial GMM (precise).** A bivariate Gaussian mixture with J = 16 components and diagonal covariance. Its means pass through tanh, so they stay in [−1, 1], and its weights pass through softmax.
- **Temporal LGMM (simple).** The same idea for duration. Durations are always positive and usually short with a long tail, so a mixture of log-normal shapes fits them well.
- **Temporal LGMM (precise).** A K = 4 log-normal mixture over log-durations. The log-durations are normalized by the training-split mean (−1.4366) and standard deviation (0.6936), and the log-scales are clamped to [−5, 3].
- **Provenance.** Both designs come from TPP-Gaze; we did not choose them.

### 18.5 How TPP-Gaze uses image features and gaze history
- **Simple.** The image is summarized once into 256 numbers. The history of fixations so far is summarized into another 256 numbers. Joined together, they are the "context" for predicting the next fixation.
- **Precise.**
  - **Image path.** DenseNet201 (frozen) gives 2,048 channels. CoordConv adds coordinate channels (2,051 in total). Three 1×1 convolutions and a linear layer then give z (256-d).
  - **History path.** Each past event is a linear embedding of its location plus a sinusoidal encoding of its normalized log-duration. These pass through 4 transformer layers (4 heads) and an LSTM, giving h (256-d).
  - **Context.** cₙ = [hₙ₋₁ ‖ z], 512-d.
- **Disclosure.** The released transformer attends across the batch rather than along each sequence (the batch-layout issue); temporal order comes from the LSTM.

### 18.6 How our ASD/TD embedding works
- **Simple.** A table with two rows of 512 numbers, one row for ASD and one for TD. The label picks a row, and we add it to the context.
- **Precise.** c̃ₙ = cₙ + e_ℓ, with e_ℓ ∈ ℝ⁵¹² and ℓ ∈ {0 = ASD, 1 = TD}. It is added before *both* heads, so it can shift both location and duration, and it is added at every generation step.
- **Size.** It is a constant shift of the context, so it cannot express image-specific group effects very flexibly. That fits the "uniform shift" we observed, although we did not test this as a cause.

### 18.7 Why the base model is frozen in Phase 2
- **Simple.** So the comparison is fair. If we also changed the base, we couldn't tell whether a difference came from the label or from retraining.
- **Precise.**
  - With zero initialization, Phase 2 starts identical to Phase 1.
  - All 1,290 base tensors are identical in both checkpoints, so Phase 1 is exactly Phase 2 with a zero embedding.
  - It also asks the narrow research question directly: what can a *minimal* label do?
  - Whether full fine-tuning would do better was **not tested**.

### 18.8 What likelihood measures
- **Simple.** How probable the model finds the real data. If real fixations land where, and last as long as, the model expected, the likelihood is high and the NLL is low.
- **Precise.**
  - NLL = −(1/N_events) Σ [log p(dₙ | c̃ₙ) + log p(xₙ | c̃ₙ)], in nats per fixation (natural log; one fixation is one event).
  - It is evaluated with the *observed* history (teacher forcing) and no temperature.
  - It splits into a duration part and a position part.

### 18.9 Why NLL can be negative, and why a small improvement matters (or doesn't)
- **Negative values.** These are probability *densities* over continuous coordinates and durations, and densities can be larger than 1, so their log can be positive and the NLL negative. Only *differences* between models on the same data matter.
- **Lower is better.** "Gain" = Phase 1 NLL − Phase 2 NLL, so a positive gain favours Phase 2.
- **What +0.0073 means.**
  - exp(0.0073) ≈ 1.007, so Phase 2 gives real fixations about 0.7% higher density on average (geometric mean). This is plain arithmetic on the reported gain.
  - That is small, and its interval includes zero.
  - Externally, −0.0172 means Phase 2 gives real fixations *lower* density than Phase 1.
- **Why it matters at all.** Likelihood on held-out data is the standard, sampling-free test of whether a model describes the data better. It is the most direct test of whether the label helps.

### 18.10 What sampling temperature changes
- **Simple.** How tightly the model samples around its blobs. It is a dial on generation only.
- **Precise.**
  - σⱼ(τ) = σⱼ^τ is applied to the spatial component scales only; means, weights and durations are untouched.
  - Every σ < 1 (median RMS σ = 0.131 on validation images), so τ = 2 *narrows* the components. On validation images it took the per-axis within-component SD from 0.138 to 0.028 for later fixations, while total spread fell only from 0.279 to 0.242.
  - NLL never uses τ, so τ cannot improve likelihood.
  - Durations can still change indirectly, because later fixations depend on earlier sampled locations.

### 18.11 Why identical random seeds are used
- **Simple.** So that the only thing that changes between panels is the model. Same dice, different player.
- **Precise.**
  - Common random numbers: P2-ASD, P2-TD and Phase 1 draw from the same seed streams.
  - Phase 1 on stream A vs stream B is a *seed-only null*: the difference you get from randomness alone.
  - Conditioning effect = (P2 ASD − P2 TD) − (P1 A − P1 B).
  - This removes most of the sampling noise, which is why d_z values are large (1–5) even for shifts of about 1% of the diagonal. d_z measures detectability, not size.

### 18.12 What the external evaluation tells us
- **Simple.** Whether the label's effect survives on different children, images and equipment. It doesn't: overall fit gets worse.
- **Precise.**
  - Phase 1 −0.5568 vs Phase 2 −0.5396; gain −0.0172 (participants [−0.0224, −0.0122]; images [−0.0324, −0.0018]).
  - TD −0.0229. ASD +0.0104, which clears zero by participants only.
  - The overall and TD results are robust across fit-to-height, stretch and Policy B. The ASD position gain is mapping-dependent.
  - The datasets differ in age, tracker, display, stimuli and site, so the cause can't be identified.
  - Duration likelihood largely transfers (−0.399 vs −0.464 in-domain). Position beats a centre-bias reference by 0.91 nats on 30 of 30 images.

### 18.13 Why group contrasts can be right while per-group realism doesn't improve
- **Simple.** A contrast measures the *distance between* two groups. Realism measures whether each group is *in the right place*. You can get the distance right by moving the wrong group, or both groups, the wrong way.
- **Example (dispersion).** The ASD condition stays at 0.113; the TD condition moves from 0.112 to 0.101, past real TD (0.108). The gap ASD − TD grows in the correct direction, but TD is now *less* realistic than before, and ASD didn't change.
- **Lesson.** Evaluations of conditional generators should always report per-group realism alongside contrasts.

---

## Section 19: Opening, closing and the main contribution

**Opening (about 30–35 s).** This is the slide 1 script above (73 words). If you need a shorter version (about 15 s):

> Good morning. I'm [name]. We added a small ASD-or-TD label to a published gaze model and tested carefully whether it helps. Mostly it doesn't, and I'll show where and why. This is research, not a diagnostic tool.

**Closing statement (after slide 16, about 15 s)**

> To sum up in one line: a tiny group label shifts generated gaze in partly the right direction, but it doesn't make the model fit better or make either group more realistic. We think reporting that honestly, with a careful design, is the useful result. Thank you.

**Inviting questions**

> I'd be happy to take questions, about the model, the statistics or the demo.

If there is a silence, seed a question yourself:

> One question people often ask is why we froze the model. I'm happy to explain that.

**"What is the main contribution of your work?" (about 20 s)**

> A controlled test of minimal ASD/TD conditioning in an existing gaze model. Because Phase 1 and Phase 2 share identical base weights, every difference comes from 1,024 parameters. We tested likelihood in-domain *and* on an external dataset, compared generated behaviour with common random numbers, and showed that a correct group contrast can arise without either group becoming more realistic. The contribution is the evaluation and that lesson, not a better model.
