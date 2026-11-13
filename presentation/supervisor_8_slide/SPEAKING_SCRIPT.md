# Speaking Script: 8-Slide Supervisor Presentation

This is a progress report to Mam on our own work. It assumes she knows eye tracking, scanpaths, temporal point processes and TPP-Gaze. It explains only what *we* did, why we did it, and what we found.

**Two versions of the deck. Only slide 4 differs.**
- **Restricted version:** `local_only_restricted/ASD_ScanPath_Supervisor_Presentation_RESTRICTED.pptx`.
  - Use this one when you present from your laptop.
  - It shows the actual Saliency4ASD test photographs and the real-vs-generated fixation plots.
  - Local use only; do not email, upload or commit it.
- **Shareable version:** `ASD_ScanPath_Supervisor_Presentation.pptx`. The photographs are withheld.

Slide 4 has a *(Restricted version)* paragraph and a *(Shareable version)* paragraph; say the one that matches the deck on screen. Each deck's speaker notes contain only its own paragraph.

**Sources.**
- Numbers: the final manuscript (`paper/final/ASD_ScanPath_Final.md`).
- Training settings: the saved configs and run summaries (`training/config*.yaml`, `training/runs/*/summary_full.json`).
- Example values: `demo/demo_results/`.

**How to use it.** The quoted text is what you say. Practise until you can say it in your own words. The questions under each slide are what Mam is most likely to ask.

## Timing

| Slide | Topic | Time | Cumulative |
|---|---|---|---|
| 1 | Research question and approach | 1:10 | 1:10 |
| 2 | What we kept and what is new | 1:05 | 2:15 |
| 3 | Implementation | 0:50 | 3:05 |
| 4 | Datasets and test images | 1:45 | 4:50 |
| 5 | Training | 1:05 | 5:55 |
| 6 | Evaluation design | 1:35 | 7:30 |
| 7 | Results | 1:35 | 9:05 |
| 8 | Interpretation and next steps | 1:10 | 10:15 |

These times are for the restricted version at a calm 130 words per minute: about 10 minutes in total (the shareable version is about 15 s shorter). At a slightly quicker 140 words per minute it takes about 9:25. If you are running long, shorten slide 6's preprocessing paragraph; keep slides 4, 7 and 8 complete.

---

## Slide 1: What does a minimal ASD/TD label change in TPP-Gaze?

**Suggested time:** 1 min 10 s

> Mam, I'll report where our project stands.
>
> We started from TPP-Gaze. It already models fixation location and duration together, but it cannot take a group label. We could have trained separate ASD and TD models. But then every weight would differ, and we couldn't tell which differences come from the group itself.
>
> So our main question was: if one shared TPP-Gaze model gets only a small ASD-or-TD label, what changes? We asked three things. Does it fit our held-out Saliency4ASD images better? Does that hold on Gaze4ASD, which it never saw? And do the scanpaths it generates for the two labels differ the way real ASD and TD children differ?
>
> To test this, we trained Phase 1 without the label, added the label in Phase 2, and compared the two on the same data and the same random seeds.
>
> The short answer is in the dark box: mostly, the label did not help.

**What the slide shows**
- The gap, the three research questions, and our three-step approach.
- The short answer, so Mam knows where the talk is going.

**Points to make clear**
- "Minimal" is deliberate: one shared model, one small label. It is a controlled test, not an attempt to build the best ASD model.

**Transition:** "First, let me be exact about what we kept and what we added."

**Likely questions**
- *Why not just train two separate models?*
  > Two models would differ in every weight. Then we couldn't say whether a difference comes from the group or from training noise. A shared model with a label isolates the group effect. We did not train separate models, so I can't compare them.
- *How is this different from Zhong et al.?*
  > They predict ASD scanpaths with semantic maps. We could only read their abstract, so we don't compare with them or claim to be better.

---

## Slide 2: What we kept from TPP-Gaze, and what is new

**Suggested time:** 1 min 5 s

> On the left is everything we kept from the released TPP-Gaze: the code, the pretrained weights, both encoders, both heads, the loss and the sampler. The released transformer mixes information across the scanpaths in a batch. We kept that, because the pretrained weights were learned that way, and we disclose it.
>
> On the right is what we added: a small wrapper with a two-row group embedding, added before both heads. We also set three things ourselves: duration normalization from our own training split, sampling temperature 1 instead of the released 2, and at least 3 fixations per scanpath instead of 4.
>
> The bottom box is the key design decision. The embedding starts at zero, and the rest of the model is frozen. So Phase 1 is exactly Phase 2 with a zero embedding. Any difference we measure between them comes from those 1,024 numbers.

**What the slide shows**
- **Left:** unchanged TPP-Gaze components.
- **Right:** everything new in our code and settings.
- **Bottom:** why the design is minimal, and what we do not claim.

**Points to make clear**
- We checked that all 1,290 base weight tensors are identical in the two checkpoints. That is what makes the Phase 1 vs Phase 2 comparison clean.

**Transition:** "Here is exactly where the embedding sits in the model."

**Likely questions**
- *Why temperature 1 when the released config uses 2?*
  > We chose it on Saliency4ASD validation images: τ = 1 had lower error for saccade amplitude and revisit rate. It is also the model's own distribution, which is the one the likelihood scores. Gaze4ASD was not used for this choice.
- *Does the batch quirk affect your numbers?*
  > For likelihood, very little: shuffling batches on Gaze4ASD changed overall NLL by about 0.0001 or less. In generation, the 20 samples drawn together can influence each other, which we disclose.

---

## Slide 3: Implementation: where the group embedding enters

**Suggested time:** 50 s

> Teal is the unchanged TPP-Gaze, and purple is our part.
>
> The image and the gaze history are encoded as before and joined into a 512-number context. The label picks one 512-number row from the embedding, and we simply add it to that context. The result feeds both heads, so the label can shift the next location and the next duration.
>
> We add it the same way in all three code paths: training, likelihood scoring and sampling. That way, the scores and the generated scanpaths come from the same model.
>
> The base has almost 24 million parameters. In Phase 2, only 1,024 of them were trainable.

**What the slide shows**
- The real information flow in `training/conditioned.py`.
- The embedding enters only at the fused context. The encoders are untouched.
- The dashed loop is used only in sampling.

**Points to make clear**
- It is an *added* vector, not a concatenated one, so the input size of the heads does not change, and the pretrained heads load unchanged.

**Transition:** "Next, the data we trained and tested on, with two actual test images."

**Likely questions**
- *Why add it there and not to the history encoder?*
  > Added just before the heads, one vector can shift both location and duration, and the pretrained encoders stay exactly as released. We did not test other positions.

---

## Slide 4: Datasets, and the actual test images we used

**Suggested time:** 1 min 45 s

> Saliency4ASD is our main dataset, with 300 images. We split it by image with a fixed seed into 210, 45 and 45 images, so no test image was seen in training. Gaze4ASD is only an external test: 30 mostly face images, and 33 ASD and 133 TD children aged 3 to 7, recorded with a different tracker and display.
>
> Below are two of our 45 test images. Each row shows the photograph, then one sample from Phase 1, Phase 2 with the ASD label, and Phase 2 with the TD label, all with the same seed. They look almost the same, so the label changes a single sample very little.
>
> *(Restricted version)* On the right we compare with the real children. The top pair is all real fixations from the ASD and TD viewers; the bottom pair is 20 generated scanpaths per label. On image 116, the real ASD fixations stay near the middle person's face, but the generated ASD fixations spread over all three faces.
>
> *(Shareable version)* On the right are the real and generated spread values. The photographs and real-gaze plots are only in our local version, because the dataset's sharing terms are unresolved.
>
> We picked these two by a fixed rule: the best and worst match between generated and real ASD-minus-TD spread. On 102, the real difference is plus 0.0133 and the model's is plus 0.014. On 116, the real difference is minus 0.0473, but the model's is plus 0.0151, the wrong direction. So they show one strength and one limitation of the same model.

**What the slide shows**
- **Top:** both datasets, with the counts after preprocessing.
- **Rows:** two held-out test images, chosen by a rule fixed before any demo output.
- **Restricted version:**
  - the actual photos, and the scanpaths exactly as the model generated and saved them;
  - real vs generated fixations: all real scanpaths for the image (102: 6 ASD and 10 TD; 116: 7 ASD and 14 TD) against E1 replicate 1 (20 per label).
- **Shareable version:** blank-canvas scanpaths and the same statistics in place of the photos.

**Points to make clear**
- **Which numbers to quote.** The row labels are 10-seed averages. The plots and statistics on the right come from one replicate, which is noisier: for 116 that replicate gives ASD 0.162 vs TD 0.109, a gap of 0.053, much larger than the average of 0.0151. Quote the averages.
- **Why 102 is not a success story.** The model's ASD−TD dispersion difference is positive on all 45 test images (between +0.002 and +0.020). On 102 the real difference happens to be positive too; on 116 it is negative. The same shift meets two different images.
- **Over-production is visible.** The generated scanpaths have 8.8–11.9 fixations, against 5.8–9.2 in the real data for these two images.

**Transition:** "With the data in place, here is how we trained the two phases."

**Likely questions**
- *Why 133 TD children when the README says 132?*
  > The release has 133 TD folders, and all of them have trials. We used the data as released and report the mismatch.
- *Can these slides be shared?*
  > Not the version with photographs. The Saliency4ASD usage agreement forbids sharing any part of the data with third parties, and its relation to the later CC BY record is unresolved. That is why there is a separate shareable version.

---

## Slide 5: Training: Phase 1 baseline, then the label only

**Suggested time:** 1 min 5 s

> In Phase 1 we fine-tuned the released TPP-Gaze on ASD and TD training data together, with no label. The DenseNet backbone stayed frozen and its features were cached, so about 3.9 million parameters were trained. This adapted the model to these children's data first.
>
> The best validation NLL was minus 0.5845, at epoch 6. After that, validation got worse while training kept improving; by epoch 26 it was minus 0.384. So early stopping really mattered.
>
> In Phase 2 we loaded the best Phase 1 checkpoint, added the zero embedding and froze everything else. We kept the base in evaluation mode, so dropout could not change it, and used a lower learning rate, 10 to the minus 4. Validation NLL improved to minus 0.5964, at epoch 8.
>
> Every choice, including the checkpoints and the temperature, used validation data only.

**What the slide shows**
- **Top:** the six steps in order, including the preprocessing decisions.
- **Table:** the actual settings from the config files and run summaries.

**Points to make clear**
- The Phase 2 validation gain (−0.5845 to −0.5964) is optimistic, because validation chose the checkpoint. The test set gives the fair number, on slide 7.
- The minimum of 3 fixations is our own choice; the TPP-Gaze paper used 4.
- **Disclosure, if asked:** the repository also has an earlier Phase 2 setup at learning rate 10⁻³. Its results could not be found, so it is not reported. This is also in the slide footnote.

**Transition:** "Next, how we evaluated the two models."

**Likely questions**
- *Why a lower learning rate in Phase 2?*
  > Only 1,024 parameters were trained, starting from zero on top of a fixed model. The config notes that 10 to the minus 4 replaced an earlier 10 to the minus 3 setup, and we have results only for the lower rate.
- *Why drop the first fixation?*
  > It is part of our preprocessing for both datasets. We didn't run a sensitivity analysis on it.

---

## Slide 6: Evaluation: likelihood and generated behaviour, kept separate

**Suggested time:** 1 min 35 s

> We evaluated in two separate ways, because they answer different questions.
>
> Part A is likelihood. Both models score the same real scanpaths. The gain is Phase 1 NLL minus Phase 2 NLL, per fixation, so a positive number means the label helped. Saliency4ASD has no participant IDs, so in-domain we resample whole images; on Gaze4ASD we resample children, images, or both.
>
> On Gaze4ASD we test three preprocessing versions. The release gives screen pixels but not where the image was drawn. Fit-to-height matches the authors' saliency maps almost perfectly, but it is inferred, so we also test a stretch mapping. And ASD children looked off the image more often: 7.35 percent of fixations, against 2.72 for TD. So we either clip those fixations to the edge, our primary choice, or remove them.
>
> Part B is experiment E1. For each test image we generate 20 scanpaths per seed, over 10 seeds, with the ASD label, the TD label and Phase 1. Phase 1 on the same two seed streams shows how much difference the seeds alone make, and we subtract that. We also compare each label with its own real group.
>
> Temperature was a separate run. It only changes sampling, never the likelihood.

**What the slide shows**
- **Left:** the two likelihood paths and the three external variants, each with its reason.
- **Right:** the E1 design. Both Phase 2 labels and Phase 1 share seed streams, which gives the conditioning effect and the per-group realism check.
- **Bottom:** the separate temperature run and the statistics.

**Points to make clear**
- The primary variant was fixed before any model was evaluated on Gaze4ASD.
- The three primary measures were fixed before E1, but they were chosen because the earlier temperature run had reproduced them. So that choice is not fully independent.
- All analyses are exploratory; nothing was preregistered.

**Transition:** "Now, the actual results."

**Likely questions**
- *Why not cluster in-domain by participant?*
  > The release has no participant IDs. Sequences may not even map one-to-one to children, so images are the only cluster we can use.
- *Why not just compare ASD and TD samples directly?*
  > Even two runs of the same model differ because of the random seeds. The Phase 1 seed-only difference tells us how large that is, so we can see what the label adds beyond it.

---

## Slide 7: Results: what improved, what did not, what is unclear

**Suggested time:** 1 min 35 s

> The top left and the first three table rows are likelihood. The rest is generated behaviour.
>
> Phase 1 clearly helped. Fine-tuning moved the test NLL from minus 0.1665 for the pretrained model to minus 0.4957.
>
> Phase 2 on top of that gained only 0.0073 nats per fixation in-domain. The 95 percent interval runs from minus 0.0012 to plus 0.0163, so it includes zero, and only 27 of 45 images favour Phase 2. So this is inconclusive, not an improvement.
>
> On Gaze4ASD, Phase 2 is worse. Phase 1 is better by 0.0172, in all three preprocessing versions. TD trials get worse in every version. ASD trials improve slightly, by 0.0104, but only when we resample children.
>
> For behaviour, the label does change the samples: 8 of 9 measures move more than the seed-only baseline. The three spatial contrasts point the right way, at about 69, 64 and 32 percent of the real size. But fixation count, total time and direction concentration point the wrong way, and neither group gets more realistic. The model also makes about 10 fixations per scanpath, against about 6 in real data.
>
> Last row: temperature 2 keeps every direction but shrinks the contrasts. The likelihood does not change.

**What the slide shows**
- **Forest plot:** the joint NLL gain with its 95% interval. In-domain crosses zero; all three Gaze4ASD versions are entirely below zero.
- **Bar chart:** the generated ASD−TD contrast as a share of the real one. These are approximate shares, not accuracy.
- **Table:** every comparison with a plain verdict.

**Points to make clear**
- **What "69 percent" means.** On Gaze4ASD the real ASD−TD dispersion difference is 0.0169 of the image diagonal. The generated difference is 0.0117, about 69 percent of it. For amplitude it is 0.0096 against 0.0151, and for revisit rate −0.0261 against −0.0805. An independent run gave 60, 50 and 29 percent, so treat the shares as approximate.
- **Why realism doesn't improve although the direction is right.** The ASD condition keeps the same dispersion as Phase 1 (0.113). The TD condition contracts from 0.112 to 0.101, past the real TD value of 0.108. The gap grows in the right direction, but neither group moved closer to its real value.
- **The ASD external gain is fragile.** Its interval clears zero only when resampling children. Under the stretch mapping it falls to +0.0027, with an interval that includes zero. The ASD embedding also beats Phase 1 on TD trials (+0.0189, post hoc), so the gain is not ASD-specific.
- **Fixation counts.** 10.19 and 9.60 generated against 6.25 and 6.41 real. The label's own shift in count (+0.66) is only about a fifth of this shared excess.

**Transition:** "So, what do we take from this, and what should we do next?"

**Likely questions**
- *Why is the external fit worse?*
  > We haven't tested the cause. The datasets differ in age, tracker, display, stimuli and site all at once. What we do know is that the TD embedding fits both Gaze4ASD groups worse than no embedding.
- *Are these effects big?*
  > Statistically they are clear, but physically small: the dispersion and amplitude shifts are 1.1 and 0.8 percent of the image diagonal.

---

## Slide 8: What we can conclude, and what comes next

**Suggested time:** 1 min 10 s

> So what can we conclude? The label does change generated behaviour, but only by about 1 percent of the image diagonal, and in roughly the same way on every image: the generated ASD-minus-TD dispersion difference is positive on all 45 test images. The spatial contrasts look right only because the two conditions move apart. And the label does not improve likelihood: inconclusive in-domain, worse on Gaze4ASD.
>
> Some questions are still open. We haven't tested why the TD embedding fits Gaze4ASD worse, or how much the extra fixations and the missing off-image gaze cause the wrong timing results.
>
> The main limitations are the inferred screen mapping and the missing participant ages.
>
> Next, I'd first ask the Gaze4ASD authors about the display geometry, the participant count and the ages. Then I'd add stopping and off-image modelling, and rerun E1 with the same baseline. Mam, I'd value your advice on which to do first.

**What the slide shows**
- What we can conclude, what is unresolved, the limitations, and the next experiments.
- The one-sentence conclusion and the not-diagnostic statement.

**Points to make clear**
- **The conclusion box, word for word:**
  > It changes generated gaze behaviour and reproduces some spatial group-contrast directions, but does not consistently improve likelihood or individual-group realism.
- **Age.** We can't adjust for it. Neither dataset has participant-level age. Within Gaze4ASD the group means differ by about half a year (5.04 vs 4.54), and across datasets the ranges are 3–7 vs 5–12.

**Likely questions**
- *Which next step matters most?*
  > Over-production of fixations is the largest error we see, so I'd start with stopping and viewing time. Confirming the mapping with the authors comes first only because it costs little and strengthens every external result.
- *Could this be used for screening?*
  > No. It uses group labels only, we never built or tested a classifier, and there is no clinical validation.

---

## Short answers to have ready

| Topic | One-breath answer |
|---|---|
| Our contribution | A controlled test of a 1,024-parameter ASD/TD label in a frozen TPP-Gaze, with an unconditioned control, an external dataset, and per-group realism alongside group contrasts |
| Why frozen and zero-initialized | So that Phase 1 equals Phase 2 with a zero embedding (1,290 identical base tensors), and every difference comes from the label |
| In-domain result | +0.0073 nats per fixation, 95% CI [−0.0012, +0.0163]: inconclusive |
| External result | Phase 1 better by 0.0172, in all three preprocessing versions; cause not tested |
| 69 / 64 / 32% | The generated ASD−TD difference as a share of the real one (dispersion, amplitude, revisit): right direction, smaller size, approximate |
| Realism | No spatial measure moves closer to its own real group; the TD condition contracts past real TD |
| Temperature | Changes generated behaviour and contrast sizes; never the likelihood |
| Test images 102 and 116 | Best and worst match by a fixed rule; one strength and one limitation of the same uniform shift, not proof |
