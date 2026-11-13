# Meeting script: walking Mam through our folder

**How to use this.** Keep this script open on your side. Open each file in the order below. Wait until the file is on screen, then read the quoted text.
- The **"If Mam asks"** lines are backup only. Use them if she asks; otherwise skip them.
- At a calm pace (140 words per minute), the whole walkthrough takes about 12 minutes without questions.

**Folder:** `presentation/supporting_material/meeting_walkthrough/`. `01_example_image/` is local only (see `README.md`); the other figures are in the repository.

| Step | Open | Time |
|---|---|---|
| 1 | `01_example_image/` (2 files) | 1:20 |
| 2 | `02_what_we_did/` (2 files) | 2:05 |
| 3 | `03_training/training.png` | 1:05 |
| 4 | `04_testing/testing.png` | 1:40 |
| 5 | `05_novelty/novelty.png` | 1:00 |
| 6 | `06_results/` (4 files) | 4:30 |
| | **Total** | **about 12:00** |

**If you need to finish in about 9–10 minutes,** skip these paragraphs. Each one is repeated somewhere else.
- **Step 2:** on `2_architecture.png`, skip the last paragraph (adding vs concatenating).
- **Step 4:** shorten the bottom paragraph to: "Gaze4ASD's screen mapping is inferred, so we tested three preprocessing versions."
- **Step 5:** skip the middle paragraph. `testing.png` already showed that evaluation.
- **Step 6:**
  - On `1_results_at_a_glance.png`, say only the first sentence and then "The details are on the next two files."
  - On `3_generated_behaviour.png`, skip the "One more point" paragraph. It is in the "Realism" quick answer if she asks.

---

## Step 1: Start with one example

### Open `01_example_image/1_beach_photo.png`

> Mam, first i will show what model produces and then i will say how we did it. 

### Open `01_example_image/2_beach_asd_vs_td_scanpaths.png`

> These are two scanpaths our model generated for this photo: on the left with the ASD label, and on the right with the TD label. It's the same model and the same image . Only the label is different.

> The numbers show the order of fixations, and a bigger circle means a longer fixation.

<!-- > Here in this photo , Both start in the same place and go to the two faces. But they differ in the details. The ASD version has 9 fixations, and the TD version has 11. And the TD version stays longer on the woman's face, about 0.84 seconds. -->

<!-- > But this is only one sample for each label. One sample can't tell us whether the label really works. For that we need many samples compared with real children's gaze, and those results come at the end. First, I'll explain what we built. -->

**If Mam asks: what temperature is this?**
> This demo used temperature 2, the released TPP-Gaze default. For all our experiments we later chose temperature 1, based on validation data. Temperature only changes sampling, not the model.

**If Mam asks: can you show an image from your test set?**
> Yes. `inference/demo/local_only/` has test images 102 and 116. They are the best and the worst match between generated and real ASD−TD spread, chosen by a fixed rule.

**If Mam asks: does this show an individual child?**
> No. It's a sample from a group-level model. It doesn't represent any particular child.

---

## Step 2: What we did

### Open `02_what_we_did/1_our_approach.png`

> Mam now coming to the approch , 

> We started from the released TPP-Gaze. In Phase 1, we fine-tuned it on Saliency4ASD, with ASD and TD data together and no label. In Phase 2, we added an ASD/TD embedding and trained only that. Then we compared Phase 1 and Phase 2 on the same images,
<!-- with the same seeds and the same settings. -->

<!-- > We did it this way for one reason. If we trained two separate models, one for ASD and one for TD, every weight would be different. Then we couldn't tell which differences come from the group and which come from training noise. With one shared model and only a label, the label is the only difference. -->
>
<!-- > We asked three questions, shown at the bottom. Does it fit our test images better? Does that hold on Gaze4ASD? And do the generated ASD and TD scanpaths differ the way real children do? -->

### Open `02_what_we_did/2_architecture.png`

> This is the architecture. Everything in blue and green is TPP-Gaze, unchanged. The orange box is our addition.
>
<!-- > The image and the gaze history are encoded as before and joined into a 512-number context. The label picks one row of 512 numbers from our embedding, and we add it to that context. The result then goes to both heads. So the label can change where the next fixation goes and how long it lasts. -->
>
<!-- > We use the embedding the same way in training, in scoring and in generating scanpaths. That way, the likelihood numbers and the generated scanpaths come from exactly the same model.
>
> We add the embedding rather than concatenate it, so the input size of the heads stays the same and the pretrained heads load unchanged. -->

**If Mam asks: why add it there and not in the history encoder?**
> Just before the heads, one vector can shift both location and duration, and the pretrained encoders stay exactly as released. We didn't test other positions.

---

## Step 3: How we trained

### Open `03_training/training.png`

> We split Saliency4ASD by image, with a fixed seed: 210 images for training, 45 for validation and 45 for test. So no test image was seen in training.
>
> In Phase 1, we started from the released weights. 
<!-- The DenseNet backbone stayed frozen, so about -->
 3.9 million parameters were trained. The best validation NLL was minus 0.5845, at epoch 6. After that the model started to overfit: by epoch 26, validation was minus 0.384. So early stopping really mattered.
>
> In Phase 2, we loaded the best Phase 1 checkpoint, added the embedding starting at zero, and froze everything else. So only 1,024 numbers were trained. We used a lower learning rate, 10 to the minus 4. Validation NLL went from minus 0.5845 to minus 0.5964.
>
> Every choice, including checkpoints and temperature, was made on validation data only, never on test.

**If Mam asks: is the Phase 2 validation gain the real gain?**
> No. It's optimistic, because validation also chose the checkpoint. The fair number is the test result, which I'll show in the results.

**If Mam asks: were there other Phase 2 runs?**
> The repository has an earlier Phase 2 configuration with learning rate 10 to the minus 3, but its results couldn't be found, so we don't report it.

---

## Step 4: How we tested

### Open `04_testing/testing.png`

> We tested in two separate ways, because they answer different questions.
>
> On the left is likelihood. Both models score the same real scanpaths. A positive gain means the label helped. On our test set we resample images for the confidence interval, because the release has no participant IDs. On Gaze4ASD we resample children, images, or both.
>
> On the right is experiment E1, for generated behaviour. For every test image we generate 20 scanpaths per seed, over 10 seeds: with the ASD label, with the TD label, and with Phase 1 twice. Even the same model gives two different sets of samples just because of the seeds. Phase 1 run twice shows how big that seed difference is, so we subtract it. What is left is the effect of the label. We also compare each label with its own real group, to check realism.
>
> At the bottom: Gaze4ASD gives screen coordinates but not where the image was drawn on the screen. Our main mapping, fit-to-height, matches the authors' own saliency maps almost perfectly, with r equal to 0.9999. But it is still inferred, so we also tested a stretch mapping. ASD children also looked off the image more often, 7.35 percent against 2.72 for TD. So we either clip those fixations to the edge, which is our main choice, or remove them.

---

## Step 5: What is new

### Open `05_novelty/novelty.png`

> To be clear about our contribution: on the left is everything we kept from TPP-Gaze unchanged. On the right is what is new.
>
> The main addition is the ASD/TD embedding. It starts at zero, it is added before both heads, and it is used the same way everywhere. We also added the evaluation: the seed-only baseline from Phase 1, the external test on Gaze4ASD with three preprocessing versions, and the check of each group's realism, not only the ASD-TD difference.
>
> The bottom box is the key point. All 1,290 base weight tensors are identical in both checkpoints. So Phase 1 is exactly Phase 2 with a zero embedding, and every difference we measure comes from the label.
>
> We do not claim that this is better than other ASD scanpath models, and it is not a diagnostic tool.

---

## Step 6: Results

### Open `06_results/1_results_at_a_glance.png`

> Now the results. Here is everything on one page.
>
> The top three rows are likelihood, meaning how well each model fits real gaze.
> - Fine-tuning clearly helped.
> - The label in Phase 2 is inconclusive on our own test images.
> - On Gaze4ASD, the dataset it never saw, the label made it worse.
>
> The next four rows are about the scanpaths it generates.
> - The label does change them.
> - Some differences go the right way.
> - Some go the wrong way.
> - Neither group becomes more realistic.
>
> The last row says that temperature changes the samples but not the likelihood.

### Open `06_results/2_likelihood.png`

> On the left: fine-tuning moved the test NLL from minus 0.1665 to minus 0.4957. Lower is better, so Phase 1 is a clear improvement over the released model.
>
> On the right is what the label adds on top of Phase 1. Right of zero means the label helped; left of zero means it hurt.
>
> On our own test images, the gain is plus 0.0073 nats per fixation. But the interval runs from minus 0.0012 to plus 0.0163. It includes zero, and only 27 of 45 images favour Phase 2. So we call this inconclusive, not an improvement.
>
> On Gaze4ASD, Phase 2 is worse, by 0.0172 in our main version. We tried three different ways of preprocessing Gaze4ASD, and it is worse in all three. So this result does not depend on how we prepared the data.

**If Mam asks: why is it worse on Gaze4ASD?**
> We haven't tested the cause. The two datasets differ in age, eye tracker, display, images and site, all at once. What we do know is that the TD embedding fits both Gaze4ASD groups worse than having no embedding.

### Open `06_results/3_generated_behaviour.png`

> This is generated behaviour on Gaze4ASD, over 10 seeds. Each bar is the generated ASD-minus-TD difference as a share of the real difference. 100 percent would match the real data exactly.
>
> The three green bars go the right way. Dispersion reaches 69 percent of the real difference, saccade amplitude 64 percent, and revisit rate 32 percent. So the direction is right, but the size is smaller. A separate run gave 60, 50 and 29 percent, so please treat these as approximate.
>
> The red bars go the wrong way. For total fixation time and direction concentration, the model reverses the real difference. Fixation count is also reversed, though the real count difference is not significant.
>
> On the right: the model makes about 10 fixations per scanpath, but real children make about 6. That over-production is bigger than anything the label does.
>
> One more point. The direction is right, but that doesn't mean each group got more realistic. The TD condition shrinks its spread from 0.112 to 0.101, past the real TD value of 0.108, and ASD stays at 0.113. The two conditions move apart, but neither moves closer to its own real group.

**If Mam asks: are these effects large?**
> Statistically they are clear: 8 of 9 measures move more than seeds alone would. But physically they are small, about 1 percent of the image diagonal.

### Open `06_results/4_conclusion_next_steps.png`

> So, in one sentence: a minimal ASD/TD embedding changes generated gaze behaviour and reproduces some spatial group differences, but it does not consistently improve likelihood or realism for each group.
>
> Two things are still open: why the TD embedding fits Gaze4ASD worse, and how much the extra fixations and the missing off-image gaze cause the wrong timing results.
>
> Our main limitations are that the Gaze4ASD screen mapping is inferred, and that we have no participant ages to adjust for. Also, the label works like one fixed shift: the generated ASD-minus-TD spread is positive on all 45 of our test images, even where the real difference is negative.
>
> For next steps, I'd first ask the Gaze4ASD authors about the display, the participant count and the ages. Then I'd model when a scanpath stops, to fix the 10-versus-6 fixation problem, and model off-image gaze. Then rerun E1 with the same baseline.
>
> Mam, I'd value your advice on which of these to do first.

**If Mam asks: could this be used for screening?**
> No. It uses group labels only. We never built or tested a classifier, and there is no clinical validation.

---

## Quick answers to keep ready

| If she asks about | Say |
|---|---|
| Our contribution | A controlled test of a 1,024-parameter ASD/TD label inside a frozen TPP-Gaze, with a seed-only control, an external dataset, and a per-group realism check |
| In-domain result | Plus 0.0073 per fixation; the interval includes zero, so inconclusive |
| External result | Phase 1 better by 0.0172, in all three preprocessing versions; cause not tested |
| 69 / 64 / 32 percent | The generated ASD−TD difference as a share of the real one: right direction, smaller size, approximate |
| Realism | No spatial measure moves closer to its own real group |
| Temperature | Changes the generated samples and the size of the differences, never the likelihood |
| The beach example | An outside photo, one sample per label at temperature 2; it shows what the model outputs, not evidence that the label works |
| Sharing these files | Keep this folder local; the beach photo's licence is unknown, and the Saliency4ASD data can't be shared with third parties |
