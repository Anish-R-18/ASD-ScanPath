// Builds the 8-slide supervisor deck (research progress report, not a concept introduction).
// Numbers come from paper/final/ASD_ScanPath_Final.md; training settings from training/config*.yaml
// and training/runs/*/summary_full.json; example values from demo/demo_results/{manifest,*/samples}.json.
// Run with NODE_PATH pointing at a node_modules folder that contains pptxgenjs 3.12.0.
const pptxgen = require("pptxgenjs");
// Theme helper from the local "pptx" skill used to build the deck; set PPTX_SKILL_DIR to its folder.
const SKILL = process.env.PPTX_SKILL_DIR || require("path").join(require("os").homedir(), ".claude/skills/synced/01c91bb0-79a0-4c38-a902-19cc77cedb90_0b33054a-965f-4dda-ba15-743d9fa092e0/pptx");
const { applyTheme } = require(SKILL + "/scripts/apply_theme.js");

const ROOT = require("path").resolve(__dirname, "..", "..", "..");  // repository root
const DIR = ROOT + "/presentation/supervisor_8_slide";
// Two versions from one script (they differ only on slide 4):
//   default            shareable deck, blank-canvas scanpaths only (no dataset photographs)
//   DECK=restricted    local-only deck with the actual Saliency4ASD test photographs (sharing terms unresolved)
const RESTRICTED = process.env.DECK === "restricted";
const RDIR = DIR + "/local_only_restricted";
// OUT_PATH overrides the output file (used when the normal file is open in PowerPoint).
const OUT = process.env.OUT_PATH || (RESTRICTED ? RDIR + "/ASD_ScanPath_Supervisor_Presentation_RESTRICTED.pptx" : DIR + "/ASD_ScanPath_Supervisor_Presentation.pptx");
const A = DIR + "/assets/";
const RA = RDIR + "/assets/";

const THEME = {
  name: "ASD-ScanPath supervisor",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B242B", lt1: "FFFFFF", dk2: "3F4E59", lt2: "EEF2F4",
    accent1: "0E6E6E", accent2: "5B4B8A", accent3: "D9701E",
    accent4: "2F6DB5", accent5: "B3261E", accent6: "2E7D4F",
    hlink: "2F6DB5", folHlink: "5B4B8A",
  },
};
// Colour roles (used consistently on every slide):
//   teal = original TPP-Gaze, purple = our addition, orange = ASD, blue = TD.
const K = {
  ink: "1B242B", slate: "3F4E59", muted: "66737D", rule: "D5DCE1", card: "F3F5F7",
  teal: "0E6E6E", tealTint: "E3F1F1", tealLight: "8CC7C7",
  ours: "5B4B8A", oursTint: "ECE8F5",
  asd: "D9701E", asdText: "A9500F", asdTint: "FBEBDD",
  td: "2F6DB5", tdTint: "E5EEF8",
  neg: "B3261E", negTint: "F8E3E1",
  amber: "8A5D00", amberTint: "F6EED9",
  pos: "2E7D4F", posTint: "E2F0E7",
  partial: "3F4E59", partialTint: "E6EAED",
};
const HEAD = "Cambria", BODY = "Calibri";

// Speaker notes = the spoken text of each slide in SPEAKING_SCRIPT.md (quoted lines), so the two never drift apart.
const SCRIPT = require("fs").readFileSync(DIR + "/SPEAKING_SCRIPT.md", "utf8");
function notesFor(n) {
  const sec = SCRIPT.split(/\n## /).find((p) => p.startsWith(`Slide ${n}:`));
  if (!sec) throw new Error(`no script section for slide ${n}`);
  const time = (sec.match(/\*\*Suggested time:\*\* ([^\n]+)/) || [])[1];
  const spoken = sec.split("**What")[0].split("\n").filter((l) => l.startsWith(">")).map((l) => l.slice(1).trim());
  const R_TAG = "*(Restricted version)* ", S_TAG = "*(Shareable version)* ";
  const paras = spoken.join("\n").split(/\n\s*\n/).map((p) => p.replace(/\n/g, " ").trim()).filter(Boolean)
    .filter((p) => !(p.startsWith(R_TAG) && !RESTRICTED) && !(p.startsWith(S_TAG) && RESTRICTED))
    .map((p) => p.replace(R_TAG, "").replace(S_TAG, ""));
  return `Suggested time: ${time}.\n\n` + paras.join("\n\n") + "\n\n(Full notes and likely questions: SPEAKING_SCRIPT.md.)";
}

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };
pres.title = RESTRICTED ? "ASD-ScanPath: supervisor presentation (RESTRICTED, local use only)" : "ASD-ScanPath: supervisor presentation";
pres.subject = "ASD/TD-conditioned TPP-Gaze: research question, implementation, evaluation and findings";
pres.author = "[Presenter name]";

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  margin: [0.4, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.6, y: 0.28, w: 9, h: 0.3, fontFace: BODY, fontSize: 12, bold: true, color: K.teal, margin: 0, valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.56, w: 12.13, h: 0.62, fontFace: HEAD, fontSize: 28, bold: true, color: K.ink, margin: 0, valign: "middle", align: "left" }, text: "" } },
  ],
  slideNumber: { x: 12.23, y: 7.06, w: 0.5, h: 0.26, fontFace: BODY, fontSize: 10, color: K.muted, align: "right" },
});

// ---------- helpers (fresh option objects on every call: pptxgenjs mutates them) ----------
function T(s, text, o) {
  if (Array.isArray(text)) text = text.map((r) => (r.options && r.options.bullet === true) ? { text: r.text, options: Object.assign({}, r.options, { bullet: { indent: 13 } }) } : r);
  s.addText(text, Object.assign({ isTextBox: true, fontFace: BODY, fontSize: 14, color: K.ink, margin: 0, valign: "top" }, o));
}
function box(s, x, y, w, h, fill, line, name, radius) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, fill: { color: fill }, line: line ? { color: line, width: 1.25 } : { type: "none" },
    rectRadius: radius === undefined ? 0.08 : radius, objectName: name,
  });
}
function arrow(s, x1, y1, x2, y2, color, width, opt) {
  opt = opt || {};
  const line = { color: color || K.slate, width: width || 1.75 };
  if (!opt.noHead) line.endArrowType = "triangle";
  if (opt.beginHead) line.beginArrowType = "triangle";
  if (opt.dash) line.dashType = "dash";
  s.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1, line, objectName: opt.name,
  });
}
function circle(s, cx, cy, d, fill, line, name) {
  s.addShape(pres.shapes.OVAL, {
    x: cx - d / 2, y: cy - d / 2, w: d, h: d,
    fill: fill ? { color: fill } : { type: "none" },
    line: line ? { color: line, width: 1.5 } : { type: "none" }, objectName: name,
  });
}
function fix(s, cx, cy, d, n, fill, fontSize) {
  circle(s, cx, cy, d, fill, null, "marker " + n);
  T(s, String(n), { x: cx - d / 2, y: cy - d / 2, w: d, h: d, align: "center", valign: "middle", bold: true, fontSize: fontSize || 12, color: "FFFFFF" });
}
const CHIP = { neg: [K.negTint, K.neg], inc: [K.amberTint, K.amber], partial: [K.partialTint, K.partial], pos: [K.posTint, K.pos],
  teal: [K.tealTint, K.teal], ours: [K.oursTint, K.ours], asd: [K.asdTint, K.asdText], td: [K.tdTint, K.td] };
const chipW = (label, fs) => 0.26 + label.length * fs * 0.0084;
function chip(s, x, y, label, kind, size) {
  const fs = size || 11, w = chipW(label, fs);
  box(s, x, y, w, 0.28, CHIP[kind][0], null, "label " + label, 0.14);
  T(s, label, { x, y, w, h: 0.28, align: "center", valign: "middle", bold: true, fontSize: fs, color: CHIP[kind][1], charSpacing: 0.3 });
  return w;
}
function footer(s, text) {
  T(s, text, { x: 0.6, y: 7.06, w: 11.4, h: 0.26, fontSize: 9, color: K.muted, valign: "middle" });
}
function slide(kicker, title) {
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText(kicker, { placeholder: "kicker" });
  s.addText(title, { placeholder: "title" });
  if (RESTRICTED) {
    const label = "RESTRICTED · LOCAL USE ONLY";
    chip(s, 12.73 - chipW(label, 10), 0.28, label, "neg", 10);
  }
  return s;
}
const bullets = (items) => items.map((t, i, a) => (typeof t === "string"
  ? { text: t, options: { bullet: true, breakLine: i < a.length - 1 } }
  : { text: t.text, options: Object.assign({ bullet: true, breakLine: i < a.length - 1 }, t.options) }));

// ======================================================================
// 1. Research question and approach
{
  const s = slide("1 · RESEARCH QUESTION AND APPROACH", "What does a minimal ASD/TD label change in TPP-Gaze?");
  box(s, 0.6, 1.42, 6.0, 1.55, K.card, null, "gap");
  T(s, [
    { text: "Gap", options: { bold: true, color: K.teal, fontFace: HEAD, fontSize: 15, breakLine: true } },
    { text: "TPP-Gaze models fixation location and duration jointly but has no group input. Separate ASD and TD models would differ in every weight, so the group effect could not be isolated. We test the smallest addition: one group label in a shared model.", options: {} },
  ], { x: 0.8, y: 1.5, w: 5.6, h: 1.4, fontSize: 13, valign: "middle" });
  T(s, "Research questions", { x: 0.6, y: 3.17, w: 6.0, h: 0.32, fontSize: 15, bold: true, fontFace: HEAD });
  const rq = [
    ["In-domain likelihood", "Does the label lower NLL on held-out Saliency4ASD test images?"],
    ["External likelihood", "Does any gain hold on Gaze4ASD: new children, images, tracker and display?"],
    ["Generated behaviour", "Do ASD- and TD-labelled samples differ like the real groups, and does each label match its own group better?"],
  ];
  rq.forEach((r, i) => {
    const y = 3.6 + i * 1.08;
    box(s, 0.6, y, 6.0, 0.96, K.card, null, "RQ" + (i + 1));
    fix(s, 0.95, y + 0.48, 0.42, "Q" + (i + 1), K.slate, 11);
    T(s, [{ text: r[0], options: { bold: true, breakLine: true } }, { text: r[1], options: { color: K.slate } }], { x: 1.3, y, w: 5.15, h: 0.96, fontSize: 13, valign: "middle" });
  });
  // Approach
  T(s, "Our approach", { x: 6.95, y: 1.42, w: 5.78, h: 0.32, fontSize: 15, bold: true, fontFace: HEAD });
  const steps = [
    [K.tealTint, K.teal, "Phase 1: unconditioned baseline", "Fine-tune the released TPP-Gaze on pooled ASD + TD training data, with no group input."],
    [K.oursTint, K.ours, "Phase 2: add the group label", "Freeze all Phase 1 weights; add a zero-initialized 2 × 512 embedding (1,024 parameters) and train only that."],
    [K.card, K.slate, "Compare Phase 2 with Phase 1", "Same scanpaths, same seeds: likelihood in-domain and on Gaze4ASD; generated behaviour vs real ASD–TD differences."],
  ];
  steps.forEach((st, i) => {
    const y = 1.82 + i * 1.18;
    box(s, 6.95, y, 5.78, 0.98, st[0], st[1], "approach " + (i + 1));
    fix(s, 7.28, y + 0.49, 0.38, i + 1, st[1], 12);
    T(s, [{ text: st[2], options: { bold: true, breakLine: true } }, { text: st[3], options: { color: K.slate } }], { x: 7.6, y, w: 5.0, h: 0.98, fontSize: 12.5, valign: "middle" });
    if (i < steps.length - 1) arrow(s, 9.84, y + 0.99, 9.84, y + 1.17, K.slate, 1.75);
  });
  box(s, 6.95, 5.45, 5.78, 1.4, K.ink, null, "short answer");
  T(s, [
    { text: "Short answer. ", options: { bold: true, color: K.tealLight } },
    { text: "In-domain gain +0.0073 nats per fixation, inconclusive. External fit worse (−0.0172). Generated spatial contrasts partly right; timing contrasts wrong; neither group more realistic.", options: { color: "FFFFFF" } },
  ], { x: 7.15, y: 5.5, w: 5.4, h: 1.3, fontSize: 13, valign: "middle" });
  footer(s, "[3] D'Amelio et al., TPP-Gaze, WACV 2025 · closest ASD scanpath work: Zhong et al., Pattern Recognition 2026 (abstract only; not compared) · manuscript §1, §2.4");
  s.addNotes(notesFor(1));
}

// 2. What we added
{
  const s = slide("2 · OUR CONTRIBUTION", "What we kept from TPP-Gaze, and what is new");
  box(s, 0.6, 1.42, 5.9, 3.85, K.tealTint, K.teal, "kept");
  T(s, "Kept unchanged from TPP-Gaze [3]", { x: 0.82, y: 1.55, w: 5.5, h: 0.38, fontSize: 16, bold: true, fontFace: HEAD, color: K.teal });
  T(s, bullets([
    "Released code and pretrained transformer weights",
    "Image encoder: DenseNet201 + CoordConv (backbone frozen)",
    "History encoder: transformer + LSTM, including the released batch-layout behaviour (kept and disclosed)",
    "Heads: 16-component Gaussian mixture (location), 4-component log-normal mixture (duration)",
    "Joint NLL objective without a survival term; autoregressive sampler",
  ]), { x: 0.82, y: 2.02, w: 5.5, h: 3.15, fontSize: 13.5, paraSpaceAfter: 6 });
  box(s, 6.8, 1.42, 5.93, 3.85, K.oursTint, K.ours, "new");
  T(s, "New in our implementation", { x: 7.02, y: 1.55, w: 5.5, h: 0.38, fontSize: 16, bold: true, fontFace: HEAD, color: K.ours });
  T(s, bullets([
    "Conditioning wrapper: a 2 × 512 group embedding, zero-initialized, added to the fused context before both heads",
    "Phase 2 trains only the embedding: 1,024 of 23,896,736 parameters",
    "Conditioned scoring and sampling: the label is added at every step",
    "Our settings: duration normalization from our training split; τ = 1 chosen on validation (released config: τ = 2); minimum 3 fixations (TPP-Gaze paper: 4)",
    "Evaluation: clustered likelihood comparison and a shared-seed behaviour comparison (E1)",
  ]), { x: 7.02, y: 2.02, w: 5.5, h: 3.15, fontSize: 13.5, paraSpaceAfter: 6 });
  box(s, 0.6, 5.45, 7.9, 1.4, K.card, null, "why minimal");
  T(s, [
    { text: "Why this design. ", options: { bold: true, color: K.ours } },
    { text: "With a zero-initialized embedding and a frozen base, Phase 1 is exactly Phase 2 with a zero embedding; all 1,290 base tensors are identical in both checkpoints. Any Phase 2 − Phase 1 difference is caused by the 1,024 label parameters.", options: {} },
  ], { x: 0.8, y: 5.5, w: 7.5, h: 1.3, fontSize: 13, valign: "middle" });
  box(s, 8.7, 5.45, 4.03, 1.4, K.negTint, null, "not claimed");
  T(s, [
    { text: "Not claimed", options: { bold: true, color: K.neg, breakLine: true } },
    ...bullets(["A new architecture", "An improvement over TPP-Gaze", "Any clinical or diagnostic use"]),
  ], { x: 8.9, y: 5.5, w: 3.7, h: 1.3, fontSize: 13, valign: "middle" });
  footer(s, "[3] D'Amelio et al., TPP-Gaze, WACV 2025 · manuscript §3.2, Table 1, §3.5, §4.4 · code: training/conditioned.py, training/generate_cond.py · run summaries: training/runs/*/summary_full.json");
  s.addNotes(notesFor(2));
}

// 3. Implementation
{
  const s = slide("3 · MODEL IMPLEMENTATION", "Implementation: where the group embedding enters");
  const base = (x, y, w, h, title, line2, name) => {
    box(s, x, y, w, h, K.tealTint, K.teal, name);
    T(s, [{ text: title, options: { bold: true, breakLine: true } }, { text: line2, options: { fontSize: 12.5, color: K.slate } }],
      { x: x + 0.08, y, w: w - 0.16, h, fontSize: 15, align: "center", valign: "middle" });
  };
  const neutral = (x, y, w, h, runs, name) => {
    box(s, x, y, w, h, "FFFFFF", K.slate, name);
    T(s, runs, { x: x + 0.08, y, w: w - 0.16, h, fontSize: 15, align: "center", valign: "middle" });
  };
  const r0 = 1.55, r1 = 2.6, r2 = 3.65, rh = 0.85;
  neutral(0.6, r0, 2.1, rh, [{ text: "ASD / TD label ℓ", options: { bold: true, breakLine: true } }, { text: "ASD = 0", options: { fontSize: 12.5, color: K.asdText, bold: true } }, { text: "   ", options: { fontSize: 12.5 } }, { text: "TD = 1", options: { fontSize: 12.5, color: K.td, bold: true } }], "input label");
  neutral(0.6, r1, 2.1, rh, [{ text: "Stimulus image", options: { bold: true, breakLine: true } }, { text: "resized to 512 × 512", options: { fontSize: 12.5, color: K.slate } }], "input image");
  neutral(0.6, r2, 2.1, rh, [{ text: "Gaze history", options: { bold: true, breakLine: true } }, { text: "past (x, y, duration)", options: { fontSize: 12.5, color: K.slate } }], "input history");
  box(s, 3.0, r0, 2.75, rh, K.oursTint, K.ours, "group embedding");
  T(s, [{ text: "Group embedding e", options: { bold: true } }, { text: "ℓ", options: { bold: true, subscript: true, breakLine: true } }, { text: "nn.Embedding(2, 512), zero-init", options: { fontSize: 12.5, color: K.slate } }],
    { x: 3.08, y: r0, w: 2.59, h: rh, fontSize: 15, align: "center", valign: "middle" });
  base(3.0, r1, 2.75, rh, "Image encoder", "DenseNet201 + CoordConv → z (256-d)", "image encoder");
  base(3.0, r2, 2.75, rh, "History encoder", "Transformer + LSTM → h (256-d)", "history encoder");
  base(6.05, r1, 1.75, 1.9, "Fused context", "c = concat(h, z)\n512-d", "fused context");
  circle(s, 8.3, 3.55, 0.52, K.ours, null, "add node");
  T(s, "+", { x: 8.04, y: 3.29, w: 0.52, h: 0.52, align: "center", valign: "middle", bold: true, fontSize: 24, color: "FFFFFF" });
  base(8.9, 2.3, 2.3, 1.1, "Spatial head", "16 Gaussian components\n→ next (x, y)", "spatial head");
  base(8.9, 3.75, 2.3, 1.1, "Temporal head", "4 log-normal components\n→ next duration", "temporal head");
  neutral(11.5, 2.95, 1.23, 1.2, [{ text: "Next fixation", options: { bold: true, breakLine: true } }, { text: "(x, y, d)", options: { fontSize: 12.5, color: K.slate } }], "output");
  [r0, r1, r2].forEach((r) => arrow(s, 2.7, r + rh / 2, 3.0, r + rh / 2));
  arrow(s, 5.75, r1 + rh / 2, 6.05, r1 + rh / 2);
  arrow(s, 5.75, r2 + rh / 2, 6.05, r2 + rh / 2);
  arrow(s, 7.8, 3.55, 8.04, 3.55);
  arrow(s, 5.75, r0 + rh / 2, 8.3, r0 + rh / 2, K.ours, 2.25, { noHead: true, name: "embedding path" });
  arrow(s, 8.3, r0 + rh / 2, 8.3, 3.29, K.ours, 2.25, { name: "embedding path" });
  T(s, [{ text: "c̃ = c + e" }, { text: "ℓ", options: { subscript: true } }, { text: " (every step)" }], { x: 5.85, y: 1.6, w: 2.4, h: 0.36, fontSize: 12, italic: true, color: K.ours, align: "center", valign: "bottom", bold: true });
  arrow(s, 8.56, 3.42, 8.9, 2.85);
  arrow(s, 8.56, 3.68, 8.9, 4.3);
  arrow(s, 11.2, 2.85, 11.5, 3.3);
  arrow(s, 11.2, 4.3, 11.5, 3.8);
  arrow(s, 12.12, 4.15, 12.12, 4.98, "7D8A93", 1.5, { noHead: true, dash: true, name: "loop" });
  arrow(s, 4.375, 4.98, 12.12, 4.98, "7D8A93", 1.5, { noHead: true, dash: true, name: "loop" });
  arrow(s, 4.375, 4.5, 4.375, 4.98, "7D8A93", 1.5, { noHead: true, beginHead: true, dash: true, name: "loop" });
  T(s, "Sampling only: generated fixations join the history until 3 s are filled; scoring uses the real history", { x: 4.5, y: 5.02, w: 7.6, h: 0.28, fontSize: 11.5, italic: true, color: K.muted, align: "center" });
  box(s, 8.9, 1.55, 0.3, 0.2, K.tealTint, K.teal, "legend original", 0.03);
  T(s, "Released TPP-Gaze (frozen in Phase 2)", { x: 9.28, y: 1.5, w: 3.45, h: 0.3, fontSize: 12, color: K.slate, valign: "middle" });
  box(s, 8.9, 1.88, 0.3, 0.2, K.oursTint, K.ours, "legend ours", 0.03);
  T(s, "Our addition (only part trained in Phase 2)", { x: 9.28, y: 1.83, w: 3.45, h: 0.3, fontSize: 12, color: K.slate, valign: "middle" });
  const C = [
    [K.tealTint, K.teal, "Unchanged modules", "Both encoders and both heads are released TPP-Gaze: fine-tuned in Phase 1, frozen in Phase 2."],
    [K.oursTint, K.ours, "Our addition", "One 512-d vector per group, broadcast over the sequence and added to the fused context."],
    [K.card, K.slate, "Same in all code paths", "Training, likelihood scoring and the sampler all add the embedding at every step."],
    [K.card, K.slate, "Parameter count", "Base 23,895,712 + embedding 1,024; only the 1,024 train in Phase 2."],
  ];
  C.forEach((c, i) => {
    const x = 0.6 + i * 3.077;
    box(s, x, 5.45, 2.9, 1.4, c[0], null, "callout " + c[2]);
    T(s, [{ text: c[2], options: { bold: true, color: c[1], fontSize: 14, breakLine: true } }, { text: c[3], options: { color: K.ink } }],
      { x: x + 0.15, y: 5.5, w: 2.62, h: 1.3, fontSize: 12, valign: "middle" });
  });
  footer(s, "Code: training/conditioned.py, train_tpp_cond.py, generate_cond.py · manuscript §3.3–3.6, Eq. 2, Figure 1 · counts: runs/phase2_embedding_only/summary_full.json");
  s.addNotes(notesFor(3));
}

// 4. Datasets and experimental stimuli
{
  const s = slide("4 · DATASETS AND EXPERIMENTAL STIMULI", "Datasets, and the actual test images we used");
  const band = (x, name, roleKind, role, l1, l2) => {
    box(s, x, 1.32, 5.98, 0.88, K.card, null, "dataset " + name);
    T(s, name, { x: x + 0.15, y: 1.35, w: 2.4, h: 0.3, fontSize: 15, bold: true, fontFace: HEAD, valign: "middle" });
    chip(s, x + 5.98 - chipW(role, 10) - 0.12, 1.37, role, roleKind, 10);
    T(s, [{ text: l1, options: { breakLine: true } }, { text: l2 }], { x: x + 0.15, y: 1.67, w: 5.75, h: 0.5, fontSize: 10.5, color: K.slate, valign: "middle" });
  };
  band(0.6, "Saliency4ASD [4, 5]", "teal", "TRAINING + IN-DOMAIN TEST",
    "300 images · split by image (seed 0): 210 / 45 / 45 → 4,229 / 897 / 903 scanpaths",
    "2,705 ASD + 3,324 TD scanpaths · age 5–12 · no participant IDs");
  band(6.75, "Gaze4ASD [6]", "ours", "EXTERNAL TEST ONLY",
    "30 stimuli, mostly faces · 33 ASD + 133 TD children (README: 132)",
    "771 + 3,605 trials · age 3–7 · other tracker, display and site");
  const pw = 2.15, gap = 0.07, colX = [0, 1, 2, 3].map((i) => 0.6 + i * (pw + gap));
  const rlx = 9.95, cw = 1.15, cx = [10.38, 10.38 + cw + 0.05];
  T(s, RESTRICTED ? "Real vs generated fixations" : "Real vs generated (statistics)", { x: rlx, y: 2.24, w: 12.73 - rlx, h: 0.18, fontSize: 9.5, bold: true, color: K.slate, align: "center", valign: "middle" });
  [["Original photograph", K.ink], ["Phase 1", K.ink], ["Phase 2, ASD label", K.asdText], ["Phase 2, TD label", K.td]].forEach((c, i) => {
    T(s, c[0], { x: colX[i], y: 2.42, w: pw, h: 0.2, fontSize: 11, bold: true, color: c[1], align: "center", valign: "middle" });
  });
  if (RESTRICTED) {
    T(s, "ASD", { x: cx[0], y: 2.42, w: cw, h: 0.2, fontSize: 11, bold: true, color: K.asdText, align: "center", valign: "middle" });
    T(s, "TD", { x: cx[1], y: 2.42, w: cw, h: 0.2, fontSize: 11, bold: true, color: K.td, align: "center", valign: "middle" });
  }
  const row = (id, y, ar, labelRuns, stats) => {
    T(s, labelRuns, { x: 0.6, y, w: 12.13, h: 0.24, fontSize: 10.5, valign: "middle" });
    const iy = y + 0.27, ph = pw / ar;
    if (RESTRICTED) {
      s.addImage({ path: RA + `photo_${id}.png`, x: colX[0], y: iy, w: pw, h: ph, altText: `Saliency4ASD held-out test image ${id}, original photograph (restricted)`, objectName: `photo ${id}` });
      ["phase1", "asd", "td"].forEach((m, k) => s.addImage({ path: RA + `overlay_${id}_${m}.png`, x: colX[k + 1], y: iy, w: pw, h: ph,
        altText: `Generated scanpath on test image ${id}: ${["Phase 1", "Phase 2 ASD label", "Phase 2 TD label"][k]} (seed 0, tau 1)`, objectName: `overlay ${id} ${m}` }));
      const ch = cw / ar;
      [["real", "Real"], ["gen", "Gen."]].forEach((r, ri) => {
        T(s, r[1], { x: rlx, y: iy + ri * (ch + 0.05), w: 0.4, h: ch, fontSize: 9.5, bold: true, color: K.slate, valign: "middle" });
        ["asd", "td"].forEach((g, gi) => s.addImage({ path: RA + `rvg_${id}_${r[0]}_${g}.png`, x: cx[gi], y: iy + ri * (ch + 0.05), w: cw, h: ch,
          altText: `${r[1] === "Real" ? "Real" : "Generated (E1 replicate 1)"} ${g.toUpperCase()} fixations on test image ${id}`, objectName: `rvg ${id} ${r[0]} ${g}` }));
      });
    } else {
      box(s, colX[0], iy, pw, ph, "E6EAED", null, `photo withheld ${id}`, 0.04);
      T(s, [{ text: "Photograph withheld", options: { bold: true, breakLine: true } }, { text: "Saliency4ASD sharing terms unresolved; shown only in the local restricted version", options: { fontSize: 10 } }],
        { x: colX[0] + 0.12, y: iy, w: pw - 0.24, h: ph, fontSize: 11.5, color: K.slate, align: "center", valign: "middle" });
      ["phase1", "asd", "td"].forEach((m, k) => s.addImage({ path: A + `blank_${id}_${m}.png`, x: colX[k + 1], y: iy, w: pw, h: ph,
        altText: `Blank-canvas generated scanpath for test image ${id}: ${["Phase 1", "Phase 2 ASD label", "Phase 2 TD label"][k]} (seed 0, tau 1)`, objectName: `blank ${id} ${m}` }));
      box(s, rlx, iy, 12.73 - rlx, ph, K.card, null, `stats ${id}`, 0.04);
      T(s, stats, { x: rlx + 0.12, y: iy, w: 12.73 - rlx - 0.24, h: ph, fontSize: 10.5, valign: "middle" });
    }
  };
  const statRuns = (ra, rt, ga, gt) => [
    { text: "Dispersion (÷ diagonal)", options: { bold: true, breakLine: true } },
    { text: "Real: ", options: { bold: true } }, { text: "ASD " + ra, options: { color: K.asdText } }, { text: " · " }, { text: "TD " + rt, options: { color: K.td, breakLine: true } },
    { text: "Generated: ", options: { bold: true } }, { text: "ASD " + ga, options: { color: K.asdText } }, { text: " · " }, { text: "TD " + gt, options: { color: K.td, breakLine: true } },
    { text: "E1 replicate 1, 20 scanpaths per group", options: { fontSize: 9.5, color: K.muted } },
  ];
  row(102, 2.66, 1024 / 768, [
    { text: "Image 102 · best-match example", options: { bold: true, color: K.pos } },
    { text: "  ASD−TD dispersion: real +0.0133, generated +0.014 (mean of 10 seeds): same sign, similar size", options: { color: K.slate } },
  ], statRuns("0.132", "0.118", "0.150", "0.138"));
  row(116, 4.72, 1024 / 685, [
    { text: "Image 116 · worst-match example", options: { bold: true, color: K.neg } },
    { text: "  ASD−TD dispersion: real −0.0473 (ASD less spread), generated +0.0151 (ASD more spread): wrong direction", options: { color: K.slate } },
  ], statRuns("0.117", "0.165", "0.162", "0.109"));
  T(s, "Best and worst agreement among the 45 test images, by a rule fixed before any demo output: one strength and one limitation, not overall performance. Generated panels: one sample each, seed 0, τ = 1. " +
    (RESTRICTED ? "Real vs generated: all real scanpaths for the image vs E1 replicate 1 (20 per group)." : "Photographs and real-gaze plots: local restricted version only."),
    { x: 0.6, y: 6.68, w: 12.13, h: 0.34, fontSize: 9.5, color: K.muted, italic: true, valign: "middle" });
  footer(s, RESTRICTED
    ? "Photos: Saliency4ASD test images 102 and 116 (from MIT1003), RESTRICTED, local use only · figures: demo/local_only/s4a_best_102, s4a_worst_116 · counts: manuscript §4.1–4.2, Table 2"
    : "[4] Duan et al., ACM MMSys 2019 · [5] Gutiérrez et al., SPIC 2021 · [6] Yang et al., ICME 2025 · counts: manuscript §4.1–4.2, Table 2 · examples: demo/demo_results");
  s.addNotes(notesFor(4));
}

// 5. Training procedure
{
  const s = slide("5 · TRAINING PROCEDURE", "Training: Phase 1 baseline, then the label only");
  const steps = [
    ["Preprocess", "Split at index resets; drop sequences with a fixation > 3 s; drop the 1st fixation and those < 60 ms; cut at 3 s; keep ≥ 3", K.slate],
    ["Split by image", "Seed 0: 210 / 45 / 45 images, so no test image is seen during training or selection", K.slate],
    ["Phase 1", "Fine-tune released TPP-Gaze on pooled ASD + TD; DenseNet201 frozen, features cached", K.teal],
    ["Set up Phase 2", "Load best Phase 1; add the zero 2 × 512 embedding; freeze every base weight", K.ours],
    ["Phase 2", "Train only the embedding; base in evaluation mode (no dropout)", K.ours],
    ["Select", "Best validation NLL checkpoint for each phase; τ also chosen on validation", K.slate],
  ];
  const w = 1.813, g = 0.25;
  steps.forEach((st, i) => {
    const x = 0.6 + i * (w + g);
    box(s, x, 1.38, w, 1.95, st[2] === K.ours ? K.oursTint : st[2] === K.teal ? K.tealTint : K.card, null, "step " + (i + 1));
    fix(s, x + 0.3, 1.68, 0.38, i + 1, st[2], 12);
    T(s, st[0], { x: x + 0.55, y: 1.49, w: w - 0.6, h: 0.38, fontSize: 13, bold: true, valign: "middle" });
    T(s, st[1], { x: x + 0.14, y: 1.98, w: w - 0.26, h: 1.32, fontSize: 11.5, color: K.slate });
    if (i < steps.length - 1) arrow(s, x + w + 0.03, 2.35, x + w + g - 0.03, 2.35, K.slate, 1.75);
  });
  const hdr = (t, c) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: c } } });
  const lab = (t) => ({ text: t, options: { bold: true, fill: { color: K.card } } });
  const rows = [
    [hdr("", K.slate), hdr("Phase 1: unconditioned baseline", K.teal), hdr("Phase 2: group conditioning", K.ours)],
    [lab("Why"), "Adapt TPP-Gaze to these children's data before adding a label", "Learn only the group difference; nothing else can change"],
    [lab("Starts from"), "Released TPP-Gaze weights", "Best Phase 1 checkpoint + zero embedding"],
    [lab("Trained parameters"), "3,881,784 (everything outside the frozen DenseNet backbone)", "1,024 (embedding only)"],
    [lab("Optimizer"), "AdamW, lr 10⁻³, weight decay 0.1, batch 128, gradient clip 5", "AdamW, lr 10⁻⁴; otherwise the same"],
    [lab("Stopping"), "20 epochs without validation improvement", "Same"],
    [lab("Best validation NLL"), "−0.5845 at epoch 6; stopped at 26, when validation NLL had worsened to −0.384", "−0.5964 at epoch 8; stopped at 28"],
  ];
  s.addTable(rows, { x: 0.6, y: 3.5, w: 12.13, colW: [2.3, 5.08, 4.75], fontFace: BODY, fontSize: 12, color: K.ink, valign: "middle",
    border: { type: "solid", pt: 0.75, color: K.rule }, rowH: 0.39, margin: [0.03, 0.1, 0.03, 0.1] });
  T(s, "NLL in nats per fixation. Duration normalization (log-duration mean −1.4366, SD 0.6936) uses the training split only. Test images and Gaze4ASD were never used for training or selection. An earlier Phase 2 configuration (lr 10⁻³) exists in the repository without local results; it is not reported.",
    { x: 0.6, y: 6.34, w: 12.13, h: 0.6, fontSize: 11, color: K.muted });
  footer(s, "Sources: training/config.yaml, training/config_phase2_embonly.yaml, training/runs/baseline/split.json, training/runs/*/summary_full.json · manuscript §4.1, §4.3–4.4, §6.1");
  s.addNotes(notesFor(5));
}

// 6. Evaluation design
{
  const s = slide("6 · EVALUATION DESIGN", "Evaluation: likelihood and generated behaviour, kept separate");
  // A. Likelihood
  T(s, "A. Likelihood: Phase 2 vs Phase 1 on identical scanpaths", { x: 0.6, y: 1.38, w: 6.1, h: 0.32, fontSize: 14.5, bold: true, fontFace: HEAD });
  const pc = (x, name, kind, label, lines) => {
    box(s, x, 1.8, 2.98, 1.95, K.card, null, name);
    chip(s, x + 0.15, 1.92, label, kind, 10);
    T(s, bullets(lines), { x: x + 0.15, y: 2.3, w: 2.72, h: 1.4, fontSize: 12, paraSpaceAfter: 3 });
  };
  pc(0.6, "in-domain", "teal", "IN-DOMAIN", ["45 held-out Saliency4ASD images, 903 scanpaths", "No participant IDs: image-cluster bootstrap and sign-flip test"]);
  pc(3.72, "external", "ours", "EXTERNAL", ["Gaze4ASD: 30 images, 4,376 trials, never used for any choice", "Participant, image and crossed bootstraps"]);
  box(s, 0.6, 3.88, 6.1, 1.62, K.amberTint, null, "variants");
  T(s, [
    { text: "Three external preprocessing variants, and why", options: { bold: true, color: K.amber, breakLine: true } },
    ...bullets([
      "Fit-to-height + clip off-image gaze: primary, fixed before modelling",
      "Stretch + clip: the screen mapping is inferred (fit-to-height r = 0.9999 vs 0.91 for stretch)",
      "Fit-to-height + remove off-image gaze: off-image gaze differs by group (7.35% ASD vs 2.72% TD of fixations)",
    ]),
  ], { x: 0.78, y: 3.93, w: 5.8, h: 1.52, fontSize: 11.5, valign: "middle", paraSpaceAfter: 2 });
  // B. Behaviour (E1)
  const bx = 6.95, bw = 5.78;
  T(s, "B. Generated behaviour: experiment E1 (τ = 1)", { x: bx, y: 1.38, w: bw, h: 0.32, fontSize: 14.5, bold: true, fontFace: HEAD });
  const gx = bx + 1.3, gy = 2.05, cwd = 1.6, chh = 0.42;
  T(s, "Phase 2", { x: gx, y: 1.78, w: cwd, h: 0.25, fontSize: 11.5, bold: true, align: "center" });
  T(s, "Phase 1", { x: gx + cwd + 0.1, y: 1.78, w: cwd, h: 0.25, fontSize: 11.5, bold: true, align: "center" });
  [["Seed stream A", "ASD label", K.asdTint, K.asd], ["Seed stream B", "TD label", K.tdTint, K.td]].forEach((r, i) => {
    const y = gy + i * (chh + 0.08);
    T(s, r[0], { x: bx, y, w: 1.25, h: chh, fontSize: 11, color: K.slate, valign: "middle" });
    box(s, gx, y, cwd, chh, r[2], r[3], r[1], 0.05); T(s, r[1], { x: gx, y, w: cwd, h: chh, fontSize: 11.5, align: "center", valign: "middle" });
    box(s, gx + cwd + 0.1, y, cwd, chh, "FFFFFF", K.slate, "P1 " + r[0], 0.05); T(s, "no label", { x: gx + cwd + 0.1, y, w: cwd, h: chh, fontSize: 11.5, align: "center", valign: "middle" });
  });
  T(s, "20 scanpaths per image and seed · 10 seeds · 30 Gaze4ASD + 45 Saliency4ASD test images", { x: bx, y: 3.05, w: bw, h: 0.26, fontSize: 11, color: K.slate });
  const card = (y, h, runs, name) => { box(s, bx, y, bw, h, K.card, null, name); T(s, runs, { x: bx + 0.15, y, w: bw - 0.3, h, fontSize: 11.5, valign: "middle" }); };
  card(3.4, 0.62, [{ text: "Conditioning effect ", options: { bold: true } }, { text: "= (P2 ASD − P2 TD) − (P1 stream A − P1 stream B). The Phase 1 term is the seed-only null." }], "effect");
  card(4.1, 0.72, [{ text: "Primary measures: ", options: { bold: true } }, { text: "dispersion, saccade amplitude, revisit rate; fixed before E1 from the earlier temperature run. Also count, duration, rate, total time, direction." }], "measures");
  card(4.9, 0.6, [{ text: "Per-group realism: ", options: { bold: true } }, { text: "each label's per-image error against its own real group." }], "realism");
  box(s, 0.6, 5.68, 12.13, 1.17, K.card, null, "temperature");
  T(s, [
    { text: "Separate temperature run. ", options: { bold: true } },
    { text: "Phase 2 sampled at τ = 1 and τ = 2 with paired seeds (10 seeds). τ only rescales the spatial component spread during sampling (σ becomes σ" }, { text: "τ", options: { superscript: true } },
    { text: "); NLL is always computed without τ. Statistics: Hedges' g, bootstrap intervals and sign-flip tests, Holm correction within families; all analyses exploratory." },
  ], { x: 0.8, y: 5.72, w: 11.75, h: 1.09, fontSize: 12, valign: "middle" });
  footer(s, "Sources: manuscript §3.8, §4.3, §5.1–5.5, §6.4 (off-image counts) · mapping evidence: manuscript §4.3 · E1: research_completion/ADDITIONAL_RESULTS.md");
  s.addNotes(notesFor(6));
}

// 7. Results and observations
{
  const s = slide("7 · RESULTS AND OBSERVATIONS", "Results: what improved, what did not, what is unclear");
  T(s, "Likelihood: Phase 2 − Phase 1 joint NLL gain (nats per fixation, 95% CI)", { x: 0.6, y: 1.32, w: 6.6, h: 0.3, fontSize: 12, bold: true });
  const px0 = 3.25, px1 = 5.55, min = -0.035, max = 0.02;
  const X = (v) => px0 + (v - min) / (max - min) * (px1 - px0);
  const fr = [
    ["In-domain (Saliency4ASD test)", "+0.0073 [−0.0012, +0.0163]", 0.0073, -0.0012, 0.0163, "INCONCLUSIVE", "inc"],
    ["External, primary version", "−0.0172 [−0.0225, −0.0122]", -0.0172, -0.0225, -0.0122, "WORSE", "neg"],
    ["External, stretch mapping", "−0.0274 [−0.0322, −0.0226]", -0.0274, -0.0322, -0.0226, "WORSE", "neg"],
    ["External, off-image removed", "−0.0087 [−0.0130, −0.0042]", -0.0087, -0.0130, -0.0042, "WORSE", "neg"],
  ];
  const y0 = 1.92, dy = 0.5, yAx = y0 + fr.length * dy - 0.2;
  arrow(s, X(0), 1.68, X(0), yAx, "9AA5AE", 1, { noHead: true, dash: true, name: "zero line" });
  fr.forEach((r, i) => {
    const y = y0 + i * dy;
    T(s, [{ text: r[0], options: { bold: i === 0, breakLine: true } }, { text: r[1], options: { fontSize: 10, color: K.muted } }], { x: 0.6, y: y - 0.23, w: 2.6, h: 0.46, fontSize: 11.5, valign: "middle" });
    const neg = r[6] === "neg";
    arrow(s, X(r[3]), y, X(r[4]), y, neg ? K.neg : K.ink, 2.25, { noHead: true, name: "interval " + r[0] });
    circle(s, X(r[2]), y, 0.17, neg ? K.neg : "FFFFFF", neg ? null : K.ink, "estimate " + r[0]);
    chip(s, 5.68, y - 0.14, r[5], r[6], 9.5);
  });
  arrow(s, px0, yAx, px1, yAx, K.slate, 1, { noHead: true, name: "axis" });
  [-0.03, -0.02, -0.01, 0, 0.01, 0.02].forEach((v) => {
    arrow(s, X(v), yAx, X(v), yAx + 0.05, K.slate, 1, { noHead: true, name: "tick" });
    T(s, (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(2), { x: X(v) - 0.3, y: yAx + 0.05, w: 0.6, h: 0.2, fontSize: 9, color: K.muted, align: "center" });
  });
  s.addChart(pres.charts.BAR, [
    { name: "Share of real contrast", labels: ["Spatial dispersion", "Saccade amplitude", "Revisit rate"], values: [69, 64, 32] },
  ], {
    x: 7.15, y: 1.28, w: 5.58, h: 2.5, barDir: "bar", chartColors: [K.slate], barGapWidthPct: 55,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '"≈"0"%"', dataLabelFontSize: 11, dataLabelColor: K.ink, dataLabelFontFace: BODY,
    showTitle: true, title: "Generated ASD−TD contrast as % of the real contrast (Gaze4ASD, E1, τ = 1)", titleFontSize: 11.5, titleColor: K.ink, titleFontFace: BODY,
    showLegend: false, valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 25, valAxisLabelFontSize: 9.5, valAxisLabelColor: K.muted, valAxisLabelFontFace: BODY,
    valAxisLabelFormatCode: '0"%"', catAxisLabelFontSize: 11, catAxisLabelColor: K.ink, catAxisLabelFontFace: BODY, catAxisOrientation: "maxMin",
    valGridLine: { color: "E3E8EC", size: 0.75 }, catGridLine: { style: "none" }, objectName: "chart shares",
  });
  T(s, "100% = real size. Approximate shares (an independent run gave 60 / 50 / 29%), not accuracy.", { x: 7.15, y: 3.78, w: 5.58, h: 0.22, fontSize: 9.5, color: K.muted, italic: true });
  const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: K.slate } } });
  const v = (t, c) => ({ text: t, options: { bold: true, color: c } });
  const grp = (g, t) => [{ text: g + " · ", options: { bold: true, color: K.muted } }, { text: t }];
  const tr = [
    [hdr("Comparison"), hdr("Result"), hdr("Verdict")],
    [{ text: grp("Likelihood", "Phase 1 vs pretrained") }, "In-domain test NLL −0.1665 → −0.4957", v("Improved", K.pos)],
    [{ text: grp("Likelihood", "Phase 2 vs Phase 1, in-domain") }, "+0.0073, 95% CI [−0.0012, +0.0163]; 27 of 45 images favour Phase 2; sign-flip p = 0.1097", v("Inconclusive", K.amber)],
    [{ text: grp("Likelihood", "Phase 2 vs Phase 1, Gaze4ASD") }, "Phase 1 better by 0.0172 (−0.5568 vs −0.5396); TD worse in all 3 versions; ASD +0.0104 (participant CI only)", v("Worse", K.neg)],
    [{ text: grp("Behaviour", "label effect") }, "8 of 9 measures shift beyond the seed-only null; shifts ≈1% of the image diagonal", v("Present, small", K.partial)],
    [{ text: grp("Behaviour", "spatial contrasts") }, "Right direction: ≈69% (dispersion), 64% (amplitude), 32% (revisit) of real size", v("Partial", K.partial)],
    [{ text: grp("Behaviour", "other contrasts") }, "Fixation count, total fixation time and direction concentration go the wrong way", v("Wrong direction", K.neg)],
    [{ text: grp("Behaviour", "per-group realism") }, "No spatial measure closer to its real group; ≈10 generated vs ≈6 real fixations per scanpath", v("Not improved", K.neg)],
    [{ text: grp("Sampling", "τ = 2 vs τ = 1") }, "Same directions, smaller contrasts (46 / 33 / 15%); likelihood unaffected", v("Caveat", K.partial)],
  ];
  s.addTable(tr, { x: 0.6, y: 4.12, w: 12.13, colW: [3.15, 7.38, 1.6], fontFace: BODY, fontSize: 11, color: K.ink, valign: "middle",
    border: { type: "solid", pt: 0.75, color: K.rule }, rowH: 0.31, margin: [0.02, 0.08, 0.02, 0.08] });
  footer(s, "Sources: manuscript §6.1, Tables 3–5 (external forest values: Table 5 CPU re-scoring; Table 4 GPU primary −0.0172), Tables 7–8, §6.6–6.8");
  s.addNotes(notesFor(7));
}

// 8. Interpretation, limitations, next steps
{
  const s = slide("8 · INTERPRETATION AND NEXT STEPS", "What we can conclude, and what comes next");
  const cols = [
    ["What we can conclude", K.pos, K.posTint, [
      "The label changes generated behaviour, but by ≈1% of the image diagonal; the generated ASD−TD dispersion difference is positive on all 45 test images",
      "Correct spatial contrasts come from the conditions moving apart (TD contracts past real TD), not per-group realism",
      "A 1,024-parameter label does not improve likelihood: inconclusive in-domain, worse externally",
    ]],
    ["Still unresolved", K.neg, K.negTint, [
      "Why the TD embedding fits both Gaze4ASD groups worse (not tested)",
      "How much fixation over-production (≈10 vs ≈6) drives the wrong timing contrasts",
      "Whether missing off-image gaze (≈2.7× more in ASD) contributes",
    ]],
    ["Limitations", K.amber, K.amberTint, [
      "Gaze4ASD screen mapping inferred, not confirmed",
      "No participant-level age (3–7 vs 5–12 years; means 5.04 vs 4.54)",
      "Stimuli selected for group differences; 33 ASD children",
      "Exploratory statistics; dataset terms unresolved",
    ]],
    ["Next experiments", K.teal, K.tealTint, [
      "Ask the Gaze4ASD authors for display geometry, the 132/133 count and ages",
      "Add stopping / viewing-time and off-image modelling; rerun E1 with the same null",
      "Richer conditioning vs the same Phase 1 control, with per-group criteria",
      "Preregistered replication on independent data",
    ]],
  ];
  cols.forEach((c, i) => {
    const x = 0.6 + i * 3.077;
    box(s, x, 1.38, 2.9, 3.95, c[2], null, "column " + c[0]);
    T(s, c[0], { x: x + 0.18, y: 1.47, w: 2.6, h: 0.4, fontSize: 15, bold: true, fontFace: HEAD, color: c[1], valign: "middle" });
    T(s, bullets(c[3]), { x: x + 0.18, y: 1.95, w: 2.6, h: 3.3, fontSize: 12.5, paraSpaceAfter: 7 });
  });
  box(s, 0.6, 5.48, 12.13, 1.4, K.ink, null, "conclusion");
  T(s, [
    { text: "Conclusion. ", options: { bold: true, color: K.tealLight } },
    { text: "We developed and evaluated an ASD/TD-conditioned extension of TPP-Gaze. It changes generated gaze behaviour and reproduces some spatial group-contrast directions, but does not consistently improve likelihood or individual-group realism.", options: { color: "FFFFFF", breakLine: true } },
    { text: "Group-level research model only: no classifier was evaluated and there is no clinical validation.", options: { color: "C9D3DA", fontSize: 12.5, italic: true } },
  ], { x: 0.85, y: 5.52, w: 11.65, h: 1.32, fontSize: 14, valign: "middle", paraSpaceAfter: 4 });
  footer(s, "Sources: manuscript §4.2 (ages), §6.3–6.7, §7–9 · demo/demo_results/manifest.json (45 of 45 images) · research_completion/UNRESOLVED_ISSUES.md");
  s.addNotes(notesFor(8));
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
