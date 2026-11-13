// Builds presentation/ASD_ScanPath_Research_Presentation.pptx.
// Every number on the slides is taken from paper/final/ASD_ScanPath_Final.md
// (demo selection values from demo/demo_results/manifest.json).
const pptxgen = require("pptxgenjs");
// Theme helper from the local "pptx" skill used to build the deck; set PPTX_SKILL_DIR to its folder.
const SKILL = process.env.PPTX_SKILL_DIR || require("path").join(require("os").homedir(), ".claude/skills/synced/01c91bb0-79a0-4c38-a902-19cc77cedb90_0b33054a-965f-4dda-ba15-743d9fa092e0/pptx");
const { applyTheme } = require(SKILL + "/scripts/apply_theme.js");

const ROOT = require("path").resolve(__dirname, "..", "..", "..", "..");  // repository root
const OUT = ROOT + "/presentation/supporting_material/research_presentation_17_slide/ASD_ScanPath_Research_Presentation.pptx";
const A = ROOT + "/presentation/supporting_material/research_presentation_17_slide/assets/";

const THEME = {
  name: "ASD-ScanPath",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B242B", lt1: "FFFFFF", dk2: "3F4E59", lt2: "EEF2F4",
    accent1: "0E6E6E", accent2: "D9701E", accent3: "2F6DB5",
    accent4: "B3261E", accent5: "9A6700", accent6: "2E7D4F",
    hlink: "2F6DB5", folHlink: "6B4E9B",
  },
};
const K = {
  ink: "1B242B", slate: "3F4E59", muted: "66737D", rule: "D5DCE1", card: "F3F5F7",
  teal: "0E6E6E", tealTint: "E3F1F1", tealLight: "8CC7C7",
  orange: "D9701E", orangeText: "A9500F", orangeTint: "FBEBDD",
  blue: "2F6DB5", blueTint: "E5EEF8",
  neg: "B3261E", negTint: "F8E3E1",
  amber: "8A5D00", amberTint: "F6EED9",
  pos: "2E7D4F", posTint: "E2F0E7",
  part: "5B4B8A", partTint: "ECE8F5",
  dark: "1B242B", dark2: "27333C",
};
const HEAD = "Cambria", BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };
pres.title = "ASD/TD-Conditioned Gaze Scanpath Generation";
pres.subject = "Research presentation: TPP-Gaze with ASD/TD conditioning";
pres.author = "[Presenter name]";

const FOOT = "ASD-ScanPath · research presentation · not a diagnostic tool";
function master(title, bg, kickerColor, titleColor, footColor) {
  pres.defineSlideMaster({
    title,
    background: { color: bg },
    margin: [0.5, 0.6, 0.6, 0.6],
    objects: [
      { placeholder: { options: { name: "kicker", type: "body", x: 0.6, y: 0.3, w: 9, h: 0.3, fontFace: BODY, fontSize: 12, bold: true, color: kickerColor, margin: 0, valign: "middle" }, text: "" } },
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.6, w: 12.13, h: 0.66, fontFace: HEAD, fontSize: 30, bold: true, color: titleColor, margin: 0, valign: "middle", align: "left" }, text: "" } },
      { text: { text: FOOT, options: { x: 0.6, y: 7.05, w: 8, h: 0.28, fontFace: BODY, fontSize: 10, color: footColor, margin: 0 } } },
    ],
    slideNumber: { x: 12.23, y: 7.05, w: 0.5, h: 0.28, fontFace: BODY, fontSize: 10, color: footColor, align: "right" },
  });
}
master("CONTENT", "FFFFFF", K.teal, K.ink, K.muted);
master("DARK", K.dark, K.tealLight, "FFFFFF", "A9B4BC");
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: K.dark },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.75, w: 8.6, h: 1.9, fontFace: HEAD, fontSize: 36, bold: true, color: "FFFFFF", margin: 0, valign: "top", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.8, w: 8.4, h: 0.9, fontFace: BODY, fontSize: 18, color: "CFD8DE", margin: 0, valign: "top", align: "left" }, text: "" } },
  ],
});

// ---------- helpers (fresh option objects on every call: pptxgenjs mutates them) ----------
function T(s, text, o) {
  // Tighter hanging indent than the pptxgenjs default for every bullet run.
  if (Array.isArray(text)) text = text.map((r) => (r.options && r.options.bullet === true) ? { text: r.text, options: Object.assign({}, r.options, { bullet: { indent: 14 } }) } : r);
  s.addText(text, Object.assign({ isTextBox: true, fontFace: BODY, fontSize: 14, color: K.ink, margin: 0, valign: "top" }, o));
}
function box(s, x, y, w, h, fill, line, name, radius) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, fill: { color: fill }, line: line ? { color: line, width: 1 } : { type: "none" },
    rectRadius: radius === undefined ? 0.08 : radius, objectName: name,
  });
}
function rect(s, x, y, w, h, fill, line, name) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill }, line: line ? { color: line, width: 1 } : { type: "none" }, objectName: name });
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
// Numbered fixation circle: the deck's visual motif.
function fix(s, cx, cy, d, n, fill, fontSize, textColor) {
  circle(s, cx, cy, d, fill, null, "fixation " + n);
  T(s, String(n), { x: cx - d / 2, y: cy - d / 2, w: d, h: d, align: "center", valign: "middle", bold: true, fontSize: fontSize || 12, color: textColor || "FFFFFF" });
}
const CHIP = {
  neg: [K.negTint, K.neg], inc: [K.amberTint, K.amber], part: [K.partTint, K.part],
  pos: [K.posTint, K.pos], note: [K.card, K.slate],
};
function chip(s, x, y, label, kind) {
  const w = 0.3 + label.length * 0.092;
  box(s, x, y, w, 0.3, CHIP[kind][0], null, "verdict " + label, 0.15);
  T(s, label, { x, y, w, h: 0.3, align: "center", valign: "middle", bold: true, fontSize: 11, color: CHIP[kind][1], charSpacing: 0.5 });
  return w;
}
function content(section, kicker, title, takeaway) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(kicker, { placeholder: "kicker" });
  s.addText(title, { placeholder: "title" });
  if (takeaway) T(s, takeaway, { x: 0.6, y: 1.27, w: 12.13, h: 0.42, fontSize: 18, color: K.slate, valign: "middle" });
  return s;
}
const sub = (t) => ({ text: t, options: { subscript: true } });
const sup = (t) => ({ text: t, options: { superscript: true } });

// ======================================================================
// 1. Title
pres.addSection({ title: "Introduction" });
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Introduction" });
  T(s, "RESEARCH PRESENTATION · SUPERVISOR REVIEW", { x: 0.8, y: 1.2, w: 8, h: 0.35, fontSize: 13, bold: true, color: K.tealLight, charSpacing: 1 });
  s.addText("ASD/TD-Conditioned Gaze Scanpath Generation with Neural Temporal Point Processes", { placeholder: "title" });
  s.addText("Likelihood, transfer and behavioural evaluation of a 1,024-parameter group embedding in TPP-Gaze", { placeholder: "body" });
  T(s, "[Presenter name]  ·  [Supervisor name]  ·  [Affiliation]  ·  [Date]", { x: 0.8, y: 5.05, w: 8.4, h: 0.4, fontSize: 16, color: "FFFFFF" });
  T(s, "Exploratory research study. The model describes group-level gaze statistics; it has no clinical or diagnostic validity.", { x: 0.8, y: 6.35, w: 8.6, h: 0.5, fontSize: 12, color: "A9B4BC" });
  // Abstract scanpath motif (illustration only)
  const pts = [[10.2, 2.0, 0.42], [11.9, 2.55, 0.62], [10.9, 3.6, 0.36], [12.2, 4.4, 0.78], [10.4, 5.0, 0.5]];
  for (let i = 0; i < pts.length - 1; i++) arrow(s, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], "4D6B75", 1.5, { name: "motif arrow" });
  pts.forEach((p, i) => fix(s, p[0], p[1], p[2], i + 1, i === 0 ? K.teal : (i === pts.length - 1 ? K.orange : "3B5560"), 12));
  T(s, "Illustration: a scanpath is an ordered sequence of fixations; circle size = duration", { x: 9.6, y: 5.6, w: 3.2, h: 0.5, fontSize: 10, color: "8796A0", italic: true });
  s.addNotes(
`Opening. Suggested time: about 30 s.
This talk presents an exploratory study: we added a tiny ASD/TD group embedding to a published scanpath model, TPP-Gaze, and asked whether it helps. The short answer is: mostly no, and the talk explains exactly where it helps, where it does not, and why that matters for evaluating conditional gaze models.
Say early: this is a research model of group-level gaze statistics. It does not predict autism and is not a diagnostic or screening tool.
Fill in presenter, supervisor, affiliation and date before presenting.`);
}

// 2. Motivation
{
  const s = content("Introduction", "INTRODUCTION · MOTIVATION", "Where, in what order and for how long we look",
    "Gaze differs between autistic and typically developing children, in space and in time.");
  const cards = [
    ["Location", "Where each fixation lands on the image"],
    ["Order", "The sequence of fixations: the scanpath"],
    ["Duration", "How long each fixation lasts"],
  ];
  cards.forEach((c, i) => {
    const x = 0.6 + i * 2.62;
    box(s, x, 1.95, 2.42, 2.55, K.card, null, "card " + c[0]);
    T(s, c[0], { x: x + 0.2, y: 3.45, w: 2.05, h: 0.35, fontSize: 17, bold: true, fontFace: HEAD });
    T(s, c[1], { x: x + 0.2, y: 3.82, w: 2.05, h: 0.6, fontSize: 13, color: K.slate });
  });
  // mini illustrations (not data)
  { const x = 0.6, y = 1.95; rect(s, x + 0.35, y + 0.3, 1.72, 1.0, "FFFFFF", K.rule, "image frame");
    [[0.7, 0.55], [1.3, 0.85], [1.75, 0.5], [0.95, 1.05]].forEach((p) => circle(s, x + p[0], y + p[1], 0.16, K.teal)); }
  { const x = 3.22, y = 1.95; const p = [[0.55, 1.0], [1.15, 0.55], [1.75, 1.05]];
    arrow(s, x + p[0][0], y + p[0][1], x + p[1][0], y + p[1][1], K.slate, 1.5); arrow(s, x + p[1][0], y + p[1][1], x + p[2][0], y + p[2][1], K.slate, 1.5);
    p.forEach((q, i) => fix(s, x + q[0], y + q[1], 0.36, i + 1, K.teal, 12)); }
  { const x = 5.84, y = 1.95; [[0.5, 0.25], [1.05, 0.45], [1.75, 0.7]].forEach((q) => circle(s, x + q[0], y + 0.8, q[1], K.tealLight, K.teal));
    T(s, "circle size ∝ duration", { x: x + 0.2, y: y + 1.25, w: 2.0, h: 0.25, fontSize: 10, color: K.muted, italic: true, align: "center" }); }

  T(s, [
    { text: "Why model it", options: { bold: true, fontFace: HEAD, fontSize: 17, breakLine: true } },
    { text: "Eye-tracking studies report ASD–TD differences in visual attention [1, 2]; effects vary across stimuli and regions of interest [1].", options: { bullet: true, breakLine: true } },
    { text: "These differences motivate computational models of gaze.", options: { bullet: true, breakLine: true } },
    { text: "What is usually missing", options: { bold: true, fontFace: HEAD, fontSize: 17, breakLine: true } },
    { text: "Most scanpath models predict an ordered sequence of locations and pay less attention to timing.", options: { bullet: true, breakLine: true } },
    { text: "TPP-Gaze models fixation location and duration jointly, as a temporal point process [3].", options: { bullet: true } },
  ], { x: 8.75, y: 1.95, w: 3.98, h: 4.4, fontSize: 14, paraSpaceAfter: 6 });

  box(s, 0.6, 4.75, 7.66, 1.4, K.tealTint, null, "study idea");
  T(s, [
    { text: "This study", options: { bold: true, fontFace: HEAD, fontSize: 16, color: K.teal, breakLine: true } },
    { text: "Give one shared TPP-Gaze model a minimal ASD/TD group label, then test whether it fits group-labelled gaze better and whether scanpaths generated under each label differ the way real groups differ.", options: {} },
  ], { x: 0.85, y: 4.75, w: 7.2, h: 1.4, fontSize: 14, valign: "middle" });
  s.addNotes(
`Suggested time: about 45 s.
Gaze has three components: where people look, in what order, and for how long. Eye-tracking studies report group differences between autistic and typically developing children [1, 2], but the effects depend on the stimulus and region [1].
Most scanpath models predict the order of locations and treat timing as secondary. TPP-Gaze [3] is different: it predicts location and duration jointly.
Our question: if we give that model only a minimal group label, does it describe group-labelled gaze better, and do its generated scanpaths differ like the real groups? The three small drawings are illustrations, not data.`);
}

// 3. Problem and objectives
{
  const s = content("Introduction", "INTRODUCTION · RESEARCH QUESTIONS", "Research question and objectives",
    "If a shared TPP-Gaze model gets only a small ASD/TD group embedding, what changes?");
  const rq = [
    ["In-domain fit", "Does the embedding describe group-labelled gaze better on held-out Saliency4ASD images? (likelihood)"],
    ["Transfer", "Does any benefit hold on a separate dataset, Gaze4ASD, never used for training or selection?"],
    ["Generated behaviour", "Do scanpaths generated under the ASD and TD labels differ the way real groups differ, and is each group more realistic?"],
    ["Sampling temperature", "Do the behavioural conclusions depend on the sampling temperature τ?"],
  ];
  rq.forEach((r, i) => {
    const x = 0.6 + (i % 2) * 3.95, y = 1.95 + Math.floor(i / 2) * 2.35;
    box(s, x, y, 3.75, 2.15, K.card, null, "RQ" + (i + 1));
    fix(s, x + 0.42, y + 0.42, 0.46, i + 1, K.teal, 14);
    T(s, r[0], { x: x + 0.8, y: y + 0.24, w: 2.8, h: 0.38, fontSize: 17, bold: true, fontFace: HEAD, valign: "middle" });
    T(s, r[1], { x: x + 0.25, y: y + 0.8, w: 3.3, h: 1.25, fontSize: 14, color: K.slate });
  });
  box(s, 8.65, 1.95, 4.08, 4.5, K.negTint, null, "not claimed");
  T(s, [
    { text: "What we do not claim", options: { bold: true, fontFace: HEAD, fontSize: 17, color: K.neg, breakLine: true } },
    { text: "The first ASD scanpath generator", options: { bullet: true, breakLine: true } },
    { text: "An improvement over TPP-Gaze", options: { bullet: true, breakLine: true } },
    { text: "Confirmed generalization", options: { bullet: true, breakLine: true } },
    { text: "Any biological mechanism", options: { bullet: true, breakLine: true } },
    { text: "Prediction or diagnosis of autism", options: { bullet: true } },
  ], { x: 8.9, y: 2.15, w: 3.6, h: 2.6, fontSize: 15, paraSpaceAfter: 6 });
  T(s, "The model describes group-level gaze statistics only.", { x: 8.9, y: 4.85, w: 3.6, h: 0.8, fontSize: 14, italic: true, color: K.slate });
  s.addNotes(
`Suggested time: about 40 s.
Four questions. One: in-domain likelihood on held-out Saliency4ASD images. Two: transfer to Gaze4ASD, which we only ever used for evaluation. Three: generated behaviour, both whether the ASD-minus-TD contrast has the real direction and whether each group's samples become more realistic. Four: whether the conclusions depend on sampling temperature.
Also state up front what we do not claim: not the first ASD scanpath generator, not an improvement over TPP-Gaze, not confirmed generalization, no mechanism, and nothing diagnostic.`);
}

// 4. Related work and gap
{
  const s = content("Introduction", "INTRODUCTION · RELATED WORK", "Related work and research gap",
    "We isolate what a minimal group label changes in a published duration-aware model.");
  const cols = [
    ["Saliency and scanpath models", ["Saliency from image features [7]", "DeepGaze III: free-viewing scanpaths [8]", "SpFormer: transformer; duration in fixation features [9]"]],
    ["Neural temporal point processes", ["History-dependent event models [13, 14]", "Transformer Hawkes process [15]", "Intensity-free, log-normal mixture [16]", "TPP-Gaze applies this to gaze [3]"]],
    ["ASD gaze data", ["Meta-analysis: group differences vary by stimulus [1]", "Saliency4ASD: 300 images, ASD and control children [4, 5]", "Gaze-based ASD identification [6, 17]: not our aim"]],
    ["ASD scanpath prediction", ["Zhong et al. [18]: semantic-based saccadic scanpath prediction for ASD", "Described from the abstract only; full text not accessible to us"]],
  ];
  cols.forEach((c, i) => {
    const x = 0.6 + i * 3.07;
    box(s, x, 1.95, 2.9, 3.05, K.card, null, "related " + c[0]);
    T(s, c[0], { x: x + 0.2, y: 2.1, w: 2.5, h: 0.62, fontSize: 15, bold: true, fontFace: HEAD, color: K.teal });
    T(s, c[1].map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < c[1].length - 1 } })),
      { x: x + 0.2, y: 2.78, w: 2.55, h: 2.15, fontSize: 13, color: K.ink, paraSpaceAfter: 5 });
  });
  box(s, 0.6, 5.2, 12.13, 1.45, K.tealTint, null, "gap");
  T(s, [
    { text: "Gap. ", options: { bold: true, color: K.teal } },
    { text: "Models already predict scanpaths with duration [3, 9] or target ASD specifically [18]. A simpler question remains open: how much does a minimal group label change a published joint location–duration model, and does the change hold on another dataset?", options: { breakLine: true } },
    { text: "Our design: ", options: { bold: true, color: K.teal } },
    { text: "keep TPP-Gaze, add one group embedding, compare against an unconditioned control, in-domain and externally, for likelihood and for generated behaviour. We claim no novelty or superiority over [18].", options: {} },
  ], { x: 0.85, y: 5.2, w: 11.65, h: 1.45, fontSize: 14, paraSpaceAfter: 6, valign: "middle" });
  s.addNotes(
`Suggested time: about 45 s.
Four strands. Saliency and scanpath models: from image-feature saliency [7] to DeepGaze III [8] and SpFormer [9], which uses duration in its fixation features. Neural temporal point processes model irregular events; TPP-Gaze [3] treats fixation durations as inter-event times and locations as marks. ASD gaze: group differences are real but stimulus-dependent [1]; Saliency4ASD [4, 5] is our training data. Some work uses gaze to identify autism [6, 17]; ours does not. Closest is Zhong et al. [18]; we only had the abstract, so we make no claims about how it compares.
The gap we address is narrow: the effect of a minimal group label on an existing joint model, with a matched unconditioned control and an external dataset.`);
}

// 5. TPP-Gaze background
pres.addSection({ title: "Method" });
{
  const s = content("Method", "METHOD · BACKGROUND", "TPP-Gaze: a scanpath as a marked point process",
    "Durations are the inter-event times; fixation locations are the marks of each event [3].");
  // Illustration: marks on an image frame + durations on a time axis (not data)
  T(s, "Illustration (not data)", { x: 0.6, y: 1.85, w: 3, h: 0.25, fontSize: 10, italic: true, color: K.muted });
  rect(s, 0.6, 2.15, 6.1, 2.55, K.card, K.rule, "image frame");
  T(s, [{ text: "Marks: locations x" }, sub("n"), { text: " ∈ [−1, 1]²" }], { x: 0.75, y: 2.22, w: 4, h: 0.3, fontSize: 12, color: K.slate });
  const P = [[1.3, 3.6], [2.6, 2.9], [3.9, 3.85], [5.1, 3.05], [5.9, 4.05]];
  const D = [0.35, 0.6, 0.3, 0.95, 0.5];
  for (let i = 0; i < P.length - 1; i++) arrow(s, P[i][0], P[i][1], P[i + 1][0], P[i + 1][1], K.slate, 1.5);
  P.forEach((p, i) => fix(s, p[0], p[1], 0.3 + D[i] * 0.45, i + 1, K.teal, 12));
  // time axis 0..3 s
  const t0 = 0.6, tw = 6.1, ty = 5.35;
  arrow(s, t0, ty + 0.5, t0 + tw + 0.05, ty + 0.5, K.slate, 1.25);
  let acc = 0.0;
  D.forEach((d, i) => {
    const x = t0 + (acc / 3) * tw, w = (d / 3) * tw - 0.04;
    box(s, x, ty, w, 0.42, i % 2 ? K.tealLight : K.tealTint, K.teal, "duration " + (i + 1), 0.04);
    T(s, [{ text: "d" }, sub(String(i + 1))], { x, y: ty, w, h: 0.42, align: "center", valign: "middle", fontSize: 12, color: K.ink });
    acc += d;
  });
  T(s, "0 s", { x: t0 - 0.1, y: ty + 0.58, w: 0.6, h: 0.25, fontSize: 11, color: K.muted });
  T(s, "3 s viewing window", { x: t0 + tw - 1.6, y: ty + 0.58, w: 1.7, h: 0.25, fontSize: 11, color: K.muted, align: "right" });
  T(s, "Inter-event times: fixation durations", { x: t0, y: 4.95, w: 4, h: 0.3, fontSize: 12, color: K.slate });

  const comp = [
    ["Image encoder", "DenseNet201 + CoordConv → image representation z, 256-d [19, 20]"],
    ["History encoder", "Transformer (4 layers, 4 heads) + LSTM → history state h, 256-d [15]"],
    ["Context", "Concatenation of h and z: 512-d"],
    ["Duration head", "Log-normal mixture, K = 4 components"],
    ["Location head", "Gaussian mixture, J = 16 components, diagonal covariance"],
    ["Objective", "Joint negative log-likelihood (NLL), nats per fixation; lower is better"],
  ];
  T(s, "At each step the model predicts the next event", { x: 7.2, y: 1.85, w: 5.5, h: 0.35, fontSize: 16, bold: true, fontFace: HEAD });
  comp.forEach((c, i) => {
    const y = 2.33 + i * 0.66;
    fix(s, 7.42, y + 0.2, 0.36, i + 1, K.teal, 12);
    T(s, [{ text: c[0] + "  ", options: { bold: true } }, { text: c[1], options: { color: K.slate } }], { x: 7.75, y, w: 4.98, h: 0.6, fontSize: 13 });
  });
  T(s, "We use the released code and pretrained weights without modification [3]. Base model: 23,895,712 parameters, 3,881,784 outside the frozen image backbone.",
    { x: 7.2, y: 6.33, w: 5.53, h: 0.6, fontSize: 11, color: K.muted });
  s.addNotes(
`Suggested time: about 50 s.
TPP-Gaze treats a scanpath as a marked temporal point process. Each fixation is an event: its duration is the inter-event time, and its location is the mark. The drawing on the left is only an illustration.
At each step the model encodes the image with DenseNet201 plus CoordConv into a 256-d vector, encodes the gaze history with a transformer and LSTM into another 256-d vector, concatenates them into a 512-d context, and predicts the next duration with a 4-component log-normal mixture and the next location with a 16-component Gaussian mixture. Training minimizes joint NLL in nats per fixation.
Two uses matter later: likelihood evaluation scores observed scanpaths; generation samples one fixation at a time and feeds it back into the history.
Disclosure if asked: the released transformer has a batch-layout issue (attention runs across scanpaths in a batch). We kept it because the pretrained weights were learned with it.`);
}

// 6. Architecture with conditioning
{
  const s = content("Method", "METHOD · PROPOSED CONDITIONING", "Our modification: one group embedding",
    "Everything except a 1,024-parameter embedding is the frozen Phase 1 model.");
  const base = (x, y, w, h, title, line2, name) => {
    box(s, x, y, w, h, K.tealTint, K.teal, name);
    T(s, [{ text: title, options: { bold: true, breakLine: true } }, { text: line2, options: { fontSize: 12, color: K.slate } }],
      { x: x + 0.1, y, w: w - 0.2, h, fontSize: 14, align: "center", valign: "middle" });
  };
  const ours = (x, y, w, h, titleRuns, line2, name) => {
    box(s, x, y, w, h, K.orangeTint, K.orange, name);
    const runs = titleRuns.map((r, i) => ({ text: r.text, options: Object.assign({ bold: true }, r.options || {}, i === titleRuns.length - 1 ? { breakLine: true } : {}) }));
    T(s, runs.concat([{ text: line2, options: { fontSize: 12, color: K.slate } }]),
      { x: x + 0.1, y, w: w - 0.2, h, fontSize: 14, align: "center", valign: "middle" });
  };
  const neutral = (x, y, w, h, title, line2, name) => {
    box(s, x, y, w, h, "FFFFFF", K.slate, name);
    T(s, [{ text: title, options: { bold: true, breakLine: !!line2 } }].concat(line2 ? [{ text: line2, options: { fontSize: 12, color: K.slate } }] : []),
      { x: x + 0.08, y, w: w - 0.16, h, fontSize: 14, align: "center", valign: "middle" });
  };
  // rows: label (top), image, history
  const r0 = 1.95, r1 = 2.85, r2 = 3.75, rh = 0.7;
  neutral(0.6, r0, 2.1, rh, "ASD / TD label ℓ", "ASD = 0, TD = 1", "input label");
  neutral(0.6, r1, 2.1, rh, "Stimulus image", "resized to 512 × 512", "input image");
  neutral(0.6, r2, 2.1, rh, "Gaze history", "past locations and durations", "input history");
  ours(3.05, r0, 2.7, rh, [{ text: "Group embedding e" }, { text: "ℓ", options: { subscript: true } }], "nn.Embedding(2, 512)", "group embedding");
  base(3.05, r1, 2.7, rh, "Image encoder", "DenseNet201 + CoordConv → z", "image encoder");
  base(3.05, r2, 2.7, rh, "History encoder", "Transformer + LSTM → h", "history encoder");
  base(6.05, r1, 1.85, 1.6, "Fused context", "concat(h, z)\n512-d", "fused context");
  // add node
  circle(s, 8.35, 3.65, 0.5, K.orange, null, "add node");
  T(s, "+", { x: 8.1, y: 3.4, w: 0.5, h: 0.5, align: "center", valign: "middle", bold: true, fontSize: 22, color: "FFFFFF" });
  base(9.0, 2.6, 2.2, 0.95, "Spatial head", "Gaussian mixture, 16 comp.\n→ location (x, y)", "spatial head");
  base(9.0, 3.85, 2.2, 0.95, "Temporal head", "Log-normal mixture, 4 comp.\n→ duration d", "temporal head");
  neutral(11.5, 3.1, 1.23, 1.2, "Next fixation", "(x, y, d)", "output");
  // arrows
  arrow(s, 2.7, r0 + rh / 2, 3.05, r0 + rh / 2);
  arrow(s, 2.7, r1 + rh / 2, 3.05, r1 + rh / 2);
  arrow(s, 2.7, r2 + rh / 2, 3.05, r2 + rh / 2);
  arrow(s, 5.75, r1 + rh / 2, 6.05, r1 + rh / 2);
  arrow(s, 5.75, r2 + rh / 2, 6.05, r2 + rh / 2);
  arrow(s, 7.9, 3.65, 8.1, 3.65);
  arrow(s, 5.75, r0 + rh / 2, 8.35, r0 + rh / 2, K.orange, 2, { noHead: true, name: "embedding path" });
  arrow(s, 8.35, r0 + rh / 2, 8.35, 3.4, K.orange, 2, { name: "embedding path" });
  T(s, "added to the context before both heads", { x: 5.85, y: 1.9, w: 2.45, h: 0.38, fontSize: 11, italic: true, color: K.orangeText, align: "center", valign: "bottom" });
  arrow(s, 8.6, 3.55, 9.0, 3.07);
  arrow(s, 8.6, 3.75, 9.0, 4.32);
  arrow(s, 11.2, 3.07, 11.5, 3.45);
  arrow(s, 11.2, 4.32, 11.5, 3.95);
  // generation loop
  arrow(s, 12.12, 4.3, 12.12, 5.05, "7D8A93", 1.5, { noHead: true, dash: true, name: "loop" });
  arrow(s, 4.4, 5.05, 12.12, 5.05, "7D8A93", 1.5, { noHead: true, dash: true, name: "loop" });
  arrow(s, 4.4, 4.45, 4.4, 5.05, "7D8A93", 1.5, { noHead: true, beginHead: true, dash: true, name: "loop" });
  T(s, "Generation only: the sampled fixation joins the history; repeat until the 3 s window ends", { x: 4.6, y: 5.1, w: 7.4, h: 0.3, fontSize: 12, italic: true, color: K.muted, align: "center" });
  // legend
  box(s, 9.0, 2.0, 0.28, 0.2, K.tealTint, K.teal, "legend base", 0.03);
  T(s, "Original TPP-Gaze (frozen in Phase 2)", { x: 9.35, y: 1.95, w: 3.4, h: 0.3, fontSize: 12, color: K.slate, valign: "middle" });
  box(s, 9.0, 2.33, 0.28, 0.2, K.orangeTint, K.orange, "legend ours", 0.03);
  T(s, "Our addition (only trained part in Phase 2)", { x: 9.35, y: 2.28, w: 3.4, h: 0.3, fontSize: 12, color: K.slate, valign: "middle" });
  // equation strip and stat callouts
  const stats = [
    ["1,024", "trained parameters: a 2 × 512 embedding, one row per group"],
    ["0", "initial value: at the start Phase 2 is numerically identical to Phase 1"],
    ["1,290", "base tensors identical in both checkpoints: the embedding is the only difference"],
  ];
  stats.forEach((st, i) => {
    const x = 0.6 + i * 4.11, nw = st[0].length > 1 ? 1.25 : 0.45;
    box(s, x, 5.6, 3.91, 1.2, K.card, null, "stat " + st[0]);
    T(s, st[0], { x: x + 0.2, y: 5.68, w: nw, h: 1.05, fontSize: 30, bold: true, fontFace: HEAD, color: K.orangeText, valign: "middle" });
    T(s, st[1], { x: x + 0.3 + nw, y: 5.68, w: 3.45 - nw, h: 1.05, fontSize: 13, color: K.slate, valign: "middle" });
  });
  s.addNotes(
`Suggested time: about 70 s. This is the key method slide.
Teal boxes are the original TPP-Gaze; the orange parts are our only addition. The ASD/TD label selects one row of a 2 x 512 embedding table, and that vector is added to the 512-d fused context immediately before both prediction heads: c-tilde = c + e_l. So the label can shift both where the next fixation goes and how long it lasts.
Training has two phases. Phase 1 fine-tunes TPP-Gaze on combined ASD and TD data, with no label. Phase 2 freezes everything and trains only the 1,024 embedding parameters. The embedding starts at zero, so Phase 2 begins identical to Phase 1, and we verified that all 1,290 base tensors are identical in the two checkpoints. Phase 1 is exactly Phase 2 with a zero embedding, so any difference between the models comes from these 1,024 numbers.
The dashed loop is only used in generation.`);
}

// 7. Datasets
{
  const s = content("Method", "METHOD · DATA", "Datasets and preprocessing",
    "Saliency4ASD for training and testing; Gaze4ASD only for external evaluation.");
  const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: K.slate } } });
  const lab = (t) => ({ text: t, options: { bold: true, color: K.ink, fill: { color: K.card } } });
  const rows = [
    [hdr(""), hdr("Saliency4ASD [4, 5]"), hdr("Gaze4ASD [6]")],
    [lab("Role"), "Training, validation, held-out test", "External evaluation only"],
    [lab("Images"), "300 (210 / 45 / 45, split by image)", "30 selected stimuli"],
    [lab("Scanpaths ASD / TD"), "2,705 / 3,324", "771 / 3,605 trials"],
    [lab("Fixations ASD / TD"), "18,475 / 25,504", "4,830 / 23,161"],
    [lab("Participants ASD / TD"), "14 / 14 per image (README); no IDs", "33 / 133 folders (README: 33 / 132)"],
    [lab("Age (years)"), "ASD 5–12, mean 8; controls mean 8", "3–7; means 5.04 ASD, 4.54 TD"],
    [lab("Viewing, tracker"), "3 s; Tobii T120, 1280 × 1024", "3 s; Tobii Pro Fusion, 2560 × 1440"],
  ];
  s.addTable(rows, { x: 0.6, y: 1.95, w: 8.2, colW: [2.2, 3.0, 3.0], fontFace: BODY, fontSize: 13, color: K.ink, valign: "middle",
    border: { type: "solid", pt: 0.75, color: K.rule }, rowH: 0.47, margin: [0.04, 0.08, 0.04, 0.08] });
  T(s, "No participant-level age in either release. Gaze4ASD stimuli are mostly smiling faces, selected for large ASD/TD differences.",
    { x: 0.6, y: 5.85, w: 8.2, h: 0.55, fontSize: 12, color: K.muted });

  T(s, "Preprocessing (both datasets)", { x: 9.2, y: 1.95, w: 3.53, h: 0.35, fontSize: 16, bold: true, fontFace: HEAD });
  const steps = ["Split into scanpaths; drop the first fixation", "Remove fixations shorter than 60 ms", "Truncate at the 3 s window", "Keep scanpaths with at least 3 fixations", "Normalize coordinates to [−1, 1]"];
  steps.forEach((t, i) => {
    const y = 2.42 + i * 0.6;
    fix(s, 9.4, y + 0.2, 0.34, i + 1, K.teal, 11);
    T(s, t, { x: 9.72, y, w: 3.01, h: 0.5, fontSize: 13, valign: "middle" });
  });
  box(s, 9.2, 5.5, 3.53, 1.3, K.amberTint, null, "gaze4asd note");
  T(s, [
    { text: "Gaze4ASD only: ", options: { bold: true, color: K.amber } },
    { text: "screen gaze mapped to image coordinates (fit-to-height, assumed); off-image gaze clipped to the edge (Policy A) or removed (Policy B).", options: {} },
  ], { x: 9.35, y: 5.58, w: 3.25, h: 1.15, fontSize: 12, valign: "middle" });
  s.addNotes(
`Suggested time: about 45 s.
Saliency4ASD: 300 natural images, children with ASD and controls, split by image into 210 training, 45 validation and 45 test images. The release has no participant IDs, so in-domain uncertainty can only be estimated across images.
Gaze4ASD: used only for external evaluation. 30 stimuli, mostly smiling faces, selected by its authors for large group differences. The README says 132 TD children but the release has 133 TD folders; we used the folders as released.
Note the age gap: 5 to 12 years in Saliency4ASD, 3 to 7 in Gaze4ASD, and no participant-level age in either.
Preprocessing is the same for both. For Gaze4ASD we must also map screen coordinates to the image; we come back to this on the external-results slide.`);
}

// 8. Experimental setup
{
  const s = content("Method", "METHOD · EXPERIMENTS", "Experimental design and evaluation metrics",
    "Matched models, matched scanpaths and matched random seeds isolate the embedding's effect.");
  const colw = 3.9, gap = 0.215;
  const head = (x, n, t) => { fix(s, x + 0.35, 2.18, 0.42, n, K.teal, 13); T(s, t, { x: x + 0.7, y: 1.97, w: colw - 0.75, h: 0.42, fontSize: 17, bold: true, fontFace: HEAD, valign: "middle" }); };
  const CH = 4.7;
  // col 1: training
  let x = 0.6; box(s, x, 1.85, colw, CH, K.card, null, "training");
  head(x, 1, "Training");
  T(s, [
    { text: "Phase 1", options: { bold: true, breakLine: true } },
    { text: "Fine-tune TPP-Gaze on combined ASD + TD training data, no label; backbone frozen. AdamW [22], learning rate 10", options: {} }, sup("−3"),
    { text: ", batch 128.", options: { breakLine: true } },
    { text: "Phase 2", options: { bold: true, breakLine: true } },
    { text: "Freeze the whole base; train only the embedding, learning rate 10", options: {} }, sup("−4"), { text: ".", options: { breakLine: true } },
    { text: "Selection", options: { bold: true, breakLine: true } },
    { text: "Validation images chose both checkpoints and τ. Test images and Gaze4ASD were never used for training or selection.", options: {} },
  ], { x: x + 0.22, y: 2.55, w: colw - 0.42, h: 4.1, fontSize: 13, paraSpaceAfter: 4 });
  // col 2: likelihood
  x = 0.6 + colw + gap; box(s, x, 1.85, colw, CH, K.card, null, "likelihood");
  head(x, 2, "Likelihood");
  T(s, [
    { text: "Joint, duration and position NLL in nats per fixation (lower is better).", options: { bullet: true, breakLine: true } },
    { text: "Gain = Phase 1 NLL − Phase 2 NLL; positive favours Phase 2.", options: { bullet: true, breakLine: true } },
    { text: "Both models score the same scanpaths in the same batches.", options: { bullet: true, breakLine: true } },
    { text: "95% intervals: paired cluster bootstrap (10,000) over images in-domain; over participants and images externally.", options: { bullet: true } },
  ], { x: x + 0.22, y: 2.55, w: colw - 0.42, h: 4.1, fontSize: 13, paraSpaceAfter: 6 });
  // col 3: E1 design
  x = 0.6 + 2 * (colw + gap); box(s, x, 1.85, colw, CH, K.card, null, "E1 design");
  head(x, 3, "Generated behaviour (E1)");
  T(s, "Common random numbers: each seed stream is shared, so models differ only through the embedding.", { x: x + 0.22, y: 2.55, w: colw - 0.42, h: 0.75, fontSize: 13 });
  // 2x2 grid
  const gx = x + 1.15, gy = 3.62, cw = 1.25, ch = 0.45;
  T(s, "Phase 2", { x: gx, y: gy - 0.32, w: cw, h: 0.28, fontSize: 12, bold: true, align: "center" });
  T(s, "Phase 1", { x: gx + cw + 0.08, y: gy - 0.32, w: cw, h: 0.28, fontSize: 12, bold: true, align: "center" });
  T(s, "Seed stream A", { x: x + 0.15, y: gy, w: 1.0, h: ch, fontSize: 11, color: K.slate, valign: "middle" });
  T(s, "Seed stream B", { x: x + 0.15, y: gy + ch + 0.08, w: 1.0, h: ch, fontSize: 11, color: K.slate, valign: "middle" });
  box(s, gx, gy, cw, ch, K.orangeTint, K.orange, "P2 ASD", 0.05); T(s, "ASD label", { x: gx, y: gy, w: cw, h: ch, fontSize: 12, align: "center", valign: "middle" });
  box(s, gx, gy + ch + 0.08, cw, ch, K.blueTint, K.blue, "P2 TD", 0.05); T(s, "TD label", { x: gx, y: gy + ch + 0.08, w: cw, h: ch, fontSize: 12, align: "center", valign: "middle" });
  box(s, gx + cw + 0.08, gy, cw, ch, "FFFFFF", K.slate, "P1 A", 0.05); T(s, "no label", { x: gx + cw + 0.08, y: gy, w: cw, h: ch, fontSize: 12, align: "center", valign: "middle" });
  box(s, gx + cw + 0.08, gy + ch + 0.08, cw, ch, "FFFFFF", K.slate, "P1 B", 0.05); T(s, "no label", { x: gx + cw + 0.08, y: gy + ch + 0.08, w: cw, h: ch, fontSize: 12, align: "center", valign: "middle" });
  T(s, [
    { text: "Effect = (ASD − TD contrast in Phase 2) − (A − B contrast in Phase 1, the seed-only null).", options: { breakLine: true } },
    { text: "30 Gaze4ASD + 45 Saliency4ASD test images; 20 scanpaths × 10 seeds; τ = 1.", options: {} },
  ], { x: x + 0.22, y: 4.72, w: colw - 0.42, h: 1.1, fontSize: 12, paraSpaceAfter: 4 });
  T(s, "Primary: dispersion, saccade amplitude, revisit rate (fixed before E1).", { x: x + 0.22, y: 5.85, w: colw - 0.42, h: 0.6, fontSize: 12, color: K.slate, italic: true });
  T(s, "Statistics: Hedges' g [23]; permutation and sign-flip tests; Holm correction [24] within prespecified families. All analyses are exploratory and were not preregistered.",
    { x: 0.6, y: 6.66, w: 12.13, h: 0.28, fontSize: 11, color: K.muted });
  s.addNotes(
`Suggested time: about 55 s.
Training: Phase 1 fine-tunes the model on combined data; Phase 2 freezes it and trains only the embedding. Validation images selected the checkpoints and the temperature; the test images and Gaze4ASD were never used for selection.
Likelihood: we report NLL in nats per fixation, lower is better, and the gain as Phase 1 minus Phase 2, so positive favours conditioning. Intervals come from cluster bootstraps, never treating fixations as independent.
Behaviour, experiment E1: common random numbers. For each seed stream we generate with the ASD label, with the TD label, and with Phase 1. The Phase 1 contrast between two seed streams is a seed-only null; the conditioning effect is the Phase 2 contrast minus that null. Dispersion, saccade amplitude and revisit rate were fixed as primary measures before E1, but they were chosen because an earlier run had reproduced them, so that choice is not independent.
Everything is exploratory and not preregistered.`);
}

// 9. In-domain likelihood
pres.addSection({ title: "Results" });
{
  const s = content("Results", "RESULTS · IN-DOMAIN (SALIENCY4ASD TEST)", "In-domain likelihood: inconclusive gain",
    "Phase 2 lowers held-out NLL slightly, but the 95% interval includes zero.");
  T(s, "+0.0073", { x: 0.6, y: 1.95, w: 4.3, h: 0.95, fontSize: 54, bold: true, fontFace: HEAD, color: K.ink, valign: "middle" });
  T(s, "nats per fixation: joint NLL gain of Phase 2 over Phase 1 (1.48%)", { x: 0.6, y: 2.92, w: 4.3, h: 0.55, fontSize: 14, color: K.slate });
  chip(s, 0.6, 3.55, "INCONCLUSIVE", "inc");
  T(s, [
    { text: "95% image-level interval [−0.0012, +0.0163]", options: { bullet: true, breakLine: true } },
    { text: "27 of 45 test images favour Phase 2", options: { bullet: true, breakLine: true } },
    { text: "Sign-flip test p = 0.1097", options: { bullet: true } },
  ], { x: 0.6, y: 4.0, w: 4.4, h: 1.15, fontSize: 14, paraSpaceAfter: 4 });
  box(s, 0.6, 5.3, 4.4, 1.5, K.card, null, "scale context");
  T(s, [
    { text: "For scale: ", options: { bold: true } },
    { text: "fine-tuning (Phase 1) moved test NLL from −0.1665 (pretrained) to −0.4957; the embedding then moved it to −0.5030.", options: {} },
  ], { x: 0.8, y: 5.4, w: 4.0, h: 1.3, fontSize: 13, valign: "middle" });

  // Native forest plot
  const lx = 5.45, px0 = 7.0, px1 = 10.35, vx = 10.55, min = -0.01, max = 0.02;
  const X = (v) => px0 + (v - min) / (max - min) * (px1 - px0);
  T(s, "Gain with 95% interval (positive favours Phase 2)", { x: lx, y: 1.92, w: 7.3, h: 0.3, fontSize: 14, bold: true });
  const rows = [
    ["Joint (primary)", 0.0073, -0.0012, 0.0163, "+0.0073 [−0.0012, +0.0163]"],
    ["Duration", 0.0019, 0.0002, 0.0035, "+0.0019 [+0.0002, +0.0035]"],
    ["Position", 0.0055, -0.0029, 0.0139, "+0.0055 [−0.0029, +0.0139]"],
    ["ASD subset", 0.0086, 0.0011, 0.0162, "+0.0086 [+0.0011, +0.0162]"],
    ["TD subset", 0.0064, -0.0058, 0.0189, "+0.0064 [−0.0058, +0.0189]"],
  ];
  const y0 = 2.65, dy = 0.58, yAxis = y0 + rows.length * dy - 0.1;
  arrow(s, X(0), 2.35, X(0), yAxis, "9AA5AE", 1, { noHead: true, dash: true, name: "zero line" });
  rows.forEach((r, i) => {
    const y = y0 + i * dy;
    T(s, r[0], { x: lx, y: y - 0.17, w: 1.5, h: 0.34, fontSize: 13, bold: i === 0, valign: "middle" });
    arrow(s, X(r[2]), y, X(r[3]), y, i === 0 ? K.ink : K.slate, i === 0 ? 2.5 : 1.75, { noHead: true, name: "interval " + r[0] });
    const excl = r[2] > 0;
    circle(s, X(r[1]), y, i === 0 ? 0.22 : 0.18, excl ? K.teal : "FFFFFF", excl ? null : K.ink, "estimate " + r[0]);
    T(s, r[4], { x: vx, y: y - 0.17, w: 2.2, h: 0.34, fontSize: 12, color: K.slate, valign: "middle" });
  });
  arrow(s, px0, yAxis, px1, yAxis, K.slate, 1, { noHead: true, name: "axis" });
  [-0.01, 0, 0.01, 0.02].forEach((v) => {
    arrow(s, X(v), yAxis, X(v), yAxis + 0.06, K.slate, 1, { noHead: true, name: "tick" });
    T(s, (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(2), { x: X(v) - 0.4, y: yAxis + 0.08, w: 0.8, h: 0.25, fontSize: 11, color: K.muted, align: "center" });
  });
  T(s, "nats per fixation", { x: px0, y: yAxis + 0.33, w: px1 - px0, h: 0.25, fontSize: 11, color: K.muted, align: "center" });
  circle(s, lx + 0.1, 6.32, 0.16, K.teal, null, "legend filled");
  T(s, "interval excludes zero (unadjusted, exploratory)", { x: lx + 0.25, y: 6.19, w: 3.4, h: 0.26, fontSize: 11, color: K.slate });
  circle(s, lx + 3.85, 6.32, 0.16, "FFFFFF", K.ink, "legend hollow");
  T(s, "interval includes zero", { x: lx + 4.0, y: 6.19, w: 2.0, h: 0.26, fontSize: 11, color: K.slate });
  T(s, "45 test images, 903 scanpaths; paired image-level cluster bootstrap (images are the only available cluster).",
    { x: lx, y: 6.5, w: 7.3, h: 0.3, fontSize: 11, color: K.muted });
  s.addNotes(
`Suggested time: about 50 s.
On the 45 held-out Saliency4ASD test images, Phase 2 lowers joint NLL by 0.0073 nats per fixation, about 1.5%. But the 95% image-level interval runs from -0.0012 to +0.0163, so it includes zero; 27 of 45 images favour Phase 2 and the sign-flip test gives p = 0.11. We therefore call the gain inconclusive, not an improvement.
The duration component and the ASD subset clear zero without correction; with several related comparisons we treat them as exploratory.
For scale: fine-tuning itself (Phase 1) improved test NLL from -0.1665 to -0.4957. The embedding's contribution is small by comparison.`);
}

// 10. External likelihood
{
  const s = content("Results", "RESULTS · EXTERNAL (GAZE4ASD)", "External likelihood: Phase 2 fits worse",
    "On Gaze4ASD the unconditioned model fits better, in every preprocessing version.");
  s.addImage({ path: A + "fig3_likelihood_gains.png", x: 0.6, y: 1.9, w: 7.0, h: 7.0 * 672 / 1566, altText: "Manuscript Figure 3: Phase 2 minus Phase 1 NLL gain with 95% intervals, in-domain and on Gaze4ASD under three preprocessing versions", objectName: "Figure 3" });
  T(s, "Manuscript Figure 3. Phase 2 − Phase 1 NLL gain, 95% intervals; positive favours Phase 2. Black: in-domain (image clusters). Colours: Gaze4ASD under three preprocessing versions (participant clusters).",
    { x: 0.6, y: 4.95, w: 7.0, h: 0.6, fontSize: 11, color: K.muted });
  box(s, 0.6, 5.68, 7.0, 1.12, K.amberTint, null, "mapping note");
  T(s, [
    { text: "Coordinate mapping. ", options: { bold: true, color: K.amber } },
    { text: "Fit-to-height placement is strongly supported (authors' saliency maps match at median r = 0.9999 over 4,533 maps, vs 0.91 stretch, 0.23 native) but not confirmed from the original presentation code.", options: {} },
  ], { x: 0.8, y: 5.73, w: 6.6, h: 1.02, fontSize: 12, valign: "middle" });

  const blk = (y, label, kind, head, value, text) => {
    const w = chip(s, 7.95, y, label, kind);
    T(s, head, { x: 7.95 + w + 0.15, y, w: 4.7 - w - 0.15, h: 0.3, fontSize: 14, bold: true, valign: "middle" });
    T(s, value, { x: 7.95, y: y + 0.38, w: 1.75, h: 0.6, fontSize: 26, bold: true, fontFace: HEAD, color: kind === "neg" ? K.neg : K.amber, valign: "middle" });
    T(s, text, { x: 9.75, y: y + 0.36, w: 2.98, h: 1.05, fontSize: 13, color: K.slate });
  };
  blk(1.95, "WORSE", "neg", "Overall", "−0.0172", "Phase 1 −0.5568 vs Phase 2 −0.5396. Interval excludes zero by participants, images and both.");
  blk(3.5, "WORSE", "neg", "TD trials", "−0.0229", "Worse under Phase 2 in all three preprocessing versions.");
  blk(5.05, "FRAGILE", "inc", "ASD trials", "+0.0104", "Clears zero by participants only (17 of 30 images); +0.0027, interval includes zero, under the stretch mapping.");
  T(s, "Post hoc label swap: the ASD embedding also beats Phase 1 on TD trials (+0.0189), so the ASD gain is not ASD-specific.",
    { x: 7.95, y: 6.25, w: 4.78, h: 0.6, fontSize: 12, color: K.muted });
  s.addNotes(
`Suggested time: about 65 s.
Externally the answer is clearly negative. On identical Gaze4ASD trials and batches, Phase 1 fits better overall by 0.0172 nats per fixation, and the interval excludes zero whether we resample participants, images or both. TD trials are worse under conditioning.
ASD trials improve slightly, +0.0104, but only at participant level, only 17 of 30 images favour Phase 2, and under the alternative stretch mapping the gain shrinks to +0.0027 with an interval including zero. A post hoc label swap shows the ASD embedding also helps TD trials, so this is not an ASD-specific effect.
The figure shows that the overall and TD results hold under all three preprocessing versions.
Mapping caveat: the release does not record where images were drawn on screen. Fit-to-height is strongly supported by the authors' own saliency maps but not confirmed.
Why the TD fit worsens is untested; age, tracker, display, stimuli and site all differ between datasets.`);
}

// 11. E1 contrasts
{
  const s = content("Results", "RESULTS · GENERATED BEHAVIOUR (E1, τ = 1)", "Generated contrasts: partly right, partly wrong",
    "The embedding shifts behaviour: spatial contrasts follow real data, temporal ones do not.");
  s.addImage({ path: A + "fig5_e1_contrast_shares.png", x: 0.6, y: 1.9, w: 7.2, h: 7.2 * 678 / 1525, altText: "Manuscript Figure 5: generated ASD minus TD contrast as a fraction of the real contrast on Gaze4ASD", objectName: "Figure 5" });
  T(s, "Manuscript Figure 5. Generated ASD−TD contrast ÷ real contrast (Gaze4ASD, 30 images). Markers: mean over 10 seeds; lines: single-seed range. 1 = real size; below 0 = wrong direction.",
    { x: 0.6, y: 5.15, w: 7.2, h: 0.55, fontSize: 11, color: K.muted });
  box(s, 0.6, 5.8, 7.2, 1.0, K.card, null, "real effects");
  T(s, [
    { text: "Real Gaze4ASD (33 ASD vs 133 TD, Hedges' g): ", options: { bold: true } },
    { text: "ASD children show larger saccades (+0.65), greater dispersion (+0.80), fewer revisits (−1.04), more concentrated directions (+0.55) and less total fixation time (−0.71).", options: {} },
  ], { x: 0.8, y: 5.85, w: 6.8, h: 0.9, fontSize: 12, valign: "middle" });

  let w = chip(s, 8.15, 1.95, "RIGHT DIRECTION, SMALLER", "part");
  T(s, "share of the real contrast", { x: 8.15 + w + 0.12, y: 1.95, w: 4.58 - w - 0.12, h: 0.3, fontSize: 12, color: K.muted, valign: "middle" });
  [["≈69%", "Spatial dispersion"], ["≈64%", "Saccade amplitude"], ["≈32%", "Revisit rate"]].forEach((r, i) => {
    const y = 2.38 + i * 0.52;
    T(s, r[0], { x: 8.15, y, w: 1.25, h: 0.48, fontSize: 24, bold: true, fontFace: HEAD, color: K.part, valign: "middle" });
    T(s, r[1], { x: 9.45, y, w: 3.28, h: 0.48, fontSize: 15, valign: "middle" });
  });
  chip(s, 8.15, 4.05, "WRONG DIRECTION", "neg");
  T(s, [
    { text: "Fixation count · total fixation time · direction concentration", options: { bold: true, breakLine: true } },
    { text: "The ASD label gives more fixations and more total time than the TD label; real data show the opposite.", options: { color: K.slate } },
  ], { x: 8.15, y: 4.45, w: 4.58, h: 1.15, fontSize: 13 });
  box(s, 8.15, 5.7, 4.58, 1.1, K.card, null, "size caveat");
  T(s, "8 of 9 measures exceed the seed-only null (Holm p ≤ 0.0009), but the shifts are small: 1.1% (dispersion) and 0.8% (amplitude) of the image diagonal.",
    { x: 8.3, y: 5.75, w: 4.3, h: 1.0, fontSize: 12, color: K.ink, valign: "middle" });
  s.addNotes(
`Suggested time: about 65 s.
First, the real data at the bottom: autistic children in Gaze4ASD make larger saccades, spread their fixations more, revisit less, have more concentrated saccade directions and spend less total time fixating.
E1 shows the embedding does change generated behaviour beyond seed noise: 8 of 9 measures shift beyond the Phase 1 seed-only null. The three primary spatial contrasts have the real direction, at roughly 69, 64 and 32 percent of the real size; these shares are approximate and vary by several points between independent runs (an earlier run gave 60, 50 and 29 percent).
But fixation count, total time and direction concentration go the wrong way: the ASD label produces more fixations and more time. Note the real count difference itself is not significant.
Keep size in perspective: highly significant under common random numbers, yet only about 1 percent of the image diagonal.`);
}

// 12. Per-group realism
{
  const s = content("Results", "RESULTS · PER-GROUP REALISM (E1, τ = 1)", "Per-group realism does not improve",
    "Correct contrasts arise because the conditions move apart, not toward their real groups.");
  // Dot plot: spatial dispersion
  T(s, "Spatial dispersion (÷ image diagonal), Gaze4ASD", { x: 0.6, y: 1.9, w: 5.9, h: 0.3, fontSize: 14, bold: true });
  const ax0 = 2.0, ax1 = 6.3, vmin = 0.098, vmax = 0.116;
  const X = (v) => ax0 + (v - vmin) / (vmax - vmin) * (ax1 - ax0);
  const yA = 2.75, yT = 3.5, yAx = 3.95;
  T(s, "ASD condition", { x: 0.6, y: yA - 0.17, w: 1.35, h: 0.34, fontSize: 13, bold: true, color: K.orangeText, valign: "middle" });
  T(s, "TD condition", { x: 0.6, y: yT - 0.17, w: 1.35, h: 0.34, fontSize: 13, bold: true, color: K.blue, valign: "middle" });
  circle(s, X(0.113), yA, 0.26, "FFFFFF", K.slate, "ASD Phase 1");
  circle(s, X(0.113), yA, 0.14, K.orange, null, "ASD Phase 2");
  T(s, "Phase 1 = Phase 2 = 0.113", { x: X(0.113) - 2.6, y: yA - 0.17, w: 2.4, h: 0.34, fontSize: 12, color: K.slate, align: "right", valign: "middle" });
  arrow(s, X(0.112) - 0.14, yT, X(0.101) + 0.1, yT, K.blue, 1.75, { name: "TD contraction" });
  circle(s, X(0.112), yT, 0.26, "FFFFFF", K.slate, "TD Phase 1");
  circle(s, X(0.101), yT, 0.18, K.blue, null, "TD Phase 2");
  arrow(s, X(0.108), yT - 0.25, X(0.108), yT + 0.25, K.ink, 2.5, { noHead: true, name: "real TD" });
  T(s, "Phase 1 0.112", { x: X(0.112) - 0.6, y: yT + 0.17, w: 1.2, h: 0.26, fontSize: 11, color: K.slate, align: "center" });
  T(s, "Phase 2 0.101", { x: X(0.101) - 0.6, y: yT + 0.17, w: 1.2, h: 0.26, fontSize: 11, color: K.blue, align: "center" });
  T(s, "real TD 0.108", { x: X(0.108) - 0.6, y: yT - 0.5, w: 1.2, h: 0.24, fontSize: 11, bold: true, align: "center" });
  arrow(s, ax0, yAx, ax1, yAx, K.slate, 1, { noHead: true, name: "axis" });
  [0.100, 0.105, 0.110, 0.115].forEach((v) => {
    arrow(s, X(v), yAx, X(v), yAx + 0.06, K.slate, 1, { noHead: true, name: "tick" });
    T(s, v.toFixed(3), { x: X(v) - 0.4, y: yAx + 0.08, w: 0.8, h: 0.24, fontSize: 11, color: K.muted, align: "center" });
  });
  T(s, "Hollow: Phase 1 · filled: Phase 2 · bar: real TD. TD contracts past real TD; ASD does not move.", { x: 0.6, y: 4.32, w: 5.9, h: 0.3, fontSize: 11, color: K.muted });

  // Native chart: over-production
  s.addChart(pres.charts.BAR, [
    { name: "Real", labels: ["ASD", "TD"], values: [6.25, 6.41] },
    { name: "Generated (Phase 2, τ = 1)", labels: ["ASD", "TD"], values: [10.19, 9.60] },
  ], {
    x: 0.6, y: 4.7, w: 5.9, h: 2.15, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: ["9AA5AE", K.teal], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
    dataLabelFontSize: 11, dataLabelColor: K.ink, dataLabelFontFace: BODY,
    showTitle: true, title: "Fixations per scanpath (Gaze4ASD)", titleFontSize: 12, titleColor: K.ink, titleFontFace: BODY,
    showLegend: true, legendPos: "r", legendFontSize: 11, legendFontFace: BODY, legendColor: K.slate,
    valAxisMinVal: 0, valAxisMaxVal: 12, valAxisMajorUnit: 4, valAxisLabelFontSize: 10, valAxisLabelColor: K.muted, valAxisLabelFontFace: BODY,
    catAxisLabelFontSize: 12, catAxisLabelColor: K.ink, catAxisLabelFontFace: BODY,
    valGridLine: { color: "E3E8EC", size: 0.75 }, catGridLine: { style: "none" }, objectName: "chart fixation count",
  });

  // Table 8 excerpt
  const g = (t) => ({ text: t, options: { color: K.pos, bold: true } });
  const r = (t) => ({ text: t, options: { color: K.neg, bold: true } });
  const n = (t) => ({ text: t, options: { color: K.muted } });
  const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: K.slate } } });
  const rows = [
    [hdr("Measure"), hdr("ASD: change in error"), hdr("TD: change in error")],
    ["Spatial dispersion", n("+0.7% (n.s.)"), n("−6.5% (n.s.)")],
    ["Saccade amplitude", n("−3.2% (n.s.)"), n("−9.1% (n.s.)")],
    ["Revisit rate", r("−11.8% (p = 0.0140)"), n("+13.0% (n.s.)")],
    ["Fixation count", r("−8.5% (p = 0.0018)"), g("+10.0% (p = 0.0018)")],
    ["Fixation rate", r("−28.1% (p = 0.0018)"), g("+22.6% (p = 0.0018)")],
    ["Mean fixation duration", r("−32.5% (p = 0.0018)"), g("+28.2% (p = 0.0121)")],
  ];
  T(s, "Phase 2 vs Phase 1: error against the matching real group", { x: 6.9, y: 1.9, w: 5.83, h: 0.3, fontSize: 14, bold: true });
  s.addTable(rows, { x: 6.9, y: 2.3, w: 5.83, colW: [2.05, 1.89, 1.89], fontFace: BODY, fontSize: 12, color: K.ink, valign: "middle",
    border: { type: "solid", pt: 0.75, color: K.rule }, rowH: 0.42, margin: [0.03, 0.08, 0.03, 0.08] });
  T(s, "Per-image absolute error, Phase 2 vs Phase 1. Positive = Phase 2 closer to the real group; negative = further away. Holm p over 18 tests; n.s. = Holm p > 0.05.",
    { x: 6.9, y: 5.32, w: 5.83, h: 0.5, fontSize: 11, color: K.muted });
  box(s, 6.9, 5.9, 5.83, 0.95, K.negTint, null, "realism summary");
  T(s, [
    { text: "No spatial measure improves for either group. ", options: { bold: true, color: K.neg } },
    { text: "Timing moves TD toward real TD and ASD away from real ASD.", options: {} },
  ], { x: 7.05, y: 5.95, w: 5.55, h: 0.85, fontSize: 13, valign: "middle" });
  s.addNotes(
`Suggested time: about 60 s.
A correct contrast does not mean either group became more realistic. Table on the right: compared with Phase 1, no spatial measure gets significantly closer to its real group for either label, and ASD revisit rate gets worse. In timing, Phase 2 moves TD toward real TD but ASD away from real ASD.
Top left shows why the dispersion contrast is nevertheless correct: the ASD condition does not move at all (0.113 in both models), while the TD condition contracts from 0.112 to 0.101, past the real TD value of 0.108. The two conditions move apart; neither moves toward its own group.
Bottom left: both labels over-produce fixations, about 10 per scanpath against about 6 real, because the sampler has no stopping model and fills the 3 s window. The conditioning shift in count (+0.66) is about a fifth of this shared excess.
MultiMatch tells the same story (real ASD scanpaths are slightly closer to TD-conditioned samples), but all those effects are below 0.01 on a 0 to 1 scale.`);
}

// 13. Temperature
{
  const s = content("Results", "RESULTS · SAMPLING TEMPERATURE", "Sampling temperature: behaviour, not likelihood",
    "τ changes how far apart generated groups are, but never the likelihood.");
  box(s, 0.6, 1.95, 3.85, 4.9, K.card, null, "how tau works");
  T(s, "How τ works", { x: 0.8, y: 2.08, w: 3.5, h: 0.35, fontSize: 16, bold: true, fontFace: HEAD });
  T(s, [{ text: "σ" }, sub("j"), { text: "(τ) = σ" }, sub("j"), sup("τ")], { x: 0.8, y: 2.5, w: 3.5, h: 0.6, fontSize: 26, fontFace: HEAD, color: K.teal, valign: "middle" });
  T(s, [
    { text: "Only the spatial component scales change; means, weights and durations are untouched.", options: { bullet: true, breakLine: true } },
    { text: "Every σ < 1, so τ = 2 narrows the components.", options: { bullet: true, breakLine: true } },
    { text: "τ = 1 reproduces the model's own distribution; chosen on validation images.", options: { bullet: true, breakLine: true } },
    { text: "NLL is computed without temperature.", options: { bullet: true } },
  ], { x: 0.8, y: 3.2, w: 3.5, h: 3.5, fontSize: 13, paraSpaceAfter: 6 });

  const common = {
    barDir: "col", showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: K.ink, dataLabelFontFace: BODY,
    showTitle: true, titleFontSize: 12, titleColor: K.ink, titleFontFace: BODY,
    valAxisMinVal: 0, valAxisMaxVal: 80, valAxisMajorUnit: 20, valAxisLabelFontSize: 10, valAxisLabelColor: K.muted, valAxisLabelFontFace: BODY,
    catAxisLabelFontSize: 11, catAxisLabelColor: K.ink, catAxisLabelFontFace: BODY,
    valGridLine: { color: "E3E8EC", size: 0.75 }, catGridLine: { style: "none" },
  };
  s.addChart(pres.charts.BAR, [
    { name: "Error reduction", labels: ["Amplitude, ASD", "Amplitude, TD", "Revisit, ASD", "Revisit, TD"], values: [52.5, 49.9, 66.1, 56.5] },
  ], Object.assign({}, common, {
    x: 4.7, y: 1.95, w: 4.0, h: 3.55, chartColors: [K.teal], showLegend: false, dataLabelFormatCode: '0.0"%"', barGapWidthPct: 70,
    title: "Per-image error reduction, τ = 1 vs τ = 2 (%)", valGridLine: { color: "E3E8EC", size: 0.75 }, catGridLine: { style: "none" }, objectName: "chart tau error",
  }));
  s.addChart(pres.charts.BAR, [
    { name: "τ = 1 (E1)", labels: ["Dispersion", "Amplitude", "Revisit rate"], values: [69, 64, 32] },
    { name: "τ = 2", labels: ["Dispersion", "Amplitude", "Revisit rate"], values: [46, 33, 15] },
  ], Object.assign({}, common, {
    x: 8.85, y: 1.95, w: 3.88, h: 3.55, barGrouping: "clustered", chartColors: [K.teal, "8CC7C7"], dataLabelFormatCode: '0"%"', barGapWidthPct: 60,
    showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: BODY, legendColor: K.slate,
    title: "Generated ASD−TD contrast, % of real", valGridLine: { color: "E3E8EC", size: 0.75 }, catGridLine: { style: "none" }, objectName: "chart tau shares",
  }));
  box(s, 4.7, 5.65, 8.03, 1.2, K.tealTint, null, "tau takeaway");
  T(s, [
    { text: "All four prespecified tests favour τ = 1 in 10 of 10 seeds (Holm p ≤ 0.0005; exploratory). ", options: {} },
    { text: "Every Phase 2 contrast keeps its direction at τ = 2, but shrinks. ", options: { bold: true } },
    { text: "Only at τ = 2 did Phase 2 match real TD significantly worse on spatial measures.", options: {} },
  ], { x: 4.9, y: 5.72, w: 7.65, h: 1.06, fontSize: 13, valign: "middle" });
  s.addNotes(
`Suggested time: about 50 s.
Temperature only rescales the spatial mixture components: sigma to the power tau. Because every sigma is below 1, tau = 2 makes components narrower. Likelihood is always computed without temperature, so tau never affects NLL; it only affects generated behaviour.
We chose tau = 1 on validation images; it is also the model's own distribution. On Gaze4ASD, tau = 1 has lower error on all four prespecified tests, in 10 of 10 seeds. Those tests were specified before generation but tau = 2 results had already been seen, so this is exploratory.
Right chart: at tau = 2 every generated contrast keeps its direction but is smaller: 46, 33 and 15 percent instead of 69, 64 and 32. So the size of generated group differences depends on a sampling setting that likelihood cannot see. Always state tau when comparing generated scanpaths.`);
}

// 14. Demo
pres.addSection({ title: "Demonstration" });
{
  const s = content("Demonstration", "DEMONSTRATION · QUALITATIVE EXAMPLES", "Demo: same image, same seed, three models",
    "Each panel is one stochastic sample, not a child and not a group profile.");
  const sw = 8.4, h1 = sw * 544 / 2232, h2 = sw * 485 / 2232;
  // column headers (panel centres at 16.3%, 49.9% and 83.4% of the strip width)
  [["Phase 1 (unconditioned)", 0.163, K.ink], ["Phase 2, ASD label", 0.499, K.orangeText], ["Phase 2, TD label", 0.834, K.blue]].forEach((c) => {
    T(s, c[0], { x: 0.6 + c[1] * sw - 1.4, y: 1.8, w: 2.8, h: 0.28, fontSize: 13, bold: true, color: c[2], align: "center", valign: "middle" });
  });
  T(s, [
    { text: "BEST MATCH · test image 102  ", options: { bold: true, color: K.pos } },
    { text: "real ASD−TD dispersion contrast +0.0133; generated +0.014", options: { color: K.slate } },
  ], { x: 0.6, y: 2.12, w: sw, h: 0.28, fontSize: 13, valign: "middle" });
  s.addImage({ path: A + "demo_best_102_panels.png", x: 0.6, y: 2.42, w: sw, h: h1, altText: "Generated scanpaths for Saliency4ASD test image 102 on a blank canvas: Phase 1, Phase 2 ASD label, Phase 2 TD label", objectName: "demo best 102" });
  const y2 = 2.42 + h1 + 0.14;
  T(s, [
    { text: "WORST MATCH · test image 116  ", options: { bold: true, color: K.neg } },
    { text: "real contrast −0.0473; generated +0.0151 (opposite sign)", options: { color: K.slate } },
  ], { x: 0.6, y: y2, w: sw, h: 0.28, fontSize: 13, valign: "middle" });
  s.addImage({ path: A + "demo_worst_116_panels.png", x: 0.6, y: y2 + 0.3, w: sw, h: h2, altText: "Generated scanpaths for Saliency4ASD test image 116 on a blank canvas: Phase 1, Phase 2 ASD label, Phase 2 TD label", objectName: "demo worst 116" });

  const rx = 9.3, rw = 3.43;
  T(s, "How to read a panel", { x: rx, y: 1.82, w: rw, h: 0.3, fontSize: 14, bold: true });
  const leg = [["4CAF50", "first fixation"], ["E57373", "last fixation"], ["F2C27B", "circle area grows with duration"]];
  leg.forEach((l, i) => { circle(s, rx + 0.15, 2.36 + i * 0.32, 0.2, l[0], "6B6B6B", "legend " + l[1]); T(s, l[1], { x: rx + 0.35, y: 2.23 + i * 0.32, w: rw - 0.35, h: 0.27, fontSize: 12, valign: "middle" }); });
  arrow(s, rx + 0.02, 3.32, rx + 0.3, 3.32, "C9A800", 2.25, { name: "legend arrow" });
  T(s, "numbers and arrows give the order", { x: rx + 0.35, y: 3.19, w: rw - 0.35, h: 0.27, fontSize: 12, valign: "middle" });

  box(s, rx, 3.6, rw, 1.65, K.amberTint, null, "one sample caution");
  T(s, [
    { text: "One sample, one seed. ", options: { bold: true, color: K.amber } },
    { text: "All three panels share seed 0 and τ = 1, so they differ only through the embedding. Changing the seed changes a panel more than switching ASD and TD.", options: {} },
  ], { x: rx + 0.15, y: 3.66, w: rw - 0.3, h: 1.53, fontSize: 12, valign: "middle" });
  T(s, "Images chosen by a rule fixed in advance from saved E1 results. The generated contrast is positive on all 45 test images, while the real contrast ranges from −0.0473 to +0.0576.",
    { x: rx, y: 5.38, w: rw, h: 0.95, fontSize: 12, color: K.slate });
  T(s, "Stimulus photographs are not shown (data terms pending); scanpaths are drawn on a blank canvas of the image size.",
    { x: rx, y: 6.33, w: rw, h: 0.55, fontSize: 11, color: K.muted, italic: true });
  s.addNotes(
`Suggested time: about 55 s. Optionally run the live demo here (about 20 s):
python inference.py --image TrainingData/TrainingData/Images/60.png --compare-phase1 --out-dir demo/local_only/live
(Show the live output only on this machine; it overlays a dataset photograph.)
Each row shows one Saliency4ASD test image: Phase 1, Phase 2 with the ASD label, and Phase 2 with the TD label, all with the same seed. Because they share random numbers, the panels look almost identical: the embedding is a small, systematic nudge, not a different behaviour.
The two images were chosen by a rule fixed before generating any demo output. Image 102 is where the model's average ASD-minus-TD dispersion contrast best matches the real one; image 116 is the worst: the real contrast is negative, but the model's is positive. In fact the generated contrast is positive on all 45 test images, while the real one changes sign: the shift is roughly uniform rather than image-specific.
Stress: one generated scanpath is one stochastic sample. It does not represent an individual child, and it is not diagnostic.`);
}

// 15. Limitations
pres.addSection({ title: "Discussion" });
{
  const s = content("Discussion", "DISCUSSION · LIMITATIONS", "Limitations and threats to validity",
    "Several limitations cannot be resolved with the released data.");
  const L = [
    ["Conditioning", "In-domain gain inconclusive; external fit worse; small ASD gain depends on the mapping."],
    ["Generated behaviour", "Fixations over-produced; count, total time and direction contrasts wrong; no spatial realism gain."],
    ["Coordinate mapping", "Gaze4ASD display geometry strongly supported but not confirmed by the authors."],
    ["Off-image gaze", "ASD children look off-image ≈2.7× as often as TD; neither clipping nor removal is group-neutral."],
    ["Age and site", "No participant-level age; 5–12 vs 3–7 years; tracker, display, stimuli and site differ."],
    ["Data and sample", "Gaze4ASD stimuli selected for group differences, mostly faces; 33 ASD vs 133 TD."],
    ["Statistics", "Exploratory, not preregistered; Holm within families only; validation chose checkpoints and τ."],
    ["Implementation", "Released transformer's batch-layout issue kept; MultiMatch uses a mean-based variant."],
    ["Clinical validity", "None established. This is not a diagnostic or screening tool."],
  ];
  L.forEach((l, i) => {
    const x = 0.6 + (i % 3) * 4.11, y = 1.88 + Math.floor(i / 3) * 1.68;
    const last = i === L.length - 1;
    box(s, x, y, 3.91, 1.52, last ? K.negTint : K.card, null, "limitation " + l[0]);
    fix(s, x + 0.33, y + 0.33, 0.38, i + 1, last ? K.neg : K.slate, 12);
    T(s, l[0], { x: x + 0.62, y: y + 0.14, w: 3.15, h: 0.38, fontSize: 15, bold: true, fontFace: HEAD, valign: "middle", color: last ? K.neg : K.ink });
    T(s, l[1], { x: x + 0.2, y: y + 0.6, w: 3.55, h: 0.86, fontSize: 13, color: K.slate });
  });
  s.addNotes(
`Suggested time: about 45 s.
The main threats: the conditioning results themselves are inconclusive or negative; the generator over-produces fixations and gets the temporal contrasts wrong.
For Gaze4ASD, the screen-to-image mapping is strongly supported but unconfirmed, and off-image gaze is itself a group difference: autistic children looked outside the image about 2.7 times as often, so both of our filtering policies affect the groups differently.
Age is a potential confound within Gaze4ASD and across datasets, and cannot be adjusted for because participant-level age is not released.
The Gaze4ASD stimuli were selected for group differences and may have been selected using Saliency4ASD.
All analyses are exploratory. The implementation keeps a disclosed batch-layout issue from the released code. And there is no clinical validity: this is not a diagnostic or screening tool.`);
}

// 16. Conclusions (dark)
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Discussion" });
  s.addText("DISCUSSION · CONCLUSIONS", { placeholder: "kicker" });
  s.addText("Conclusions and future work", { placeholder: "title" });
  const F = [
    ["INCONCLUSIVE", "inc", "In-domain likelihood: small gain (+0.0073 nats per fixation); interval includes zero."],
    ["NEGATIVE", "neg", "External likelihood: worse overall (−0.5568 vs −0.5396), in every preprocessing version."],
    ["PARTIAL", "part", "Generated spatial contrasts follow the real direction at reduced size (≈69 / 64 / 32%)."],
    ["NEGATIVE", "neg", "Count, total time and direction contrasts wrong; neither group becomes more realistic."],
    ["CAVEAT", "note", "Temperature changes the size of generated differences, not the likelihood."],
  ];
  F.forEach((f, i) => {
    const y = 1.6 + i * 0.74;
    chip(s, 0.6, y + 0.08, f[0], f[1]);
    T(s, f[2], { x: 2.35, y, w: 5.6, h: 0.62, fontSize: 14, color: "FFFFFF", valign: "middle" });
  });
  box(s, 0.6, 5.45, 7.35, 1.35, K.dark2, null, "contribution");
  T(s, [
    { text: "Contribution. ", options: { bold: true, color: K.tealLight } },
    { text: "A controlled, transparently reported test of minimal group conditioning: likelihood in-domain and externally, a common-random-numbers behavioural comparison, and per-group realism reported alongside group contrasts.", options: { color: "E6ECEF" } },
  ], { x: 0.8, y: 5.5, w: 6.95, h: 1.25, fontSize: 14, valign: "middle" });
  box(s, 8.35, 1.6, 4.38, 5.2, K.dark2, null, "future work");
  T(s, "Future work", { x: 8.6, y: 1.75, w: 3.9, h: 0.4, fontSize: 18, bold: true, fontFace: HEAD, color: "FFFFFF" });
  T(s, [
    "Confirm the display mapping and participant count with the Gaze4ASD authors",
    "Obtain participant-level age to test age effects",
    "Model viewing time, stopping and off-image gaze",
    "Compare richer conditioning against the same control and per-group realism criteria",
    "Replicate on independent data with preregistered tests",
  ].map((t, j, a) => ({ text: t, options: { bullet: true, breakLine: j < a.length - 1 } })),
  { x: 8.6, y: 2.25, w: 3.95, h: 4.4, fontSize: 14, color: "E6ECEF", paraSpaceAfter: 8 });
  s.addNotes(
`Suggested time: about 45 s.
To summarize: a 1,024-parameter group embedding gives a small, inconclusive in-domain gain and a worse external fit, robustly across preprocessing versions. Generated scanpaths reproduce the direction of the main real spatial differences at reduced size, get the temporal and direction-concentration differences wrong, and do not become more realistic for either group. Temperature changes the size of generated differences but not the likelihood.
The contribution is methodological and honest: a controlled comparison with a matched unconditioned control, external testing, and the observation that a correct group contrast is weak evidence of group-specific realism.
Next steps: confirm the mapping and participant count with the dataset authors, obtain age, model stopping and off-image gaze, and replicate with preregistered tests.
Code is public at github.com/Anish-R-18/ASD-ScanPath; checkpoints are not redistributed.`);
}

// 17. References (backup)
pres.addSection({ title: "Backup" });
{
  const s = content("Backup", "BACKUP", "References", null);
  const R = [
    "[1] Frazier et al. A meta-analysis of gaze differences to social and nonsocial information between individuals with and without autism. J. Am. Acad. Child Adolesc. Psychiatry, 2017.",
    "[2] Guillon et al. Visual social attention in autism spectrum disorder: insights from eye tracking studies. Neurosci. Biobehav. Rev., 2014.",
    "[3] D'Amelio et al. TPP-Gaze: modelling gaze dynamics in space and time with neural temporal point processes. WACV, 2025.",
    "[4] Duan et al. A dataset of eye movements for the children with autism spectrum disorder. ACM MMSys, 2019.",
    "[5] Gutiérrez et al. Saliency4ASD: challenge, dataset and tools for visual attention modeling for ASD. Signal Process. Image Commun., 2021.",
    "[6] Yang et al. Gaze4ASD: a novel dataset and visual saliency map-based method for autism screening. IEEE ICME, 2025.",
    "[7] Itti, Koch, Niebur. A model of saliency-based visual attention for rapid scene analysis. IEEE TPAMI, 1998.",
    "[8] Kümmerer, Bethge, Wallis. DeepGaze III: modeling free-viewing human scanpaths with deep learning. J. Vision, 2022.",
    "[9] Zhong et al. SpFormer: spatio-temporal modeling for scanpaths with transformer. AAAI, 2024.",
    "[10] Jarodzka, Holmqvist, Nyström. A vector-based, multidimensional scanpath similarity measure. ETRA, 2010.",
    "[11] Dewhurst et al. Scanpath comparison in multiple dimensions with MultiMatch. Behav. Res. Methods, 2012.",
    "[12] Wagner, Halchenko, Hanke. multimatch-gaze: the MultiMatch algorithm for gaze path comparison in Python. JOSS, 2019.",
    "[13] Du et al. Recurrent marked temporal point processes: embedding event history to vector. ACM SIGKDD, 2016.",
    "[14] Shchur et al. Neural temporal point processes: a review. IJCAI, 2021.",
    "[15] Zuo et al. Transformer Hawkes process. ICML, 2020.",
    "[16] Shchur, Biloš, Günnemann. Intensity-free learning of temporal point processes. ICLR, 2020.",
    "[17] Liaqat et al. Predicting ASD diagnosis in children with synthetic and image-based eye gaze data. Signal Process. Image Commun., 2021.",
    "[18] Zhong et al. Semantic-based saccadic scanpath prediction for autism spectrum disorder. Pattern Recognition, 2026.",
    "[19] Huang et al. Densely connected convolutional networks. CVPR, 2017.",
    "[20] Liu et al. An intriguing failing of convolutional neural networks and the CoordConv solution. NeurIPS, 2018.",
    "[21] Judd et al. Learning to predict where humans look. ICCV, 2009.",
    "[22] Loshchilov, Hutter. Decoupled weight decay regularization. ICLR, 2019.",
    "[23] Hedges. Distribution theory for Glass's estimator of effect size and related estimators. J. Educ. Stat., 1981.",
    "[24] Holm. A simple sequentially rejective multiple test procedure. Scand. J. Stat., 1979.",
  ];
  const col = (items, x) => T(s, items.map((t, j) => ({ text: t, options: { breakLine: j < items.length - 1 } })),
    { x, y: 1.5, w: 5.9, h: 5.4, fontSize: 10.5, color: K.ink, paraSpaceAfter: 4.5 });
  col(R.slice(0, 12), 0.6);
  col(R.slice(12), 6.83);
  s.addNotes("Backup slide. Numbering follows the manuscript (paper/final/ASD_ScanPath_Final.md); full bibliographic details and DOIs are in the manuscript reference list and paper/final/references.bib.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
