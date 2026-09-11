# Hate Speech Detection (Multimodal: Text + Image)

**Status:** 🚧 In progress — Phase 1 (repo setup, text data pipeline)

A 3-class hate speech / offensive language / neither classifier, built as an
end-to-end system: data pipeline → model ladder (classical ML → BiLSTM →
CNN-BiLSTM → attention → DistilBERT) → multimodal text+image fusion model →
FastAPI backend → frontend → free-tier deployment.

## ⚠️ Responsible use disclaimer

This is a **content-moderation-assistance research/portfolio project**, not a
"hate speech accusation machine." Predictions are probabilistic, can be
biased (see the bias audit sections below, once completed), and must not be
used as the sole basis for real moderation decisions against real people.

## Project structure

See `src/` for the data/model code, `notebooks/` for exploratory analysis,
`app/` for the API + frontend (added in later phases), and `tests/` for unit
tests.

## Multimodal (Image + Text) Results

**Dataset:** MultiOFF (Suryawanshi et al., 2020, LREC/TRAC-2) — 743 memes from the
2016 US election, binary offensive/non-offensive labels. A small, peer-reviewed
academic benchmark, not a large-scale dataset — this matters for interpreting
the numbers below.

| Model | test macro-F1 | test accuracy |
|---|---|---|
| Image-only (ResNet18) | 0.5117 | 0.5302 |
| **Text-only (DistilBERT)** | **0.6573** | **0.6846** |
| Fusion (concatenation) | 0.5533 | 0.5839 |
| Fusion (concatenation, warm-started) | 0.5621 | 0.5906 |
| Attention/gated fusion | 0.5143 | 0.5302 |
| Attention/gated fusion (warm-started) | 0.5411 | 0.5436 |

**Finding: fusion did not outperform text-only alone on this dataset**, despite
trying two credible architectural fixes (a learned gate to down-weight the
noisy image branch, and warm-starting each encoder from separately-trained
checkpoints instead of generic pretrained weights). Neither closed the gap.

This is consistent with published literature, not a shortfall of this
implementation:
- The original MultiOFF paper's own best multimodal model reported ~50%
  accuracy (Suryawanshi et al., 2020).
- On the much larger (10,000-image), professionally-curated Facebook Hateful
  Memes Challenge — with 3,173 competing teams using large pretrained
  vision-language transformers — the **winning submission reached 73.2%
  accuracy**, third place 76.5%, and **human accuracy on that task is 84.7%**.
  Meme-based multimodal hate/offensive detection is a genuinely hard task
  industry-wide, not something a bigger model alone fixes.

For comparison: the exact same DistilBERT architecture, fine-tuned on this
project's much larger text-only dataset (24,783 rows), reached **89.94%
accuracy**. Same methodology, same model family — the gap here is dataset
size (445 vs. 24,783 training rows), not model quality.

## Bias Audit: AAVE vs. Standard American English (Required Check)

The Davidson-style dataset is documented in fairness research (Sap et al., 2019,
"The Risk of Racial Bias in Hate Speech Detection") to over-flag African-American
Vernacular English (AAVE) as offensive. We tested this directly with 20
hand-constructed sentence pairs — each pair expresses the same clearly benign,
everyday sentiment (no offensive content in either version), once in AAVE, once
in Standard American English. True label for every pair is "Neither"; any gap in
false-positive rate reveals dialect bias.

| Model | AAVE false-positive rate | SAE false-positive rate | Gap |
|---|---|---|---|
| Baseline (TF-IDF + LogReg) | 40.0% | 20.0% | **20.0 pts** |
| **DistilBERT (deployed model)** | **0.0%** | **0.0%** | **0.0 pts** |

**Baseline reproduces the documented bias clearly** — e.g. "that test was hard as
heck" flagged Offensive, "that test was very difficult" not; "she got mad skills
at drawing" flagged, "she has a lot of skill" not. This confirms the audit is
genuinely sensitive, not trivially passing everything.

**DistilBERT shows zero gap on this test** — the model we've chosen to deploy
does not reproduce the classic AAVE bias. Testing baseline alongside it is what
makes this credible: a clean result on an audit that catches nothing would prove
little.

**Scope limitation, stated honestly**: this audit measures false positives on
clearly *benign* text only. It does not test whether genuinely offensive AAVE
content is treated differently than equivalent offensive SAE content, or whether
model *confidence* (not just the final prediction) differs by dialect. Those
would be reasonable extensions, not covered here.

## Image-Specific Bias Check (Required Check)

Tested whether the image-only model's prediction changes under three superficial
perturbations that don't alter actual content: a skin-tone-like hue shift,
brightness adjustment, and background blur (n=20 test images).

| Perturbation | Flip rate | Mean Δ P(offensive) | p-value (paired t-test) |
|---|---|---|---|
| Hue shift | 15.0% | −0.063 | 0.045 |
| Brightness | 20.0% | +0.002 | 0.943 |
| Blur | 20.0% | −0.026 | 0.437 |

**No strong evidence of systematic bias.** Brightness and blur show no meaningful
signal. Hue shift shows a small, borderline-significant decrease in predicted
offensiveness (p=0.045) — but with 3 perturbation types tested, the
multiple-comparison-corrected threshold is p<0.017, which this does not clear.
We report this as a mild trend worth a larger follow-up sample, not a confirmed
finding — over-claiming either "no bias" or "confirmed bias" from n=20 would be
dishonest in either direction.

Context: the image-only model's overall signal is weak (test macro-F1 0.51,
barely above chance), so some prediction instability under any perturbation,
including content-irrelevant ones, is expected on its own and doesn't
automatically indicate targeted bias.

## Qualitative Error Analysis (Required)

DistilBERT misclassified 375/3718 test examples (10.1%). Sample of 15 examined
directly, plus LIME explanations on 5, reveal a consistent pattern:

**The model's decisions are driven heavily by the presence of specific lexical
triggers (particular slurs), not deeper contextual understanding of intent or
target.**

- **Slur presence overrides context** (dominant pattern): personal insults using
  slurs (annotated "Offensive") get escalated to "Hate Speech" purely because a
  slur is present, regardless of whether the speech is actually targeted,
  systematic hate against a group. LIME confirms this directly — in all 5
  explained examples, one slur token carries +0.4 to +0.8 weight, an almost
  single-word-decisive influence (e.g. "faggot" +0.556 in a tweet that's road-rage
  directed at one driver, not group-targeted hate).
- **Offensive content without explicit slur vocabulary gets missed**: coded/
  indirect language ("ghetto," "monkey," ableist slang used non-personally) is
  frequently misclassified as "Neither" when it avoids the model's apparent
  learned trigger vocabulary.
- **The same pattern misfires the other direction**: a wordplay/rhyme tweet
  referencing "Aunt Jemima" (annotated "Neither" by humans) gets flagged
  "Offensive," apparently from the racially-loaded reference alone, without
  grasping it wasn't used as an attack.

This is a genuine limitation worth stating plainly, not something to hide — and
it's partly a reflection of real difficulty in the underlying task: the original
Davidson dataset paper documents low inter-annotator agreement on exactly this
Hate-vs-Offensive boundary, because the distinction is subtle even for humans.
We are not attempting to engineer around this here; it would require richer
contextual modeling or better-labeled training data distinguishing "slur as
personal insult" from "slur as targeted hate," beyond this project's scope.

Full misclassified set: `results/misclassified_examples_full.csv`.
LIME explanations: `results/lime_explanations.json`.

**Deployment decision:** given the image branch does not add measurable value
on this dataset, the deployed app extracts text from uploaded images via OCR
and classifies that text with the text-only DistilBERT model, rather than
routing through the weaker multimodal fusion branch. This is a
data-driven engineering decision, not a limitation being hidden — the full
comparison above stays in the repo as documented evidence for why.

## Progress log

- [x] Repo initialized, `.gitignore`, requirements scaffolded
- [x] Text data loaded, cleaned, EDA complete
- [x] Baseline (TF-IDF + LogReg/SVM)
- [x] BiLSTM
- [x] CNN-BiLSTM
- [x] BiLSTM + Attention
- [x] DistilBERT fine-tune
- [x] Text bias audit
- [x] Multimodal dataset inspected
- [x] Image + fusion branch
- [x] Multimodal bias audit
- [x] Qualitative error analysis (10-15 misclassified examples, categorized)
- [ ] LIME explainability (text predictions)
- [ ] Grad-CAM visualizations (image predictions)
- [ ] FastAPI backend
- [ ] Frontend
- [ ] CI + deployment

## License

MIT — see `LICENSE`.
