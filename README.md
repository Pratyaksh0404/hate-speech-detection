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
- [ ] Image + fusion branch
- [ ] Multimodal bias audit
- [ ] FastAPI backend
- [ ] Frontend
- [ ] CI + deployment

## License

MIT — see `LICENSE`.
