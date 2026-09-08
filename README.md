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

## Progress log

- [x] Repo initialized, `.gitignore`, requirements scaffolded
- [x] Text data loaded, cleaned, EDA complete (class-weighted loss chosen for imbalance — see notebooks/EDA_text.ipynb section 7)
- [x] Baseline (TF-IDF + LogReg/SVM) — LogReg test macro-F1 0.7362 / acc 0.8693, LinearSVM test macro-F1 0.7332 / acc 0.8946 — see results/baseline_logreg.json, results/baseline_linearsvm.json
- [x] BiLSTM (test macro-F1 0.7310, essentially tied with baseline — see results/bilstm.json; CNN-BiLSTM next)
- [x] CNN-BiLSTM — test macro-F1 0.7462, test accuracy 0.8808 — best model so far, beats both baselines and plain BiLSTM (multi-kernel CNN feature extraction + BiLSTM sequential context)
- [x] BiLSTM + Attention — test macro-F1 0.7358, test accuracy 0.8687 — additive attention over BiLSTM outputs (masked correctly at padding), attention weights available for later explainability/error-analysis
- [ ] DistilBERT fine-tune
- [ ] Text bias audit
- [ ] Multimodal dataset inspected
- [ ] Image + fusion branch
- [ ] Multimodal bias audit
- [ ] FastAPI backend
- [ ] Frontend
- [ ] CI + deployment

*(Full results tables, architecture diagram, and reproduction instructions
will be filled in as each phase lands — no numbers go here until they come
from an actual run.)*

## License

MIT — see `LICENSE`.
