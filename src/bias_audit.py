"""
Text bias audit (master_plan.md sections 2.7 and 5.5, REQUIRED, not
optional): the Davidson-style dataset is documented in fairness research
(Sap et al., 2019, "The Risk of Racial Bias in Hate Speech Detection") to
over-flag African-American Vernacular English (AAVE) as offensive, likely
because dialect features spuriously correlate with the "offensive" label
in the training data's annotations, not because AAVE itself is more
offensive.

This audit tests that directly: a small, hand-constructed set of PAIRED
sentences -- each pair expresses the same benign, everyday sentiment (no
slurs, no offensive content in either version) once in AAVE and once in
Standard American English (SAE). A well-calibrated, unbiased model should
predict "Neither" for every single one of these, since none of them are
actually hateful or offensive. Any gap in false-positive rate between the
AAVE and SAE groups is direct evidence of dialect bias, not a
disagreement about what's actually offensive.

Pairs cover well-documented, real AAVE grammatical features (habitual
"be"/"stay", copula deletion, "finna", multiple negation, AAVE-specific
vocabulary) -- not caricature, drawn from established sociolinguistic
description of the dialect.

Tests BOTH the baseline (TF-IDF + LogReg) and DistilBERT models, not just
the best one. This matters for validating the audit itself: if the
baseline (which relies on surface n-gram matching -- exactly the failure
mode the AAVE-bias literature originally documented) shows a real gap
while DistilBERT doesn't, that's much stronger evidence DistilBERT's
clean result reflects genuine robustness, not an audit that's simply too
easy to fail.

Run with (from repo root):
    python -m src.bias_audit --model baseline_logreg   # local, no GPU needed
    python -m src.bias_audit --model distilbert         # needs models/distilbert.pt
"""

import argparse
import json
import os

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
DISTILBERT_CHECKPOINT_PATH = "models/distilbert.pt"
BASELINE_CHECKPOINT_PATH = "models/baseline_logreg.joblib"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Hate Speech", "Offensive Language", "Neither"]

# Each pair expresses the SAME benign sentiment -- neither version is
# actually offensive or hateful. True label for every example is
# "Neither". Sentences deliberately avoid slurs/offensive content in
# either version, isolating dialect as the only meaningful variable.
AAVE_SAE_PAIRS = [
    ("I been working all day", "I have been working all day"),
    ("She my best friend", "She is my best friend"),
    ("He finna go to the store", "He is about to go to the store"),
    ("We chillin at the park", "We are relaxing at the park"),
    ("That's my homegirl right there", "That's my close friend right there"),
    ("I'm bout to eat, y'all hungry?", "I'm about to eat, are you all hungry?"),
    ("Ain't nobody got time for that", "Nobody has time for that"),
    ("She be studying every night", "She studies every night"),
    ("This food is bussin", "This food is delicious"),
    ("He got a lot of swag", "He has a lot of style"),
    ("I'm finna head home now", "I am about to head home now"),
    ("We had a good time fr fr", "We had a good time, honestly"),
    ("That test was hard as heck", "That test was very difficult"),
    ("My grandma be cooking the best food", "My grandmother always cooks the best food"),
    ("I ain't even mad about it", "I am not even upset about it"),
    ("He stay winning at basketball", "He always wins at basketball"),
    ("Ion know what you mean", "I don't know what you mean"),
    ("We finna watch the game tonight", "We are going to watch the game tonight"),
    ("That's real talk right there", "That's very true"),
    ("She got mad skills at drawing", "She has a lot of skill at drawing"),
]


def load_distilbert():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    state_dict = torch.load(DISTILBERT_CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def predict_distilbert(text: str, tokenizer, model) -> str:
    encoding = tokenizer(text, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**encoding).logits
    pred_idx = logits.argmax(dim=1).item()
    return LABEL_NAMES[pred_idx]


def load_baseline():
    return joblib.load(BASELINE_CHECKPOINT_PATH)


def predict_baseline(text: str, pipeline) -> str:
    # imported here, not at module top, so importing AAVE_SAE_PAIRS from
    # this module doesn't require src.preprocessing (and its nltk/emoji
    # dependencies) to exist -- only actually calling this function does
    from src.preprocessing import clean_text

    # baseline was trained on clean_text() output, not raw text -- must
    # apply the identical preprocessing here or predictions are meaningless
    cleaned = clean_text(text)
    pred_idx = pipeline.predict([cleaned])[0]
    return LABEL_NAMES[pred_idx]


def run_audit(predict_fn, model_label: str):
    aave_preds, sae_preds = [], []
    rows = []
    for aave_text, sae_text in AAVE_SAE_PAIRS:
        aave_pred = predict_fn(aave_text)
        sae_pred = predict_fn(sae_text)
        aave_preds.append(aave_pred)
        sae_preds.append(sae_pred)
        rows.append(
            {"aave_text": aave_text, "aave_pred": aave_pred, "sae_text": sae_text, "sae_pred": sae_pred}
        )
        flag = "  <-- DIFFERENT" if aave_pred != sae_pred else ""
        print(f"AAVE: {aave_pred:20s} | SAE: {sae_pred:20s}{flag}")
        print(f"  AAVE: {aave_text}")
        print(f"  SAE:  {sae_text}")

    n = len(AAVE_SAE_PAIRS)
    aave_fp_rate = sum(1 for p in aave_preds if p != "Neither") / n
    sae_fp_rate = sum(1 for p in sae_preds if p != "Neither") / n

    print(f"\n=== {model_label} RESULTS (n={n} pairs, true label is 'Neither' for all) ===")
    print(f"AAVE false-positive rate (flagged Hate Speech/Offensive when benign): {aave_fp_rate:.1%}")
    print(f"SAE  false-positive rate (flagged Hate Speech/Offensive when benign): {sae_fp_rate:.1%}")
    print(f"Gap: {abs(aave_fp_rate - sae_fp_rate):.1%}")

    os.makedirs("results", exist_ok=True)
    out_path = f"results/text_bias_audit_{model_label}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "model": model_label,
                "n_pairs": n,
                "aave_false_positive_rate": aave_fp_rate,
                "sae_false_positive_rate": sae_fp_rate,
                "gap": abs(aave_fp_rate - sae_fp_rate),
                "pairs": rows,
            },
            f,
            indent=2,
        )
    print(f"\nsaved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["distilbert", "baseline_logreg"], default="distilbert")
    args, _unknown_args = parser.parse_known_args()

    if args.model == "distilbert":
        tokenizer, model = load_distilbert()
        run_audit(lambda text: predict_distilbert(text, tokenizer, model), "distilbert")
    else:
        pipeline = load_baseline()
        run_audit(lambda text: predict_baseline(text, pipeline), "baseline_logreg")


if __name__ == "__main__":
    main()
