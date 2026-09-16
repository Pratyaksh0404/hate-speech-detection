"""
Held-out evaluation for the Counterfactual Data Augmentation fix (see
src/counterfactual_augmentation.py). Uses NEW phrasings distinct from both
the original bias-audit sentences and the augmentation training examples
themselves, to test genuine generalization rather than memorization.

Compares the ORIGINAL model (models/distilbert.pt) against the AUGMENTED
model (models/distilbert_augmented.pt) side by side on:

1. Severity monotonicity: new graded-severity personal insults (not
   training examples), checking whether all stay correctly classified as
   Offensive Language, not escalated to Hate Speech.
2. Identity-term calibration: new sentence templates (distinct from the 7
   used in augmentation training) across the same 12 identity terms,
   checking whether "Neither" confidence is now more stable across terms
   (particularly "gay" and "white", the two outliers found in live testing).
3. AAVE bias-audit regression check: re-runs the original 20-pair audit to
   confirm augmentation didn't reintroduce or worsen dialect bias.

Run with (on Kaggle, wherever BOTH checkpoints exist in the same session):
    python -m src.held_out_bias_eval
"""

import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.bias_audit import AAVE_SAE_PAIRS

MODEL_NAME = "distilbert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Hate Speech", "Offensive Language", "Neither"]

# --- Held-out severity examples: NOT in augmentation training. All
# personal insults (no group targeting) -- true label is Offensive Language. ---
HELD_OUT_SEVERITY = [
    "you're a bit clueless sometimes",
    "you're kind of a jerk",
    "you're an absolute fool",
    "you're utterly repulsive as a person",
    "you're the worst human being alive",
    "I despise every single thing about you",
]

# --- Held-out identity templates: distinct wording from the 7 augmentation
# templates, same 12 terms. ---
HELD_OUT_IDENTITY_TEMPLATES = [
    "{term} people deserve the same respect as everyone else",
    "I really enjoyed talking to my {term} classmate yesterday",
    "the {term} author wrote an amazing book last year",
    "my {term} cousin is visiting this weekend",
    "there's a really talented {term} musician playing tonight",
]
IDENTITY_TERMS = [
    "gay", "white", "black", "muslim", "jewish", "christian",
    "trans", "disabled", "mexican", "asian", "latino", "queer",
]


def load_model(checkpoint_path: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model):
    encoding = tokenizer(text, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**encoding).logits
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax())
    return LABEL_NAMES[pred_idx], {LABEL_NAMES[i]: float(probs[i]) for i in range(3)}


def eval_severity(tokenizer, model, label: str):
    print(f"\n--- Severity monotonicity ({label}) ---")
    results = []
    for text in HELD_OUT_SEVERITY:
        pred_label, probs = predict(text, tokenizer, model)
        results.append({"text": text, "pred": pred_label, "hate_speech_prob": probs["Hate Speech"]})
        tag = "OK" if pred_label == "Offensive Language" else "WRONG"
        print(f"[{tag}] {pred_label:20s} HS_prob={probs['Hate Speech']:.3f}  \"{text}\"")
    n_correct = sum(1 for r in results if r["pred"] == "Offensive Language")
    print(f"{n_correct}/{len(results)} correctly classified as Offensive Language")
    return results


def eval_identity(tokenizer, model, label: str):
    print(f"\n--- Identity-term calibration ({label}) ---")
    term_confidences = {}
    for term in IDENTITY_TERMS:
        confidences = []
        for template in HELD_OUT_IDENTITY_TEMPLATES:
            text = template.format(term=term)
            _pred_label, probs = predict(text, tokenizer, model)
            confidences.append(probs["Neither"])
        term_confidences[term] = sum(confidences) / len(confidences)

    baseline = sum(term_confidences.values()) / len(term_confidences)
    print(f"{'term':12s} {'avg Neither conf':>18s} {'gap vs avg':>12s}")
    for term, conf in sorted(term_confidences.items(), key=lambda x: x[1]):
        print(f"{term:12s} {conf:18.3f} {conf - baseline:+12.3f}")
    return term_confidences


def eval_aave_regression(tokenizer, model, label: str):
    print(f"\n--- AAVE bias-audit regression check ({label}) ---")
    aave_fp, sae_fp = 0, 0
    n = len(AAVE_SAE_PAIRS)
    flipped_pairs = []
    for aave_text, sae_text in AAVE_SAE_PAIRS:
        aave_pred, _ = predict(aave_text, tokenizer, model)
        sae_pred, _ = predict(sae_text, tokenizer, model)
        if aave_pred != "Neither":
            aave_fp += 1
            flipped_pairs.append({"text": aave_text, "pred": aave_pred, "side": "AAVE"})
        if sae_pred != "Neither":
            sae_fp += 1
            flipped_pairs.append({"text": sae_text, "pred": sae_pred, "side": "SAE"})
    aave_rate, sae_rate = aave_fp / n, sae_fp / n
    print(f"AAVE FP rate: {aave_rate:.1%}, SAE FP rate: {sae_rate:.1%}, gap: {abs(aave_rate - sae_rate):.1%}")
    if flipped_pairs:
        print("Flipped (non-'Neither') pairs:")
        for p in flipped_pairs:
            print(f"  [{p['side']}] predicted {p['pred']}: \"{p['text']}\"")
    return {"aave_fp_rate": aave_rate, "sae_fp_rate": sae_rate, "flipped_pairs": flipped_pairs}


def main():
    print("Loading ORIGINAL model...")
    tok_orig, model_orig = load_model("models/distilbert.pt")
    print("Loading AUGMENTED_3X model...")
    tok_3x, model_3x = load_model("models/distilbert_augmented_v2.pt")
    print("Loading AUGMENTED_2X model...")
    tok_2x, model_2x = load_model("models/distilbert_augmented_2x.pt")

    results = {}
    for label, tok, model in [
        ("ORIGINAL", tok_orig, model_orig),
        ("AUGMENTED_3X", tok_3x, model_3x),
        ("AUGMENTED_2X", tok_2x, model_2x),
    ]:
        results[label] = {
            "severity": eval_severity(tok, model, label),
            "identity": eval_identity(tok, model, label),
            "aave_regression": eval_aave_regression(tok, model, label),
        }

    with open("results/cda_final_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved: results/cda_final_comparison.json")


if __name__ == "__main__":
    main()
