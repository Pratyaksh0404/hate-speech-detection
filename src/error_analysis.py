"""
Qualitative error analysis + LIME explainability (master_plan.md section
6, REQUIRED): pulls the DistilBERT model's actual misclassified test
examples (not hypothetical ones) and generates LIME explanations for a
sample, showing which tokens actually drove specific predictions.

Uses the RAW `text` column (not clean_text), matching how DistilBERT was
actually trained/evaluated (see src/bert_dataset.py's design rationale).

num_samples=500 for LIME (not the library's default 5000) -- each LIME
sample requires a forward pass through DistilBERT, and 500 perturbed
samples per explanation is enough for stable word-importance rankings on
sentences this short, at a fraction of the compute cost.

Run with (wherever models/distilbert.pt + data/text/test.csv exist):
    python -m src.error_analysis
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
CHECKPOINT_PATH = "models/distilbert.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Hate Speech", "Offensive Language", "Neither"]


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def predict_proba_batch(texts, tokenizer, model, max_length: int = 128, batch_size: int = 32):
    """Returns an (N, 3) numpy array of class probabilities -- the
    interface LIME's LimeTextExplainer expects."""
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        encoding = tokenizer(
            batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**encoding).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def main():
    test_df = pd.read_csv("data/text/test.csv").reset_index(drop=True)
    test_df["text"] = test_df["text"].fillna("")
    tokenizer, model = load_model()

    probs = predict_proba_batch(test_df["text"].tolist(), tokenizer, model)
    preds = probs.argmax(axis=1)

    test_df["pred_label"] = preds
    test_df["pred_label_name"] = [LABEL_NAMES[p] for p in preds]
    test_df["true_label_name"] = test_df["label"].apply(lambda i: LABEL_NAMES[i])
    test_df["correct"] = test_df["label"] == test_df["pred_label"]

    misclassified = test_df[~test_df["correct"]].copy()
    print(f"Total test examples: {len(test_df)}")
    print(f"Misclassified: {len(misclassified)} ({len(misclassified) / len(test_df):.1%})")

    os.makedirs("results", exist_ok=True)
    misclassified[["text", "true_label_name", "pred_label_name"]].to_csv(
        "results/misclassified_examples_full.csv", index=False
    )
    print("saved: results/misclassified_examples_full.csv (all misclassified examples)")

    sample = misclassified.sample(n=min(15, len(misclassified)), random_state=42)
    print(f"\n=== SAMPLE OF {len(sample)} MISCLASSIFIED EXAMPLES ===")
    for _, row in sample.iterrows():
        print(f"\nTRUE: {row['true_label_name']:20s} | PRED: {row['pred_label_name']:20s}")
        print(f"TEXT: {row['text']}")

    sample.to_csv("results/misclassified_examples_sample.csv", index=False)
    print("\nsaved: results/misclassified_examples_sample.csv")

    # --- LIME explanations for a handful of these misclassified examples ---
    explainer = LimeTextExplainer(class_names=LABEL_NAMES)

    def predict_fn(texts):
        return predict_proba_batch(texts, tokenizer, model)

    lime_sample = sample.head(5)
    lime_results = []
    print(f"\n=== LIME EXPLANATIONS (first {len(lime_sample)} misclassified examples) ===")
    for _, row in lime_sample.iterrows():
        text = row["text"]
        pred_label_int = int(row["pred_label"])
        exp = explainer.explain_instance(
            text, predict_fn, num_features=8, num_samples=500, labels=(pred_label_int,)
        )
        word_weights = exp.as_list(label=pred_label_int)
        print(f"\nTEXT: {text}")
        print(f"TRUE: {row['true_label_name']} | PRED: {row['pred_label_name']}")
        print("Top words driving this prediction:")
        for word, weight in word_weights:
            print(f"  {word:20s} {weight:+.3f}")
        lime_results.append(
            {
                "text": text,
                "true_label": row["true_label_name"],
                "pred_label": row["pred_label_name"],
                "word_weights": word_weights,
            }
        )

    with open("results/lime_explanations.json", "w") as f:
        json.dump(lime_results, f, indent=2)
    print("\nsaved: results/lime_explanations.json")


if __name__ == "__main__":
    main()
