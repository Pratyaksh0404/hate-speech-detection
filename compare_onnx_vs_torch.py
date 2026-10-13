"""
ONE-TIME validation: compare the quantized ONNX model's predictions against
the original float32 model across your REAL held-out test set, not just a
handful of hand-picked sentences.

WHY THIS, ON TOP OF export_onnx_model.py's built-in check:

That script's 3-sentence smoke test flagged one sentence with a 0.072
max-probability difference -- above the 0.05 threshold, though the
predicted label still matched. Three anecdotal sentences can't tell you
whether that's normal int8 quantization noise or something that will flip
predictions near the decision boundary on real data. This script runs
BOTH models over data/text/test.csv (the same split used for this
project's own reported README metrics) and reports:

  - overall accuracy and macro-F1 for EACH model separately, so you can
    see directly whether the ONNX version's reported metrics would differ
    from the ones already written up in the README
  - how many test examples the two models disagree on (different argmax),
    and what fraction that is
  - the single largest per-example probability difference anywhere in the
    whole test set, not just 3 sentences

Run this from the repo root, in the same venv you used for
export_onnx_model.py (it already has torch, transformers, onnxruntime,
and this adds sklearn's classification_report, which you already have via
scikit-learn in requirements.txt):

    python compare_onnx_vs_torch.py

If accuracy/macro-F1 match closely (expect differences in the thousandths,
not whole percentage points) and disagreements are rare, the ONNX model is
safe to rely on for production. If not, paste the output back before
deploying -- that would mean the quantization is doing real damage and we
should investigate rather than ship it.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_REPO = "Prat-04/hate-speech-distilbert"
ONNX_PATH = "distilbert_int8.onnx"
TEST_CSV = "data/text/test.csv"
MAX_LENGTH = 128
BATCH_SIZE = 32
LABEL_NAMES = ["Hate Speech", "Offensive Language", "Neither"]

print(f"loading test set from {TEST_CSV}...")
test_df = pd.read_csv(TEST_CSV).reset_index(drop=True)
test_df["text"] = test_df["text"].fillna("")
texts = test_df["text"].tolist()
true_labels = test_df["label"].to_numpy()
print(f"  {len(texts)} examples")

print(f"loading float32 model from {MODEL_REPO}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
torch_model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
torch_model.eval()

print(f"loading quantized ONNX model from {ONNX_PATH}...")
import onnxruntime as ort

session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])


def torch_predict_proba(batch_texts):
    enc = tokenizer(
        batch_texts, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = torch_model(**enc).logits
    return torch.softmax(logits, dim=1).numpy()


def onnx_predict_proba(batch_texts):
    enc = tokenizer(
        batch_texts, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="np"
    )
    ort_inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    (logits,) = session.run(["logits"], ort_inputs)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


print("running both models over the test set (this takes a minute)...")
torch_probs_all = []
onnx_probs_all = []
for start in range(0, len(texts), BATCH_SIZE):
    batch = texts[start : start + BATCH_SIZE]
    torch_probs_all.append(torch_predict_proba(batch))
    onnx_probs_all.append(onnx_predict_proba(batch))
    if (start // BATCH_SIZE) % 20 == 0:
        print(f"  {start + len(batch)}/{len(texts)}", flush=True)

torch_probs = np.concatenate(torch_probs_all, axis=0)
onnx_probs = np.concatenate(onnx_probs_all, axis=0)

torch_preds = torch_probs.argmax(axis=1)
onnx_preds = onnx_probs.argmax(axis=1)

torch_acc = accuracy_score(true_labels, torch_preds)
onnx_acc = accuracy_score(true_labels, onnx_preds)
torch_f1 = f1_score(true_labels, torch_preds, average="macro")
onnx_f1 = f1_score(true_labels, onnx_preds, average="macro")

disagreements = (torch_preds != onnx_preds).sum()
max_diff = np.abs(torch_probs - onnx_probs).max()
mean_diff = np.abs(torch_probs - onnx_probs).mean()

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"{'':20s} {'float32 (original)':>22s} {'ONNX int8':>15s}")
print(f"{'accuracy':20s} {torch_acc:22.4f} {onnx_acc:15.4f}")
print(f"{'macro-F1':20s} {torch_f1:22.4f} {onnx_f1:15.4f}")
print()
print(f"Predictions that DISAGREE between the two models: {disagreements} / {len(texts)} "
      f"({disagreements / len(texts):.2%})")
print(f"Largest per-example probability difference anywhere in the test set: {max_diff:.4f}")
print(f"Mean per-example probability difference: {mean_diff:.5f}")
print()

if abs(torch_acc - onnx_acc) < 0.01 and abs(torch_f1 - onnx_f1) < 0.01 and disagreements / len(texts) < 0.02:
    print("LOOKS SAFE: accuracy and macro-F1 are within 1 point, disagreements are rare.")
    print("The earlier 3-sentence warning was quantization noise on one example, not a")
    print("real accuracy problem. Safe to deploy distilbert_int8.onnx.")
else:
    print("CAUTION: differences are larger than expected for int8 dynamic quantization.")
    print("Paste this output back before deploying -- worth investigating further rather")
    print("than shipping a model that measurably underperforms the one your README reports.")

# Show a few of the actual disagreements, if any, so you can eyeball them.
if disagreements > 0:
    print()
    print("Sample of disagreeing examples (up to 10):")
    disagree_idx = np.where(torch_preds != onnx_preds)[0][:10]
    for i in disagree_idx:
        print(
            f"  true={LABEL_NAMES[true_labels[i]]:20s} "
            f"torch={LABEL_NAMES[torch_preds[i]]:20s} "
            f"onnx={LABEL_NAMES[onnx_preds[i]]:20s} "
            f"text={texts[i][:60]!r}"
        )
