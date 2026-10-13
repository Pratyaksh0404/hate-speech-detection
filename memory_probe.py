"""
Staged memory probe -- run this locally instead of eyeballing Task Manager.

WHY THIS INSTEAD OF TASK MANAGER:
Task Manager only gives you two snapshots (idle, after-one-request), which
tells you baseline-vs-total but not WHICH import or load step is actually
costing the memory. This script imports/loads things one stage at a time,
in the SAME process, and prints resident memory after each stage -- so you
get a breakdown instead of a single number.

USAGE:
    python memory_probe.py

Run it from the repo root (same venv you use for `uvicorn src.api:app`).
It does NOT start a server -- it just walks through the same imports and
model-loading steps src/api.py does at module-import time, in order, and
reports memory after each one. Paste the full output back to Claude.
"""

import os
import sys

try:
    import psutil
    _proc = psutil.Process(os.getpid())
    def rss_mb():
        return _proc.memory_info().rss / (1024 * 1024)
except ImportError:
    import resource
    def rss_mb():
        # ru_maxrss is KB on Linux, bytes on macOS -- this repo targets
        # Linux (Render), so assume KB. It's also a HIGH-WATER MARK, not a
        # current value, so stage deltas after this point will look smaller
        # than they really are once something else has pushed the mark up.
        # Installing psutil (`pip install psutil`) gives true current RSS
        # per stage instead -- strongly preferred if you can.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

STAGES = []
_last = [0.0]

def mark(label):
    now = rss_mb()
    delta = now - _last[0]
    _last[0] = now
    STAGES.append((label, now, delta))
    print(f"{label:<55} {now:9.1f} MB   (+{delta:.1f} MB)", flush=True)

mark("baseline (python + os + psutil only)")

import numpy  # noqa: E402
mark("+ numpy")

import torch  # noqa: E402
mark("+ torch")

import torchvision  # noqa: E402
mark("+ torchvision")

from PIL import Image  # noqa: E402
mark("+ Pillow")

import pytesseract  # noqa: E402
mark("+ pytesseract")

import fastapi, uvicorn  # noqa: E402
mark("+ fastapi + uvicorn")

import sklearn  # noqa: E402
mark("+ scikit-learn (lime dependency)")

import scipy  # noqa: E402
mark("+ scipy (lime dependency)")

import lime.lime_text  # noqa: E402
mark("+ lime.lime_text itself")

import transformers  # noqa: E402
mark("+ transformers")

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("Prat-04/hate-speech-distilbert")
mark("+ tokenizer loaded")

# ---- the actual quantized text model (the thing we've been trying to
# shrink). Loads the same file src/api.py loads at runtime. ----
from huggingface_hub import hf_hub_download  # noqa: E402
weights_path = hf_hub_download(
    repo_id="Prat-04/hate-speech-distilbert",
    filename="distilbert_quantized_int8.pt",
)
mark("+ weights file downloaded to disk (no RAM cost expected)")

model = torch.load(weights_path, map_location="cpu", weights_only=False)
model.eval()
mark("+ quantized text model loaded into memory")

# ---- one plain inference (tokenize + forward pass, no LIME) ----
inputs = tok("this is a completely harmless test sentence", return_tensors="pt")
with torch.no_grad():
    _ = model(**inputs)
mark("+ one plain forward pass (tokenize + predict, no LIME)")

# ---- one LIME explanation call, same num_samples as src/api.py uses (30) ----
explainer = lime.lime_text.LimeTextExplainer(class_names=["not_hate", "hate"])

def predict_proba(texts):
    import numpy as np
    outs = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).numpy()[0]
        outs.append(probs)
    return np.array(outs)

_ = explainer.explain_instance(
    "this is a completely harmless test sentence",
    predict_proba,
    num_features=10,
    num_samples=30,
)
mark("+ one LIME explain_instance call (num_samples=30, matches api.py)")

print()
print("=" * 70)
print("SUMMARY -- biggest single jumps, largest first:")
print("=" * 70)
for label, now, delta in sorted(STAGES, key=lambda s: -s[2])[:8]:
    print(f"  {delta:7.1f} MB   {label}")

print()
print(f"TOTAL at end: {_last[0]:.1f} MB")
print()
print("Paste this entire output back. The 'baseline' through '+ transformers'")
print("rows are the library-import cost (what we'd pay even with a tiny")
print("model); everything from 'tokenizer loaded' onward is the model/")
print("inference cost. That split tells us whether dropping LIME or")
print("shrinking the model further is the lever that actually matters.")
