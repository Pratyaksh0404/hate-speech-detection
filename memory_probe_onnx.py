"""
Staged memory probe for the NEW onnxruntime-based text pipeline -- same
idea as memory_probe.py, but mirrors what src/api.py actually imports now
(no torch/torchvision at module level at all; onnxruntime instead).

Run this AFTER you've exported and uploaded distilbert_int8.onnx via
export_onnx_model.py, so hf_hub_download below can actually fetch it.

USAGE (from repo root, same venv -- plus the new deps):
    pip install onnxruntime psutil
    python memory_probe_onnx.py

This deliberately does NOT import torch or torchvision anywhere -- that's
the whole point: a plain /predict/text request in the real app never
imports them either (see src/api.py's lazy _ensure_image_model_loaded).
If you also want to measure the rare /predict/image path's cost, that's a
separate, expected one-time hit on first use -- not measured here since
it doesn't affect the number that matters (steady-state memory for the
traffic Render actually bills/limits you on).

UPDATED: now sets USE_TORCH=0 / USE_TF=0 before importing transformers,
matching the fix just added to src/api.py -- the first run of this script
showed "+ tokenizer loaded" costing +242MB instead of the expected ~28MB,
because transformers silently imports torch internally when it detects
torch is installed, even though nothing here asks for it. These env vars
stop that.
"""

import os

os.environ.setdefault("USE_TORCH", "0")
os.environ.setdefault("USE_TF", "0")

try:
    import psutil
    _proc = psutil.Process(os.getpid())
    def rss_mb():
        return _proc.memory_info().rss / (1024 * 1024)
except ImportError:
    import resource
    def rss_mb():
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

import onnxruntime  # noqa: E402
mark("+ onnxruntime")

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

from tokenizers import Tokenizer  # noqa: E402
mark("+ tokenizers (lightweight -- NOT transformers.AutoTokenizer)")

import sys  # noqa: E402
sys.path.insert(0, ".")
from src.light_tokenizer import LightBertTokenizer  # noqa: E402

tok = LightBertTokenizer("Prat-04/hate-speech-distilbert", max_length=128)
mark("+ light tokenizer loaded (no torch import -- see src/light_tokenizer.py)")

from huggingface_hub import hf_hub_download  # noqa: E402
onnx_path = hf_hub_download(
    repo_id="Prat-04/hate-speech-distilbert",
    filename="distilbert_int8.onnx",
)
mark("+ onnx file downloaded to disk (no RAM cost expected)")

session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
mark("+ onnx session created (model loaded into onnxruntime)")

# ---- one plain inference (tokenize + run, no LIME) ----
import numpy as np  # noqa: E402

ort_inputs = tok(["this is a completely harmless test sentence"])
(logits,) = session.run(["logits"], ort_inputs)
mark("+ one plain forward pass (tokenize + predict, no LIME)")

# ---- one LIME explanation call, same num_samples as src/api.py uses (30) ----
explainer = lime.lime_text.LimeTextExplainer(class_names=["Hate Speech", "Offensive Language", "Neither"])

def predict_proba(texts):
    oi = tok(list(texts))
    (lg,) = session.run(["logits"], oi)
    shifted = lg - lg.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)

_ = explainer.explain_instance(
    "this is a completely harmless test sentence",
    predict_proba,
    num_features=8,
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
print("Compare this TOTAL against the old memory_probe.py run (643.2 MB).")
print("If this comes in comfortably under ~450MB (leaving headroom for")
print("uvicorn/OS overhead Render adds on top), the ONNX migration worked")
print("and it's safe to deploy. Paste the output back either way.")
