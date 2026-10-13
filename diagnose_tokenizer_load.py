"""
Diagnostic: the previous fix (USE_TORCH=0/USE_TF=0) had zero effect on the
+242MB jump when loading the tokenizer, which means the "transformers
silently imports torch" theory was wrong. Rather than guess again, this
prints exactly which top-level packages get added to sys.modules during
that one call -- the real, direct answer instead of another hypothesis.

Run this from the repo root, same venv:
    python diagnose_tokenizer_load.py
"""

import os
import sys

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

import numpy  # noqa: E402
import onnxruntime  # noqa: E402
from PIL import Image  # noqa: E402
import pytesseract  # noqa: E402
import fastapi, uvicorn  # noqa: E402
import sklearn  # noqa: E402
import scipy  # noqa: E402
import lime.lime_text  # noqa: E402
import transformers  # noqa: E402

before_rss = rss_mb()
before_modules = set(sys.modules.keys())
print(f"RSS before tokenizer load: {before_rss:.1f} MB")
print("Is 'torch' already in sys.modules before tokenizer load?", "torch" in sys.modules)
print()

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("Prat-04/hate-speech-distilbert")

after_rss = rss_mb()
after_modules = set(sys.modules.keys())
print(f"RSS after tokenizer load: {after_rss:.1f} MB  (delta: {after_rss - before_rss:+.1f} MB)")
print("Is 'torch' in sys.modules after tokenizer load?", "torch" in sys.modules)
print()

new_modules = after_modules - before_modules
# Collapse to top-level package names (e.g. "torch.nn.functional" -> "torch")
top_level_new = sorted({m.split(".")[0] for m in new_modules if not m.startswith("_")})
print(f"New top-level modules imported during tokenizer load ({len(top_level_new)} total):")
for m in top_level_new:
    print(f"  {m}")

print()
print(f"(Full new module count including submodules: {len(new_modules)})")
print()
print("Paste this entire output back -- the 'New top-level modules' list above")
print("tells us definitively what's actually responsible for the +240MB,")
print("instead of guessing again.")
