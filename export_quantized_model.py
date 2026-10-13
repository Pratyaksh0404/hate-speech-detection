"""
ONE-TIME offline step: load the trained DistilBERT text model in full
precision, apply the same dynamic int8 quantization technique we already
tried to do at serve time, and save the ALREADY-QUANTIZED model to a
single file -- then you upload that file to your Hugging Face model repo.

WHY THIS HAS TO HAPPEN HERE (your machine), NOT ON RENDER:

Quantizing AFTER loading a model needs the full float32 model AND the
brand-new quantized copy to exist in memory AT THE SAME TIME, even if
only for a moment during the conversion. On Render's free 512MB tier,
that transient peak is enough to get the whole process OOM-killed -- we
saw this happen on an actual deploy. Doing the Docker BUILD step instead
doesn't dodge it either: Render's free build machine turned out to have
the exact same 512MB cap (a separate deploy attempt was OOM-killed during
the build for the same reason). Your own machine has plenty of RAM, so
doing the quantization here and uploading the FINISHED quantized model
means the Render server never needs to hold a float32 copy at all --
it just loads the already-quantized file directly.

THIS ALSO FIXES THE 3-4 MINUTE RESPONSE TIMES:

Real int8 dynamically-quantized Linear layers run through PyTorch's
optimized quantized GEMM kernels, which are genuinely faster on CPU.
The bfloat16 version we tried as a stopgap only solved the memory
problem -- it didn't speed up computation, because most CPUs (including
whatever Render's free instances run on) don't have fast native bfloat16
math and effectively emulate it, which can be AS SLOW AS, or slower
than, plain float32. This int8 version should be genuinely faster on
top of fitting in memory from the start.

USAGE (run once, from anywhere, in your normal Python environment that
already has torch + transformers installed -- the same one you've been
using for `uvicorn src.api:app --reload`):

    python export_quantized_model.py

This writes `distilbert_quantized_int8.pt` to the current directory.
The script prints the upload command to run next once it finishes.
"""

import torch
from transformers import AutoModelForSequenceClassification

MODEL_REPO = "Prat-04/hate-speech-distilbert"
OUTPUT_PATH = "distilbert_quantized_int8.pt"

print(f"loading {MODEL_REPO} in full precision...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
model.eval()

print("applying dynamic int8 quantization to Linear layers...")
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Saving the WHOLE module object (torch.save(quantized_model, ...)), not
# just its state_dict(), is deliberate: dynamically-quantized Linear
# layers are a special PyTorch class (torch.ao.nn.quantized.dynamic.Linear)
# with their own packed int8 weight format. Reconstructing that from a
# plain state_dict at load time is fiddly and version-sensitive; pickling
# the whole already-quantized module and torch.load()-ing it back is the
# standard, reliable way to round-trip a quantized model.
torch.save(quantized_model, OUTPUT_PATH)
print(f"saved quantized model to {OUTPUT_PATH}")

print()
print("=" * 70)
print("Next step -- upload this file to your Hugging Face model repo:")
print("=" * 70)
print()
print("  pip install huggingface_hub   # if you don't already have it")
print("  huggingface-cli login         # if you haven't logged in before")
print("                                # (needs a token with WRITE access")
print(f"                                #  to {MODEL_REPO})")
print()
print("  python -c \"from huggingface_hub import upload_file; "
      f"upload_file(path_or_fileobj='{OUTPUT_PATH}', "
      f"path_in_repo='{OUTPUT_PATH}', repo_id='{MODEL_REPO}')\"")
print()
print("Once that finishes, tell Claude -- the Dockerfile and api.py are")
print("already updated to download and load this file directly.")
