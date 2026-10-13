"""
ONE-TIME offline step: export the trained DistilBERT text model to ONNX and
quantize it with ONNX Runtime, then upload the result to the Hugging Face
model repo -- replacing export_quantized_model.py's torch-based approach.

WHY THIS REPLACES export_quantized_model.py:

The int8-quantized torch.save() file that script produces genuinely IS
smaller on disk (139MB vs ~270MB float32), but the memory numbers from a
staged import/load probe (see chat / memory_probe.py) showed the real
problem was never really the model file size:

  - `import torch` alone costs ~165MB of RAM, before a single weight loads.
    That's the PyTorch framework itself (autograd, its CPU kernel
    libraries, etc.) -- quantizing the MODEL doesn't touch this cost at
    all, because it isn't model weight.
  - Loading even the already-quantized .pt file cost ~187MB more on top of
    that -- noticeably more than the 139MB file size on disk, almost
    certainly because the first use of a dynamically-quantized module
    triggers PyTorch to load its quantized-GEMM backend (fbgemm/qnnpack)
    into the process, which is a near-fixed cost unrelated to model size.
  - Combined with the rest of the library stack (transformers, lime's
    scikit-learn + scipy, fastapi, pytesseract), total steady-state memory
    came out to ~590MB even on the already-quantized path -- over Render's
    512MB free-tier cap, which matches exactly where the real deploy died.

Serving through ONNX Runtime instead of torch avoids both costs: `import
onnxruntime` has a much smaller footprint than `import torch` (no
autograd, no training-time machinery, a single lean C++ runtime), and
loading a pre-quantized .onnx file doesn't trigger any comparable
backend-initialization tax -- ONNX Runtime's int8 kernels are already part
of its normal execution providers, not a separately-loaded extension.

This does NOT replace the pre-quantized .pt file for the image-only
ResNet18 diagnostic path (src/api.py's lazy `_ensure_image_model_loaded`)
-- that path still uses torch/torchvision, deliberately, since it's rare
(only /predict/image, lazy-loaded on first use) and converting it isn't
worth the effort right now. This script only touches the TEXT model,
which is on the hot path for every single request.

USAGE (run once, from anywhere, in your normal Python environment that
already has torch + transformers installed):

    pip install onnx onnxruntime   # if you don't already have them
    python export_onnx_model.py

This writes `distilbert_int8.onnx` to the current directory. The script
prints the upload command to run next, and verifies the exported model's
predictions match the original float32 model's before telling you it's
safe to upload.
"""

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_REPO = "Prat-04/hate-speech-distilbert"
ONNX_PATH = "distilbert_int8.onnx"
_FP32_ONNX_PATH = "_distilbert_fp32_tmp.onnx"
MAX_LENGTH = 128

print(f"loading {MODEL_REPO} in full precision...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
model.eval()

print("exporting to ONNX (float32, dynamic batch/sequence axes)...")
dummy = tokenizer(
    "this is a dummy sentence used only to trace the model graph",
    return_tensors="pt",
    truncation=True,
    max_length=MAX_LENGTH,
)
torch.onnx.export(
    model,
    (dummy["input_ids"], dummy["attention_mask"]),
    _FP32_ONNX_PATH,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch"},
    },
    opset_version=14,
    # Recent torch versions default to the newer dynamo-based exporter,
    # which needs the separate `onnxscript` package installed. The legacy
    # TorchScript-based exporter (dynamo=False) has no such dependency and
    # handles a plain classification head like this one without issue --
    # no need to add another package just for this one-time offline step.
    dynamo=False,
)
print(f"float32 ONNX graph written to {_FP32_ONNX_PATH}")

print("applying dynamic int8 quantization via onnxruntime.quantization...")
from onnxruntime.quantization import QuantType, quantize_dynamic

quantize_dynamic(
    model_input=_FP32_ONNX_PATH,
    model_output=ONNX_PATH,
    weight_type=QuantType.QInt8,
    # Default op set (MatMul/Attention-equivalent nodes) -- this is the
    # direct analog of torch's quantize_dynamic({nn.Linear}, ...): it
    # quantizes the same Linear-layer weights and leaves embeddings/
    # LayerNorm in float32, same trade-off as before. Embedding
    # quantization (adding "Gather" to op_types_to_quantize) is possible
    # but has had accuracy/compatibility rough edges across onnxruntime
    # versions, and the embedding table isn't the dominant cost here (the
    # torch import + backend-load tax is) -- so it's deliberately left out
    # rather than risk silently degrading predictions for a smaller win.
)
print(f"quantized ONNX model written to {ONNX_PATH}")

print("verifying the quantized ONNX model's predictions match the original...")
import onnxruntime as ort

session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

test_sentences = [
    "this is a completely harmless test sentence",
    "I really appreciate your help today",
    "get out of here you worthless piece of trash",
]
max_abs_diff = 0.0
for sentence in test_sentences:
    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        torch_probs = torch.softmax(model(**enc).logits, dim=1).numpy()[0]

    enc_np = tokenizer(sentence, return_tensors="np", truncation=True, max_length=MAX_LENGTH)
    (onnx_logits,) = session.run(
        ["logits"],
        {
            "input_ids": enc_np["input_ids"].astype(np.int64),
            "attention_mask": enc_np["attention_mask"].astype(np.int64),
        },
    )
    onnx_exp = np.exp(onnx_logits - onnx_logits.max(axis=1, keepdims=True))
    onnx_probs = (onnx_exp / onnx_exp.sum(axis=1, keepdims=True))[0]

    diff = np.abs(torch_probs - onnx_probs).max()
    max_abs_diff = max(max_abs_diff, diff)
    agree = torch_probs.argmax() == onnx_probs.argmax()
    print(
        f"  {'OK ' if agree else 'MISMATCH '} argmax agree={agree}  "
        f"max prob diff={diff:.4f}  text={sentence!r}"
    )

print()
if max_abs_diff > 0.05:
    print(
        f"WARNING: max probability difference across test sentences is "
        f"{max_abs_diff:.4f}, which is larger than expected for int8 "
        f"dynamic quantization. Double-check before uploading -- this can "
        f"happen if the wrong opset or a different MODEL_REPO revision was "
        f"used for export vs. comparison."
    )
else:
    print(f"Looks good -- max probability difference is {max_abs_diff:.4f}.")

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
      f"upload_file(path_or_fileobj='{ONNX_PATH}', "
      f"path_in_repo='{ONNX_PATH}', repo_id='{MODEL_REPO}')\"")
print()
print("Once that finishes, tell Claude -- Dockerfile, requirements-deploy.txt,")
print("and src/api.py are already updated to download and serve this file")
print("through onnxruntime instead of the old torch .pt file.")
print()
print(f"(You can delete the intermediate {_FP32_ONNX_PATH} now -- only")
print(f"{ONNX_PATH} gets uploaded.)")
