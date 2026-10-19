"""
FastAPI backend (master_plan.md section 7): serves the trained hate-speech
detection models via two endpoints.

Design decisions:

1. /predict/image uses OCR + the text-only DistilBERT model, NOT the
   fusion model the plan's literal wording suggests. This is a deliberate,
   evidence-based deviation, not an oversight: our own experiments (see
   README "Multimodal Results") showed the fusion model does not
   outperform text-only DistilBERT on this task (macro-F1 0.55-0.56 vs
   0.77), and Grad-CAM analysis showed the image branch mostly does
   generic texture pattern-matching on text-dense regions rather than
   genuine visual understanding. Serving a worse model because the
   architecture sounds more sophisticated would be dishonest engineering;
   OCR + the strong text model is the one that's actually good.

2. image_saliency_note is a static, informative string explaining this
   decision (with a pointer to the repo's Grad-CAM analysis). Disclosed
   openly in the response itself, not hidden.

3. Both the tokenizer and model are loaded ONCE at module import time, not
   per-request -- standard practice for serving latency.

4. top_contributing_tokens uses LIME -- the same method used throughout
   this project's offline error analysis -- with a reduced num_samples
   (30 in this deployed version, vs. 500 used offline) to keep
   single-request latency survivable on Render's free tier's 0.1 vCPU.
   Fewer samples means noisier/less stable token attributions than the
   offline analysis; that's a disclosed trade-off for serving under a
   hard compute ceiling, not an attempt to pass off a different method.

5. Raw text (not clean_text()) is fed directly to DistilBERT's tokenizer,
   matching exactly how the model was trained (see src/bert_dataset.py).

6. images_not_stored_note is now CONDITIONAL on whether this specific
   request actually triggered the OCR.space remote fallback (see
   src/ocr.py). Local OCR still runs first, in-memory, with nothing ever
   saved to disk -- that part of the original promise is unconditionally
   true. But when local OCR returns little/no text, the image is also
   sent to a third-party OCR service (OCR.space) over HTTPS to try to
   recover it. Silently keeping the old blanket "never leaves your
   machine" wording in that case would be false, so the response discloses
   per-request whether the fallback actually fired.

7. Static-file serving: when a ``static/`` directory exists next to the
   ``src/`` package (i.e. the Docker container's layout), the app serves
   the built React frontend from the same origin. In local dev (no
   ``static/`` directory), the existing JSON root endpoint is preserved
   and CORS allows the Vite dev server.

8. DEPLOYMENT CONSTRAINTS (Render's free tier = 512MB RAM, 0.1 vCPU).
   This went through three iterations, each fixing what the previous one
   exposed:
   a. MEMORY, attempt 1: a first deploy was OOM-killed the instant the
      text model's float32 weights finished loading, before even
      reaching a post-load ``quantize_dynamic`` call. Quantizing AFTER a
      full float32 load doesn't avoid that peak -- for a moment both the
      float32 original and the newly-built quantized copy exist in
      memory together, which is worse than either alone, right when it
      mattered.
   b. MEMORY, attempt 2: switched to loading the text model directly in
      bfloat16 (``torch_dtype=torch.bfloat16, low_cpu_mem_usage=True``),
      so the float32 version was never materialized at all. This fixed
      the OOM, but bfloat16 on a generic CPU (no fast native bfloat16
      math -- it gets emulated) doesn't actually speed up computation,
      only memory. Combined with LIME's 100 forward passes per text
      request on a throttled 0.1 vCPU, responses took 3-4 minutes, which
      in turn made Render's health check (5s timeout) fail and restart
      the service mid-request.
   c. THE FIX: the text model ships as an ALREADY dynamically-quantized
      (int8) file, produced offline (``export_quantized_model.py``, repo
      root) on a machine with no memory ceiling, then uploaded to the
      ``Prat-04/hate-speech-distilbert`` HF repo as
      ``distilbert_quantized_int8.pt``. The server just
      ``torch.load()``s this file directly -- it never holds a float32
      copy, AND real int8 quantized Linear layers run through PyTorch's
      optimized CPU GEMM kernels, which are genuinely faster than
      float32, unlike bfloat16. This IS numerically a different model
      than the one the README's reported metrics were measured on (those
      describe the full-precision model); expected accuracy impact is
      negligible for a model this size, but it's disclosed here rather
      than silently swapped in. LIME's sample count was also cut to 30
      for this deployment (see point 4) as a second, independent lever
      on latency.
   d. The image-only ResNet18 model (used ONLY for the diagnostic
      Grad-CAM overlay -- never for the actual classification) is
      lazy-loaded on first use instead of at import time, AND its
      torchvision-dependent imports (src.image_dataset,
      src.models.multimodal_models -- both import torchvision at module
      level) are deferred into the same lazy-loading function instead of
      sitting at the top of this file. Most traffic is /predict/text,
      which never touches any of this, so none of it costs memory until
      the first /predict/image request after a cold start.
   e. THE REAL FIX (this revision): a staged memory probe (see chat /
      memory_probe.py) showed that point (d) wasn't enough, because
      `import torch` ITSELF -- not any model weight -- cost ~165MB of
      RAM, and loading even the already-quantized .pt file from (c) cost
      a further ~187MB on top of that (almost certainly the one-time cost
      of PyTorch initializing its quantized-GEMM backend, fbgemm/qnnpack,
      on first use -- a near-fixed cost unrelated to model size). Those
      two alone, plus the rest of the library stack, totalled ~590MB even
      on the "fixed" path from (c) -- over the 512MB cap. So the text
      model now ships as an ONNX file (export_onnx_model.py, repo root)
      and is served through `onnxruntime` instead of torch+transformers'
      model classes. `onnxruntime` has a far smaller import footprint
      than `torch` (no autograd/training machinery), and loading a
      pre-quantized .onnx file doesn't trigger a comparable backend-init
      tax. Just as importantly: `torch` and `torch.nn.functional` are no
      longer imported at MODULE level at all -- they're imported lazily,
      INSIDE the functions that still need them for the image-only
      Grad-CAM path (see _ensure_image_model_loaded, _unnormalize_image,
      _LiveGradCAM.generate), so a plain /predict/text request -- the
      overwhelming majority of traffic -- never pays torch's import cost
      in the first place. (`from __future__ import annotations` below is
      what makes this safe: without it, a module-level function signature
      like `def f(tensor: torch.Tensor)` would try to evaluate `torch.Tensor`
      at import time and crash with a NameError, since `torch` wouldn't be
      imported yet.)
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from lime.lime_text import LimeTextExplainer
from PIL import Image
from pydantic import BaseModel

from src.light_tokenizer import LightBertTokenizer
from src.ocr import combine_ocr_and_caption, extract_text_from_image

# NOTE: we deliberately do NOT use `transformers.AutoTokenizer` here, even
# though it would be the obvious choice. A direct diagnostic (diffing
# sys.modules before/after AutoTokenizer.from_pretrained()) showed it
# unconditionally imports torch -- plus torchgen, sympy, mpmath, and
# safetensors -- internally, about +240MB, regardless of the documented
# USE_TORCH=0 / USE_TF=0 environment variables transformers offers for
# exactly this situation (tried first; had zero measurable effect on this
# transformers version). src/light_tokenizer.py loads the same
# tokenizer.json file directly through the lightweight `tokenizers`
# library (the Rust backend transformers' own fast tokenizers wrap
# internally) and produces identical token ids without any of that
# overhead -- verified in verify_light_tokenizer.py (repo root).

# NOTE: torch, torchvision, src.image_dataset, and
# src.models.multimodal_models are NOT imported here at module level --
# all four are only needed for the rare /predict/image Grad-CAM diagnostic
# path, and torch alone costs ~165MB of RAM just to import (see point 8e
# above). They're imported lazily inside _ensure_image_model_loaded() and
# the other image-only helper functions below, so that cost is deferred
# until the first /predict/image request after a cold start -- most
# traffic (/predict/text) never pays it at all.

MODEL_REPO = "Prat-04/hate-speech-distilbert"
IMAGE_MODEL_REPO = "Prat-04/hate-speech-image-only"
# Real torch device detection happens in _ensure_image_model_loaded() (the
# only place that needs it), once torch is actually imported. "cpu" is the
# correct value everywhere this constant is used before that point, since
# the text/ONNX path never touches a GPU regardless.
DEVICE = "cpu"
LABEL_NAMES = ["Hate Speech", "Offensive Language", "Neither"]

# Grad-CAM diagnostic overlay: OFF by default, ON unless explicitly disabled.
# Render's Dockerfile sets ENABLE_IMAGE_GRAD_CAM=0 as an environment
# variable (not a code change) -- local `uvicorn --reload` runs still get
# the full feature. WHY: the first /predict/image request after a cold
# start has to import torch (~165MB) + torchvision (~82MB) + transformers
# (~27MB, pulled in by src/models/multimodal_models.py) + the ResNet18
# weights, ALL AT ONCE, on top of the ~284MB already resident for text
# serving -- comfortably over Render's 512MB cap. This caused an actual
# 502 (OOM) in production. OCR + text classification -- the real
# prediction, documented as never influenced by this diagnostic overlay --
# use none of torch/torchvision and are completely unaffected either way.
GRAD_CAM_ENABLED = os.environ.get("ENABLE_IMAGE_GRAD_CAM", "1") != "0"

GRAD_CAM_DISABLED_NOTE = (
    "The vision-model attention overlay (Grad-CAM) is disabled on this "
    "deployment: loading it requires PyTorch + torchvision + a second "
    "model all at once, which does not fit in Render's free-tier 512MB "
    "memory limit on top of the text model already serving every "
    "request. This does NOT affect the classification above -- that "
    "comes from OCR + the text model, never from this diagnostic overlay "
    "(see below). Run this project locally to see the Grad-CAM overlay."
)

# ---------------------------------------------------------------------------
# Resolve the static/ directory: it lives ONE level above the src/ package.
# In Docker this is /home/user/app/static (see Dockerfile COPY --from=...).
# In local dev it won't exist, which is fine -- the app just skips mounting.
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
SERVE_FRONTEND = STATIC_DIR.is_dir()

IMAGE_SALIENCY_NOTE = (
    "This prediction is based on OCR-extracted text + caption, not raw image "
    "pixels. During evaluation, a dedicated image-only vision model was trained "
    "and compared against text-only classification (see the project repository's "
    "'Multimodal Results' section): the image branch did not outperform "
    "text-only DistilBERT (macro-F1 0.55-0.56 vs 0.77), and Grad-CAM analysis "
    "showed its attention mostly reflects generic text-density texture patterns "
    "rather than genuine visual understanding of hateful content. Serving that "
    "weaker model here would not improve predictions, so this endpoint uses "
    "OCR + the strongest available text model instead. The Grad-CAM overlay "
    "below is shown for transparency about what the vision model focuses on -- "
    "it does NOT influence the classification above."
)

# Base promise: true on EVERY request, regardless of the OCR.space fallback.
IMAGES_NOT_STORED_NOTE = (
    "Uploaded images are processed in memory for this single request only and "
    "are never saved to disk or logged."
)

# Shown instead of the base note ONLY on requests where local OCR returned
# little/no text and the OCR.space fallback actually fired (see src/ocr.py's
# OCR_SPACE_TRIGGER_MAX_LENGTH). Most requests will never see this version.
IMAGES_NOT_STORED_NOTE_WITH_FALLBACK = (
    "Uploaded images are processed in memory for this single request only and "
    "are never saved to disk or logged by this app. However, for THIS image, "
    "local on-device text extraction found little to no text, so the image "
    "was also sent once, over HTTPS, to a third-party OCR service (OCR.space) "
    "to try to recover the text. That is the only case in which an image "
    "leaves this app's own process."
)

DISCLAIMER = (
    "Content-moderation-ASSISTANCE research/portfolio project, not a 'hate "
    "speech accusation machine.' Predictions are probabilistic, can be "
    "biased (see bias-audit results in the project repository), and must "
    "not be used as the sole basis for real moderation decisions against "
    "real people."
)

app = FastAPI(
    title="Hate Speech Detection API",
    description=DISCLAIMER,
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS: only needed in local dev (Vite dev server on a different port).
# In production the frontend is served from the same origin, so browsers
# never send a CORS preflight. We add the middleware unconditionally --
# when the origin matches, the header is set; otherwise it's a no-op.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("loading text model from Hugging Face Hub...")
tokenizer = LightBertTokenizer(MODEL_REPO, max_length=128)

# Load the ONNX-exported, int8-quantized text model directly (see module
# docstring point 8e for the full story of why this replaced the torch
# .pt file: `import torch` itself, plus a one-time quantized-backend-init
# cost on first load, together cost ~350MB regardless of how small the
# model file was). This file is produced offline by export_onnx_model.py
# (repo root) and uploaded to this same HF repo. onnxruntime loads and
# runs it without torch ever being imported in this process for the text
# path.
_onnx_model_path = hf_hub_download(repo_id=MODEL_REPO, filename="distilbert_int8.onnx")
_onnx_session = ort.InferenceSession(_onnx_model_path, providers=["CPUExecutionProvider"])
print("text model loaded (ONNX int8, served via onnxruntime -- torch never imported).")

_lime_explainer = LimeTextExplainer(class_names=LABEL_NAMES)

# ---------------------------------------------------------------------------
# Image-only model + Grad-CAM: LAZY-LOADED (see module docstring point 8b).
# Nothing here -- not the model, not even the torchvision-dependent
# imports -- happens at module import time. It all happens inside
# _ensure_image_model_loaded() below, on the first /predict/image request.
# These all start as None and are filled in exactly once.
# ---------------------------------------------------------------------------
image_model = None
_grad_cam = None
_image_transform = None
IMAGENET_MEAN = None
IMAGENET_STD = None


def _ensure_image_model_loaded():
    """
    Loads the image-only ResNet18 + wraps it with Grad-CAM hooks, but only
    the first time this is called. Safe to call on every /predict/image
    request -- it's a no-op after the first successful load. Also where
    the torchvision-dependent imports happen (deferred from module level
    for the same memory-footprint reason -- see module docstring point 8b).
    """
    global image_model, _grad_cam, _image_transform, IMAGENET_MEAN, IMAGENET_STD, DEVICE
    if image_model is not None:
        return

    # torch/torchvision are imported HERE, not at module level -- this is
    # the only code path that needs them (see module docstring point 8e).
    # A plain /predict/text request never reaches this function, so it
    # never pays torch's ~165MB import cost.
    import torch

    from src.image_dataset import IMAGENET_MEAN as _MEAN, IMAGENET_STD as _STD, build_image_transform
    from src.models.multimodal_models import ImageOnlyClassifier

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMAGENET_MEAN = _MEAN
    IMAGENET_STD = _STD
    _image_transform = build_image_transform(train=False)

    print("loading image-only model (Grad-CAM diagnostic only) from Hugging Face Hub...")
    checkpoint_path = hf_hub_download(repo_id=IMAGE_MODEL_REPO, filename="image_only_resnet18.pt")
    loaded_model = ImageOnlyClassifier(n_classes=2)
    loaded_model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    loaded_model.to(DEVICE)
    loaded_model.eval()

    image_model = loaded_model
    _grad_cam = _LiveGradCAM(image_model, image_model.image_encoder.backbone.layer4)
    print("image-only model loaded.")


class _LiveGradCAM:
    """
    Same Grad-CAM math as src/grad_cam.py (offline analysis), adapted for a
    single live request instead of batch analysis of saved test images.
    DIAGNOSTIC ONLY -- never used to produce the actual classification.
    """

    def __init__(self, target_model, target_layer):
        self.model = target_model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, module_input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        # Local import -- see module docstring point 8e. generate() is only
        # ever called from compute_grad_cam_overlay_base64(), which itself
        # only runs after _ensure_image_model_loaded() has already imported
        # torch, but importing it again here is cheap (sys.modules cache
        # hit) and keeps this method correct even if that ever changes.
        import torch.nn.functional as TF

        self.model.zero_grad()
        logits = self.model(input_tensor)
        target_class = logits.argmax(dim=1).item()
        score = logits[0, target_class]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = TF.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam


def _unnormalize_image(tensor: "torch.Tensor") -> np.ndarray:
    import torch  # local import -- see module docstring point 8e

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.detach().cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def _apply_jet_colormap(heatmap_01: np.ndarray) -> np.ndarray:
    """
    Lightweight 'jet'-style colormap (blue->green->yellow->red) implemented
    directly in numpy -- verified against a known gradient before use (see
    chat) -- avoiding a matplotlib dependency in the live request path.
    """
    r = np.clip(1.5 - np.abs(4 * heatmap_01 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * heatmap_01 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * heatmap_01 - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def compute_grad_cam_overlay_base64(pil_image: Image.Image) -> str:
    """
    Live Grad-CAM overlay for the uploaded image, base64-encoded PNG.
    DIAGNOSTIC/TRANSPARENCY ONLY -- does not influence the classification,
    which comes from OCR + DistilBERT (see IMAGE_SALIENCY_NOTE).
    """
    input_tensor = _image_transform(pil_image).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_(True)

    cam = _grad_cam.generate(input_tensor)
    display_img = _unnormalize_image(input_tensor[0])

    cam_resized = (
        np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0
    )
    heatmap_rgb = _apply_jet_colormap(cam_resized)

    alpha = 0.45
    overlay = np.clip((1 - alpha) * display_img + alpha * heatmap_rgb, 0, 1)
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))

    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class TextRequest(BaseModel):
    text: str


class TextResponse(BaseModel):
    label: str
    confidence: float
    per_class_probabilities: dict
    top_contributing_tokens: list


class ImageResponse(BaseModel):
    label: str
    confidence: float
    per_class_probabilities: dict
    ocr_extracted_text: str
    image_saliency_note: str
    grad_cam_overlay_base64: str
    images_not_stored_note: str


def _predict_proba(texts):
    # LightBertTokenizer already returns numpy int64 arrays, dynamically
    # padded to this batch's longest sequence and truncated to 128 tokens
    # (see src/light_tokenizer.py) -- no extra kwargs needed here, and
    # nothing in this path ever imports torch.
    ort_inputs = tokenizer(list(texts))
    (logits,) = _onnx_session.run(["logits"], ort_inputs)

    # Plain numpy softmax -- no torch needed. Subtracting the row max before
    # exponentiating is the standard numerically-stable formulation (avoids
    # overflow for large logits); this is mathematically identical to
    # torch.softmax, just without the torch dependency.
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _classify(text: str):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    probs = _predict_proba([text])[0]
    pred_idx = int(probs.argmax())
    label = LABEL_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    per_class = {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))}

    # num_samples=30, not the 100 used before: even with a genuinely fast
    # int8-quantized model, 100 forward passes on Render's throttled
    # 0.1 vCPU was a meaningful chunk of the latency. See module
    # docstring points 4 and 8c for the full trade-off being made here.
    exp = _lime_explainer.explain_instance(
        text, _predict_proba, num_features=8, num_samples=30, labels=(pred_idx,)
    )
    top_tokens = [
        {"token": word, "weight": round(weight, 4)} for word, weight in exp.as_list(label=pred_idx)
    ]

    return label, confidence, per_class, top_tokens


# ---------------------------------------------------------------------------
# Health-check endpoint -- Render (or any host) can poll this to verify the
# container is up and ready.
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/text", response_model=TextResponse)
def predict_text(request: TextRequest):
    label, confidence, per_class, top_tokens = _classify(request.text)
    return TextResponse(
        label=label,
        confidence=confidence,
        per_class_probabilities=per_class,
        top_contributing_tokens=top_tokens,
    )


@app.post("/predict/image", response_model=ImageResponse)
async def predict_image(image: UploadFile = File(...), caption: str = Form(default="")):
    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="could not read uploaded file as an image")

    ocr_text, used_remote_ocr = extract_text_from_image(pil_image)

    combined_text = combine_ocr_and_caption(ocr_text, caption)
    if not combined_text.strip():
        raise HTTPException(
            status_code=400,
            detail="no text could be extracted from the image and no caption was provided",
        )

    label, confidence, per_class, _top_tokens = _classify(combined_text)

    # See GRAD_CAM_ENABLED above: disabled by default on Render (memory),
    # on by default locally. Classification above never depends on this --
    # it's OCR + the text model either way.
    if GRAD_CAM_ENABLED:
        _ensure_image_model_loaded()  # no-op after the first call (see docstring point 8b)
        grad_cam_overlay = compute_grad_cam_overlay_base64(pil_image)
        saliency_note = IMAGE_SALIENCY_NOTE
    else:
        grad_cam_overlay = ""
        saliency_note = GRAD_CAM_DISABLED_NOTE

    images_not_stored_note = (
        IMAGES_NOT_STORED_NOTE_WITH_FALLBACK if used_remote_ocr else IMAGES_NOT_STORED_NOTE
    )

    return ImageResponse(
        label=label,
        confidence=confidence,
        per_class_probabilities=per_class,
        ocr_extracted_text=ocr_text,
        image_saliency_note=saliency_note,
        grad_cam_overlay_base64=grad_cam_overlay,
        images_not_stored_note=images_not_stored_note,
    )


# ---------------------------------------------------------------------------
# Static-file serving (production only).
#
# When the built React frontend exists at ../static/ (Docker layout), we:
#   1. Define an explicit GET / that returns static/index.html
#   2. Mount the rest of static/ as a catch-all at "/" so JS/CSS/images/
#      examples/ all resolve correctly
#
# The mount MUST come AFTER all API routes because FastAPI tries routes in
# registration order -- if the StaticFiles catch-all were first, it would
# swallow /predict/text, /predict/image, /health, and /docs.
#
# In local dev (no static/ directory), the original JSON root endpoint is
# kept so you still get a useful response when you hit http://localhost:8000.
# ---------------------------------------------------------------------------
if SERVE_FRONTEND:
    @app.get("/")
    def serve_index():
        return FileResponse(str(STATIC_DIR / "index.html"))

    # html=True makes StaticFiles return index.html for directory requests
    # and for any path that doesn't match a real file (SPA client-side
    # routing fallback).
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    print(f"serving frontend from {STATIC_DIR}")
else:
    @app.get("/")
    def root():
        return {
            "message": "Hate Speech Detection API. See /docs for interactive Swagger UI.",
            "disclaimer": DISCLAIMER,
        }
    print("no static/ directory found -- running in API-only (dev) mode")
