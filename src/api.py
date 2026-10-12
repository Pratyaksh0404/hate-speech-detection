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
   (100, vs. 500 used offline) to keep single-request latency reasonable.

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

8. DEPLOYMENT MEMORY FOOTPRINT (Render's free tier = 512MB RAM, 0.1 CPU):
   a. The text DistilBERT model is dynamically quantized (int8 weights for
      its Linear layers) right after loading, CPU-only -- a standard,
      no-retraining technique for serving BERT-family models on CPU. It
      shrinks the model's largest layers and speeds up CPU inference.
      Expected accuracy impact is negligible, but it IS numerically a
      different model than the one the README's reported metrics were
      measured on: those describe the full-precision model; this
      quantized copy is a serving-time optimization, disclosed here
      rather than silently swapped in.
   b. The image-only ResNet18 model (used ONLY for the diagnostic
      Grad-CAM overlay -- never for the actual classification) is
      lazy-loaded on first use instead of at import time. Most traffic
      is /predict/text, which never touches it, so this keeps baseline
      memory low; the first /predict/image request after a cold start
      pays a one-time load cost instead of every process paying for a
      model that may go unused.
"""

import base64
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from lime.lime_text import LimeTextExplainer
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.image_dataset import IMAGENET_MEAN, IMAGENET_STD, build_image_transform
from src.models.multimodal_models import ImageOnlyClassifier
from src.ocr import combine_ocr_and_caption, extract_text_from_image

MODEL_REPO = "Prat-04/hate-speech-distilbert"
IMAGE_MODEL_REPO = "Prat-04/hate-speech-image-only"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Hate Speech", "Offensive Language", "Neither"]

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
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
model.to(DEVICE)
model.eval()

# Dynamic quantization (int8 weights for nn.Linear layers) -- CPU-only in
# PyTorch, which is exactly our deployment target. DistilBERT is mostly
# Linear layers, so this meaningfully shrinks the model's memory footprint
# and speeds up CPU inference, with no retraining and negligible accuracy
# impact. See module docstring point 8a for why this is disclosed rather
# than silently swapped in.
if DEVICE.type == "cpu":
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    print("text model loaded (dynamically quantized for CPU serving).")
else:
    print("text model loaded (GPU available -- skipping CPU-only quantization).")

_lime_explainer = LimeTextExplainer(class_names=LABEL_NAMES)

# ---------------------------------------------------------------------------
# Image-only model + Grad-CAM: LAZY-LOADED (see module docstring point 8b).
# Not loaded here at import time -- only on the first /predict/image
# request, via _ensure_image_model_loaded() below. image_model, _grad_cam
# and _image_transform start as None/undefined and are filled in once.
# ---------------------------------------------------------------------------
image_model = None
_grad_cam = None
_image_transform = build_image_transform(train=False)  # cheap: no weights, just transform config


def _ensure_image_model_loaded():
    """
    Loads the image-only ResNet18 + wraps it with Grad-CAM hooks, but only
    the first time this is called. Safe to call on every /predict/image
    request -- it's a no-op after the first successful load.
    """
    global image_model, _grad_cam
    if image_model is not None:
        return

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


def _unnormalize_image(tensor: torch.Tensor) -> np.ndarray:
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
    encoding = tokenizer(
        list(texts), truncation=True, max_length=128, padding=True, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**encoding).logits
    return torch.softmax(logits, dim=1).cpu().numpy()


def _classify(text: str):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    probs = _predict_proba([text])[0]
    pred_idx = int(probs.argmax())
    label = LABEL_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    per_class = {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))}

    exp = _lime_explainer.explain_instance(
        text, _predict_proba, num_features=8, num_samples=100, labels=(pred_idx,)
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

    _ensure_image_model_loaded()  # no-op after the first call (see docstring point 8b)
    grad_cam_overlay = compute_grad_cam_overlay_base64(pil_image)

    images_not_stored_note = (
        IMAGES_NOT_STORED_NOTE_WITH_FALLBACK if used_remote_ocr else IMAGES_NOT_STORED_NOTE
    )

    return ImageResponse(
        label=label,
        confidence=confidence,
        per_class_probabilities=per_class,
        ocr_extracted_text=ocr_text,
        image_saliency_note=IMAGE_SALIENCY_NOTE,
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
