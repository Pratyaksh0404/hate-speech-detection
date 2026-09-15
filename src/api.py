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
   decision (with a pointer to the repo's Grad-CAM analysis) rather than
   live per-request Grad-CAM computation -- the image-only model's signal
   is weak enough that running it live would add latency and a second
   model to serve for no prediction-quality benefit. Disclosed openly in
   the response itself, not hidden.

3. Both the tokenizer and model are loaded ONCE at module import time, not
   per-request -- standard practice for serving latency; re-loading a
   transformer on every request would make the API unusably slow.

4. top_contributing_tokens uses LIME -- the same method used throughout
   this project's offline error analysis, kept consistent rather than
   introducing a different explainability method just for the live API --
   with a reduced num_samples (100, vs. 500 used offline) to keep
   single-request latency reasonable.

5. Raw text (not clean_text()) is fed directly to DistilBERT's tokenizer,
   matching exactly how the model was trained (see src/bert_dataset.py) --
   using different preprocessing at serve time than train time would
   silently degrade predictions.
"""

import io

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from lime.lime_text import LimeTextExplainer
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.ocr import combine_ocr_and_caption, extract_text_from_image

MODEL_REPO = "Prat-04/hate-speech-distilbert"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Hate Speech", "Offensive Language", "Neither"]

IMAGE_SALIENCY_NOTE = (
    "This prediction is based on OCR-extracted text + caption, not raw image "
    "pixels. During evaluation, a dedicated image-only vision model was trained "
    "and compared against text-only classification (see the project repository's "
    "'Multimodal Results' section): the image branch did not outperform "
    "text-only DistilBERT (macro-F1 0.55-0.56 vs 0.77), and Grad-CAM analysis "
    "showed its attention mostly reflects generic text-density texture patterns "
    "rather than genuine visual understanding of hateful content. Serving that "
    "weaker model here would not improve predictions, so this endpoint uses "
    "OCR + the strongest available text model instead."
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

print("loading model from Hugging Face Hub...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
model.to(DEVICE)
model.eval()
print("model loaded.")

_lime_explainer = LimeTextExplainer(class_names=LABEL_NAMES)


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

    ocr_text = extract_text_from_image(pil_image)

    combined_text = combine_ocr_and_caption(ocr_text, caption)
    if not combined_text.strip():
        raise HTTPException(
            status_code=400,
            detail="no text could be extracted from the image and no caption was provided",
        )

    label, confidence, per_class, _top_tokens = _classify(combined_text)

    return ImageResponse(
        label=label,
        confidence=confidence,
        per_class_probabilities=per_class,
        ocr_extracted_text=ocr_text,
        image_saliency_note=IMAGE_SALIENCY_NOTE,
    )


@app.get("/")
def root():
    return {"message": "Hate Speech Detection API. See /docs for interactive Swagger UI.", "disclaimer": DISCLAIMER}
