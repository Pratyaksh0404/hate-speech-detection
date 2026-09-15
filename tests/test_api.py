"""
Tests for src/api.py. These are genuine integration tests, not pure unit
tests -- importing src.api loads the real model from Hugging Face Hub at
module level (by design, for serving-latency reasons), which needs real
internet access. Not runnable in the sandbox that wrote this (no network);
this is meant to run on your machine.

First run will be slow-ish (downloading DistilBERT + tokenizer from the
Hub, cached after). LIME explanations also add a few seconds per text
request (100 samples) -- expect these tests to take longer than our other
test suites, that's expected, not a bug.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from src.api import LABEL_NAMES, app

client = TestClient(app)


def test_root_returns_disclaimer():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "disclaimer" in body
    assert len(body["disclaimer"]) > 0


def test_predict_text_valid_response_schema():
    response = client.post("/predict/text", json={"text": "have a wonderful day everyone"})
    assert response.status_code == 200
    body = response.json()

    assert body["label"] in LABEL_NAMES
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["per_class_probabilities"].keys()) == set(LABEL_NAMES)
    assert abs(sum(body["per_class_probabilities"].values()) - 1.0) < 1e-3
    assert isinstance(body["top_contributing_tokens"], list)
    assert len(body["top_contributing_tokens"]) > 0
    assert "token" in body["top_contributing_tokens"][0]
    assert "weight" in body["top_contributing_tokens"][0]


def test_predict_text_empty_returns_400():
    response = client.post("/predict/text", json={"text": ""})
    assert response.status_code == 400


def test_predict_text_whitespace_only_returns_400():
    response = client.post("/predict/text", json={"text": "   "})
    assert response.status_code == 400


def _load_test_font(size: int = 40):
    """
    Tries common cross-platform TrueType font paths, falling back to
    Pillow's scalable default font (load_default(size=...), added in
    Pillow 10.1+) if none are found. A single hardcoded Linux-only path
    broke this test on Windows: PIL silently fell back to its OLD tiny,
    non-scalable bitmap font, rendering text too small for Tesseract to
    read reliably -- OCR correctly returned nothing given genuinely
    unreadable input, so the 400 we saw was actually the API behaving
    correctly on a flawed test image, not an API bug.
    """
    candidate_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        "C:\\Windows\\Fonts\\arialbd.ttf",  # Windows (Arial Bold)
        "C:\\Windows\\Fonts\\arial.ttf",  # Windows (Arial)
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS
    ]
    for path in candidate_paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow 10.1+, scalable
    except TypeError:
        return ImageFont.load_default()  # older Pillow, tiny fixed size


def _make_test_image_with_text(text: str) -> bytes:
    img = Image.new("RGB", (900, 200), color="white")
    draw = ImageDraw.Draw(img)
    font = _load_test_font(40)
    draw.text((20, 70), text, fill="black", font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def test_predict_image_with_readable_text():
    image_bytes = _make_test_image_with_text("HAVE A WONDERFUL DAY")
    response = client.post(
        "/predict/image",
        files={"image": ("test.png", image_bytes, "image/png")},
        data={"caption": ""},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["label"] in LABEL_NAMES
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["ocr_extracted_text"].strip()) > 0
    assert "WONDERFUL" in body["ocr_extracted_text"].upper() or "DAY" in body["ocr_extracted_text"].upper()
    assert len(body["image_saliency_note"]) > 0


def test_predict_image_blank_no_caption_returns_400():
    blank = Image.new("RGB", (200, 200), color="white")
    buf = io.BytesIO()
    blank.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict/image",
        files={"image": ("blank.png", buf.read(), "image/png")},
        data={"caption": ""},
    )
    assert response.status_code == 400


def test_predict_image_no_text_but_has_caption_still_works():
    blank = Image.new("RGB", (200, 200), color="white")
    buf = io.BytesIO()
    blank.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict/image",
        files={"image": ("blank.png", buf.read(), "image/png")},
        data={"caption": "this is a caption with no image text"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in LABEL_NAMES


def test_predict_image_not_an_image_returns_400():
    response = client.post(
        "/predict/image",
        files={"image": ("not_image.txt", b"this is not an image", "text/plain")},
        data={"caption": ""},
    )
    assert response.status_code == 400
