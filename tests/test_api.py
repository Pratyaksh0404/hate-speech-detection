"""
Tests for src/api.py. Genuine integration tests -- importing src.api loads
real models from Hugging Face Hub (text model + image-only model for
Grad-CAM) at module level, needs real internet access. Not runnable in the
sandbox that wrote this; meant to run on your machine.
"""

import base64
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
    candidate_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for path in candidate_paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _make_test_image_with_text(text: str) -> bytes:
    img = Image.new("RGB", (900, 200), color="white")
    draw = ImageDraw.Draw(img)
    font = _load_test_font(40)
    draw.text((20, 70), text, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _assert_valid_grad_cam_overlay(body):
    assert "grad_cam_overlay_base64" in body
    assert len(body["grad_cam_overlay_base64"]) > 0
    decoded_bytes = base64.b64decode(body["grad_cam_overlay_base64"])
    decoded_img = Image.open(io.BytesIO(decoded_bytes))
    assert decoded_img.size == (224, 224)
    assert decoded_img.mode == "RGB"

    assert "images_not_stored_note" in body
    assert len(body["images_not_stored_note"]) > 0


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
    _assert_valid_grad_cam_overlay(body)


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
    _assert_valid_grad_cam_overlay(body)


def test_predict_image_not_an_image_returns_400():
    response = client.post(
        "/predict/image",
        files={"image": ("not_image.txt", b"this is not an image", "text/plain")},
        data={"caption": ""},
    )
    assert response.status_code == 400
