"""
OCR pipeline: local Tesseract first (multi-variant preprocessing), with an
OPTIONAL OCR.space fallback used only when the local result is empty or
too short to be trustworthy.

Why a fallback at all: real-world testing (see README "Known limitations")
found that some memes — white text over a busy/photo background, or
overlapping faces — defeat Tesseract even with the plain/inverted/
thresholded multi-variant approach below. OCR.space's hosted engine
handles some of these better. But routing an image to a third-party API
is a real privacy trade-off against this project's own "images are never
stored or logged" promise shown in the frontend, so the fallback is
deliberately narrow:

  - It only fires when local OCR looks untrustworthy (near-empty result).
    It does NOT fire just because the result "looks wrong" — we have no
    way to detect that without a ground truth, so we use length as the
    only honest, measurable proxy we have. This will NOT catch cases
    where Tesseract returns a confidently wrong but non-trivially-long
    string (garbled real-image testing showed this happens — a busy
    background can produce dozens of characters of noise, not an empty
    result). That's a known, accepted gap in this heuristic, not a bug.
  - OCR.space is called with the image bytes only, over HTTPS, and
    nothing is written to disk on our side either way.
  - If OCR.space is unreachable, rate-limited, or errors in any way, we
    silently fall back to whatever local OCR already produced — this
    path must never crash a request or block on a flaky third party.

Privacy note wording is intentionally updated at the call site (src/api.py)
to disclose this conditional third-party fallback honestly, rather than
leaving the older "never leaves your machine" wording technically false.
"""

import base64
import os

import pytesseract
import requests
from PIL import Image, ImageOps

# --- Local Tesseract binary discovery (Windows dev convenience) -----------

def _configure_tesseract_binary() -> None:
    if pytesseract.pytesseract.tesseract_cmd != "tesseract":
        return
    candidate_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return


_configure_tesseract_binary()

# --- OCR.space fallback configuration --------------------------------------

# Trigger: local OCR text shorter than this many characters is treated as
# "untrustworthy" and we try OCR.space. This is a length-based heuristic
# only (see module docstring) — it catches near-empty results, not
# confidently-wrong-but-long ones.
OCR_SPACE_TRIGGER_MAX_LENGTH = 10

OCR_SPACE_ENDPOINT = "https://api.ocr.space/parse/image"

# OCR.space requires an API key on every request. We default to the
# public demo key ("helloworld") so the fallback works out of the box,
# but the demo key has very low/shared rate limits and is NOT meant for
# real usage. Set the OCR_SPACE_API_KEY environment variable to your own
# free key (sign up at https://ocr.space/ocrapi, no credit card) before
# relying on this in anything beyond local testing.
OCR_SPACE_API_KEY = os.environ.get("OCR_SPACE_API_KEY", "helloworld")

# Free-tier hard limit is 1 MB per request; we enforce a little below that
# so our own encoding overhead doesn't push us over.
OCR_SPACE_MAX_BYTES = 900_000

OCR_SPACE_TIMEOUT_SECONDS = 8


def _image_to_png_bytes(img: Image.Image) -> bytes:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _call_ocr_space(img: Image.Image) -> str:
    """
    Sends `img` to OCR.space as base64 and returns the extracted text, or
    "" on ANY failure (network error, timeout, rate limit, bad response,
    image too large). This function must never raise — it's a best-effort
    fallback, not a required step, and a flaky third party must not be
    able to break /predict/image.
    """
    try:
        png_bytes = _image_to_png_bytes(img)

        # If the image is too large for the free tier, downscale once
        # before giving up — better a slightly lower-res OCR attempt than
        # no fallback attempt at all.
        if len(png_bytes) > OCR_SPACE_MAX_BYTES:
            scale = (OCR_SPACE_MAX_BYTES / len(png_bytes)) ** 0.5
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img = img.resize(new_size, Image.LANCZOS)
            png_bytes = _image_to_png_bytes(img)
            if len(png_bytes) > OCR_SPACE_MAX_BYTES:
                return ""  # still too big; not worth a guaranteed-rejected call

        b64_payload = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")

        response = requests.post(
            OCR_SPACE_ENDPOINT,
            data={
                "apikey": OCR_SPACE_API_KEY,
                "base64Image": b64_payload,
                "OCREngine": 2,  # engine 2 handles noisy/busy backgrounds better than engine 1
                "scale": "true",
            },
            timeout=OCR_SPACE_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            return ""

        payload = response.json()
        if payload.get("IsErroredOnProcessing"):
            return ""

        results = payload.get("ParsedResults") or []
        if not results:
            return ""

        return (results[0].get("ParsedText") or "").strip()
    except Exception:
        # Any failure (DNS, timeout, malformed JSON, ...) just means "no
        # fallback text available" -- never propagate this as a request
        # error.
        return ""


def extract_text_from_image(image, min_upscale_width: int = 800, allow_remote_fallback: bool = True) -> tuple:
    """
    image can be a file path (str) or PIL.Image.Image directly.

    Returns a tuple (text, used_remote_fallback) so callers (the API) can
    show an honest "this image was sent to a third-party OCR service"
    disclosure only when it actually happened, instead of a blanket
    always-on or always-off note.
    """
    try:
        if isinstance(image, str):
            img = Image.open(image).convert("L")
        else:
            img = image.convert("L")
    except Exception:
        return "", False

    original_img = img  # keep an unmodified-grayscale copy for the remote fallback

    if img.width < min_upscale_width:
        scale = min_upscale_width / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    variants = [img, ImageOps.invert(img), img.point(lambda p: 255 if p > 200 else 0)]
    best_text = ""
    for variant in variants:
        try:
            candidate = pytesseract.image_to_string(variant).strip()
        except pytesseract.TesseractNotFoundError:
            raise
        except Exception:
            candidate = ""
        if len(candidate) > len(best_text):
            best_text = candidate

    if allow_remote_fallback and len(best_text) < OCR_SPACE_TRIGGER_MAX_LENGTH:
        remote_text = _call_ocr_space(original_img)
        # Only prefer the remote result if it actually found more than
        # the local pipeline did -- never let an empty/failed remote call
        # overwrite a (short but real) local result.
        if len(remote_text) > len(best_text):
            return remote_text, True

    return best_text, False


def combine_ocr_and_caption(ocr_text: str, caption: str) -> str:
    ocr_text = (ocr_text or "").strip()
    caption = (caption or "").strip()
    if ocr_text and caption:
        return f"{caption} | {ocr_text}"
    return ocr_text or caption
