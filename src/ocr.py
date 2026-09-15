"""
OCR extraction for the multimodal meme dataset (master_plan.md section 4A.3).

Design decisions:

1. Uses pytesseract (a thin Python wrapper around the tesseract OCR
   engine). Verified the underlying tesseract binary directly before
   writing this: a low-resolution synthetic test image produced garbled,
   spaceless output ("THISISATESTMEMECAPTION"), but the SAME text
   rendered at a larger font size / higher effective resolution came out
   correctly with proper word spacing ("THIS IS A TEST MEME CAPTION").
   This confirms OCR quality here is highly resolution-dependent — so we
   upscale small images before running OCR rather than feeding them in
   at their original (sometimes tiny, per our EDA: widths from 471px) size.

2. Convert to grayscale before OCR — a standard, well-established
   preprocessing step: it removes color-channel noise that doesn't help
   character recognition and tends to improve accuracy on real-world
   images (memes with colored backgrounds/gradients) versus feeding raw
   RGB in directly.

3. Images with no readable text (OCR returns empty/whitespace) fall back
   to an empty string, never a crash or placeholder token — the fusion
   step then falls back to the dataset's own `sentence` caption alone for
   that row, exactly the fallback behavior the plan asks us to document.

4. combine_ocr_and_caption() keeps BOTH sources rather than one
   overriding the other. Manual inspection of real examples (see chat)
   showed the dataset's own `sentence` field is an inconsistent mix of
   literal OCR-quality transcripts and paraphrased annotator captions —
   neither source alone is fully reliable, so we give the model both and
   let it learn which is informative per example, falling back to
   whichever single source is non-empty if the other found nothing.
5. Auto-detects the Tesseract-OCR binary on Windows. Installing the
   pytesseract PIP package only installs the Python wrapper -- the actual
   OCR engine is a separate native program that needs its own installer
   (https://github.com/UB-Mannheim/tesseract/wiki on Windows), and that
   installer does not add it to PATH by default. Rather than requiring
   manual PATH edits, this checks the standard install locations and
   points pytesseract directly at the binary if found. No-op on Linux/Mac,
   where the binary is normally already on PATH.

6. TesseractNotFoundError is NOT swallowed by the broad except below --
   it's a genuine configuration/installation problem, not "this image has
   no text," and silently treating it as the latter would make a real
   setup bug indistinguishable from a normal empty result (exactly what
   happened before this was separated out: see chat).
"""

import os

import pytesseract
from PIL import Image


def _configure_tesseract_binary() -> None:
    if pytesseract.pytesseract.tesseract_cmd != "tesseract":
        return  # already explicitly configured elsewhere -- don't override
    candidate_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return


_configure_tesseract_binary()


def extract_text_from_image(image, min_upscale_width: int = 800) -> str:
    """
    Runs OCR on a single image, returning "" (never None, never a crash)
    if no text is found or the image can't be read.

    `image` can be either a file path (str) or an already-loaded
    PIL.Image.Image. Accepting both matters in practice: the FastAPI
    /predict/image endpoint already has the uploaded image in memory, and
    round-tripping it through a temp file just to satisfy a path-only
    function is unnecessary overhead -- and on Windows specifically,
    tempfile.NamedTemporaryFile's exclusive file lock means a second
    open() on that same path (e.g. from PIL) raises PermissionError,
    which is exactly what happened before this function accepted images
    directly (see chat).
    """
    try:
        if isinstance(image, str):
            img = Image.open(image).convert("L")  # grayscale
        else:
            img = image.convert("L")
    except Exception:
        return ""

    if img.width < min_upscale_width:
        scale = min_upscale_width / img.width
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)

    try:
        text = pytesseract.image_to_string(img)
    except pytesseract.TesseractNotFoundError:
        raise  # genuine setup/config error -- must NOT look like "no text found"
    except Exception:
        return ""

    return text.strip()


def combine_ocr_and_caption(ocr_text: str, caption: str) -> str:
    """
    Combines OCR-extracted meme text with the dataset's provided caption.
    See design decision 4 above for why both are kept rather than one
    overriding the other.
    """
    ocr_text = (ocr_text or "").strip()
    caption = (caption or "").strip()

    if ocr_text and caption:
        return f"{caption} | {ocr_text}"
    return ocr_text or caption
