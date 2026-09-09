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
"""

import pytesseract
from PIL import Image


def extract_text_from_image(image_path: str, min_upscale_width: int = 800) -> str:
    """
    Runs OCR on a single image, returning "" (never None, never a crash)
    if no text is found or the file can't be read.
    """
    try:
        img = Image.open(image_path).convert("L")  # grayscale
    except Exception:
        return ""

    if img.width < min_upscale_width:
        scale = min_upscale_width / img.width
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)

    try:
        text = pytesseract.image_to_string(img)
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
