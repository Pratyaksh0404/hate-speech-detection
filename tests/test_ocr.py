"""
Tests for src/ocr.py.

test_combine_* tests are pure Python string logic — I verified these
myself directly. test_extract_text_from_image_* tests need `pytesseract`
(pip) AND the tesseract system binary installed — neither is available in
the sandbox that wrote this. Kaggle's environment has the tesseract binary
already (common for their image-processing stack), but installing
pytesseract on Windows also requires the separate Tesseract-OCR system
installer (not just `pip install pytesseract`) — given that extra
friction, it's more practical to verify extract_text_from_image() directly
on Kaggle rather than requiring a Windows-side system install just for
this.
"""

import pytest

from src.ocr import combine_ocr_and_caption

pytesseract = pytest.importorskip("pytesseract")
from src.ocr import extract_text_from_image  # noqa: E402


def test_combine_both_present():
    result = combine_ocr_and_caption("NORTH KOREA IS EVIL", "Bernie Sanders meme")
    assert result == "Bernie Sanders meme | NORTH KOREA IS EVIL"


def test_combine_ocr_only():
    assert combine_ocr_and_caption("SOME MEME TEXT", "") == "SOME MEME TEXT"


def test_combine_caption_only():
    assert combine_ocr_and_caption("", "just a caption") == "just a caption"


def test_combine_neither_present():
    assert combine_ocr_and_caption(None, None) == ""


def test_combine_strips_whitespace():
    assert combine_ocr_and_caption("  padded text  ", "") == "padded text"


def test_extract_text_from_image_missing_file_returns_empty_not_crash():
    assert extract_text_from_image("/nonexistent/path/fake.png") == ""
