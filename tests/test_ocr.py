"""
Tests for src/ocr.py.

Two groups:
1. test_combine_*  -- pure Python string logic, no external dependencies.
2. test_extract_*  -- need pytesseract import; skipped if not installed.
3. test_ocr_space_* -- mock-based tests for the OCR.space fallback path.
   These mock BOTH pytesseract.image_to_string (so we control what "local
   OCR" returns) AND requests.post (so no real network call is made).
   They run without Tesseract installed.
"""

import pytest

from src.ocr import combine_ocr_and_caption

pytesseract = pytest.importorskip("pytesseract")
from src.ocr import extract_text_from_image  # noqa: E402


# ── combine_ocr_and_caption tests (pure string logic) ───────────────────

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


# ── extract_text_from_image edge case ────────────────────────────────────

def test_extract_text_from_image_missing_file_returns_empty_not_crash():
    # extract_text_from_image returns (text, used_remote_fallback) tuple
    text, used_fallback = extract_text_from_image("/nonexistent/path/fake.png")
    assert text == ""
    assert used_fallback is False


# ── OCR.space fallback path tests (all mocked, no network) ──────────────
#
# These test the fallback logic in extract_text_from_image() by mocking:
#   - pytesseract.image_to_string  → controls what "local OCR" returns
#   - requests.post                → controls what "OCR.space" returns
#   - Image.open                   → returns a tiny dummy PIL image
#
# This means they run without Tesseract installed and without internet.

from unittest.mock import patch, MagicMock
from PIL import Image


def _make_dummy_image():
    """Create a tiny 10x10 white image for testing."""
    return Image.new("RGB", (10, 10), color="white")


def _make_ocr_space_response(parsed_text: str, is_error: bool = False,
                              status_code: int = 200):
    """Build a mock requests.Response mimicking OCR.space JSON output."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    if is_error:
        mock_resp.json.return_value = {"IsErroredOnProcessing": True}
    else:
        mock_resp.json.return_value = {
            "IsErroredOnProcessing": False,
            "ParsedResults": [{"ParsedText": parsed_text}],
        }
    return mock_resp


@patch("src.ocr.pytesseract.image_to_string")
@patch("src.ocr.requests.post")
def test_ocr_space_fallback_fires_when_local_too_short(mock_post, mock_tess):
    """
    When local Tesseract returns very short text (< 10 chars), the
    OCR.space fallback should fire and its result should be preferred
    if it found more text.
    """
    # Local OCR returns only 2 chars (below the 10-char threshold)
    mock_tess.return_value = "ab"

    # OCR.space returns a longer, better result
    mock_post.return_value = _make_ocr_space_response("VOTE FOR NOBODY 2024")

    text, used_fallback = extract_text_from_image(_make_dummy_image())

    assert text == "VOTE FOR NOBODY 2024"
    assert used_fallback is True
    mock_post.assert_called_once()  # fallback was actually invoked


@patch("src.ocr.pytesseract.image_to_string")
@patch("src.ocr.requests.post")
def test_ocr_space_fallback_skipped_when_local_long_enough(mock_post, mock_tess):
    """
    When local Tesseract returns >=10 chars, OCR.space should NOT be
    called at all — the local result is trusted.
    """
    mock_tess.return_value = "plenty of text here from tesseract"

    text, used_fallback = extract_text_from_image(_make_dummy_image())

    assert text == "plenty of text here from tesseract"
    assert used_fallback is False
    mock_post.assert_not_called()  # no network request made


@patch("src.ocr.pytesseract.image_to_string")
@patch("src.ocr.requests.post")
def test_ocr_space_fallback_disabled_by_flag(mock_post, mock_tess):
    """
    When allow_remote_fallback=False (used by bulk offline data prep),
    OCR.space is never called even if local result is short.
    """
    mock_tess.return_value = "ab"  # short, would normally trigger fallback

    text, used_fallback = extract_text_from_image(
        _make_dummy_image(), allow_remote_fallback=False
    )

    assert text == "ab"
    assert used_fallback is False
    mock_post.assert_not_called()


@patch("src.ocr.pytesseract.image_to_string")
@patch("src.ocr.requests.post")
def test_ocr_space_fallback_handles_network_error(mock_post, mock_tess):
    """
    If OCR.space is unreachable (ConnectionError, timeout, DNS failure,
    etc.), the function must NOT crash — it should return whatever local
    OCR found, with used_remote_fallback=False.
    """
    mock_tess.return_value = "ab"
    mock_post.side_effect = ConnectionError("DNS resolution failed")

    text, used_fallback = extract_text_from_image(_make_dummy_image())

    # Falls back to local result without crashing
    assert text == "ab"
    assert used_fallback is False


@patch("src.ocr.pytesseract.image_to_string")
@patch("src.ocr.requests.post")
def test_ocr_space_fallback_handles_error_response(mock_post, mock_tess):
    """
    If OCR.space returns IsErroredOnProcessing=True (bad image, rate
    limit, etc.), it should be treated as "no result" and the local
    OCR text should be kept.
    """
    mock_tess.return_value = "ab"
    mock_post.return_value = _make_ocr_space_response("", is_error=True)

    text, used_fallback = extract_text_from_image(_make_dummy_image())

    assert text == "ab"
    assert used_fallback is False


@patch("src.ocr.pytesseract.image_to_string")
@patch("src.ocr.requests.post")
def test_ocr_space_fallback_empty_remote_keeps_local(mock_post, mock_tess):
    """
    If OCR.space returns empty text (or shorter than local), the local
    result should be kept — an empty remote result should never overwrite
    a short-but-real local result.
    """
    mock_tess.return_value = "ab"
    mock_post.return_value = _make_ocr_space_response("")  # empty remote

    text, used_fallback = extract_text_from_image(_make_dummy_image())

    # Local "ab" is kept because remote "" is not longer
    assert text == "ab"
    assert used_fallback is False
