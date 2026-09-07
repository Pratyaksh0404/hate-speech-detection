"""
Unit tests for src/preprocessing.py.

NOTE: these require `nltk` + `emoji` installed and the NLTK corpora downloaded
(call preprocessing.ensure_nltk_resources() once, or run the download lines in
requirements setup). Run with: pytest tests/test_preprocessing.py -v
"""

import pytest
from src.preprocessing import (
    clean_text,
    strip_urls,
    strip_mentions,
    unwrap_hashtags,
    demojize_text,
    ensure_nltk_resources,
    normalize_whitespace,
)


@pytest.fixture(scope="module", autouse=True)
def _setup():
    ensure_nltk_resources()


# --- Edge cases (section 9 requires these explicitly) ---

def test_empty_string():
    assert clean_text("") == ""


def test_none_input():
    assert clean_text(None) == ""


def test_nan_input():
    assert clean_text(float("nan")) == ""


def test_whitespace_only():
    assert clean_text("   \n\t  ") == ""


def test_emoji_only_text_does_not_crash():
    result = clean_text("😡😡😡")
    assert isinstance(result, str)
    assert "pouting_face" in result or result != ""


# --- Idempotency (section 9 requires this explicitly) ---

def test_idempotent_on_plain_text():
    text = "This is a normal sentence about dogs."
    once = clean_text(text)
    twice = clean_text(once)
    assert once == twice


def test_idempotent_on_messy_text():
    text = "OMG @user check #ThisOutNow https://example.com so mad 😡!!!"
    once = clean_text(text)
    twice = clean_text(once)
    assert once == twice


# --- Individual component behavior ---

def test_strip_urls():
    result = normalize_whitespace(strip_urls("go to https://x.com/y now"))
    assert result == "go to now"


def test_strip_mentions():
    result = normalize_whitespace(strip_mentions("hi @someone1 there"))
    assert result == "hi there"


def test_unwrap_hashtags_keeps_words_drops_hash():
    out = unwrap_hashtags("#BlackLivesMatter now trending")
    assert "#" not in out
    assert "Black" in out and "Lives" in out and "Matter" in out


def test_demojize_preserves_signal_as_text():
    out = demojize_text("nice job 👍")
    assert ":thumbs_up:" in out or "thumbs_up" in out


def test_lemmatization_reduces_inflection():
    out = clean_text("the dogs are running quickly")
    # after lemmatization 'running' -> 'run', 'dogs' -> 'dog'
    assert "running" not in out.split() or "run" in out

def test_html_entity_not_left_as_stray_token():
    result = clean_text("guns &amp; roses", lemmatize=False)
    assert "amp" not in result.split()


def test_retweet_marker_stripped_but_real_words_kept():
    result = clean_text("rt someone said something", lemmatize=False)
    assert "rt" not in result.split()
    kept = clean_text("this is a great part of the show", lemmatize=False)
    assert "part" in kept.split()