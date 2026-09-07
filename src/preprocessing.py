"""
Text cleaning pipeline for the hate-speech-detection text branch.

Design decisions (documented per master_plan.md section 5.2, not left implicit):

1. URLs -> stripped entirely. A URL carries essentially no generalizable
   signal for hate-speech classification, and leaving raw URLs in creates
   huge, mostly-useless vocabulary entries.

2. @mentions -> stripped entirely (handle text included). Usernames are
   near-unique tokens; keeping them would let models "memorize" specific
   accounts instead of learning language patterns that generalize.

3. #hashtags -> the '#' is dropped but the WORD is kept, because hashtag
   text frequently carries real semantic/topical signal (e.g. "#BuildTheWall",
   "#BlackLivesMatter"). We also split CamelCase hashtags into separate words
   ("BuildTheWall" -> "Build The Wall") so the tokenizer/embedding layer sees
   real words instead of one rare concatenated token.

4. Emojis -> NOT blindly stripped. They are converted to text descriptors
   via the `emoji` library's demojize (e.g. "😡" -> ":pouting_face:"). Emojis
   often carry sentiment/intent that's directly relevant to hate/offensive
   classification (e.g. a slur followed by a laughing emoji reads very
   differently than one followed by an angry emoji), so we preserve that
   signal as text tokens rather than deleting it or leaving raw unicode that
   most tokenizers can't handle consistently.

5. Lemmatization -> POS-aware lemmatization via NLTK's WordNetLemmatizer.
   We tag each token's part of speech first (so "meeting" the noun and
   "meeting" the verb lemmatize correctly) rather than defaulting every
   word to noun-lemmatization, which is a common shortcut that quietly
   mis-lemmatizes verbs and adjectives.

6. Stopwords -> NOT removed here by design. Removing stopwords is a
   classical-ML-era trick that can hurt sequence models (BiLSTM/CNN/
   DistilBERT) which rely on word order and function words for meaning.
   Instead, stopword removal is exposed as an OPTION at the baseline
   (TF-IDF) stage only, via sklearn's `stop_words='english'` param in
   src/models/baseline.py — it is not baked into this shared cleaner.

7. HTML entities (e.g. "&amp;") -> unescaped BEFORE anything else runs.
   Twitter's API frequently returns raw, un-decoded HTML entities in tweet
   text. Without this step, "&amp;" becomes a stray "amp" token once
   punctuation stripping removes the "&" — this actually showed up as a
   real artifact in the class-1/2 wordclouds during EDA, which is how we
   caught it. Decoding first means "&amp;" correctly becomes "&", which
   then gets dropped by punctuation stripping like any other symbol,
   instead of leaving noise behind.

8. Retweet marker "rt" -> stripped as a standalone token (word-boundary
   match, so it never touches real words containing "rt" like "part").
   This is a Twitter platform artifact, not language content — the exact
   same category as URLs and @mentions (metadata about how the tweet was
   posted, not what it says) — and it was visibly dominating the "Neither"
   class wordcloud during EDA, drowning out actual signal words.

clean_text() is designed to be idempotent: clean_text(clean_text(x)) == clean_text(x).
This matters both for correctness and because tests/test_preprocessing.py
checks idempotency directly.
"""

import html
import re
import emoji
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

# --- One-time NLTK resource downloads ---
# These need network access once, on whichever machine actually runs training
# (your local machine or Colab — see the note in the chat about this sandbox
# having no network access). Safe to call repeatedly; NLTK skips re-downloading
# if the resource is already present.
def ensure_nltk_resources() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, pkg_id in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg_id, quiet=True)


_lemmatizer = WordNetLemmatizer()

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_RETWEET_MARKER_RE = re.compile(r"\brt\b")
_CAMELCASE_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WHITESPACE_RE = re.compile(r"\s+")
# Keep letters, numbers, whitespace, and the ':' '_' characters that
# emoji.demojize() uses (e.g. ":face_with_tears_of_joy:"). Drop other punctuation.
_NON_ALLOWED_CHARS_RE = re.compile(r"[^a-z0-9\s:_]")


def strip_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)


def strip_mentions(text: str) -> str:
    return _MENTION_RE.sub(" ", text)


def split_camel_case(word: str) -> str:
    """'BuildTheWall' -> 'Build The Wall'"""
    return _CAMELCASE_RE.sub(" ", word)


def unwrap_hashtags(text: str) -> str:
    """'#BuildTheWall is bad' -> 'Build The Wall is bad' (drop '#', keep + split words)"""
    return _HASHTAG_RE.sub(lambda m: split_camel_case(m.group(1)), text)


def strip_retweet_marker(text: str) -> str:
    """'rt someone said something' -> ' someone said something' (whole-token 'rt' only)"""
    return _RETWEET_MARKER_RE.sub(" ", text)


def demojize_text(text: str) -> str:
    """'so mad 😡' -> 'so mad :pouting_face:'"""
    return emoji.demojize(text, delimiters=(" :", ": "))


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _wordnet_pos(treebank_tag: str) -> str:
    """Map a Penn Treebank POS tag to the one-letter tag WordNetLemmatizer expects."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # default fallback


def lemmatize_text(text: str) -> str:
    if not text:
        return text
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmas = [_lemmatizer.lemmatize(tok, _wordnet_pos(tag)) for tok, tag in tagged]
    return " ".join(lemmas)


def clean_text(text, lemmatize: bool = True, strip_rt: bool = True) -> str:
    """
    Full cleaning pipeline, applied in this order:
    unescape HTML entities -> lowercase -> strip URLs -> strip mentions ->
    unwrap hashtags -> (optional) strip retweet marker -> demojize emojis ->
    drop other punctuation -> normalize whitespace -> (optional) lemmatize.

    Handles edge cases explicitly: None/NaN -> "", empty string -> "",
    emoji-only text -> its demojized token text (never crashes).

    strip_rt=True by default (see design decision 8 in the module docstring);
    set False if you ever want to keep the raw retweet marker for some
    other analysis.
    """
    if text is None or (isinstance(text, float)):  # covers NaN from pandas
        return ""
    if not isinstance(text, str):
        text = str(text)
    if text.strip() == "":
        return ""

    text = html.unescape(text)
    text = text.lower()
    text = strip_urls(text)
    text = strip_mentions(text)
    text = unwrap_hashtags(text)
    if strip_rt:
        text = strip_retweet_marker(text)
    text = demojize_text(text)
    text = _NON_ALLOWED_CHARS_RE.sub(" ", text)
    text = normalize_whitespace(text)

    if lemmatize and text:
        text = lemmatize_text(text)
        text = normalize_whitespace(text)

    return text