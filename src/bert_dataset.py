"""
Dataset for DistilBERT fine-tuning — rung 5 (final) of the text ladder.

Unlike BiLSTM/CNN-BiLSTM/attention (rungs 2-4), which used our own
vocab.py + whitespace tokenization on already-cleaned text, DistilBERT
uses its own pretrained SUBWORD tokenizer. This means:

1. We tokenize the RAW `text` column, not `clean_text`. DistilBERT was
   pretrained on natural, largely unprocessed text (real casing, real
   punctuation). Our clean_text() pipeline (lowercasing, stripping
   punctuation, demojizing emojis into ":tag:" text) was built specifically
   because a from-scratch embedding layer can't otherwise handle casing/
   punctuation/emoji well — but DistilBERT's own subword vocabulary
   already handles all of that better than our pipeline could, so applying
   our cleaning here would only push the input further from what the
   model was actually pretrained on, for no benefit.

2. We DO still strip URLs from the raw text using a small self-contained
   regex (not importing src.preprocessing.strip_urls) — that module also
   imports nltk/emoji at the top level for its other functions, which
   would be an unnecessary dependency chain here just to reuse one regex.
   A URL carries no classification signal and just eats context-window
   budget as wasted subword tokens, so it's worth stripping regardless.

3. Padding is left to HuggingFace's DataCollatorWithPadding at the
   DataLoader level, not done here — this is the standard, correctly-
   tested approach for transformer models: tokenize without padding per
   example, pad dynamically per batch in the collator. It also produces
   the attention_mask the model needs to correctly ignore padding.
"""

import re

from torch.utils.data import Dataset

_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def strip_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)


class BertTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.texts = [strip_urls(str(t)) for t in texts]
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        encoding["labels"] = self.labels[idx]
        return encoding
