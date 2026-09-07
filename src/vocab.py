"""
Vocabulary and pretrained GloVe embedding loading, used by the BiLSTM,
CNN-BiLSTM, and attention models (rungs 2-4 of the text ladder in
master_plan.md section 4).

Design decisions:

1. Tokenization: simple whitespace split on the already-cleaned text from
   src/preprocessing.py. clean_text() already lowercases, strips
   punctuation, and demojizes emojis into space-separated ":tag:"-style
   tokens, so whitespace splitting is sufficient -- no need for a second,
   separate tokenizer library here.

2. Special tokens: <pad> (index 0) for padding shorter sequences to match
   the longest one in a batch, <unk> (index 1) for words seen at inference
   time that weren't in the training vocabulary (or were too rare to keep).

3. Vocabulary is built ONLY from the training split, never val/test.
   Using val/test text to build the vocabulary would leak information
   about those sets into what the model can even represent -- an easy
   mistake in from-scratch NLP pipelines that quietly inflates reported
   performance.

4. min_freq threshold: words appearing fewer than min_freq times in
   training map to <unk> instead of getting their own embedding row. This
   keeps the embedding table a sane size and avoids trying to learn a
   meaningful representation for a word seen once.

5. GloVe loading: for vocabulary words GloVe has a vector for, we use it.
   For words in OUR vocabulary that GloVe doesn't have (our own demojized
   emoji tokens like ":pouting_face:", or dataset-specific slang), we
   initialize with a small random vector instead of zeros -- since the
   embedding layer is fine-tunable (master_plan.md section 4, rung 2),
   those words can still learn a useful representation during training.
   An all-zero start would give them no gradient signal to ever become
   distinguishable from one another.
"""

from collections import Counter

import numpy as np

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class Vocabulary:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.token_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx_to_token = {0: PAD_TOKEN, 1: UNK_TOKEN}

    def build(self, texts):
        """texts: iterable of already-cleaned, whitespace-tokenizable strings."""
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        next_idx = len(self.token_to_idx)
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.token_to_idx:
                self.token_to_idx[token] = next_idx
                self.idx_to_token[next_idx] = token
                next_idx += 1
        return self

    def encode(self, text) -> list:
        """text -> list[int] token indices; unknown words map to <unk>."""
        unk_idx = self.token_to_idx[UNK_TOKEN]
        return [self.token_to_idx.get(tok, unk_idx) for tok in text.split()]

    def __len__(self) -> int:
        return len(self.token_to_idx)


def load_glove_embeddings(
    vocab: Vocabulary, glove_path: str, embedding_dim: int, seed: int = 42
) -> np.ndarray:
    """
    Returns a (len(vocab), embedding_dim) numpy array ready to hand to
    nn.Embedding.from_pretrained(). Words found in GloVe use GloVe's
    vector; words in our vocab that GloVe doesn't have get a small random
    vector so they still start out distinguishable from each other and can
    learn something useful via fine-tuning during training.
    """
    rng = np.random.default_rng(seed)
    embedding_matrix = rng.normal(scale=0.1, size=(len(vocab), embedding_dim)).astype(
        np.float32
    )
    embedding_matrix[0] = np.zeros(embedding_dim, dtype=np.float32)  # <pad> is always zero

    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab.token_to_idx:
                vector = np.asarray(parts[1:], dtype=np.float32)
                embedding_matrix[vocab.token_to_idx[word]] = vector
                found += 1

    coverage_pct = found / len(vocab) * 100
    print(f"GloVe coverage: {found}/{len(vocab)} vocab words found ({coverage_pct:.1f}%)")
    return embedding_matrix
