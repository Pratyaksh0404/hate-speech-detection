"""
Unit tests for src/vocab.py. Pure Python/numpy -- no torch needed, runs
anywhere.
"""

import numpy as np

from src.vocab import PAD_TOKEN, UNK_TOKEN, Vocabulary, load_glove_embeddings


def test_pad_and_unk_are_reserved_at_fixed_indices():
    vocab = Vocabulary()
    assert vocab.token_to_idx[PAD_TOKEN] == 0
    assert vocab.token_to_idx[UNK_TOKEN] == 1


def test_build_includes_frequent_words_excludes_rare():
    texts = ["cat dog cat dog cat", "bird bird", "onlyonce"]
    vocab = Vocabulary(min_freq=2).build(texts)
    assert "cat" in vocab.token_to_idx
    assert "dog" in vocab.token_to_idx
    assert "bird" in vocab.token_to_idx
    assert "onlyonce" not in vocab.token_to_idx  # freq=1, below min_freq=2


def test_encode_maps_unknown_words_to_unk():
    vocab = Vocabulary(min_freq=1).build(["cat dog"])
    encoded = vocab.encode("cat elephant")
    unk_idx = vocab.token_to_idx[UNK_TOKEN]
    assert encoded[0] == vocab.token_to_idx["cat"]
    assert encoded[1] == unk_idx


def test_vocab_built_only_from_given_texts_not_leaked():
    train_texts = ["cat dog"]
    vocab = Vocabulary(min_freq=1).build(train_texts)
    assert "elephant" not in vocab.token_to_idx


def test_load_glove_embeddings_uses_real_vectors_when_available(tmp_path):
    vocab = Vocabulary(min_freq=1).build(["cat dog fish"])
    glove_path = tmp_path / "fake_glove.txt"
    with open(glove_path, "w") as f:
        f.write("cat 1.0 2.0 3.0\n")
        f.write("dog 4.0 5.0 6.0\n")
        # "fish" intentionally left out to test the random-init fallback

    matrix = load_glove_embeddings(vocab, str(glove_path), embedding_dim=3)

    assert matrix.shape == (len(vocab), 3)
    np.testing.assert_allclose(matrix[vocab.token_to_idx["cat"]], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(matrix[vocab.token_to_idx["dog"]], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(matrix[0], np.zeros(3))  # <pad> row must be all-zero
    # "fish" wasn't in the fake glove file -- must NOT be all-zero, or it
    # could never learn a distinguishable representation via fine-tuning
    assert not np.allclose(matrix[vocab.token_to_idx["fish"]], np.zeros(3))
