"""
Smoke test for BiLSTMAttentionClassifier + the shared training loop.
Same pattern as the other smoke tests, plus one that's specific to this
model: verifying attention weights are a valid probability distribution
over REAL tokens only (near-zero at padding positions). That's the one
correctness property that's easy to get subtly wrong in an attention
implementation and wouldn't be caught by an output-shape check alone.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.bilstm_attention import BiLSTMAttentionClassifier
from src.text_dataset import HateSpeechTextDataset, collate_batch
from src.train import run_epoch
from src.vocab import Vocabulary


def _make_synthetic_data():
    class_words = {
        0: ["alpha", "beta", "gamma"],
        1: ["delta", "epsilon", "zeta"],
        2: ["eta", "theta", "iota"],
    }
    rng = np.random.default_rng(0)
    texts, labels = [], []
    for label, words in class_words.items():
        for _ in range(40):
            texts.append(" ".join(rng.choice(words, size=5)))
            labels.append(label)
    return texts, labels


def test_output_shape():
    vocab = Vocabulary(min_freq=1).build(["alpha beta gamma delta"])
    fake_embedding_matrix = np.random.default_rng(0).normal(size=(len(vocab), 16)).astype(np.float32)
    model = BiLSTMAttentionClassifier(fake_embedding_matrix, hidden_dim=8, n_classes=3, dropout=0.1)

    ds = HateSpeechTextDataset(["alpha beta", "gamma delta gamma"], [0, 1], vocab)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_batch)
    padded, lengths, _ = next(iter(loader))

    logits = model(padded, lengths)
    assert logits.shape == (2, 3)


def test_attention_weights_are_valid_distribution_over_real_tokens_only():
    vocab = Vocabulary(min_freq=1).build(["alpha beta gamma delta"])
    fake_embedding_matrix = np.random.default_rng(0).normal(size=(len(vocab), 16)).astype(np.float32)
    model = BiLSTMAttentionClassifier(fake_embedding_matrix, hidden_dim=8, n_classes=3, dropout=0.0)
    model.eval()

    # two sequences of deliberately different real length, padded to the same T
    ds = HateSpeechTextDataset(["alpha", "gamma delta gamma beta"], [0, 1], vocab)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_batch)
    padded, lengths, _ = next(iter(loader))

    logits, attn_weights = model(padded, lengths, return_attention=True)
    assert attn_weights.shape == padded.shape

    for i, length in enumerate(lengths.tolist()):
        row = attn_weights[i]
        assert abs(row.sum().item() - 1.0) < 1e-4, "attention weights must sum to ~1"
        if length < row.shape[0]:
            assert row[length:].sum().item() < 1e-4, "padding positions must get ~0 attention weight"


def test_loss_decreases_on_trivially_separable_data():
    texts, labels = _make_synthetic_data()
    vocab = Vocabulary(min_freq=1).build(texts)
    fake_embedding_matrix = np.random.default_rng(0).normal(size=(len(vocab), 16)).astype(np.float32)

    model = BiLSTMAttentionClassifier(fake_embedding_matrix, hidden_dim=16, n_classes=3, dropout=0.0)
    ds = HateSpeechTextDataset(texts, labels, vocab)
    loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_batch)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    first_epoch_loss, _ = run_epoch(model, loader, criterion, optimizer)
    loss = first_epoch_loss
    for _ in range(9):
        loss, _ = run_epoch(model, loader, criterion, optimizer)

    assert loss < first_epoch_loss * 0.5, (
        f"expected loss to drop substantially on trivially-separable data, "
        f"went from {first_epoch_loss:.4f} to {loss:.4f}"
    )
