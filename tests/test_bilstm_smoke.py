"""
Smoke test for BiLSTMClassifier + the shared training loop in src/train.py,
on tiny synthetic data. Confirms the architecture is wired correctly
(right output shape, loss actually decreases) before spending any real
time or Kaggle GPU-hours on it. Needs torch — this is one I could not run
myself (see chat), so this is the real verification step.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.bilstm import BiLSTMClassifier
from src.text_dataset import HateSpeechTextDataset, collate_batch
from src.train import run_epoch
from src.vocab import Vocabulary


def _make_synthetic_data():
    # Deliberately trivial: each class uses its own disjoint vocabulary.
    # A correctly-wired BiLSTM should memorize this near-perfectly within
    # a handful of epochs -- if it doesn't, something upstream is broken
    # (wrong loss, broken backward pass, shape mismatch, bad indexing).
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


def test_bilstm_forward_pass_output_shape():
    vocab = Vocabulary(min_freq=1).build(["alpha beta gamma delta"])
    fake_embedding_matrix = np.random.default_rng(0).normal(size=(len(vocab), 16)).astype(np.float32)
    model = BiLSTMClassifier(fake_embedding_matrix, hidden_dim=8, n_classes=3, dropout=0.1)

    ds = HateSpeechTextDataset(["alpha beta", "gamma delta gamma"], [0, 1], vocab)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_batch)
    padded, lengths, _ = next(iter(loader))

    logits = model(padded, lengths)
    assert logits.shape == (2, 3)


def test_bilstm_loss_decreases_on_trivially_separable_data():
    texts, labels = _make_synthetic_data()
    vocab = Vocabulary(min_freq=1).build(texts)
    fake_embedding_matrix = np.random.default_rng(0).normal(size=(len(vocab), 16)).astype(np.float32)

    model = BiLSTMClassifier(fake_embedding_matrix, hidden_dim=16, n_classes=3, dropout=0.0)
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
