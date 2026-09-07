"""
Unit tests for src/text_dataset.py. Requires torch (installed locally per
project convention for CPU smoke-testing before anything runs on Kaggle).
"""

import torch
from torch.utils.data import DataLoader

from src.text_dataset import HateSpeechTextDataset, collate_batch
from src.vocab import Vocabulary


def _tiny_vocab():
    return Vocabulary(min_freq=1).build(["cat dog", "bird fish snake"])


def test_dataset_returns_tensor_and_label():
    vocab = _tiny_vocab()
    ds = HateSpeechTextDataset(["cat dog"], [1], vocab)
    token_ids, label = ds[0]
    assert isinstance(token_ids, torch.Tensor)
    assert token_ids.dtype == torch.long
    assert label == 1


def test_empty_text_encodes_as_single_unk_not_crash():
    vocab = _tiny_vocab()
    ds = HateSpeechTextDataset([""], [2], vocab)
    token_ids, label = ds[0]
    assert len(token_ids) == 1
    assert token_ids[0].item() == vocab.token_to_idx["<unk>"]


def test_collate_pads_to_batch_max_length():
    vocab = _tiny_vocab()
    ds = HateSpeechTextDataset(["cat dog", "bird fish snake", "cat"], [0, 1, 2], vocab)
    loader = DataLoader(ds, batch_size=3, collate_fn=collate_batch)
    padded, lengths, labels = next(iter(loader))

    assert padded.shape == (3, 3)  # padded to the longest sequence (3 tokens)
    assert lengths.tolist() == [2, 3, 1]
    assert labels.tolist() == [0, 1, 2]
    # the shortest sequence ("cat", length 1) should have 2 pad tokens
    # (index 0) trailing after its single real token
    assert padded[2, 1].item() == 0
    assert padded[2, 2].item() == 0
