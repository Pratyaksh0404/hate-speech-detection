"""
PyTorch Dataset + collate function, used with src/vocab.py by the
BiLSTM/CNN-BiLSTM/attention models.

Design decision: DYNAMIC padding via a custom collate_fn, not a fixed
max_len decided up front. Each batch is padded only to the length of its
own longest sequence, not some global max — avoids wasting compute on pad
tokens for the many short tweets, and is the standard approach for
variable-length sequence models. Sequence lengths are also returned so the
BiLSTM training loop can use pack_padded_sequence, which stops the LSTM
from accumulating hidden state over pad tokens (important for correctness,
not just speed).
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.vocab import PAD_TOKEN, UNK_TOKEN


class HateSpeechTextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        """
        texts: list/Series of already-cleaned strings (output of clean_text())
        labels: list/Series of int labels (0/1/2)
        vocab: a fitted src.vocab.Vocabulary (built on TRAIN split only)
        """
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx):
        token_ids = self.vocab.encode(self.texts[idx])
        if len(token_ids) == 0:
            # a row that cleaned down to an empty string (e.g. was only a
            # URL) -- encode as a single <unk> so the model always sees at
            # least one timestep instead of crashing on a zero-length sequence
            token_ids = [self.vocab.token_to_idx[UNK_TOKEN]]
        return torch.tensor(token_ids, dtype=torch.long), self.labels[idx]


def collate_batch(batch):
    """
    batch: list of (token_id_tensor, label) tuples from __getitem__.
    Returns: (padded_ids [B, T], lengths [B], labels [B]).
    lengths are needed for pack_padded_sequence in the BiLSTM training loop.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)  # 0 == <pad>
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels
