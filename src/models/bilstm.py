"""
BiLSTM text classifier — rung 2 of the text model ladder (master_plan.md
section 4): Embedding (GloVe-initialized, fine-tunable) -> Bidirectional
LSTM -> Dense -> Dropout -> Softmax(3).

Design decisions:

1. Embedding initialized from GloVe but NOT frozen (freeze=False) — the
   plan explicitly calls for "GloVe-initialized, fine-tunable" embeddings,
   so word vectors can shift toward what's actually useful for hate-speech
   classification specifically, not just stay at generic word similarity.

2. pack_padded_sequence/pad_packed_sequence with enforce_sorted=False —
   tells the LSTM to skip computation on padding tokens entirely. This is
   not just a speed optimization: without it, the LSTM's final hidden
   state would be contaminated by however many pad tokens trail the real
   content, which is especially bad for short tweets sitting in a batch
   next to much longer ones.

3. We concatenate the final forward and backward hidden states (not a
   max/mean pool over all timesteps) — the standard, simplest approach for
   a first BiLSTM. CNN-BiLSTM and the attention variant (later rungs) will
   use pooling/attention instead; keeping this one architecturally simple
   is intentional, matching the plan's framing of BiLSTM as the "core"
   model before adding complexity.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim: int = 128, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=False,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        # *2 because bidirectional concatenates forward + backward final states
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)  # [B, T, embedding_dim]

        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        # hidden: [num_layers * num_directions, B, hidden_dim].
        # For a single-layer bidirectional LSTM, the last two entries are
        # always this layer's forward and backward final states, in order.
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)  # [B, hidden_dim*2]

        dropped = self.dropout(combined)
        logits = self.fc(dropped)  # [B, n_classes]
        return logits
