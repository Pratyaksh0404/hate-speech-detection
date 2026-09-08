"""
CNN-BiLSTM text classifier — rung 3 of the text model ladder
(master_plan.md section 4): Embedding (GloVe-initialized, fine-tunable)
-> parallel multi-kernel Conv1D -> BiLSTM -> Dense -> Dropout -> Softmax(3).

Design decisions:

1. Multiple parallel kernel sizes (3, 5, 7) rather than one. Each kernel
   size acts as an n-gram detector over a different local context window
   (3-word, 5-word, 7-word patterns), and their outputs are concatenated so
   the BiLSTM that follows sees a richer, multi-scale local feature at
   every timestep instead of only one fixed window size. This is the
   standard multi-kernel CNN idea (Kim, 2014) used as a feature extractor
   feeding INTO the BiLSTM, rather than pooled directly into a classifier
   the way a pure CNN would be — the goal is to combine local pattern
   detection (what beat plain BiLSTM at the TF-IDF baseline stage) with
   the BiLSTM's sequential/contextual modeling.

2. All kernel sizes are ODD (3, 5, 7), with "same" padding = (k-1)//2.
   This keeps every kernel's output at exactly the same sequence length T,
   which is required to concatenate their outputs along the channel
   dimension before handing the combined feature sequence to the BiLSTM.
   Even kernel sizes would need asymmetric padding to get the same effect,
   which Conv1d doesn't support directly — odd sizes sidestep that.

3. Real sequence lengths are unchanged by "same" padding, so we still pass
   the original `lengths` to pack_padded_sequence for the BiLSTM step —
   same correctness concern as plain BiLSTM (rung 2): the LSTM must not
   accumulate hidden state over pad positions. Note the conv layers
   themselves DO blend a few real embeddings with pad-embedding vectors
   right near each sequence's true end (padding_idx=0 makes pad embeddings
   all-zero, so this is a small, bounded edge effect, not arbitrary noise)
   — this is standard, accepted behavior for CNNs over padded sequences.

4. Combining the LSTM's output the same way as rung 2 (concatenated final
   forward + backward hidden states, not pooled) — kept consistent so the
   only architectural difference between rungs 2 and 3 in the comparison
   table is genuinely "does a CNN feature-extraction layer help", not two
   unrelated design changes at once.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class CNNBiLSTMClassifier(nn.Module):
    def __init__(
        self,
        embedding_matrix,
        cnn_out_channels: int = 64,
        kernel_sizes=(3, 5, 7),
        hidden_dim: int = 64,
        n_classes: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=False,
            padding_idx=0,
        )

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=cnn_out_channels,
                    kernel_size=k,
                    padding=(k - 1) // 2,
                )
                for k in kernel_sizes
            ]
        )

        conv_output_dim = cnn_out_channels * len(kernel_sizes)

        self.lstm = nn.LSTM(
            input_size=conv_output_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)  # [B, T, E]
        embedded_t = embedded.transpose(1, 2)  # [B, E, T] -- Conv1d wants channels-first

        conv_outputs = [F.relu(conv(embedded_t)) for conv in self.convs]  # each [B, C, T]
        combined = torch.cat(conv_outputs, dim=1)  # [B, C*len(kernel_sizes), T]
        combined = combined.transpose(1, 2)  # [B, T, C*len(kernel_sizes)] -- back to batch_first

        packed = pack_padded_sequence(
            combined, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)

        dropped = self.dropout(combined_hidden)
        logits = self.fc(dropped)
        return logits
