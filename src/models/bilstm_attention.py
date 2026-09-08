"""
BiLSTM + Attention text classifier — rung 4 of the text model ladder
(master_plan.md section 4): Embedding (GloVe-initialized, fine-tunable)
-> BiLSTM (all timestep outputs, not just the final hidden state) ->
additive attention over those outputs -> Dense -> Dropout -> Softmax(3).

Design decisions:

1. Additive (Bahdanau-style) attention: score(h_t) = v^T tanh(W h_t) for
   each BiLSTM output h_t, softmax over t to get a weight per timestep,
   then a weighted sum ("context vector") replaces the final-hidden-state
   trick used in rungs 2-3. This is the standard, well-established
   attention formulation for text classification (see e.g. Zhou et al.
   2016 attention-BiLSTM; Yang et al. 2016 Hierarchical Attention
   Networks), and — unlike a black-box final hidden state — it gives us a
   genuinely useful per-word weight we can visualize later for the plan's
   error-analysis/explainability sections: which words the model actually
   focused on for a given prediction.

2. Padding is explicitly masked OUT of the attention scores (set to -inf
   before softmax) using the real sequence lengths, not the padded length.
   Without this, padding tokens (which are all-zero embeddings, but still
   pass through the LSTM and get a nonzero score from the attention
   layer) could receive nonzero attention weight purely as a padding
   artifact, corrupting both the prediction and any attention
   visualization we build later. This is verified directly by a test
   asserting attention weight is ~0 at every position at or beyond a
   sequence's real length.

3. forward() has an optional return_attention flag. Training/evaluation
   uses it in default (logits-only) mode so it's a drop-in replacement in
   the shared training loop (same call signature as BiLSTM/CNN-BiLSTM) --
   but returning the attention weights on demand is exactly what we'll
   need later to visualize "which words triggered this prediction" for
   the frontend/error-analysis, without needing a second model.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim: int = 64, n_classes: int = 3, dropout: float = 0.5):
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

        lstm_out_dim = hidden_dim * 2
        self.attn_W = nn.Linear(lstm_out_dim, lstm_out_dim)
        self.attn_v = nn.Linear(lstm_out_dim, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, n_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor, return_attention: bool = False):
        embedded = self.embedding(input_ids)  # [B, T, E]

        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, T, hidden_dim*2]

        max_len = lstm_out.size(1)
        # True at real-token positions, False at padding -- built from the
        # ORIGINAL lengths (pre-padding), not the padded sequence length.
        position_idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # [1, T]
        mask = position_idx < lengths.unsqueeze(1)  # [B, T], bool

        scores = self.attn_v(torch.tanh(self.attn_W(lstm_out))).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=1)  # [B, T], pad positions -> ~0

        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # [B, hidden_dim*2]

        dropped = self.dropout(context)
        logits = self.fc(dropped)  # [B, n_classes]

        if return_attention:
            return logits, attn_weights
        return logits
