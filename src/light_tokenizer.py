"""
Minimal, torch-free wrapper around the `tokenizers` library -- the same
Rust-backed library that `transformers.AutoTokenizer`'s "fast" tokenizers
wrap internally -- used in place of `transformers.AutoTokenizer` for
serving src/api.py's text model.

WHY THIS EXISTS: a direct diagnostic (diffing sys.modules before/after
`AutoTokenizer.from_pretrained(...)`) showed it unconditionally imports
torch, torchgen, sympy, mpmath, and safetensors internally -- roughly
+240MB -- regardless of the `USE_TORCH=0` / `USE_TF=0` environment
variables transformers documents for exactly this situation, which had
ZERO measurable effect when tried. Since this app only ever needs plain
tokenization (text in, input_ids/attention_mask out), not any of
transformers' Python-level convenience API (chat templates, saving/
pushing to hub, etc.), loading the SAME `tokenizer.json` file directly
through `tokenizers.Tokenizer` -- the library transformers itself uses
under the hood for fast tokenizers -- produces identical token ids
without pulling in any of that.

Verified identical to `AutoTokenizer.from_pretrained(repo)(texts,
truncation=True, max_length=128, padding=True)` via
verify_light_tokenizer.py (repo root) across short text, text long enough
to require truncation, and batches of mixed lengths requiring padding --
run that script yourself if you change this file.
"""

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


class LightBertTokenizer:
    """
    Loads a HuggingFace fast-tokenizer's `tokenizer.json` directly and
    replicates `padding=True, truncation=True, max_length=N` behavior:
    dynamic padding to the longest sequence IN EACH CALL's batch (not a
    fixed length), special tokens ([CLS]/[SEP]) handled by the
    post-processor already embedded in tokenizer.json (the same one
    `save_pretrained()` wrote), matching exactly how this model was
    trained (see src/bert_dataset.py).
    """

    def __init__(self, repo_id: str, max_length: int = 128):
        path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
        self._tok = Tokenizer.from_file(path)

        pad_id = self._tok.token_to_id("[PAD]")
        if pad_id is None:
            raise ValueError(
                f"tokenizer.json from {repo_id!r} has no [PAD] token in its "
                "vocabulary -- this wrapper assumes a standard BERT/"
                "DistilBERT-style vocabulary and needs updating for anything else."
            )

        self._tok.enable_truncation(max_length=max_length)
        # No fixed `length=` here -- this pads each call's batch to ITS OWN
        # longest sequence, which is what `padding=True` (dynamic padding)
        # means in transformers, not a fixed max_length pad.
        self._tok.enable_padding(pad_id=pad_id, pad_token="[PAD]")

    def __call__(self, texts):
        """
        texts: list[str] -> {"input_ids": np.ndarray[int64], "attention_mask": np.ndarray[int64]}
        Shape is always (len(texts), longest_sequence_in_this_call).
        """
        encodings = self._tok.encode_batch(list(texts))
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
