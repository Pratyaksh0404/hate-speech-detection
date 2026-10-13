"""
ONE-TIME validation: confirm LightBertTokenizer (src/light_tokenizer.py,
built on the `tokenizers` library directly) produces IDENTICAL input_ids
and attention_mask to transformers.AutoTokenizer for the same inputs,
across cases that actually exercise different code paths: short text,
text long enough to require truncation at max_length=128, and a batch of
mixed-length texts that requires dynamic padding.

This matters because LightBertTokenizer exists specifically to avoid
transformers.AutoTokenizer's torch-import side effect (see
src/light_tokenizer.py's docstring) -- if its output ever diverged from
AutoTokenizer's, predictions would be silently wrong in a way nothing else
in this pipeline would catch (the model would run fine, just on the wrong
token ids).

Run from the repo root, same venv:
    python verify_light_tokenizer.py
"""

import numpy as np
from transformers import AutoTokenizer

from src.light_tokenizer import LightBertTokenizer

MODEL_REPO = "Prat-04/hate-speech-distilbert"
MAX_LENGTH = 128

print(f"loading transformers.AutoTokenizer for {MODEL_REPO}...")
hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)

print(f"loading LightBertTokenizer for {MODEL_REPO}...")
light_tokenizer = LightBertTokenizer(MODEL_REPO, max_length=MAX_LENGTH)

test_cases = {
    "short, plain text": ["this is a completely harmless test sentence"],
    "text with punctuation/emoji-like chars": ["wow!! 😡 really?? #hateful @someone"],
    "empty-ish text": ["   "],
    "text long enough to require truncation at 128 tokens": [
        "word " * 300  # ~300 tokens before truncation, way over max_length=128
    ],
    "batch of mixed lengths (tests dynamic padding)": [
        "short one",
        "a somewhat longer sentence that has more words in it than the first one",
        "the longest sentence of this batch, deliberately written with many more "
        "words than the other two so padding actually has to pad the shorter "
        "sequences in this batch up to this one's length",
    ],
}

all_match = True
for case_name, texts in test_cases.items():
    hf_encoding = hf_tokenizer(
        texts, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="np"
    )
    light_encoding = light_tokenizer(texts)

    ids_match = np.array_equal(hf_encoding["input_ids"], light_encoding["input_ids"])
    mask_match = np.array_equal(hf_encoding["attention_mask"], light_encoding["attention_mask"])
    shapes_match = hf_encoding["input_ids"].shape == light_encoding["input_ids"].shape

    ok = ids_match and mask_match and shapes_match
    all_match = all_match and ok
    print(f"  {'OK ' if ok else 'MISMATCH '} {case_name}")
    if not ok:
        print(f"      hf shape={hf_encoding['input_ids'].shape}  light shape={light_encoding['input_ids'].shape}")
        print(f"      hf input_ids[0][:20]   ={hf_encoding['input_ids'][0][:20].tolist()}")
        print(f"      light input_ids[0][:20]={light_encoding['input_ids'][0][:20].tolist()}")
        print(f"      hf attention_mask[0][:20]   ={hf_encoding['attention_mask'][0][:20].tolist()}")
        print(f"      light attention_mask[0][:20]={light_encoding['attention_mask'][0][:20].tolist()}")

print()
if all_match:
    print("ALL CASES MATCH EXACTLY -- LightBertTokenizer is safe to use in production.")
else:
    print("MISMATCH FOUND -- do NOT deploy src/light_tokenizer.py as-is. Paste this")
    print("entire output back; the token ids diverging means predictions would be")
    print("silently wrong, not just slower or using more memory.")
