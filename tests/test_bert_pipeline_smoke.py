"""
Smoke test for the DistilBERT dataset/training pipeline. Unlike every
other smoke test in this repo, this one needs real network access to
download the pretrained tokenizer/model from HuggingFace Hub (a few
hundred MB, cached locally after the first run) — my sandbox has none,
so this is meant to be run on YOUR machine, which does have internet.
This is the real verification step for this rung.

Kept small on purpose so it's still reasonably fast on CPU (expect roughly
a minute or so total, mostly the one-time model download).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from src.bert_dataset import BertTextDataset
from src.train_bert import run_epoch

MODEL_NAME = "distilbert-base-uncased"


def test_dataset_and_collator_produce_correct_batch_shapes():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    texts = ["I really love this!", "This is a much longer sentence to test padding behavior properly."]
    labels = [2, 0]
    ds = BertTextDataset(texts, labels, tokenizer, max_length=32)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=2, collate_fn=collator)
    batch = next(iter(loader))

    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].tolist() == [2, 0]


def test_urls_are_stripped_before_tokenizing():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = BertTextDataset(["check this out https://example.com/spam now"], [1], tokenizer, max_length=32)
    assert "https" not in ds.texts[0] and "example.com" not in ds.texts[0]


def test_loss_decreases_over_a_few_steps_on_tiny_separable_data():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    texts = [
        "I hate you so much", "you are disgusting and awful",
        "this is stupid nonsense", "shut up idiot",
        "have a wonderful lovely day", "thank you so much for this",
    ]
    labels = [0, 0, 1, 1, 2, 2]
    ds = BertTextDataset(texts, labels, tokenizer, max_length=32)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=6, collate_fn=collator)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    first_loss, _ = run_epoch(model, loader, criterion, optimizer)
    loss = first_loss
    for _ in range(4):
        loss, _ = run_epoch(model, loader, criterion, optimizer)

    assert loss < first_loss, f"expected loss to decrease even slightly, went from {first_loss:.4f} to {loss:.4f}"
