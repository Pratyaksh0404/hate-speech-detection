"""
Fine-tuning script for DistilBERT — rung 5 (final) of the text ladder
(master_plan.md section 4). Kept as a separate script from src/train.py
(which handles BiLSTM/CNN-BiLSTM/attention) because the training regime is
genuinely different: HuggingFace's own tokenizer/data collator instead of
our vocab.py pipeline, AdamW with a much lower learning rate (standard for
fine-tuning a pretrained transformer — a from-scratch-scale LR would wreck
the pretrained weights in a couple of batches instead of gently adapting
them), and no embedding matrix to load.

Still uses the SAME evaluation code (src/evaluate.py) and the SAME
weighted-loss imbalance strategy as every other rung, so the final
comparison table stays apples-to-apples.

Run with (from repo root):
    python -m src.train_bert --n_epochs 3
"""

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from src.bert_dataset import BertTextDataset
from src.evaluate import compute_metrics, measure_latency_ms, save_results

RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "distilbert-base-uncased"


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_splits(data_dir: str = "data/text"):
    train = pd.read_csv(f"{data_dir}/train.csv")
    val = pd.read_csv(f"{data_dir}/val.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    for df in (train, val, test):
        df["text"] = df["text"].fillna("")
    return train, val, test


def load_class_weights(path: str = "data/text/class_weights.json") -> torch.Tensor:
    with open(path) as f:
        raw = json.load(f)
    ordered = [raw[str(i)] for i in sorted(int(k) for k in raw.keys())]
    return torch.tensor(ordered, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.enable_grad() if is_train else torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics


def train_model(model, train_loader, val_loader, class_weights, name: str,
                 n_epochs: int = 3, lr: float = 2e-5, patience: int = 2):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    os.makedirs("models", exist_ok=True)
    checkpoint_path = f"models/{name}.pt"

    history = []
    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, optimizer=None)
        epoch_time = time.perf_counter() - t0

        print(
            f"[{name}] epoch {epoch}/{n_epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} ({epoch_time:.1f}s)"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[{name}] early stopping at epoch {epoch} (no val macro-F1 improvement for {patience} epochs)")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    return model, history


def evaluate_on_test(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"]
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())
    return compute_metrics(all_labels, all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--patience", type=int, default=2)
    args, _unknown_args = parser.parse_known_args()

    set_seed()
    print(f"device: {DEVICE}")

    train, val, test = load_splits()
    class_weights = load_class_weights()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_ds = BertTextDataset(train["text"], train["label"].tolist(), tokenizer, max_length=args.max_length)
    val_ds = BertTextDataset(val["text"], val["label"].tolist(), tokenizer, max_length=args.max_length)
    test_ds = BertTextDataset(test["text"], test["label"].tolist(), tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    model, history = train_model(
        model, train_loader, val_loader, class_weights,
        name="distilbert", n_epochs=args.n_epochs, lr=args.lr, patience=args.patience,
    )
    test_metrics = evaluate_on_test(model, test_loader)

    def predict_one(text: str) -> None:
        encoding = tokenizer(str(text), truncation=True, max_length=args.max_length, return_tensors="pt")
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)
        model.eval()
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)

    latency_ms = measure_latency_ms(predict_one, test["text"].tolist(), n_repeats=100)

    results = {
        "history": history,
        "test_metrics": test_metrics,
        "latency_ms_per_sample": latency_ms,
        "device_used_for_training": str(DEVICE),
        "pretrained_checkpoint": MODEL_NAME,
    }
    results_path = save_results(results, "distilbert")

    print("\n=== distilbert FINAL ===")
    print(f"test macro-F1: {test_metrics['macro_f1']:.4f} | test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"latency: {latency_ms:.3f} ms/sample")
    print("model saved: models/distilbert.pt")
    print(f"results saved: {results_path}")


if __name__ == "__main__":
    main()
