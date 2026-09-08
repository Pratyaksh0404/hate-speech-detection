"""
Shared training loop for the RNN-based text ladder rungs: BiLSTM,
CNN-BiLSTM, and BiLSTM+Attention (master_plan.md section 4, rungs 2-4).
One script, selectable model, so all three train/evaluate under the exact
identical regime (same data, same weighted loss, same optimizer setup) —
apples-to-apples for the final comparison table, same reasoning as
src/evaluate.py being shared across the whole ladder.

Run with (from repo root, so imports resolve):
    python -m src.train --model bilstm --glove_path data/text/glove.6B.100d.txt
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

from src.evaluate import compute_metrics, measure_latency_ms, save_results
from src.text_dataset import HateSpeechTextDataset, collate_batch
from src.vocab import Vocabulary, load_glove_embeddings

RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        df["clean_text"] = df["clean_text"].fillna("")
    return train, val, test


def load_class_weights(path: str = "data/text/class_weights.json") -> torch.Tensor:
    with open(path) as f:
        raw = json.load(f)
    # JSON round-trips int keys as strings; sort numerically, not lexically
    ordered = [raw[str(i)] for i in sorted(int(k) for k in raw.keys())]
    return torch.tensor(ordered, dtype=torch.float32)


def build_dataloaders(train, val, test, vocab, batch_size: int = 64):
    train_ds = HateSpeechTextDataset(train["clean_text"], train["label"].tolist(), vocab)
    val_ds = HateSpeechTextDataset(val["clean_text"], val["label"].tolist(), vocab)
    test_ds = HateSpeechTextDataset(test["clean_text"], test["label"].tolist(), vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.enable_grad() if is_train else torch.no_grad():
        for padded, lengths, labels in loader:
            padded, lengths, labels = padded.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            logits = model(padded, lengths)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * padded.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics


def train_model(model, train_loader, val_loader, class_weights, name: str,
                 n_epochs: int = 10, lr: float = 1e-3, patience: int = 3, weight_decay: float = 0.0):
    """
    Early stopping on VAL MACRO-F1, not val loss — macro-F1 is our headline
    metric (master_plan.md section 6), and on a dataset this imbalanced,
    loss can keep improving on the majority class while macro-F1 stagnates
    or gets worse. Checkpointing on the wrong signal would quietly pick a
    worse model.
    """
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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
        for padded, lengths, labels in test_loader:
            padded, lengths = padded.to(DEVICE), lengths.to(DEVICE)
            logits = model(padded, lengths)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())
    return compute_metrics(all_labels, all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bilstm", "cnn_bilstm", "bilstm_attention"], default="bilstm")
    parser.add_argument("--glove_path", default="data/text/glove.6B.100d.txt")
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=10)
    # defaults below carry forward what actually worked for BiLSTM (rung 2)
    # after tuning -- hidden_dim=128/dropout=0.3/weight_decay=0 overfit
    # badly (test macro-F1 0.7184, below baseline); hidden_dim=64/
    # dropout=0.5/weight_decay=1e-5 fixed it (test macro-F1 0.7310).
    # Still overridable via CLI for further tuning per model.
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--cnn_out_channels", type=int, default=64)
    parser.add_argument("--patience", type=int, default=3)
    # parse_known_args (not parse_args) so this is robust even if something
    # else's arguments leak into sys.argv -- e.g. Jupyter/Colab kernels pass
    # their own "-f <connection_file>.json" flag when a script's main() runs
    # inside the notebook process instead of as a real subprocess.
    args, _unknown_args = parser.parse_known_args()

    set_seed()
    print(f"device: {DEVICE}")

    train, val, test = load_splits()
    class_weights = load_class_weights()

    vocab = Vocabulary(min_freq=2).build(train["clean_text"])
    print(f"vocab size: {len(vocab)}")

    embedding_matrix = load_glove_embeddings(vocab, args.glove_path, args.embedding_dim)
    train_loader, val_loader, test_loader = build_dataloaders(train, val, test, vocab)

    if args.model == "bilstm":
        from src.models.bilstm import BiLSTMClassifier

        model = BiLSTMClassifier(
            embedding_matrix=embedding_matrix,
            hidden_dim=args.hidden_dim,
            n_classes=3,
            dropout=args.dropout,
        )
    elif args.model == "cnn_bilstm":
        from src.models.cnn_bilstm import CNNBiLSTMClassifier

        model = CNNBiLSTMClassifier(
            embedding_matrix=embedding_matrix,
            cnn_out_channels=args.cnn_out_channels,
            hidden_dim=args.hidden_dim,
            n_classes=3,
            dropout=args.dropout,
        )
    elif args.model == "bilstm_attention":
        from src.models.bilstm_attention import BiLSTMAttentionClassifier

        model = BiLSTMAttentionClassifier(
            embedding_matrix=embedding_matrix,
            hidden_dim=args.hidden_dim,
            n_classes=3,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"unknown model {args.model}")

    model, history = train_model(
        model,
        train_loader,
        val_loader,
        class_weights,
        name=args.model,
        n_epochs=args.n_epochs,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )
    test_metrics = evaluate_on_test(model, test_loader)

    def predict_one(text: str) -> None:
        token_ids = vocab.encode(text) or [vocab.token_to_idx["<unk>"]]
        x = torch.tensor([token_ids], dtype=torch.long).to(DEVICE)
        length = torch.tensor([len(token_ids)]).to(DEVICE)
        model.eval()
        with torch.no_grad():
            model(x, length)

    latency_ms = measure_latency_ms(predict_one, test["clean_text"].tolist(), n_repeats=200)

    results = {
        "history": history,
        "test_metrics": test_metrics,
        "latency_ms_per_sample": latency_ms,
        "vocab_size": len(vocab),
        "device_used_for_training": str(DEVICE),
    }
    results_path = save_results(results, args.model)

    print(f"\n=== {args.model} FINAL ===")
    print(f"test macro-F1: {test_metrics['macro_f1']:.4f} | test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"latency: {latency_ms:.3f} ms/sample")
    print(f"model saved: models/{args.model}.pt")
    print(f"results saved: {results_path}")


if __name__ == "__main__":
    main()
