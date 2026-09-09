"""
Training script for the three required multimodal baselines
(master_plan.md section 4A.5): image-only, text-only, and fusion — all
trained under the identical regime so the comparison actually proves
whether fusion adds value, not just that a different training setup
happened to work better.

Run with (from repo root, on Kaggle where images are mounted):
    python -m src.train_multimodal --model fusion --n_epochs 10 \
        --image_dir "/kaggle/input/datasets/aditya1220/hate-speech-detection-dataset/Dataset/Labelled Images"
"""

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.evaluate import compute_metrics, measure_latency_ms, save_results
from src.image_dataset import MultimodalMemeDataset, build_image_transform
from src.models.multimodal_models import (
    TEXT_MODEL_NAME,
    AttentionFusionClassifier,
    FusionClassifier,
    ImageOnlyClassifier,
    TextOnlyClassifier,
)

RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_NAMES = ["Non-offensive", "Offensive"]


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_splits(data_dir: str = "data/multimodal"):
    train = pd.read_csv(f"{data_dir}/train.csv")
    val = pd.read_csv(f"{data_dir}/val.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    for df in (train, val, test):
        df["combined_text"] = df["combined_text"].fillna("")
    return train, val, test


def build_collate_fn(tokenizer, max_length: int = 64):
    def collate(batch):
        images, texts, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        encoding = tokenizer(
            list(texts), truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        labels = torch.tensor(labels, dtype=torch.long)
        return images, encoding["input_ids"], encoding["attention_mask"], labels

    return collate


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.enable_grad() if is_train else torch.no_grad():
        for images, input_ids, attention_mask, labels in loader:
            images = images.to(DEVICE)
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            logits = model(images, input_ids, attention_mask)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, target_names=TARGET_NAMES)
    return avg_loss, metrics


def train_model(model, train_loader, val_loader, class_weights, name: str,
                 n_epochs: int = 10, lr: float = 2e-5, patience: int = 3, weight_decay: float = 1e-5):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    # only optimize UNFROZEN params -- everything else has requires_grad=False
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

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
            f"[{name}] epoch {epoch}/{n_epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} ({epoch_time:.1f}s)"
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_macro_f1": val_metrics["macro_f1"]}
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
        for images, input_ids, attention_mask, labels in test_loader:
            images, input_ids, attention_mask = images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE)
            logits = model(images, input_ids, attention_mask)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())
    return compute_metrics(all_labels, all_preds, target_names=TARGET_NAMES)


def load_encoder_weights(model, image_only_ckpt: str = None, text_only_ckpt: str = None) -> None:
    """
    Warm-starts a fusion model's image_encoder/text_encoder from
    separately-trained image_only.pt / text_only.pt checkpoints, instead
    of starting cold from generic ImageNet/DistilBERT-base weights.

    Tried this because: plain concatenation fusion (0.5533 macro-F1) and
    gated fusion (0.5143) both underperformed text-only alone (0.6573).
    The gate targeted "let the model down-weight the noisy image branch"
    and didn't fix it -- this targets a different hypothesis: with only
    445 training examples, fusion may be struggling to learn good
    per-modality features AND a good combination simultaneously. Starting
    each branch from weights that already adapted to this task
    individually removes that burden, leaving only the combination itself
    to be learned from scratch.

    Uses strict=False + prefix filtering since the checkpoint's full
    state_dict includes an image_only/text_only-specific final layer
    (self.fc) that doesn't exist in the fusion model and shouldn't be
    loaded anyway -- only image_encoder.*/text_encoder.* keys are used.
    """
    if image_only_ckpt:
        state = torch.load(image_only_ckpt, map_location=DEVICE)
        encoder_state = {k: v for k, v in state.items() if k.startswith("image_encoder.")}
        model.load_state_dict(encoder_state, strict=False)
        print(f"warm-started image_encoder from {image_only_ckpt} ({len(encoder_state)} tensors)")
    if text_only_ckpt:
        state = torch.load(text_only_ckpt, map_location=DEVICE)
        encoder_state = {k: v for k, v in state.items() if k.startswith("text_encoder.")}
        model.load_state_dict(encoder_state, strict=False)
        print(f"warm-started text_encoder from {text_only_ckpt} ({len(encoder_state)} tensors)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["image_only", "text_only", "fusion", "attention_fusion"], required=True)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--data_dir", default="data/multimodal")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--warm_start", action="store_true")
    parser.add_argument("--image_only_ckpt", default="models/multimodal_image_only.pt")
    parser.add_argument("--text_only_ckpt", default="models/multimodal_text_only.pt")
    args, _unknown_args = parser.parse_known_args()

    set_seed()
    print(f"device: {DEVICE}")

    train, val, test = load_splits(args.data_dir)

    class_weights_arr = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=train["label"])
    class_weights = torch.tensor(class_weights_arr, dtype=torch.float32)
    print(f"class weights: {dict(zip([0, 1], class_weights_arr))}")

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    collate_fn = build_collate_fn(tokenizer, max_length=args.max_length)

    train_ds = MultimodalMemeDataset(train, args.image_dir, transform=build_image_transform(train=True))
    val_ds = MultimodalMemeDataset(val, args.image_dir, transform=build_image_transform(train=False))
    test_ds = MultimodalMemeDataset(test, args.image_dir, transform=build_image_transform(train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model_map = {
        "image_only": ImageOnlyClassifier,
        "text_only": TextOnlyClassifier,
        "fusion": FusionClassifier,
        "attention_fusion": AttentionFusionClassifier,
    }
    model = model_map[args.model](n_classes=2)

    run_name = args.model
    if args.warm_start and args.model in ("fusion", "attention_fusion"):
        load_encoder_weights(model, image_only_ckpt=args.image_only_ckpt, text_only_ckpt=args.text_only_ckpt)
        run_name = f"{args.model}_warmstart"  # distinct name so this doesn't overwrite the cold-start results

    model, history = train_model(
        model, train_loader, val_loader, class_weights, name=f"multimodal_{run_name}",
        n_epochs=args.n_epochs, lr=args.lr, patience=args.patience,
    )
    test_metrics = evaluate_on_test(model, test_loader)

    def predict_one(idx: int) -> None:
        image, text, _ = test_ds[idx]
        image = image.unsqueeze(0).to(DEVICE)
        encoding = tokenizer([text], truncation=True, max_length=args.max_length, return_tensors="pt")
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)
        model.eval()
        with torch.no_grad():
            model(image, input_ids, attention_mask)

    latency_ms = measure_latency_ms(predict_one, list(range(len(test_ds))), n_repeats=50)

    results = {
        "history": history,
        "test_metrics": test_metrics,
        "latency_ms_per_sample": latency_ms,
        "device_used_for_training": str(DEVICE),
        "warm_started": bool(args.warm_start and args.model in ("fusion", "attention_fusion")),
    }
    results_path = save_results(results, f"multimodal_{run_name}")

    print(f"\n=== multimodal_{run_name} FINAL ===")
    print(f"test macro-F1: {test_metrics['macro_f1']:.4f} | test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"latency: {latency_ms:.3f} ms/sample")
    print(f"results saved: {results_path}")


if __name__ == "__main__":
    main()
