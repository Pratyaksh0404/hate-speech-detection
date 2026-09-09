"""
Smoke tests for src/models/multimodal_models.py + src/train_multimodal.py.
Needs torch, torchvision, and transformers, plus a one-time download of
pretrained ResNet18 + DistilBERT weights — none of which is available in
the sandbox that wrote this. This is the REAL verification step for the
most complex, least-tested code in the whole project so far.

Checks, for each of the three model types:
1. Forward pass produces the correct output shape.
2. Backward pass runs without error.
3. Frozen parameters (early ResNet layers, early DistilBERT layers)
   actually stay frozen (no gradient), and unfrozen ones (layer4,
   last transformer layer, classifier head) DO get a gradient — this is
   the one thing that's easy to silently get wrong in a freeze/fine-tune
   setup and would NOT be caught by an output-shape check alone.
"""

import os
import tempfile

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer

from src.image_dataset import MultimodalMemeDataset, build_image_transform
from src.models.multimodal_models import (
    TEXT_MODEL_NAME,
    AttentionFusionClassifier,
    FusionClassifier,
    ImageOnlyClassifier,
    TextOnlyClassifier,
)
from src.train_multimodal import build_collate_fn


def _make_fake_batch(tmp_dir, tokenizer):
    Image.new("RGB", (500, 300), color=(200, 100, 50)).save(os.path.join(tmp_dir, "a.png"))
    Image.new("RGBA", (600, 900), color=(10, 20, 30, 128)).save(os.path.join(tmp_dir, "b.png"))
    df = pd.DataFrame(
        {
            "image_name": ["a.png", "b.png"],
            "combined_text": ["some caption text", "another caption | ocr text"],
            "label": [0, 1],
        }
    )
    ds = MultimodalMemeDataset(df, image_dir=tmp_dir, transform=build_image_transform(train=False))
    collate_fn = build_collate_fn(tokenizer, max_length=32)
    batch = collate_fn([ds[0], ds[1]])
    return batch  # (images, input_ids, attention_mask, labels)


def _assert_forward_backward_ok(model, batch):
    images, input_ids, attention_mask, labels = batch
    logits = model(images, input_ids, attention_mask)
    assert logits.shape == (2, 2), f"expected [2, 2] logits, got {logits.shape}"

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    loss.backward()
    return loss


def test_image_only_forward_backward_and_freezing():
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    with tempfile.TemporaryDirectory() as tmp_dir:
        batch = _make_fake_batch(tmp_dir, tokenizer)
        model = ImageOnlyClassifier(n_classes=2)
        _assert_forward_backward_ok(model, batch)

        # frozen early ResNet layer must have NO gradient
        early_param = next(model.image_encoder.backbone.layer1.parameters())
        assert early_param.grad is None, "layer1 should be frozen (no gradient)"

        # unfrozen layer4 must HAVE a gradient
        late_param = next(model.image_encoder.backbone.layer4.parameters())
        assert late_param.grad is not None, "layer4 should be unfrozen (gradient expected)"
        assert not torch.all(late_param.grad == 0), "layer4's gradient should be nonzero"


def test_text_only_forward_backward_and_freezing():
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    with tempfile.TemporaryDirectory() as tmp_dir:
        batch = _make_fake_batch(tmp_dir, tokenizer)
        model = TextOnlyClassifier(n_classes=2)
        _assert_forward_backward_ok(model, batch)

        early_param = next(model.text_encoder.bert.transformer.layer[0].parameters())
        assert early_param.grad is None, "first transformer layer should be frozen"

        late_param = next(model.text_encoder.bert.transformer.layer[-1].parameters())
        assert late_param.grad is not None, "last transformer layer should be unfrozen"
        assert not torch.all(late_param.grad == 0)


def test_fusion_forward_backward_and_freezing():
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    with tempfile.TemporaryDirectory() as tmp_dir:
        batch = _make_fake_batch(tmp_dir, tokenizer)
        model = FusionClassifier(n_classes=2)
        loss = _assert_forward_backward_ok(model, batch)
        assert loss.item() > 0

        # both branches should have their last-block gradients flowing
        image_late = next(model.image_encoder.backbone.layer4.parameters())
        text_late = next(model.text_encoder.bert.transformer.layer[-1].parameters())
        assert image_late.grad is not None
        assert text_late.grad is not None


def test_attention_fusion_forward_backward_and_gate_weights():
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    with tempfile.TemporaryDirectory() as tmp_dir:
        batch = _make_fake_batch(tmp_dir, tokenizer)
        images, input_ids, attention_mask, labels = batch

        model = AttentionFusionClassifier(n_classes=2)
        logits, gate_weights = model(images, input_ids, attention_mask, return_attention=True)
        assert logits.shape == (2, 2)
        assert gate_weights.shape == (2, 2)  # [B, 2 modalities]

        # gate weights must be a valid distribution over the 2 modalities, per example
        row_sums = gate_weights.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(2), atol=1e-5)
        assert torch.all(gate_weights >= 0)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        loss.backward()

        image_late = next(model.image_encoder.backbone.layer4.parameters())
        text_late = next(model.text_encoder.bert.transformer.layer[-1].parameters())
        assert image_late.grad is not None
        assert text_late.grad is not None
