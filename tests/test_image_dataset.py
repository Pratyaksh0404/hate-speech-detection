"""
Smoke tests for src/image_dataset.py. Needs torch + torchvision, which
aren't in the sandbox that wrote this — I only verified the RGBA->RGB
PIL conversion logic directly (see chat). This is the real check for
everything else: correct output shape/type, RGBA images handled without
crashing, and that train-split augmentation actually differs from eval
(no randomness) between calls.
"""

import os
import tempfile

import pandas as pd
import torch
from PIL import Image

from src.image_dataset import MultimodalMemeDataset, build_image_transform


def _make_fake_dataset(tmp_dir):
    # one RGB image, one RGBA image -- deliberately mirrors the mixed
    # modes our EDA found in the real MultiOFF dataset
    Image.new("RGB", (500, 300), color=(200, 100, 50)).save(os.path.join(tmp_dir, "a.png"))
    Image.new("RGBA", (600, 900), color=(10, 20, 30, 128)).save(os.path.join(tmp_dir, "b.png"))

    df = pd.DataFrame(
        {
            "image_name": ["a.png", "b.png"],
            "combined_text": ["some caption text", "another caption | ocr text"],
            "label": [0, 1],
        }
    )
    return df


def test_dataset_returns_correct_shapes_and_types():
    with tempfile.TemporaryDirectory() as tmp_dir:
        df = _make_fake_dataset(tmp_dir)
        transform = build_image_transform(train=False)
        ds = MultimodalMemeDataset(df, image_dir=tmp_dir, transform=transform)

        assert len(ds) == 2

        image, text, label = ds[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)  # 3 channels even for the RGBA source image
        assert isinstance(text, str)
        assert label == 0

        image_rgba_source, _, label2 = ds[1]
        assert image_rgba_source.shape == (3, 224, 224)  # RGBA input still comes out as 3 channels
        assert label2 == 1


def test_eval_transform_is_deterministic_train_transform_is_not():
    with tempfile.TemporaryDirectory() as tmp_dir:
        df = _make_fake_dataset(tmp_dir)

        eval_transform = build_image_transform(train=False)
        ds_eval = MultimodalMemeDataset(df, image_dir=tmp_dir, transform=eval_transform)
        img_a, _, _ = ds_eval[0]
        img_b, _, _ = ds_eval[0]
        assert torch.allclose(img_a, img_b), "eval transform (no augmentation) should be deterministic"

        train_transform = build_image_transform(train=True)
        ds_train = MultimodalMemeDataset(df, image_dir=tmp_dir, transform=train_transform)
        torch.manual_seed(0)
        img_c, _, _ = ds_train[0]
        torch.manual_seed(1)
        img_d, _, _ = ds_train[0]
        assert not torch.allclose(img_c, img_d), "train transform should apply randomized augmentation"
