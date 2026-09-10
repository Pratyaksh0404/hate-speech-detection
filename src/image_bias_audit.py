"""
Image-specific bias check (master_plan.md section 5.7, REQUIRED): spot-
checks whether the vision branch's prediction changes when only
SUPERFICIAL visual properties are altered (a skin-tone-like hue shift,
brightness, background blur) while the actual image content stays fixed.
Same spirit as the text bias audit: if predictions flip based on
superficial cues unrelated to genuine content, that's evidence of bias
rather than the model learning real signal.

Uses the image-only model specifically -- this is the branch being
checked, independent of the separate deployment decision to not use it in
production (that decision was about macro-F1 not adding value over
text-only; this is a different, still-required question about bias).

Honest interpretation note baked into the results: our image-only model's
overall signal is weak (test macro-F1 0.51, barely above chance -- see
results/multimodal_image_only.json), so some baseline instability under
ANY perturbation is expected and isn't automatically evidence of targeted
bias. What matters more than raw flip rate is whether there's a
CONSISTENT DIRECTIONAL shift in P(offensive) under a specific perturbation
(e.g. darker/warmer hues systematically increasing predicted offensiveness)
-- that's a meaningfully different, more concerning finding than noisy,
undirected flipping from an already-unstable model. Both are reported.

Run with (on Kaggle, wherever MultiOFF images are mounted):
    python -m src.image_bias_audit --image_dir "/kaggle/input/datasets/aditya1220/hate-speech-detection-dataset/Dataset/Labelled Images"
"""

import argparse
import json
import os

import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter

from src.image_dataset import build_image_transform
from src.models.multimodal_models import ImageOnlyClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Non-offensive", "Offensive"]


def load_model(checkpoint_path: str = "models/multimodal_image_only.pt"):
    model = ImageOnlyClassifier(n_classes=2)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(pil_image, model, transform):
    tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = probs.argmax().item()
    return LABEL_NAMES[pred_idx], probs[1].item()  # (label, P(offensive))


def hue_shift(pil_image, degrees: int):
    """Simulates a skin-tone-like superficial color shift -- rotates hue
    in HSV space, leaving actual content/composition untouched."""
    hsv = pil_image.convert("HSV")
    h, s, v = hsv.split()
    h = h.point(lambda x: (x + degrees) % 256)
    return Image.merge("HSV", (h, s, v)).convert("RGB")


def brightness_shift(pil_image, factor: float):
    return ImageEnhance.Brightness(pil_image).enhance(factor)


def background_blur(pil_image, radius: int = 8):
    """Stand-in for 'background clutter' change -- blurs the whole frame
    uniformly, since no segmentation is available to isolate a true
    background region from the foreground content."""
    return pil_image.filter(ImageFilter.GaussianBlur(radius=radius))


def run_audit(image_dir: str, test_csv: str = "data/multimodal/test.csv", n_samples: int = 20, seed: int = 42):
    model = load_model()
    transform = build_image_transform(train=False)

    df = pd.read_csv(test_csv).sample(n=min(n_samples, 999), random_state=seed)

    rows = []
    n_flipped = {"hue": 0, "brightness": 0, "blur": 0}
    delta_sum = {"hue": 0.0, "brightness": 0.0, "blur": 0.0}

    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, row["image_name"])
        try:
            original = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        orig_label, orig_p = predict_image(original, model, transform)

        variants = {
            "hue": hue_shift(original, degrees=60),
            "brightness": brightness_shift(original, factor=1.6),
            "blur": background_blur(original, radius=8),
        }

        row_result = {
            "image_name": row["image_name"],
            "true_label": int(row["label"]),
            "original_pred": orig_label,
            "original_p_offensive": orig_p,
        }

        line = f"{row['image_name']}: orig={orig_label} ({orig_p:.2f})"
        for name, variant_img in variants.items():
            label, p = predict_image(variant_img, model, transform)
            flipped = label != orig_label
            n_flipped[name] += flipped
            delta_sum[name] += p - orig_p
            row_result[f"{name}_pred"] = label
            row_result[f"{name}_p_offensive"] = p
            row_result[f"{name}_flipped"] = flipped
            line += f" | {name}={label} ({p:.2f}){' <--FLIP' if flipped else ''}"

        rows.append(row_result)
        print(line)

    n = len(rows)
    print(f"\n=== IMAGE BIAS AUDIT RESULTS (n={n} images) ===")
    summary = {}
    for name in ("hue", "brightness", "blur"):
        flip_rate = n_flipped[name] / n
        mean_delta = delta_sum[name] / n
        print(
            f"{name}: flip rate {flip_rate:.1%}, mean change in P(offensive) "
            f"{mean_delta:+.3f} ({'higher' if mean_delta > 0 else 'lower'} after perturbation, on average)"
        )
        summary[f"{name}_flip_rate"] = flip_rate
        summary[f"{name}_mean_p_offensive_delta"] = mean_delta

    os.makedirs("results", exist_ok=True)
    with open("results/image_bias_audit.json", "w") as f:
        json.dump({"n_images": n, **summary, "rows": rows}, f, indent=2)
    print("\nsaved: results/image_bias_audit.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--test_csv", default="data/multimodal/test.csv")
    parser.add_argument("--n_samples", type=int, default=20)
    args, _unknown_args = parser.parse_known_args()
    run_audit(args.image_dir, args.test_csv, args.n_samples)


if __name__ == "__main__":
    main()
