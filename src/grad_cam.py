"""
Grad-CAM visualization for image predictions (master_plan.md section 6,
REQUIRED): visualizes which regions of a meme image the image-only
model's CNN attended to for a given prediction, by weighting the last
convolutional layer's (layer4) feature maps by the gradient of the
predicted class score with respect to them (Selvaraju et al., 2017).

Implemented from scratch with forward/backward hooks, not a third-party
library -- consistent with this project's "write and understand every
piece" approach, and avoids an extra, less-controllable dependency. The
core weighting/normalization math was verified against synthetic data
before writing the full PyTorch version (see chat) -- this file adds the
model-specific plumbing (hooks on the real ResNet18, real forward/
backward pass) on top of already-checked arithmetic.

Target layer is image_encoder.backbone.layer4 -- the same last residual
block we chose to fine-tune (see src/models/multimodal_models.py), and
the standard Grad-CAM target for a ResNet: the last spatial feature map
before global average pooling collapses it to a single vector.

Run with (on Kaggle, wherever MultiOFF images + models/multimodal_image_only.pt exist):
    python -m src.grad_cam --image_dir "..." --n_samples 6
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from src.image_dataset import IMAGENET_MEAN, IMAGENET_STD, build_image_transform
from src.models.multimodal_models import ImageOnlyClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Non-offensive", "Offensive"]


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class: int = None):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        score = logits[0, target_class]
        score.backward()

        # global-average-pool gradients over spatial dims -> one weight per channel
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        pred_prob = torch.softmax(logits, dim=1)[0, target_class].item()
        return cam, target_class, pred_prob


def unnormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Reverses ImageNet normalization for display purposes."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.detach().cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def load_model(checkpoint_path: str = "models/multimodal_image_only.pt"):
    model = ImageOnlyClassifier(n_classes=2)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def run_grad_cam(
    image_dir: str,
    test_csv: str = "data/multimodal/test.csv",
    n_samples: int = 6,
    seed: int = 42,
    out_path: str = "results/grad_cam_examples.png",
):
    model = load_model()
    target_layer = model.image_encoder.backbone.layer4
    grad_cam = GradCAM(model, target_layer)
    transform = build_image_transform(train=False)

    df = pd.read_csv(test_csv).sample(n=min(n_samples, 999), random_state=seed).reset_index(drop=True)

    fig, axes = plt.subplots(2, len(df), figsize=(4 * len(df), 8))
    if len(df) == 1:
        axes = axes.reshape(2, 1)

    rows_summary = []
    for i, row in df.iterrows():
        img_path = os.path.join(image_dir, row["image_name"])
        original = Image.open(img_path).convert("RGB")
        input_tensor = transform(original).unsqueeze(0).to(DEVICE)
        input_tensor.requires_grad_(True)

        cam, pred_class, pred_prob = grad_cam.generate(input_tensor)
        display_img = unnormalize_image(input_tensor[0])

        cam_resized = (
            np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0
        )

        true_label = LABEL_NAMES[int(row["label"])]
        pred_label = LABEL_NAMES[pred_class]

        axes[0, i].imshow(display_img)
        axes[0, i].set_title(f"True: {true_label}\nPred: {pred_label} ({pred_prob:.2f})", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(display_img)
        axes[1, i].imshow(cam_resized, cmap="jet", alpha=0.5)
        axes[1, i].set_title("Grad-CAM", fontsize=9)
        axes[1, i].axis("off")

        rows_summary.append(
            {"image_name": row["image_name"], "true_label": true_label, "pred_label": pred_label, "pred_prob": pred_prob}
        )
        print(f"{row['image_name']}: true={true_label}, pred={pred_label} ({pred_prob:.2f})")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"\nsaved: {out_path}")

    for r in rows_summary:
        print(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--test_csv", default="data/multimodal/test.csv")
    parser.add_argument("--n_samples", type=int, default=6)
    args, _unknown_args = parser.parse_known_args()
    run_grad_cam(args.image_dir, args.test_csv, args.n_samples)


if __name__ == "__main__":
    main()
