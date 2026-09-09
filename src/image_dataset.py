"""
PyTorch Dataset for the MultiOFF multimodal meme dataset (master_plan.md
sections 4A / 5.6). Returns (image_tensor, combined_text, label) triples —
ONE dataset class shared by all three required baselines (image-only,
text-only, fusion), per master_plan.md section 4A.5: each model-specific
script uses only the parts of the triple it needs (image-only ignores the
text, text-only ignores the image, fusion uses both), rather than
duplicating a near-identical Dataset class three times.

Design decisions:

1. Images are force-converted to RGB before any transform. Our EDA found
   both RGB and RGBA images in this dataset (some with an alpha channel);
   torchvision's ImageNet-pretrained ResNet18 expects exactly 3 channels,
   so RGBA images would crash or silently misbehave without this.

2. Resize to 224x224 with ImageNet normalization stats — the standard
   input contract for torchvision's ImageNet-pretrained backbones,
   including ResNet18 (our chosen image backbone). Our EDA also found
   image sizes varying wildly (471-2668px wide, 346-5312px tall, some
   extreme aspect ratios) — resizing to a fixed square necessarily
   distorts aspect ratio for some images, which is standard, accepted
   practice for this kind of transfer learning, not something we
   specially correct for.

3. Augmentation (horizontal flip, slight rotation, color jitter) is
   applied ONLY to the train split, never val/test — per master_plan.md
   section 5.6, to reduce overfitting on this "likely-small multimodal
   dataset" (743 examples total) without corrupting the val/test numbers
   we actually report.

4. combined_text (OCR + caption, from src/prepare_multimodal_data.py) is
   returned as a raw string, not tokenized here — tokenization differs
   per text encoder (DistilBERT's tokenizer vs. our own vocab.py), so
   it's left to whichever model-specific training script consumes this
   dataset, exactly like we did for the text-only DistilBERT rung.
"""

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_transform(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class MultimodalMemeDataset(Dataset):
    def __init__(self, df, image_dir: str, transform=None, text_column: str = "combined_text"):
        self.image_names = df["image_name"].tolist()
        self.texts = df[text_column].fillna("").tolist()
        self.labels = df["label"].tolist()
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = f"{self.image_dir}/{self.image_names[idx]}"
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.texts[idx], self.labels[idx]
