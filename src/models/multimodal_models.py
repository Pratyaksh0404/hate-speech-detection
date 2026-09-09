"""
Multimodal models for the MultiOFF dataset (master_plan.md section 4A):
image-only, text-only, and fusion baselines, sharing an image encoder and
a text encoder so the three required comparison models (section 4A.5)
are directly comparable — same encoders, different combination.

Design decisions:

1. Image branch: ResNet18 (ImageNet-pretrained), early layers frozen,
   only layer4 (the last residual block) fine-tuned — per master_plan.md
   section 4A.2, keeps this feasible on free GPU compute and limits
   overfitting risk given only 445 training images.

2. Text branch: DistilBERT (per master_plan.md's recommendation to reuse
   the best text-ladder encoder). UNLIKE the text-only ladder (rung 5,
   ~17k training rows), we freeze most of DistilBERT here and fine-tune
   only its last transformer layer + classifier head — mirroring the
   image branch's freeze-most/fine-tune-last-block strategy. This is a
   deliberate departure from full fine-tuning: 445 training examples is
   roughly 40x smaller than the text-only ladder's training set, and
   fully fine-tuning a 66M-parameter model on that little data would
   overfit far more severely than it's worth.

3. Fusion: simple concatenation of the two fixed-size embeddings
   (512-dim image + 768-dim text) -> dense -> dropout -> classifier head.
   Per master_plan.md section 4A.4, this is the "simple concatenation
   first" approach — attention-based fusion is a stretch goal if time
   allows, not a requirement.

4. All three model classes share the SAME forward(images, input_ids,
   attention_mask) signature, even though image-only ignores the text
   args and text-only ignores the image arg. This lets one shared
   training loop (src/train_multimodal.py) call any of the three
   identically, rather than branching per model type.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from transformers import AutoModel

TEXT_MODEL_NAME = "distilbert-base-uncased"
IMAGE_EMBED_DIM = 512
TEXT_EMBED_DIM = 768


class ImageEncoder(nn.Module):
    """ResNet18 backbone, early layers frozen, only layer4 fine-tuned.
    Outputs a 512-dim embedding (final fc removed)."""

    def __init__(self):
        super().__init__()
        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad = False
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        resnet.fc = nn.Identity()  # remove classification head, keep 512-dim pooled features
        self.backbone = resnet

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)  # [B, 512]


class TextEncoder(nn.Module):
    """DistilBERT backbone, only the last transformer layer fine-tuned.
    Outputs the first-token ([CLS]-equivalent) hidden state, matching how
    DistilBertForSequenceClassification itself pools for classification."""

    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(TEXT_MODEL_NAME)
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.transformer.layer[-1].parameters():
            param.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [B, 768]


class ImageOnlyClassifier(nn.Module):
    def __init__(self, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(IMAGE_EMBED_DIM, n_classes)

    def forward(self, images, input_ids=None, attention_mask=None):
        embed = self.image_encoder(images)
        return self.fc(self.dropout(embed))


class TextOnlyClassifier(nn.Module):
    def __init__(self, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(TEXT_EMBED_DIM, n_classes)

    def forward(self, images=None, input_ids=None, attention_mask=None):
        embed = self.text_encoder(input_ids, attention_mask)
        return self.fc(self.dropout(embed))


class FusionClassifier(nn.Module):
    def __init__(self, n_classes: int = 2, dropout: float = 0.3, hidden_dim: int = 256):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        combined_dim = IMAGE_EMBED_DIM + TEXT_EMBED_DIM
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, images, input_ids, attention_mask):
        image_embed = self.image_encoder(images)
        text_embed = self.text_encoder(input_ids, attention_mask)
        combined = torch.cat([image_embed, text_embed], dim=1)
        hidden = torch.relu(self.fc1(combined))
        return self.fc2(self.dropout(hidden))


class AttentionFusionClassifier(nn.Module):
    """
    Gated fusion: instead of blindly concatenating both embeddings (which
    forces the model to use both modalities equally every time), this
    projects each embedding to a shared dimension, then learns a per-
    example softmax gate over the two modalities (similar to a Gated
    Multimodal Unit — Arevalo et al., 2017).

    Built specifically because plain FusionClassifier (concatenation)
    measurably underperformed text-only alone on this dataset (test
    macro-F1 0.56 vs 0.66) — confirmed NOT a training-time issue (numbers
    were stable across a 2.5x-longer training run). Concatenation gives
    the model no mechanism to down-weight the image branch, which we
    separately confirmed carries a weak, noisy signal on its own
    (image-only test macro-F1 0.51, barely above majority-class guessing).
    This gate gives the model exactly that mechanism.
    """

    def __init__(self, n_classes: int = 2, dropout: float = 0.3, proj_dim: int = 256):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_proj = nn.Linear(IMAGE_EMBED_DIM, proj_dim)
        self.text_proj = nn.Linear(TEXT_EMBED_DIM, proj_dim)
        self.gate = nn.Linear(proj_dim * 2, 2)  # outputs one weight per modality
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(proj_dim, n_classes)

    def forward(self, images, input_ids, attention_mask, return_attention: bool = False):
        image_embed = self.image_encoder(images)
        text_embed = self.text_encoder(input_ids, attention_mask)

        image_proj = torch.relu(self.image_proj(image_embed))  # [B, proj_dim]
        text_proj = torch.relu(self.text_proj(text_embed))  # [B, proj_dim]

        gate_input = torch.cat([image_proj, text_proj], dim=1)  # [B, proj_dim*2]
        gate_weights = torch.softmax(self.gate(gate_input), dim=1)  # [B, 2] -- sums to 1 per example

        combined = gate_weights[:, 0:1] * image_proj + gate_weights[:, 1:2] * text_proj  # [B, proj_dim]

        dropped = self.dropout(combined)
        logits = self.fc(dropped)

        if return_attention:
            return logits, gate_weights
        return logits
