"""
Uploads the trained image-only vision model (ResNet18) to Hugging Face
Hub, giving the FastAPI backend a permanent home for it -- same reasoning
as src/push_to_hub.py for DistilBERT.

Unlike DistilBERT, this isn't a transformers PreTrainedModel, so there's
no model.push_to_hub() convenience method -- instead we upload the raw
state_dict file directly via HfApi, and the API downloads it with
huggingface_hub.hf_hub_download() at startup.

IMPORTANT CONTEXT for the model card: this model is NOT used to produce
the actual /predict/image classification (see src/api.py's documented
decision -- the vision branch doesn't outperform OCR + text-only
DistilBERT). It's hosted here solely so the API can compute a LIVE
Grad-CAM overlay for transparency/diagnostic purposes on whatever image a
user uploads, per master_plan.md section 8's UI requirement. The model
card says this explicitly so nobody mistakes it for the deployed
classifier.

Run with (on Kaggle, wherever models/multimodal_image_only.pt exists):
    python -m src.push_image_model_to_hub --repo_id Prat-04/hate-speech-image-only
"""

import argparse

from huggingface_hub import HfApi, login

MODEL_CARD = """---
license: mit
tags:
- image-classification
- hate-speech-detection
- grad-cam
---

# Image-Only Vision Model (ResNet18) -- Diagnostic/Grad-CAM Use Only

**This model does NOT produce the deployed hate-speech classification.**
It is hosted here solely so the live API can compute a Grad-CAM overlay
(transparency/diagnostic visualization) for uploaded images. The actual
`/predict/image` prediction comes from OCR-extracted text run through
[Prat-04/hate-speech-distilbert](https://huggingface.co/Prat-04/hate-speech-distilbert).

## Why this model isn't used for the actual prediction

Trained and evaluated on the MultiOFF dataset (Suryawanshi et al., 2020) as
part of a required image-only/text-only/fusion comparison:

- Image-only test macro-F1: **0.5117** (barely above majority-class baseline)
- Text-only test macro-F1: 0.6573
- Grad-CAM analysis (see source repository) found this model's attention on
  text-heavy memes mostly reflects generic text-density texture patterns,
  not genuine visual understanding of hateful content -- it cannot "read"
  meme text the way OCR + a text model can.

Full comparison, methodology, and the decision rationale are documented in
the [source repository](https://github.com/Pratyaksh0404/hate-speech-detection)
under "Multimodal Results."

## Architecture
ResNet18 (ImageNet-pretrained), early layers frozen, only the last residual
block (`layer4`) fine-tuned. Binary classifier head (Non-offensive/Offensive).

## Intended use
Visualizing model attention (Grad-CAM) for transparency in a live demo only.
Not intended for any actual classification use -- use the DistilBERT model
above for that.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--checkpoint_path", default="models/multimodal_image_only.pt")
    args, _unknown_args = parser.parse_known_args()

    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret("HF_TOKEN")
    login(token=token)

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=False)

    print(f"uploading {args.checkpoint_path} to {args.repo_id} ...")
    api.upload_file(
        path_or_fileobj=args.checkpoint_path,
        path_in_repo="image_only_resnet18.pt",
        repo_id=args.repo_id,
        repo_type="model",
    )

    with open("IMAGE_MODEL_CARD.md", "w") as f:
        f.write(MODEL_CARD)
    api.upload_file(
        path_or_fileobj="IMAGE_MODEL_CARD.md",
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )

    print(f"done. model live at https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
