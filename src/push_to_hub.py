"""
Uploads the trained DistilBERT hate-speech classifier to Hugging Face Hub
(master_plan.md: "Large trained models go to HuggingFace Hub (free), not
committed directly to git"). Gives the FastAPI backend a permanent, free,
publicly-loadable home for the model instead of depending on ephemeral
Kaggle sessions.

Reads the HF token from Kaggle Secrets (Add-ons -> Secrets), never
hardcoded or passed on the command line -- keeps it out of notebook
history/version snapshots entirely.

Sets proper id2label/label2id on the model config before pushing, so
anyone loading this model gets meaningful label names ("Hate Speech")
instead of generic ones ("LABEL_0").

Run with (on Kaggle, after training produces models/distilbert.pt):
    python -m src.push_to_hub --repo_id Prat-04/hate-speech-distilbert
"""

import argparse

import torch
from huggingface_hub import HfApi, login
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
CHECKPOINT_PATH = "models/distilbert.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ID2LABEL = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}
LABEL2ID = {"Hate Speech": 0, "Offensive Language": 1, "Neither": 2}

MODEL_CARD = """---
license: mit
language: en
tags:
- text-classification
- hate-speech-detection
datasets:
- tdavidson/hate_speech_offensive
metrics:
- f1
---

# Hate Speech Detection (DistilBERT, fine-tuned)

DistilBERT fine-tuned for 3-class hate speech / offensive language detection,
trained on the Davidson et al. (2017) hate speech and offensive language
dataset (24,783 tweets).

## Labels
- 0: Hate Speech
- 1: Offensive Language
- 2: Neither

## Performance (held-out test set, n=3718)
- Macro-F1: 0.7663
- Accuracy: 0.8994

Best-performing model out of a 6-model ladder (TF-IDF+LogReg/SVM, BiLSTM,
CNN-BiLSTM, BiLSTM+Attention, DistilBERT) -- full comparison and methodology
in the source repository.

## Bias audit
Tested against a documented failure mode in this dataset family (Sap et al.,
2019): over-flagging African-American Vernacular English (AAVE) as offensive.
On 20 hand-constructed AAVE/Standard-English sentence pairs with equivalent
(benign) meaning, this model showed a **0% false-positive-rate gap** between
dialects, versus a 20-point gap in a TF-IDF+LogReg baseline tested the same
way (see repo for full methodology, including statistical tests).

## Known limitations
Qualitative error analysis (see repo, backed by LIME explanations) found the
model's decisions are driven heavily by the presence of specific slur tokens
as near-decisive lexical triggers, rather than deeper contextual understanding
of intent or target. Personal insults using slurs can be escalated to "Hate
Speech" when annotators labeled them "Offensive Language," and coded/indirect
offensive language without explicit slur vocabulary is sometimes missed
entirely.

## Intended use
Content-moderation assistance / research and portfolio demonstration. NOT
intended as a sole or automatic basis for real moderation decisions against
real people. Predictions are probabilistic and can be wrong -- see the full
disclaimer and bias-audit results in the source repository before any other use.
"""


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    return tokenizer, model


def get_hf_token() -> str:
    """Reads the HF token from Kaggle Secrets -- never hardcoded, never
    passed as a CLI arg (which would leak into notebook cell history)."""
    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()
    return secrets.get_secret("HF_TOKEN")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True)
    args, _unknown_args = parser.parse_known_args()

    token = get_hf_token()
    login(token=token)

    tokenizer, model = load_model()

    print(f"pushing model to {args.repo_id} ...")
    model.push_to_hub(args.repo_id, private=False, token=token)
    tokenizer.push_to_hub(args.repo_id, token=token)

    with open("MODEL_CARD.md", "w") as f:
        f.write(MODEL_CARD)

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj="MODEL_CARD.md",
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"done. model live at https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
