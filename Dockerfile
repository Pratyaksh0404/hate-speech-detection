# =============================================================================
# Multi-stage Dockerfile for Render.com's free Web Service tier.
#
# (Originally written for HuggingFace Spaces' Docker SDK, but HF now
# requires a PRO subscription for Docker/Gradio Spaces on free accounts --
# only their Static SDK is free. Render's free tier genuinely supports
# Docker with no card required, so we deploy there instead. The container
# itself needed only two changes: (1) listen on Render's $PORT instead of
# a hardcoded 7860, and (2) pre-download model weights at build time,
# since Render's free tier has no persistent disk -- every cold start
# after the ~15min idle spin-down would otherwise re-download from HF Hub.)
#
# Stage 1 — Build the React frontend with Vite.
# Stage 2 — Slim Python image that serves FastAPI + the built static files.
# =============================================================================

# ── Stage 1: React build ─────────────────────────────────────────────────────
FROM node:20-slim AS frontend-build

WORKDIR /build
COPY app/frontend/package*.json ./
RUN npm ci --ignore-scripts
COPY app/frontend/ ./
RUN npm run build
# Output: /build/dist/  (index.html + assets/ + examples/)

# ── Stage 2: Python runtime ──────────────────────────────────────────────────
FROM python:3.12-slim

# System dependencies: Tesseract OCR engine + English language data
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user (good practice regardless of host; harmless on Render)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    # HuggingFace cache goes here (model downloads)
    HF_HOME=/home/user/.cache/huggingface

WORKDIR $HOME/app

# ── Install Python deps ──────────────────────────────────────────────────────
# CPU-only PyTorch first (saves ~1.8 GB vs the full CUDA build)
RUN pip install --no-cache-dir --user \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements-deploy.txt ./
RUN pip install --no-cache-dir --user -r requirements-deploy.txt

# ── Pre-download model weights at BUILD time ─────────────────────────────────
# Render's free tier has no persistent disk, so without this, every cold
# start after the container spins down (idle ~15min) would re-download
# ~300MB from HF Hub before it could serve a single request. Baking the
# weights into the image here means that only happens once, at deploy
# time, not on every wake-up. Repo IDs are duplicated from src/api.py's
# MODEL_REPO / IMAGE_MODEL_REPO constants -- update both places if those
# ever change.
#
# IMPORTANT: this uses huggingface_hub's snapshot_download/hf_hub_download
# directly -- it just fetches and caches the raw repo files over HTTP. It
# deliberately does NOT import transformers or construct the actual model
# (e.g. via AutoModelForSequenceClassification.from_pretrained), because
# that would load ~270MB of float32 weights into RAM on top of torch's own
# import footprint. Render's FREE BUILD MACHINE is also capped at 512MB --
# an earlier version of this step did exactly that and was OOM-killed
# during the build (not even at runtime). Plain file downloads use at most
# a few tens of MB, well clear of that limit.
# ALSO fetches distilbert_quantized_int8.pt -- the pre-quantized text
# model produced offline by export_quantized_model.py (repo root) and
# uploaded to the SAME hate-speech-distilbert HF repo. api.py loads this
# file directly (torch.load) instead of the regular fp32 weights that
# snapshot_download also happens to pull down as part of the repo; those
# extra fp32 files go unused at runtime but are harmless, just a bit of
# wasted image size.
RUN python -c "\
from huggingface_hub import snapshot_download, hf_hub_download; \
snapshot_download(repo_id='Prat-04/hate-speech-distilbert'); \
hf_hub_download(repo_id='Prat-04/hate-speech-distilbert', filename='distilbert_quantized_int8.pt'); \
hf_hub_download(repo_id='Prat-04/hate-speech-image-only', filename='image_only_resnet18.pt')"

# From here on, loading these two repos should hit the local cache only --
# never the network. (Set AFTER the download step above, which needs
# network access; Docker ENV only affects subsequent layers, not the one
# that just ran.)
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# ── Copy application code ────────────────────────────────────────────────────
COPY --chown=user src/ ./src/

# ── Copy built frontend from Stage 1 ─────────────────────────────────────────
COPY --from=frontend-build --chown=user /build/dist ./static

# Render assigns its own port via $PORT at runtime (not a fixed value we can
# hardcode) -- shell form is needed here instead of JSON-array form so that
# ${PORT:-7860} actually gets expanded. The 7860 fallback keeps this image
# runnable with a plain `docker run` locally, where $PORT isn't set.
CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-7860}
