# =============================================================================
# Multi-stage Dockerfile for HuggingFace Spaces (free Docker SDK tier).
#
# Stage 1 — Build the React frontend with Vite.
# Stage 2 — Slim Python image that serves FastAPI + the built static files.
#
# HF Spaces runs containers as uid 1000; we create that user explicitly.
# The app listens on port 7860 (HF Spaces default).
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

# Create non-root user (HF Spaces requirement: uid 1000)
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

# ── Copy application code ────────────────────────────────────────────────────
COPY --chown=user src/ ./src/

# ── Copy built frontend from Stage 1 ─────────────────────────────────────────
COPY --from=frontend-build --chown=user /build/dist ./static

EXPOSE 7860

# Uvicorn serves FastAPI; static files are mounted inside the app.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
