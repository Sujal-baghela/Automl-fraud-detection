# ── Stage 1: Base image ───────────────────────────────────────
# python:3.10-slim is the standard production choice:
# - Full Python 3.10 without dev tools or docs (~150MB vs ~900MB)
# - Matches your local venv Python version exactly
FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────
LABEL maintainer="Sujal Baghela <sujal@example.com>"
LABEL description="AutoML-X Fraud Detection API"
LABEL version="1.0"

# ── System dependencies ───────────────────────────────────────
# libgomp1 is required by LightGBM for OpenMP parallelization.
# Without it LightGBM will crash on import inside the container.
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────
# Copy requirements first — Docker caches this layer separately.
# If only your code changes (not requirements), this layer is
# reused and pip install is skipped on rebuild. Saves ~2 minutes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────
# Copy everything except what's in .dockerignore
COPY . .

# ── Create required directories ───────────────────────────────
# These are gitignored so they won't exist in the repo clone.
# The API needs them to exist at startup.
RUN mkdir -p models reports/evaluation reports/shap logs

# ── Port ──────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────
# Docker will ping /health every 30s.
# If it fails 3 times the container is marked unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# ── Start command ─────────────────────────────────────────────
# --host 0.0.0.0  → accept connections from outside the container
# --port 8000     → match EXPOSE above
# --workers 1     → single worker (scale with docker-compose if needed)
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
