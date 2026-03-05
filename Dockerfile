# ══════════════════════════════════════════════════════════════
# Dockerfile  —  AutoML-X v6.0  (HuggingFace Space)
# Runs: app_universal.py on port 7860
# ══════════════════════════════════════════════════════════════

FROM python:3.11-slim

LABEL maintainer="AutoML-X"

# ── System dependencies ───────────────────────────────────────
# libgomp1 is required by LightGBM on Linux
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (required by HuggingFace Spaces) ───────────
RUN useradd -m -u 1000 appuser

# ── Set workdir AFTER creating user ──────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────
COPY . .

# ── Create runtime directories and hand off to appuser ───────
RUN mkdir -p models logs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

# ── Launch Streamlit on HF port ───────────────────────────────
CMD ["streamlit", "run", "app_universal.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.maxUploadSize=2048", \
     "--browser.gatherUsageStats=false"]