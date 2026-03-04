FROM python:3.11-slim

LABEL maintainer="Sujal Baghela"
LABEL description="AutoML-X Fraud Detection API"

# ── System dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user (required by HF Spaces) ─────────────
RUN useradd -m -u 1000 appuser

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────
COPY . .

# ── Create required directories ───────────────────────────────
RUN mkdir -p models reports/evaluation reports/shap logs \
    && chown -R appuser:appuser /app

# ── Switch to non-root user ───────────────────────────────────
USER appuser

# ── HF Spaces default port ────────────────────────────────────
EXPOSE 7860

# ── Start FastAPI only (simple and reliable) ──────────────────
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]