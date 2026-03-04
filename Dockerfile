# ── AutoML-X Fraud Detection — Hugging Face Spaces Dockerfile ─
# Runs 3 services in one container:
#   - FastAPI     → port 7860 (HF Spaces default)
#   - Streamlit   → port 8501
#   - MLflow UI   → port 5000
# supervisord manages all 3 processes.

FROM python:3.11-slim

LABEL maintainer="Sujal Baghela"
LABEL description="AutoML-X Fraud Detection — Full Stack"
LABEL version="2.0"

# ── System dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y \
    libgomp1 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# ── Hugging Face Spaces requires a non-root user ─────────────
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

# ── Supervisord config — manages all 3 services ───────────────
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# ── Switch to non-root user (required by HF Spaces) ──────────
USER appuser

# ── HF Spaces uses port 7860 by default ──────────────────────
EXPOSE 7860 8501 5000

# ── Start all services via supervisord ───────────────────────
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]