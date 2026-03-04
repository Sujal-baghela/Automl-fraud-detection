FROM python:3.11-slim

LABEL maintainer="Sujal Baghela"

# ── System dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user ──────────────────────────────────────
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────
COPY . .

# ── Create required directories & permissions ─────────────────
RUN mkdir -p models logs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

# ── Run Streamlit on port 7860 (HF Spaces default) ───────────
CMD ["streamlit", "run", "app_dashboard.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]