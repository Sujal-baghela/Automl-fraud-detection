#!/bin/bash
# ══════════════════════════════════════════════════════════════
# start_local.sh  —  LOCAL DEVELOPMENT ONLY
#
# Starts both apps directly (without Docker) for fast dev.
# Use this when running locally without Docker.
#
# Usage:
#   chmod +x start_local.sh
#   ./start_local.sh
#
# Or with Docker:
#   docker compose up --build
# ══════════════════════════════════════════════════════════════
set -e

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║         AutoML-X  —  Local Dev Mode              ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  🛡️  Fraud Detection  →  http://localhost:8501   ║"
echo "║  🤖  Universal        →  http://localhost:8502   ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Activate venv if present ──────────────────────────────────
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtualenv…"
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    echo "Activating .venv…"
    source .venv/bin/activate
fi

# ── Start Fraud Detection on 8501 ────────────────────────────
echo "Starting 🛡️  Fraud Detection on port 8501…"
streamlit run app_fraud_local.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.fileWatcherType=auto &
FRAUD_PID=$!
echo "  Fraud PID: $FRAUD_PID"

# ── Start Universal Trainer on 8502 ──────────────────────────
echo "Starting 🤖  Universal Trainer on port 8502…"
streamlit run app_universal.py \
    --server.port=8502 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.fileWatcherType=auto &
UNIVERSAL_PID=$!
echo "  Universal PID: $UNIVERSAL_PID"

echo ""
echo "Both services running. Press Ctrl+C to stop."
echo ""

# ── Cleanup on exit ───────────────────────────────────────────
cleanup() {
    echo ""
    echo "Stopping services…"
    kill $FRAUD_PID     2>/dev/null || true
    kill $UNIVERSAL_PID 2>/dev/null || true
    echo "Done."
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Wait ──────────────────────────────────────────────────────
wait $FRAUD_PID $UNIVERSAL_PID