#!/bin/bash
# ══════════════════════════════════════════════════════════════
# start_local.sh  —  LOCAL DEVELOPMENT ONLY
#
# Starts both apps directly (without Docker) for fast dev.
#
# Usage:
#   chmod +x start_local.sh
#   ./start_local.sh          (Linux/Mac/Git Bash on Windows)
# ══════════════════════════════════════════════════════════════
set -e

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║         AutoML-X  —  Local Dev Mode              ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Fraud Detection  →  http://localhost:8501       ║"
echo "║  Universal        →  http://localhost:8502       ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Check ports are free ──────────────────────────────────────
for PORT in 8501 8502; do
    if lsof -i ":$PORT" > /dev/null 2>&1; then
        echo "ERROR: Port $PORT is already in use. Stop the existing process first."
        exit 1
    fi
done

# ── Activate venv if present ──────────────────────────────────
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtualenv…"
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    echo "Activating .venv…"
    source .venv/bin/activate
fi

# ── Create log directory ──────────────────────────────────────
mkdir -p logs

# ── Start Fraud Detection on 8501 ────────────────────────────
echo "Starting Fraud Detection on port 8501…"
streamlit run app_fraud_local.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.maxUploadSize=2048 \
    --browser.gatherUsageStats=false \
    --server.fileWatcherType=auto \
    > logs/fraud.log 2>&1 &
FRAUD_PID=$!
echo "  Fraud PID: $FRAUD_PID  (logs → logs/fraud.log)"

# ── Start Universal Trainer on 8502 ──────────────────────────
echo "Starting Universal Trainer on port 8502…"
streamlit run app_universal.py \
    --server.port=8502 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.maxUploadSize=2048 \
    --browser.gatherUsageStats=false \
    --server.fileWatcherType=auto \
    > logs/universal.log 2>&1 &
UNIVERSAL_PID=$!
echo "  Universal PID: $UNIVERSAL_PID  (logs → logs/universal.log)"

# ── Wait for apps to start then verify ───────────────────────
echo ""
echo "Waiting for apps to start…"
sleep 5

check_app() {
    local NAME=$1
    local PORT=$2
    local PID=$3
    if ! kill -0 $PID 2>/dev/null; then
        echo "  ERROR: $NAME failed to start. Check logs/${NAME,,}.log"
        return 1
    fi
    echo "  ✓ $NAME is running (PID $PID)"
}

check_app "Fraud"     8501 $FRAUD_PID
check_app "Universal" 8502 $UNIVERSAL_PID

echo ""
echo "Both services running. Press Ctrl+C to stop."
echo "Logs: logs/fraud.log | logs/universal.log"
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