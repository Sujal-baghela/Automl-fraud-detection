#!/bin/bash
set -e

echo "===== Starting AutoML-X Services ====="

# Start FastAPI in background
echo "Starting FastAPI on port 7860..."
uvicorn app.api:app --host 0.0.0.0 --port 7860 --workers 1 &
FASTAPI_PID=$!

# Start Streamlit in background
echo "Starting Streamlit on port 8501..."
streamlit run app_dashboard.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false &
STREAMLIT_PID=$!

echo "FastAPI PID: $FASTAPI_PID"
echo "Streamlit PID: $STREAMLIT_PID"

# Wait for either process to exit
wait -n $FASTAPI_PID $STREAMLIT_PID

echo "A service exited. Keeping container alive..."
wait