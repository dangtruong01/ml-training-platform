#!/bin/bash

# Script to start FastAPI server with cloud training enabled
cd /Users/truonghaidang/Desktop/open-trainer/backend

# Load environment variables from .env file
export $(cat .env | grep -v '^#' | xargs)

# Set Python path
export PYTHONPATH=/Users/truonghaidang/Desktop/open-trainer/backend

# Kill any existing server
pkill -f "uvicorn app.main:app" 2>/dev/null || true

echo "🚀 Starting FastAPI server with cloud training enabled..."
echo "📁 Working directory: $(pwd)"
echo "🔑 Environment variables loaded from .env"
echo "🌐 Server will be available at: http://127.0.0.1:8000"
echo "☁️ Cloud training: ENABLED"

# Start the server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir /Users/truonghaidang/Desktop/open-trainer/backend
