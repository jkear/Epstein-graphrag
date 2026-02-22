#!/bin/bash
# ============================================================
# Volunteer OCR Script - Contribute Your GPU to Process Files
# ============================================================
#
# This script:
# 1. Downloads an unprocessed manifest chunk
# 2. Runs OCR on your local GPU
# 3. Uploads results when complete
#
# Requirements:
# - NVIDIA GPU with 12GB+ VRAM (RTX 3060, 3070, 3080, 4070, etc.)
# - ~20GB free disk space
# - Python 3.11+
#
# Usage:
#   ./scripts/volunteer.sh
#
# ============================================================

set -e

echo "=============================================="
echo "  Epstein GraphRAG - Volunteer OCR Worker"
echo "=============================================="
echo ""

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. You need an NVIDIA GPU."
    exit 1
fi

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "Detected GPU with ${GPU_MEM}MB VRAM"

if [ "$GPU_MEM" -lt 11000 ]; then
    echo "WARNING: Your GPU has less than 12GB VRAM."
    echo "The model may not fit. Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

# Setup virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install dependencies
if ! command -v egr &> /dev/null; then
    echo "Installing dependencies..."
    pip install -q uv
    uv sync --frozen
fi

# Install vLLM if not present
if ! python -c "import vllm" 2>/dev/null; then
    echo "Installing vLLM (this may take a few minutes)..."
    pip install -q vllm
fi

# Configuration
CHUNK_SERVER="${CHUNK_SERVER:-https://your-server.com/chunks}"
RESULTS_SERVER="${RESULTS_SERVER:-https://your-server.com/upload}"
WORKER_ID="${WORKER_ID:-$(hostname)-$(date +%s)}"

echo ""
echo "Worker ID: $WORKER_ID"
echo ""

# Fetch available chunk
echo "Fetching available work..."
# TODO: Replace with actual chunk distribution server
# For now, use local manifest if available

if [ -f "data/manifest.json" ]; then
    MANIFEST="data/manifest.json"
    echo "Using local manifest: $MANIFEST"
else
    echo "ERROR: No manifest found."
    echo "Please run 'egr classify /path/to/pdfs' first."
    exit 1
fi

DOC_COUNT=$(python3 -c "import json; print(len(json.load(open('$MANIFEST')).get('documents', [])))")
echo "Found $DOC_COUNT documents to process"

# Start vLLM server in background
echo ""
echo "Starting vLLM server (this takes 1-2 minutes to load the model)..."
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --port 8000 \
    > /tmp/vllm.log 2>&1 &

VLLM_PID=$!

# Wait for server
echo "Waiting for vLLM server..."
for i in {1..120}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ vLLM server ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: vLLM server failed to start. Check /tmp/vllm.log"
        cat /tmp/vllm.log
        exit 1
    fi
    sleep 2
    echo -n "."
done
echo ""

# Calculate estimated time
# ~5 sec/page with vLLM, assuming ~2 pages/doc average
EST_SECONDS=$((DOC_COUNT * 2 * 5))
EST_HOURS=$((EST_SECONDS / 3600))
EST_MINS=$(((EST_SECONDS % 3600) / 60))
echo "Estimated time: ${EST_HOURS}h ${EST_MINS}m"
echo ""

# Run OCR
echo "Starting OCR processing..."
echo "Press Ctrl+C to stop (progress is saved, you can resume later)"
echo ""

egr ocr \
    --manifest "$MANIFEST" \
    --ocr-provider vllm \
    --vllm-url http://localhost:8000/v1 \
    --num-workers 3

echo ""
echo "=============================================="
echo "  OCR Complete!"
echo "=============================================="
echo ""
echo "Results saved to: data/processed/"
echo ""
echo "To continue with entity extraction:"
echo "  egr extract --num-workers 3"
echo ""

# Cleanup
kill $VLLM_PID 2>/dev/null || true

echo "Thank you for contributing!"
