#!/bin/bash
# ============================================================
# RTX 3080 Ti Setup - vLLM Server
# ============================================================
#
# Run this on your Windows/Linux PC with RTX 3080 Ti
# Starts a vLLM server that your Mac can connect to
#
# ============================================================

set -e

echo "=============================================="
echo "  Epstein GraphRAG - RTX 3080 Ti Server"
echo "=============================================="

# Check NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Install vLLM if needed
if ! python -c "import vllm" 2>/dev/null; then
    echo "Installing vLLM..."
    pip install vllm
fi

# Configuration
MODEL="${MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.90}"

echo "Starting vLLM server..."
echo "  Model: $MODEL"
echo "  Port: $PORT"
echo "  GPU utilization: $GPU_UTIL"
echo ""
echo "Server will be available at: http://$(hostname -I | awk '{print $1}'):$PORT/v1"
echo "Use this URL on your Mac to connect."
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --trust-remote-code \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len 4096 \
    --host 0.0.0.0 \
    --port "$PORT"
