#!/bin/bash
# vLLM Worker Entrypoint - FASTEST OCR option
# 
# vLLM continuous batching = 5-10x faster than Ollama
# Processes multiple pages simultaneously on single GPU

set -e

echo "=============================================="
echo "Epstein GraphRAG - vLLM OCR Worker"
echo "=============================================="
echo "Model: Qwen2-VL-7B-Instruct"
echo "Engine: vLLM (5-10x faster than Ollama)"
echo "=============================================="

# Configuration
MODEL="${VLLM_MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
GPU_MEMORY="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# Start vLLM server in background
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --trust-remote-code \
    --gpu-memory-utilization "$GPU_MEMORY" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype auto \
    --port 8000 \
    &

VLLM_PID=$!

# Wait for server to be ready
echo "Waiting for vLLM server to initialize..."
for i in {1..120}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ vLLM server ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: vLLM server failed to start"
        exit 1
    fi
    sleep 2
done

# Check for manifest
MANIFEST="${MANIFEST_PATH:-/data/manifest.json}"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest not found at $MANIFEST"
    echo "Mount your data with: -v /path/to/data:/data"
    exit 1
fi

DOC_COUNT=$(python3 -c "import json; print(len(json.load(open('$MANIFEST')).get('documents', [])))")
echo "Found $DOC_COUNT documents to process"

# Worker count (vLLM handles batching internally, so more workers = more concurrent PDFs)
NUM_WORKERS="${NUM_WORKERS:-5}"
echo "Running with $NUM_WORKERS parallel PDF workers"
echo "(vLLM batches pages internally for even higher throughput)"

# Run OCR with vLLM backend
echo "Starting OCR pipeline..."
egr ocr \
    --manifest "$MANIFEST" \
    --ocr-provider vllm \
    --num-workers "$NUM_WORKERS"

echo "=============================================="
echo "OCR complete!"
echo "Results in /data/processed/"
echo "=============================================="

# Cleanup
kill $VLLM_PID 2>/dev/null || true

# Keep alive for result collection if requested
if [ "${KEEP_ALIVE:-false}" = "true" ]; then
    echo "Container staying alive..."
    tail -f /dev/null
fi
