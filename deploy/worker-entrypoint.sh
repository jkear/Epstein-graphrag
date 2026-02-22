#!/bin/bash
# Worker entrypoint for Vast.ai / Runpod deployment
# Starts Ollama and runs OCR pipeline

set -e

echo "=============================================="
echo "Epstein GraphRAG OCR Worker"
echo "=============================================="

# Start Ollama in background
echo "Starting Ollama..."
ollama serve &
OLLAMA_PID=$!
sleep 5

# Verify model is available
echo "Checking minicpm-v model..."
if ! ollama list | grep -q "minicpm-v"; then
    echo "Pulling minicpm-v:8b model..."
    ollama pull minicpm-v:8b
fi

# Check for manifest
MANIFEST="${MANIFEST_PATH:-/data/manifest.json}"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest not found at $MANIFEST"
    echo "Mount your data volume with: -v /path/to/data:/data"
    echo "Or set MANIFEST_PATH environment variable"
    exit 1
fi

# Count documents
DOC_COUNT=$(python3 -c "import json; print(len(json.load(open('$MANIFEST')).get('documents', [])))")
echo "Found $DOC_COUNT documents to process"

# Get worker count (default 3 for RTX 3060, adjust based on VRAM)
NUM_WORKERS="${NUM_WORKERS:-3}"
echo "Running with $NUM_WORKERS parallel workers"

# Run OCR
echo "Starting OCR pipeline..."
egr ocr \
    --manifest "$MANIFEST" \
    --ocr-provider ollama \
    --num-workers "$NUM_WORKERS"

echo "=============================================="
echo "OCR complete!"
echo "Results in /data/processed/"
echo "=============================================="

# Keep container alive for result collection (optional)
if [ "${KEEP_ALIVE:-false}" = "true" ]; then
    echo "Container staying alive for result collection..."
    tail -f /dev/null
fi
