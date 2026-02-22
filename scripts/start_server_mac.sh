#!/bin/bash
# ============================================================
# M4 MacBook Pro Setup - vLLM-MLX or LM Studio Server
# ============================================================
#
# Run this on your M4 MacBook Pro with 48GB RAM
# Options:
#   1. vLLM-MLX (recommended - fastest)
#   2. LM Studio (GUI, already installed)
#
# ============================================================

set -e

echo "=============================================="
echo "  Epstein GraphRAG - M4 Mac Server"
echo "=============================================="

# Check if running on Mac
if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: This script is for macOS"
    exit 1
fi

# Check Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "ERROR: This script requires Apple Silicon (M1/M2/M3/M4)"
    exit 1
fi

echo "System: $(system_profiler SPHardwareDataType | grep "Chip" | awk -F: '{print $2}')"
echo "Memory: $(system_profiler SPHardwareDataType | grep "Memory" | awk -F: '{print $2}')"
echo ""

# Check which backend to use
BACKEND="${BACKEND:-auto}"

if [ "$BACKEND" = "auto" ]; then
    if python3 -c "import vllm_mlx" 2>/dev/null; then
        BACKEND="vllm-mlx"
    elif command -v lms &> /dev/null; then
        BACKEND="lmstudio"
    else
        BACKEND="vllm-mlx"
    fi
fi

echo "Using backend: $BACKEND"
echo ""

case "$BACKEND" in
    "vllm-mlx")
        # Install vllm-mlx if needed
        if ! python3 -c "import vllm_mlx" 2>/dev/null; then
            echo "Installing vllm-mlx..."
            pip3 install vllm-mlx
        fi
        
        MODEL="${MODEL:-mlx-community/Qwen2-VL-7B-Instruct-4bit}"
        PORT="${PORT:-8001}"
        
        echo "Starting vLLM-MLX server..."
        echo "  Model: $MODEL"
        echo "  Port: $PORT"
        echo ""
        echo "Server will be available at: http://localhost:$PORT/v1"
        echo ""
        
        # vllm-mlx serve command
        python3 -m vllm_mlx.server \
            --model "$MODEL" \
            --host 0.0.0.0 \
            --port "$PORT"
        ;;
        
    "lmstudio")
        echo "Using LM Studio backend"
        echo ""
        echo "Make sure LM Studio is running with:"
        echo "  1. A Qwen2-VL or similar vision model loaded"
        echo "  2. Local server enabled on port 1234"
        echo ""
        echo "Then use this URL in the coordinator:"
        echo "  http://localhost:1234/v1"
        echo ""
        echo "Opening LM Studio..."
        open -a "LM Studio" 2>/dev/null || echo "LM Studio not found. Install from https://lmstudio.ai"
        ;;
        
    *)
        echo "Unknown backend: $BACKEND"
        echo "Options: vllm-mlx, lmstudio"
        exit 1
        ;;
esac
