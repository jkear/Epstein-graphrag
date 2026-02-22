#!/bin/bash
# ============================================================
# Vast.ai Deployment Script for Epstein GraphRAG
# ============================================================
# 
# COST BREAKDOWN (as of Feb 2026):
#   - RTX 3060 (12GB): $0.05/hr on Vast.ai spot
#   - T4 (16GB): $0.02/hr on Vast.ai spot
#   - RTX 2080 Ti (11GB): $0.06/hr
#
# For 2.7M files at ~30 sec/page:
#   - 22,500 GPU-hours total
#   - 20 workers = $450-1,125 total over ~47 days
#   - 50 workers = $450-1,125 total over ~19 days
#
# YOUR MODEL: 7GB model, 13GB max VRAM = RTX 3060 is PERFECT
# ============================================================

set -e

# Configuration - EDIT THESE
DOCKER_IMAGE="your-dockerhub-username/epstein-graphrag:latest"
MANIFEST_CHUNK="manifest_chunk_0.json"  # Which chunk this worker processes
WORKER_ID="${1:-0}"  # Pass worker ID as argument

# Vast.ai template settings (copy these to Vast.ai UI)
cat << 'VAST_TEMPLATE'
============================================================
VAST.AI INSTANCE TEMPLATE - Copy to Vast.ai UI
============================================================

Image: your-dockerhub-username/epstein-graphrag:latest

Docker Options:
  -v /workspace/data:/data
  -e NEO4J_URI=bolt://YOUR_NEO4J_IP:7687
  -e NEO4J_USER=neo4j
  -e NEO4J_PASSWORD=YOUR_PASSWORD

On-start Script:
  # Download your manifest chunk
  curl -o /data/manifest.json YOUR_MANIFEST_URL
  
  # Start Ollama in background (if using local model)
  ollama serve &
  sleep 5
  ollama pull minicpm-v:8b
  
  # Run OCR
  egr ocr --manifest /data/manifest.json --num-workers 3

Disk Space: 50GB minimum (for model + processed files)
GPU RAM: 12GB+ (RTX 3060, T4, RTX 2080 Ti)
============================================================
VAST_TEMPLATE

echo ""
echo "============================================================"
echo "STEP-BY-STEP INSTRUCTIONS FOR $0 BUDGET PROCESSING"
echo "============================================================"
echo ""
echo "1. SPLIT YOUR MANIFEST INTO CHUNKS:"
echo "   python deploy/split_manifest.py --chunks 20"
echo ""
echo "2. UPLOAD CHUNKS TO ACCESSIBLE LOCATION:"
echo "   - GitHub Gist (free, public)"
echo "   - S3/GCS bucket"
echo "   - Any HTTP server"
echo ""
echo "3. BUILD AND PUSH DOCKER IMAGE:"
echo "   docker build -t $DOCKER_IMAGE ."
echo "   docker push $DOCKER_IMAGE"
echo ""
echo "4. RENT GPUS ON VAST.AI:"
echo "   a. Go to https://vast.ai/console/create/"
echo "   b. Filter: GPU RAM >= 12GB, $/hr <= 0.10"
echo "   c. Sort by: Price (ascending)"
echo "   d. Select 20 instances (RTX 3060 @ \$0.05/hr = \$1/hr total)"
echo ""
echo "5. CONFIGURE EACH INSTANCE:"
echo "   - Use the template above"
echo "   - Change MANIFEST_URL to point to different chunks"
echo ""
echo "6. COLLECT RESULTS:"
echo "   - Each instance writes to /data/processed/"
echo "   - Download and merge when complete"
echo ""
echo "============================================================"
echo "ESTIMATED COSTS:"
echo "============================================================"
echo "  20x RTX 3060 @ \$0.05/hr = \$1.00/hr"
echo "  47 days runtime = 1,128 hours"
echo "  TOTAL: ~\$1,128"
echo ""
echo "  Want faster? 50 workers = 19 days, same \$1,128 total"
echo "============================================================"
