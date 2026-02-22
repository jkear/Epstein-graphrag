# Epstein GraphRAG - Cheap GPU Cloud Deployment
# Optimized for Vast.ai / Runpod with 12-16GB VRAM GPUs (RTX 3060, T4, etc.)
# Model size: ~7GB, Max VRAM usage: ~13GB

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create app directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install dependencies
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv sync --frozen

# Set venv as default
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Create data directories
RUN mkdir -p /data/processed /data/extracted /data/manifest

# Default environment variables (override at runtime)
ENV NEO4J_URI="bolt://host.docker.internal:7687"
ENV NEO4J_USER="neo4j"
ENV NEO4J_PASSWORD="password"
ENV LMSTUDIO_BASE_URL="http://host.docker.internal:1234/v1"

# For Ollama running on host
ENV OLLAMA_HOST="http://host.docker.internal:11434"

# Expose nothing - this is a worker container
EXPOSE 0

# Default command - show help
ENTRYPOINT ["egr"]
CMD ["--help"]

# ============================================================
# USAGE EXAMPLES:
# ============================================================
#
# BUILD:
#   docker build -t epstein-graphrag .
#
# RUN OCR (with Ollama on host):
#   docker run --gpus all -v /path/to/data:/data \
#     -e OLLAMA_HOST=http://host.docker.internal:11434 \
#     epstein-graphrag ocr --manifest /data/manifest.json --num-workers 3
#
# RUN OCR (with LM Studio on host):
#   docker run --gpus all -v /path/to/data:/data \
#     -e LMSTUDIO_BASE_URL=http://host.docker.internal:1234/v1 \
#     epstein-graphrag ocr --ocr-provider lmstudio --manifest /data/manifest.json -w 3
#
# VAST.AI / RUNPOD:
#   See deploy/vast-ai-template.sh for cloud deployment
# ============================================================
