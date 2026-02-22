# CLAUDE.md - Project Context for AI Assistants

## Project Overview

**Epstein GraphRAG** - A forensic document processing pipeline that extracts structured knowledge from 850K+ PDF documents related to the Epstein case and builds a queryable Neo4j knowledge graph.

## Current Status (Feb 2026)

| Metric | Value |
|--------|-------|
| Total Documents | 849,503 |
| Total Pages | 1,506,429 |
| Already Processed | 749,569 (88%) |
| Remaining | ~100,000 |
| Text Extracted (no OCR needed) | 756,340 (89%) |
| Needs VLM OCR | 93,163 (11%) |

**Key Discovery**: 89% of PDFs have embedded text - pdfplumber extracts this during classification, skipping expensive VLM OCR.

## Architecture

```
PDF Files → classify → manifest.json
                ↓
         ┌─────┴─────┐
         │           │
    text_document   photograph/mixed
         │           │
    pdfplumber      VLM OCR (vLLM/LMStudio/Ollama)
         │           │
         └─────┬─────┘
               ↓
        data/processed/*.json
               ↓
           extract (entity extraction)
               ↓
        data/extracted/*.json
               ↓
           ingest (Neo4j)
               ↓
         Knowledge Graph
```

## Key Commands

```bash
# Classify PDFs (extracts text from digital PDFs automatically)
egr classify /path/to/pdfs -w 12

# Run OCR on remaining scanned documents
egr ocr --ocr-provider vllm --vllm-url http://localhost:8000/v1 -w 5

# Extract entities
egr extract --num-workers 3

# Setup Neo4j schema and ingest
egr schema --create
egr ingest

# Generate embeddings
egr embed --label all
```

## OCR Providers (ordered by speed)

1. **vLLM** (fastest, 5-10x faster than Ollama)
   - Model: `Qwen/Qwen2-VL-7B-Instruct`
   - Requires: NVIDIA GPU 12GB+ VRAM
   - Start: `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-VL-7B-Instruct --port 8000`

2. **vLLM-MLX** (for Apple Silicon)
   - Model: `mlx-community/Qwen2-VL-7B-Instruct-4bit`
   - Start: `python -m vllm_mlx.server --model mlx-community/Qwen2-VL-7B-Instruct-4bit --port 8001`

3. **LM Studio** (GUI, good for Mac)
   - Model: `Qwen3-VL-8B-MLX`
   - Default URL: `http://localhost:1234/v1`

4. **Ollama** (simplest, but sequential)
   - Model: `minicpm-v:8b`
   - Note: Does not handle concurrency well

## Hardware Setup

Owner has:
- **RTX 3080 Ti PC** - For vLLM inference server
- **M4 MacBook Pro 48GB** - Main development machine, runs operations

**Dual-machine workflow**:
```bash
# On PC: Start vLLM server (expose to network)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --host 0.0.0.0 --port 8000

# On Mac: Run OCR pointing to PC
egr ocr --ocr-provider vllm --vllm-url http://PC_IP:8000/v1 --manifest data/manifest_rtx.json -w 5
```

## File Structure

```
epstein-graphrag/
├── data/
│   ├── manifest.json           # Full classification (849K docs) - NOT in git (215MB)
│   ├── manifest_rtx.json       # Split for RTX 3080 Ti (60K docs)
│   ├── manifest_mac.json       # Split for M4 Mac (40K docs)
│   ├── processed/              # OCR output (750K+ files)
│   └── extracted/              # Entity extraction output
├── src/epstein_graphrag/
│   ├── cli.py                  # Main CLI (egr command)
│   ├── classify/classifier.py  # Parallel PDF classification
│   ├── ocr/
│   │   ├── vllm_ocr.py         # vLLM backend (fastest)
│   │   ├── lmstudio_ocr.py     # LM Studio backend
│   │   ├── ollama_ocr.py       # Ollama backend
│   │   ├── marker_pipeline_ollama.py  # Main OCR orchestration
│   │   └── forensic_prompts.py # Forensic-aware prompts
│   ├── extract/                # Entity extraction
│   └── graph/                  # Neo4j operations
├── scripts/
│   ├── start_server_rtx.sh     # Start vLLM on NVIDIA
│   ├── start_server_mac.sh     # Start vLLM-MLX on Mac
│   └── multi_machine_ocr.py    # Coordinate multiple machines
├── deploy/
│   ├── split_manifest.py       # Split manifest for distributed work
│   └── vast-ai-template.sh     # Cloud GPU deployment
├── Dockerfile.vllm             # vLLM container (fastest)
├── Dockerfile.worker           # Ollama container
└── docs/
    └── DISTRIBUTED_DEPLOYMENT.md
```

## Data Locations

PDFs are stored at: `~/locDocuments/files/`
```
DataSet 1:     3,158 PDFs (1.2GB)   ✓ Processed
DataSet 2:       574 PDFs (633MB)
DataSet 3:        67 PDFs (600MB)
DataSet 4:       152 PDFs (359MB)
DataSet 5:       120 PDFs (62MB)
DataSet 6:        13 PDFs (53MB)
DataSet 7:        17 PDFs (98MB)
DataSet 8:    10,594 PDFs (11GB)
VOL00010:   503,153 PDFs (76GB)    ← Largest
VOL00011:   331,655 PDFs (27GB)
VOL00012:       152 PDFs (120MB)
```

## Common Tasks

### Resume OCR after interruption
OCR is resume-safe - just run the same command again. It skips files already in `data/processed/`.

### Check progress
```bash
# Count processed files
find data/processed -name "*.json" ! -name "*.error.json" | wc -l

# Count errors
find data/processed -name "*.error.json" | wc -l
```

### Regenerate manifest for unprocessed files
```python
# See scripts in deploy/split_manifest.py for reference
```

## Key Design Decisions

1. **pdfplumber first** - 89% of files have embedded text, no OCR needed
2. **vLLM for speed** - Continuous batching = 5-10x faster than Ollama
3. **Forensic prompts** - OCR prompts tell model it's reading legal evidence
4. **Resume-safe** - All operations skip completed work
5. **MERGE not CREATE** - Neo4j ingestion is idempotent

## Environment Variables

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=change_me
GEMINI_API_KEY=...      # Optional, for extraction fallback
DEEPSEEK_API_KEY=...    # Optional, for extraction fallback
```

## Git Notes

- `data/manifest.json` is in `.gitignore` (215MB)
- `data/processed/*.json` is in `.gitignore`
- Split manifests (`manifest_rtx.json`, `manifest_mac.json`) ARE tracked
