---
name: epstein-graphrag-pipeline
description: >-
  Deep knowledge of the Epstein Evidence GraphRAG pipeline: 7-stage document
  processing system (classify → OCR → extract → validate → ingest → embed →
  query) using Neo4j, Ollama, vLLM, LM Studio, DeepSeek, and Gemini. Covers
  all CLI commands, AI models, file paths, config, error handling, and
  resume-safe patterns.
user-invocable: true
---

# Epstein GraphRAG Pipeline

Expert knowledge of the Epstein Evidence GraphRAG system — a 7-stage forensic document processing pipeline that turns 850K+ PDFs into a queryable Neo4j knowledge graph.

## Pipeline Stages

```
PDF Documents
      │
      ▼
Stage 1: classify    → manifest.json          (LLaVA 1.6 34B via Ollama)
      │
      ▼
Stage 2: ocr         → data/processed/*.json  (minicpm-v:8b / Qwen3-VL)
      │
      ▼
Stage 3: extract     → data/extracted/*.json  (multi-provider fallback)
      │
      ▼
Stage 4: validate    → data/schema_alerts.jsonl
      │
      ▼
Stage 5: ingest      → Neo4j graph            (MERGE-based, idempotent)
      │
      ▼
Stage 6: embed       → Neo4j vector indexes   (nomic-embed-text, 768d)
      │
      ▼
Stage 7: query       (planned — hybrid vector + graph search)
```

## Key Commands

```bash
# Classify PDFs (extracts embedded text automatically via pdfplumber)
egr classify /path/to/pdfs -w 12

# OCR scanned documents
egr ocr --ocr-provider vllm --vllm-url http://localhost:8000/v1 -w 5
egr ocr --ocr-provider lmstudio --num-workers 4
egr ocr --ocr-provider ollama

# Extract entities (multi-provider fallback)
egr extract --num-workers 5

# Neo4j schema setup
egr schema --create --verify

# Ingest entities into graph
egr ingest

# Generate embeddings
egr embed              # all node types
egr embed -l Person    # single type
```

## Stage 1 — Classification Agent

**File:** `src/epstein_graphrag/classify/classifier.py`
**Model:** LLaVA 1.6 34B (via Ollama)
**Output types:** `photograph` | `text_document` | `mixed`

89% of PDFs have embedded text — pdfplumber extracts it at classification time, skipping expensive VLM OCR entirely.

## Stage 2 — OCR Agent

**File:** `src/epstein_graphrag/ocr/marker_pipeline_ollama.py`

| Provider | Model | Concurrency | Notes |
|----------|-------|-------------|-------|
| vLLM | Qwen/Qwen2-VL-7B-Instruct | Parallel (`-w N`) | Fastest; requires NVIDIA GPU 12GB+ |
| vLLM-MLX | mlx-community/Qwen2-VL-7B-Instruct-4bit | Parallel | Apple Silicon |
| LM Studio | qwen/qwen3-vl-8b | Parallel (`--num-workers N`) | Recommended for Mac batch work |
| Ollama | minicpm-v:8b | Sequential only | Simplest setup |

**Forensic prompts** (`src/epstein_graphrag/ocr/forensic_prompts.py`):
- `FORENSIC_OCR_PROMPT` — general documents
- `REDACTION_AWARE_OCR_PROMPT` — documents with redactions
- `FORENSIC_PHOTOGRAPH_PROMPT` — scene/photo analysis
- `IDENTITY_DOCUMENT_OCR_PROMPT` — IDs, passports
- `LEGAL_DOCUMENT_OCR_PROMPT` — court filings, subpoenas

**Sub-components:**

| Component | File | Purpose |
|-----------|------|---------|
| Duplicate Detector | `ocr/duplicate_detector.py` | File/text/perceptual hash fingerprinting |
| Redaction Merger | `ocr/redaction_merger.py` | Fills redacted text from duplicate copies |
| Quality Check | `ocr/quality_check.py` | Detects repetition, hallucination, gibberish |
| Forensic Marker Processor | `ocr/forensic_marker_processor.py` | Extends Marker's LLM processor |

**Output JSON:**
```json
{
  "doc_id": "EFTA00002001",
  "text": "extracted text content...",
  "page_count": 3,
  "confidence": 0.95,
  "processing_engine": "lmstudio-vision"
}
```

## Stage 3 — Entity Extraction Agent

**File:** `src/epstein_graphrag/extract/multi_provider_extractor.py`

**Provider fallback order:**

| Priority | Provider | Model | Cost | Requires |
|----------|----------|-------|------|----------|
| 1 | Ollama (local) | minicpm-v:8b | Free | Ollama running |
| 2 | DeepSeek API | deepseek-chat | Per-token | `DEEPSEEK_API_KEY` |
| 3 | Gemini API | gemini-2.5-flash | Free tier | `GEMINI_API_KEY` |
| 4 | MLX (local) | Qwen3-VL | Free | Apple Silicon + MLX |

**13 extracted entity categories:**

| # | Category | Key Fields |
|---|----------|-----------|
| 1 | People | name, aliases, role (perpetrator/victim/witness/associate/legal/law_enforcement) |
| 2 | Locations | name, type (residence/island/airport/office/hotel/yacht/property) |
| 3 | Organizations | name, type (company/foundation/school/government/legal/financial) |
| 4 | Events | description, type (flight/meeting/transaction/assault/arrest/testimony) |
| 5 | Allegations | description, severity (critical/severe/moderate/minor), status |
| 6 | Associations | person pairs, nature, context |
| 7 | Identity Documents | passport, ID, birth certificate |
| 8 | Communications | emails, phone calls, letters |
| 9 | Legal Documents | subpoenas, depositions, warrants |
| 10 | Transactions | financial transactions |
| 11 | Physical Evidence | documents, devices, recordings |
| 12 | Redacted Entities | tracked redactions for potential de-redaction |
| 13 | Objects of Interest | photo analysis objects (legacy) |

## Stage 4 — Schema Validator

**File:** `src/epstein_graphrag/extract/schema_validator.py`

Validates entity types, allowed enum values (roles, severity, status, location types, org types), and required fields. Pipe-separated enum values are handled by taking the first value. Alerts logged to `data/schema_alerts.jsonl`.

## Stage 5 — Graph Ingestion Agent

**File:** `src/epstein_graphrag/graph/ingest.py` — `GraphIngestor`

- **Always MERGE, never CREATE** — idempotent by design
- Alias resolution via `AliasResolver` (`data/alias_table.json`)
- Hallucination filtering: rejects prompt echoes, empty names, doc IDs used as names
- Pipe-separated enum cleanup (takes first value)

**18 Node Types:** Person, Organization, Location, Property, BankAccount, Transaction, Vessel, Aircraft, PhysicalItem, DigitalEvidence, Event, PropertyDevelopment, Document, DocumentChunk, VisualEvidence, Allegation, LegalProceeding, Contact

**Relationship Types:** `MENTIONED_IN`, `PARTICIPATED_IN`, `OCCURED_AT`, `ALLEGED_IN`, `VICTIM_OF`, `ASSOCIATED_WITH`, `DOCUMENTED_IN`, `OWNS`, `HAS_CONTACT`, `ACCESSED_VIA`, `TRAVELED_VIA`

## Stage 6 — Embedding Agent

**File:** `src/epstein_graphrag/embeddings/embed.py`
**Model:** nomic-embed-text (via Ollama) — 768 dimensions

| Node Type | Embedding Property | Text Source |
|-----------|--------------------|-------------|
| Event | description_embedding | description |
| Allegation | embedding | description |
| Person | description_embedding | name + description |
| Location | description_embedding | name + description |

Resume-safe: only processes nodes where embedding IS NULL.

## Neo4j Schema

**File:** `src/epstein_graphrag/graph/schema.py`

18 NODE KEY constraints (existence + uniqueness).

**Vector indexes (768d):** document_chunk_embeddings, visual_evidence_embeddings, event_embeddings, allegation_embeddings, person_description_embeddings, location_description_embeddings

**Fulltext indexes:** person_fulltext (name, description), document_fulltext (doc_id, text_content), allegation_fulltext (description), event_fulltext (description)

## Configuration

**File:** `src/epstein_graphrag/config.py`

| Setting | Default | Env Override |
|---------|---------|-------------|
| Neo4j URI | `bolt://localhost:7687` | `NEO4J_URI` |
| Neo4j User | `neo4j` | `NEO4J_USER` |
| Neo4j Password | `password` | `NEO4J_PASSWORD` |
| Gemini API Key | (none) | `GEMINI_API_KEY` |
| DeepSeek API Key | (none) | `DEEPSEEK_API_KEY` |
| Gemini Model | `gemini-2.5-flash` | — |
| LM Studio URL | `http://localhost:1234/v1` | `LMSTUDIO_BASE_URL` |
| LM Studio Model | `qwen/qwen3-vl-8b` | `LMSTUDIO_MODEL` |
| Embedding Model | `nomic-embed-text` | — |
| Embedding Dimensions | 768 | — |
| Batch Size | 100 | — |
| OCR Confidence Threshold | 0.7 | — |

## File Locations

| Data Type | Path | Format |
|-----------|------|--------|
| Manifest | `data/manifest.json` | JSON (not in git — 215MB) |
| OCR Results | `data/processed/*.json` | JSON per document |
| Extractions | `data/extracted/*.json` | JSON per document |
| Aliases | `data/alias_table.json` | JSON |
| Schema Alerts | `data/schema_alerts.jsonl` | JSONL |
| OCR Quality Report | `data/ocr_quality_report.json` | JSON |

## Error Handling & Resume Patterns

All stages are **resume-safe** — rerunning a command skips already-completed work:

1. **OCR**: checks if output file exists in `data/processed/`
2. **Extraction**: checks `data/extracted/` for existing results
3. **Ingestion**: MERGE operations are idempotent
4. **Embeddings**: only processes nodes with NULL embeddings

Failed operations write `.error.json` files:
```json
{
  "doc_id": "EFTA00002012",
  "error": "Error message",
  "timestamp": "2026-02-11T12:00:00Z"
}
```

Check progress:
```bash
find data/processed -name "*.json" ! -name "*.error.json" | wc -l
find data/processed -name "*.error.json" | wc -l
```

## Hardware Notes

- **Apple Silicon (MPS)**: surya-ocr 0.17.x has bfloat16 bugs — pipeline forces `TORCH_DEVICE=cpu` automatically. Use LM Studio (MLX backend) for OCR.
- **Dual-machine workflow**: Run vLLM on NVIDIA PC (`--host 0.0.0.0`), point Mac at it via `--vllm-url http://PC_IP:8000/v1`.
- **vLLM** is 5–10× faster than Ollama for batch OCR.
