# AI Agents and Pipeline Components

This document describes the AI agents and processing components used in the Epstein Evidence GraphRAG system.

## Overview

The system uses a multi-stage pipeline where each stage employs different AI models and strategies optimized for specific tasks. OCR supports two providers: Ollama (sequential) and LM Studio (concurrent). Entity extraction uses a multi-provider fallback strategy across local and cloud models.

## Pipeline Architecture

```
PDF Documents
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Classification                                     │
│ Model: LLaVA 1.6 34B (via Ollama)                          │
│ Output: manifest.json (document types)                      │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: OCR (Text Extraction)                              │
│                                                             │
│ Provider A: Ollama (sequential)                             │
│   Model: minicpm-v:8b                                      │
│                                                             │
│ Provider B: LM Studio (concurrent, recommended)             │
│   Model: qwen/qwen3-vl-8b                                  │
│   Supports --num-workers for parallel processing            │
│                                                             │
│ Prompts: FORENSIC_OCR_PROMPT, REDACTION_AWARE_OCR_PROMPT   │
│ Output: data/processed/*.json (extracted text)              │
│                                                             │
│ Sub-components:                                             │
│   - Duplicate Detection (file/text/perceptual hashing)     │
│   - Redaction Merger (fills gaps across duplicate copies)   │
│   - Quality Validation (repetition, hallucination checks)  │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Entity Extraction (Multi-Provider Strategy)        │
│                                                             │
│ Provider 1: MiniCPM-V (local, via Ollama)                  │
│   - First attempt, no cost                                  │
│   - Model: minicpm-v:8b                                    │
│                                                             │
│ Provider 2: DeepSeek API (fallback)                        │
│   - Model: deepseek-chat                                   │
│   - Requires: DEEPSEEK_API_KEY                             │
│                                                             │
│ Provider 3: Gemini 2.5 Flash (final fallback)              │
│   - Model: gemini-2.5-flash                                │
│   - Requires: GEMINI_API_KEY                               │
│                                                             │
│ Provider 4: MLX Qwen3-VL (optional, Apple Silicon)         │
│   - Local inference via MLX                                 │
│                                                             │
│ Output: data/extracted/*.json (structured entities)         │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Schema Validation                                  │
│ Component: SchemaValidator                                  │
│ Validates entity types, allowed values, required fields     │
│ Logs violations to data/schema_alerts.jsonl                 │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 5: Graph Ingestion                                    │
│ Database: Neo4j                                             │
│ Features:                                                   │
│   - MERGE-based idempotent entity creation                 │
│   - Alias resolution (AliasResolver)                       │
│   - Hallucination filtering (prompt echoes, doc IDs)       │
│   - Pipe-separated enum cleanup                            │
│   - 18 node types, 12+ relationship types                  │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 6: Embeddings                                         │
│ Model: nomic-embed-text (via Ollama)                       │
│ Dimensions: 768                                             │
│ Target Nodes: Person, Location, Event, Allegation          │
│ Resume-capable: only processes nodes with NULL embeddings  │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 7: Query Engine (Not Yet Implemented)                 │
│ Planned: Vector + Graph hybrid search                      │
│ Planned: Natural language to Cypher generation             │
│ Planned: Pattern detection / anomaly detection             │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### Classification Agent

**File:** `src/epstein_graphrag/classify/classifier.py`
**Model:** LLaVA 1.6 34B (via Ollama)
**Task:** Determine document type from first page image

**Output Types:**
- `photograph` - Interior/exterior shots
- `text_document` - Scanned text documents
- `mixed` - Combination of text and images

**CLI:** `egr classify <data_dir>`

---

### OCR Agent

**File:** `src/epstein_graphrag/ocr/marker_pipeline_ollama.py`

**Providers:**

| Provider | Model | Concurrency | Use Case |
|----------|-------|-------------|----------|
| Ollama | minicpm-v:8b | Sequential only | Default, simple setup |
| LM Studio | qwen/qwen3-vl-8b | Parallel (`--num-workers`) | Recommended for batch processing |

**Prompts** (defined in `src/epstein_graphrag/ocr/forensic_prompts.py`):
- `FORENSIC_OCR_PROMPT` - General documents with forensic context
- `FORENSIC_PHOTOGRAPH_PROMPT` - Photograph scene analysis
- `REDACTION_AWARE_OCR_PROMPT` - Documents with redactions
- `IDENTITY_DOCUMENT_OCR_PROMPT` - IDs, passports, licenses
- `LEGAL_DOCUMENT_OCR_PROMPT` - Court filings, subpoenas

**Sub-components:**

| Component | File | Purpose |
|-----------|------|---------|
| Duplicate Detector | `ocr/duplicate_detector.py` | File hash, text hash, perceptual hash fingerprinting |
| Redaction Merger | `ocr/redaction_merger.py` | Fills redacted text using duplicate copies |
| Quality Check | `ocr/quality_check.py` | Detects repetition, hallucination, gibberish |
| Forensic Marker Processor | `ocr/forensic_marker_processor.py` | Extends Marker's LLM processor with forensic context |

**CLI:**
```bash
# Ollama (sequential)
egr ocr --ocr-provider ollama

# LM Studio with Qwen3-VL (parallel, recommended)
egr ocr --ocr-provider lmstudio --num-workers 4
```

**Output Structure:**
```json
{
  "doc_id": "EFTA00002001",
  "text": "extracted text content...",
  "page_count": 3,
  "confidence": 0.95,
  "processing_engine": "lmstudio-vision"
}
```

---

### Entity Extraction Agent

**File:** `src/epstein_graphrag/extract/multi_provider_extractor.py`
**Strategy:** Multi-provider fallback with parallel workers

| Provider | Model | Cost | Speed | Requires |
|----------|-------|------|-------|----------|
| Ollama (local) | minicpm-v:8b | Free | Slowest | Ollama running |
| DeepSeek API | deepseek-chat | Per-token | Fast | `DEEPSEEK_API_KEY` |
| Gemini API | gemini-2.5-flash | Free tier | Fast | `GEMINI_API_KEY` |
| MLX (local) | Qwen3-VL | Free | Medium | Apple Silicon, MLX |

**Extracted Entity Types (13 categories):**

| # | Category | Description |
|---|----------|-------------|
| 1 | People | Names, aliases, roles (perpetrator, victim, witness, associate, legal, law_enforcement) |
| 2 | Locations | Residences, islands, airports, offices, schools, hotels, yachts, properties |
| 3 | Organizations | Companies, foundations, schools, government agencies, legal firms, financial institutions |
| 4 | Events | Flights, meetings, transactions, phone calls, visits, assaults, arrests, testimony, court hearings |
| 5 | Allegations | Criminal allegations with severity (critical/severe/moderate/minor) and status |
| 6 | Associations | Relationships between people with nature and context |
| 7 | Identity Documents | Passports, IDs, birth certificates |
| 8 | Communications | Emails, phone calls, letters |
| 9 | Legal Documents | Subpoenas, depositions, warrants |
| 10 | Transactions | Financial transactions |
| 11 | Physical Evidence | Documents as objects, devices, recordings |
| 12 | Redacted Entities | Tracked redactions for potential de-redaction |
| 13 | Objects of Interest | Photo analysis objects (legacy) |

**CLI:**
```bash
egr extract --num-workers 5
```

---

### Schema Validator

**File:** `src/epstein_graphrag/extract/schema_validator.py`

**Validates:**
- Entity type existence
- Allowed enum values (roles, severity, status, location types, org types)
- Required fields
- Pipe-separated value handling (takes first value)

**Alerts logged to:** `data/schema_alerts.jsonl`

---

### Graph Ingestion Agent

**File:** `src/epstein_graphrag/graph/ingest.py`
**Component:** `GraphIngestor`

**Features:**
- MERGE-based idempotent entity creation (never CREATE)
- Alias resolution via `AliasResolver` (`data/alias_table.json`)
- Hallucination filtering: rejects prompt template fragments, empty names, doc IDs used as names
- Pipe-separated enum cleanup (takes first value)
- Document-to-entity linking via MENTIONED_IN relationships

**Node Types (18):**
Person, Organization, Location, Property, BankAccount, Transaction, Vessel, Aircraft, PhysicalItem, DigitalEvidence, Event, PropertyDevelopment, Document, DocumentChunk, VisualEvidence, Allegation, LegalProceeding, Contact

**Relationship Types:**
- `MENTIONED_IN` - Entity mentioned in document
- `PARTICIPATED_IN` - Person participated in event
- `OCCURED_AT` - Event occurred at location
- `ALLEGED_IN` - Person alleged in allegation
- `VICTIM_OF` - Person victim of allegation
- `ASSOCIATED_WITH` - Person connected to person
- `DOCUMENTED_IN` - Event/allegation documented in source
- `OWNS` - Person owns property
- `HAS_CONTACT` - Person has contact info
- `ACCESSED_VIA` - Person accessed account
- `TRAVELED_VIA` - Person traveled via vessel/aircraft

**CLI:** `egr ingest`

---

### Neo4j Schema

**File:** `src/epstein_graphrag/graph/schema.py`

**Constraints:** 18 NODE KEY constraints (existence + uniqueness)

**Vector Indexes (6):**
| Index | Label | Property | Dimensions |
|-------|-------|----------|------------|
| document_chunk_embeddings | DocumentChunk | embedding | 768 |
| visual_evidence_embeddings | VisualEvidence | embedding | 768 |
| event_embeddings | Event | description_embedding | 768 |
| allegation_embeddings | Allegation | embedding | 768 |
| person_description_embeddings | Person | description_embedding | 768 |
| location_description_embeddings | Location | description_embedding | 768 |

**Fulltext Indexes (4):**
| Index | Labels | Properties |
|-------|--------|------------|
| person_fulltext | Person | name, description |
| document_fulltext | Document | doc_id, text_content |
| allegation_fulltext | Allegation | description |
| event_fulltext | Event | description |

**CLI:** `egr schema --create --verify`

---

### Embedding Agent

**File:** `src/epstein_graphrag/embeddings/embed.py`
**Model:** nomic-embed-text (via Ollama)
**Dimensions:** 768

**Embedding Targets:**
| Node Type | Embedding Property | Text Source | Key Property |
|-----------|--------------------|-------------|--------------|
| Event | description_embedding | description | event_id |
| Allegation | embedding | description | allegation_id |
| Person | description_embedding | name + description | name |
| Location | description_embedding | name + description | name |

**Process:**
1. Query nodes where embedding IS NULL and text IS NOT NULL
2. Build text from node properties (single field or name + description)
3. Generate 768d vector via Ollama nomic-embed-text
4. Write vector back to Neo4j node
5. Resume-safe: skips already-embedded nodes

**CLI:**
```bash
egr embed              # all node types
egr embed -l Person    # single type
```

---

## Parallel Processing

### OCR Pipeline

| Provider | Concurrency | Notes |
|----------|-------------|-------|
| Ollama | Sequential (1 worker) | Ollama serves one request at a time |
| LM Studio | Configurable (`--num-workers N`) | Supports concurrent requests |

### Entity Extraction

Configurable parallel workers (default: 3):
```bash
egr extract --num-workers 5
```

Each worker reads from `data/processed/`, skips completed files in `data/extracted/`, runs extraction with provider fallback, and writes results.

---

## Error Handling

### Resume Capability

All stages support resuming:
1. **OCR**: Checks if output file exists before processing
2. **Extraction**: Checks `data/extracted/` for existing results
3. **Ingestion**: MERGE operations are idempotent
4. **Embeddings**: Only processes nodes with NULL embeddings

### Error Files

Failed operations create `.error.json` files:
```json
{
  "doc_id": "EFTA00002012",
  "error": "Error message",
  "timestamp": "2026-02-11T12:00:00Z"
}
```

### Provider Fallback (Extraction)

Attempts providers in order:
1. Local Ollama (no cost) - fails on quality/validation
2. DeepSeek API - fails on API errors
3. Gemini API - final attempt

---

## File Locations

| Data Type | Location | Format |
|-----------|----------|--------|
| Manifest | `data/manifest.json` | JSON |
| OCR Results | `data/processed/*.json` | JSON per document |
| Extractions | `data/extracted/*.json` | JSON per document |
| Aliases | `data/alias_table.json` | JSON |
| Schema Alerts | `data/schema_alerts.jsonl` | JSONL |
| OCR Quality Report | `data/ocr_quality_report.json` | JSON |

---

## Configuration

**File:** `src/epstein_graphrag/config.py`

| Setting | Default | Env Override |
|---------|---------|-------------|
| Neo4j URI | `bolt://localhost:7687` | `NEO4J_URI` |
| Neo4j User | `neo4j` | `NEO4J_USER` |
| Neo4j Password | `password` | `NEO4J_PASSWORD` |
| Gemini API Key | (none) | `GEMINI_API_KEY` |
| DeepSeek API Key | (none) | `DEEPSEEK_API_KEY` |
| Gemini Model | `gemini-2.5-flash` | - |
| LM Studio URL | `http://localhost:1234/v1` | `LMSTUDIO_BASE_URL` |
| LM Studio Model | `qwen/qwen3-vl-8b` | `LMSTUDIO_MODEL` |
| Embedding Model | `nomic-embed-text` | - |
| Embedding Dimensions | 768 | - |
| Batch Size | 100 | - |
| OCR Confidence Threshold | 0.7 | - |

---

## Apple Silicon Notes

- **Surya/Marker OCR**: Must use CPU mode on MPS due to bfloat16 bugs in surya-ocr 0.17.x. The pipeline sets `TORCH_DEVICE=cpu` automatically.
- **LM Studio**: Runs MLX backend natively on Apple Silicon. Recommended provider for OCR.
- **MLX Extraction**: Optional Qwen3-VL provider for entity extraction runs natively via MLX.
