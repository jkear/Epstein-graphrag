# Epstein Evidence GraphRAG

A system for extracting structured knowledge from the Epstein files using GraphRAG.

## Current Status

| Metric | Value |
|--------|-------|
| **Total Documents** | 849,503 |
| **Total Pages** | 1,506,429 |
| **Text Extracted (no OCR needed)** | 756,340 (89%) |
| **Needs VLM OCR** | 93,163 (11%) |

**89% of documents have embedded text** - the classifier extracts this automatically. Only 11% need vision model OCR.

## Source Files

Download the Epstein evidence PDFs from: https://github.com/yung-megafone/Epstein-Files

## Overview

This system processes PDF documents through a multi-stage pipeline:

1. **Classification** - Categorize documents and **extract text from digital PDFs** (skips OCR for 89% of files!)
2. **OCR** - Extract text from scanned documents using vision models (vLLM, LM Studio, or Ollama)
3. **Entity Extraction** - Extract structured entities using multi-provider LLM strategy
4. **Graph Ingestion** - Load entities into Neo4j with relationship mapping
5. **Embeddings** - Generate vector embeddings for semantic search
6. **Query** - Natural language queries against the evidence graph

### Prerequisites

- Python 3.11+
- One of:
  - **vLLM** (fastest - 5-10x faster than Ollama) with Qwen2-VL-7B
  - LM Studio with Qwen3-VL-8B-MLX (for Mac)
  - Ollama with minicpm-v:8b
- Neo4j 5+ (for graph storage)
- Optional: Gemini API key, DeepSeek API key

### Hardware Requirements

| Hardware | OCR Speed | Notes |
|----------|-----------|-------|
| RTX 3060+ (12GB VRAM) | ~100-200 pages/min | vLLM recommended |
| M1/M2/M3/M4 Mac (16GB+) | ~50-100 pages/min | vLLM-MLX or LM Studio |
| Cloud GPU (T4/A100) | ~150-300 pages/min | Vast.ai ~$0.02-0.05/hr |

### Setup

```bash
cd epstein-graphrag
uv venv
source .venv/bin/activate
uv sync
```

### OCR Backend Setup

**Option 1: vLLM (Recommended - Fastest)**
```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

**Option 2: vLLM-MLX (Mac Apple Silicon)**
```bash
pip install vllm-mlx
vllm-mlx serve mlx-community/Qwen2-VL-7B-Instruct-4bit --port 8001
```

**Option 3: LM Studio (Mac GUI)**
```bash
lms pull lmstudio-community/Qwen3-VL-8B-MLX
lms server start lmstudio-community/Qwen3-VL-8B-MLX
```

**Option 4: Ollama (Simplest)**
```bash
ollama pull minicpm-v:8b
ollama pull nomic-embed-text
```

### Neo4j

```bash
docker run -d \
  --name neo4j-epstein \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/change_me \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Optional API keys for extraction providers
GEMINI_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here

# Neo4j connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=change_me
```

## Usage

The CLI entry point is `egr` (defined in `pyproject.toml`).

### 1. Classify Documents (Extracts text from 89% of files!)

```bash
egr classify /path/to/pdf/directory -w 12
```

This command:
- Classifies all PDFs by type (text_document, photograph, mixed, unknown)
- **Automatically extracts text** from digital PDFs using pdfplumber
- Saves extracted text to `data/processed/` (these files skip OCR!)
- Uses parallel workers for speed (`-w 12` = 12 workers)

Outputs: `data/manifest.json` + `data/processed/*.json` for text documents

### 2. Run OCR (Only for scanned documents - 11% of files)

```bash
# With vLLM (fastest)
egr ocr --ocr-provider vllm --num-workers 5

# With LM Studio
egr ocr --ocr-provider lmstudio --num-workers 3

# With Ollama
egr ocr --ocr-provider ollama --num-workers 1
```

OCR automatically **skips files already in `data/processed/`** (from classify step).

Outputs to `data/processed/`.

### 3. Extract Entities

```bash
egr extract --num-workers 3
```

Multi-provider strategy (fallback chain):
1. Local model (Ollama/LM Studio)
2. DeepSeek API
3. Gemini 2.5 Flash

Outputs to `data/extracted/`.

### 4. Setup Neo4j Schema

```bash
egr schema --create --verify
```

Creates constraints, vector indexes, and fulltext indexes.

### 5. Ingest into Graph

```bash
egr ingest
```

MERGEs all extracted entities into Neo4j with alias resolution.

### 6. Generate Embeddings

```bash
egr embed --label all
```

Generate 768d vectors using `nomic-embed-text` for semantic search.

## Distributed Processing

For processing at scale, see `docs/DISTRIBUTED_DEPLOYMENT.md`.

### Multi-Machine Setup

Run OCR across multiple machines:

```bash
# On RTX PC
./scripts/start_server_rtx.sh

# On Mac
./scripts/start_server_mac.sh

# Coordinate from any machine
python scripts/multi_machine_ocr.py \
    --servers http://pc-ip:8000/v1,http://mac-ip:8001/v1 \
    --manifest data/manifest.json
```

### Cloud Deployment (Vast.ai)

Process on cheap cloud GPUs (~$0.02-0.05/hr):

```bash
# Split manifest for distributed workers
python deploy/split_manifest.py --chunks 20

# Deploy workers on Vast.ai
# See deploy/vast-ai-template.sh for instructions
```

### Docker

```bash
# vLLM-based (fastest)
docker build -f Dockerfile.vllm -t epstein-graphrag:vllm .

# Ollama-based (simpler)
docker build -f Dockerfile.worker -t epstein-graphrag:worker .
```

## Project Structure

```
epstein-graphrag/
├── data/
│   ├── manifest.json           # Document classification manifest (849K docs)
│   ├── processed/              # Extracted text (756K from pdfplumber + OCR results)
│   ├── extracted/              # Entity extraction JSON files
│   ├── alias_table.json        # Name resolution mappings
│   └── schema_alerts.jsonl     # Schema validation alerts
├── src/epstein_graphrag/
│   ├── cli.py                  # Command-line interface (egr)
│   ├── config.py               # Configuration management
│   ├── classify/
│   │   └── classifier.py       # Parallel PDF classification + text extraction
│   ├── ocr/
│   │   ├── vllm_ocr.py         # vLLM backend (fastest)
│   │   ├── lmstudio_ocr.py     # LM Studio backend
│   │   ├── ollama_ocr.py       # Ollama backend
│   │   ├── forensic_prompts.py # Forensic-aware OCR prompts
│   │   ├── quality_check.py    # Hallucination detection
│   │   ├── duplicate_detector.py
│   │   └── redaction_merger.py # Merge redacted duplicates
│   ├── extract/
│   │   ├── multi_provider_extractor.py
│   │   ├── prompts.py          # Entity extraction prompts
│   │   └── schema_validator.py
│   ├── graph/
│   │   ├── schema.py           # Neo4j schema creation
│   │   ├── ingest.py           # MERGE-based ingestion
│   │   └── dedup.py            # Alias resolution
│   ├── embeddings/
│   │   └── embed.py            # Vector embedding generation
│   └── query/
│       ├── engine.py           # Query engine
│       ├── cypher_gen.py       # Cypher generation
│       └── hybrid_search.py    # Vector + graph search
├── scripts/
│   ├── start_server_rtx.sh     # Start vLLM on NVIDIA GPU
│   ├── start_server_mac.sh     # Start vLLM-MLX on Mac
│   ├── multi_machine_ocr.py    # Coordinate multiple machines
│   └── volunteer.sh            # Easy contributor script
├── deploy/
│   ├── split_manifest.py       # Split work for distributed processing
│   ├── vast-ai-template.sh     # Cloud deployment guide
│   ├── vllm-entrypoint.sh      # Docker entrypoint (vLLM)
│   └── worker-entrypoint.sh    # Docker entrypoint (Ollama)
├── docs/
│   └── DISTRIBUTED_DEPLOYMENT.md
├── Dockerfile                  # Basic container
├── Dockerfile.vllm             # vLLM-based (fastest)
├── Dockerfile.worker           # Ollama-based (simpler)
├── pyproject.toml
├── uv.lock
└── AGENTS.md
```

## Entity Schema

The graph supports 18 entity types:

| Entity | Description |
| ------ | ----------- |
| Person | Individuals (victims, associates, staff, witnesses) |
| Organization | Companies, foundations, government agencies |
| Location | Properties, addresses, cities, countries |
| Property | Real estate (islands, mansions, ranches) |
| BankAccount | Financial accounts |
| Transaction | Financial transactions |
| Vessel | Boats, yachts, watercraft |
| Aircraft | Planes, helicopters |
| PhysicalItem | Evidence items (hard drives, computers) |
| DigitalEvidence | Files, images, videos from devices |
| Event | Flights, meetings, allegations |
| PropertyDevelopment | Construction projects |
| Document | Source documents |
| DocumentChunk | Text chunks for vector search |
| VisualEvidence | Analyzed photograph content |
| Allegation | Claims, charges, testimony |
| LegalProceeding | Court cases, investigations |
| Contact | Phone numbers, email addresses |

See `docs/entity-schema.md` for complete schema details.

## Error Handling

All pipelines support resume capability:

- Failed OCR writes `.error.json` files
- Failed extraction writes `.error.json` files
- Re-running any command skips completed work

## Development

### Running Tests

```bash
uv run pytest tests/
```

### Linting

```bash
uv run ruff check src/
uv run ruff format src/
```

### Type Checking

```bash
npx pyright src/
```

## Configuration

Edit `src/epstein_graphrag/config.py` to modify:

- Default directories
- Model names
- API endpoints
- Neo4j connection settings

## Notes

## Mapping the Epstein Files

Thousands of pages of court filings, flight logs, deposition transcripts, and photographs from the Jeffrey Epstein case sit in public document repositories. Individually, each page tells a fragment of a story. A name here, a flight date there, a location mentioned in passing. The connections between them — the patterns that reveal who knew what, who went where, and who looked the other way — remain buried in scanned PDFs that no single person could read in a lifetime.

I built a pipeline to read them all, extract every name, date, location, and allegation, and load the results into a graph database where those connections surface on their own.

## Designed to Stay Hidden

The documents released, like other "publicly available" records, have data that are obscured to protect congress' wealthy benefactors and keep the hand of the oligarchy, which is strangling democracy, hidden. Many are scanned at poor resolution, contain redactions — over names and emails of perpatrators, dates, and descriptions. Others are photographs with no searchable text at all. Government agencies released these files in formats that make systematic review difficult: thousands of individual PDFs, no index, no descriptive naming, no metadata.

This is not accidental. When you scatter evidence across 12 separate document sets, redact key names, and release it all as images rather than text, you build a wall between the public and the truth. A determined reader might connect two documents. Connecting 850,000 requires a machine.

## Reading the Docs

The first assumption was that all 850,000 documents were scanned images requiring expensive vision-model OCR. **This turned out to be wrong.**

After classifying every PDF, we discovered:
- **89% (756,340 files) are digital PDFs** with embedded, extractable text
- **11% (93,163 files) are scanned images** requiring vision-model OCR

The classifier extracts text from digital PDFs automatically using pdfplumber. Only the 11% of scanned documents need the expensive VLM pipeline.

For those scanned documents, the pipeline uses forensic OCR prompts that tell the vision model what it is looking at. The model knows it is reading legal evidence. It knows to preserve every name exactly as written, to flag redacted regions, and to note text still visible beneath sloppy black-marker redactions. That last point matters: some redactions were done so poorly that the underlying text remains legible. The pipeline catches those.

Each document goes through quality validation after OCR. The system checks for hallucinated text (a common failure where AI models invent plausible-sounding content), repetitive output, and gibberish sequences. Bad OCR output gets flagged and reprocessed rather than silently polluting the dataset.

Photographs get separate treatment. A vision model describes the scene — the location type, the people visible, objects in frame — and that description enters the same extraction pipeline as text documents.

## Duplicate Detection and Redaction Recovery

The same document often appears multiple times across the eight dataset volumes. Sometimes these copies are identical. Sometimes they carry different redactions — a name blacked out in one copy but visible in another.

The pipeline fingerprints every document three ways: a file hash for exact matches, a text hash for content similarity, and a perceptual hash for visual layout. When it finds duplicates with different redactions, a merger combines them. Text visible in one copy fills gaps left by redactions in another. The result is a more complete version than any single release intended to provide.

This is where the incompetence begins to work against the interests of Patel's FBI and the DoJ. It seems like the work of redacting particular intel was divided up without the forsight to consider how it would come back together. Inconsistent redaction across duplicate releases is a gift with this sort of system.

## Extracting Structured Evidence

Raw text, however clean, cannot answer questions like "Who flew to Little St. James on the same dates as Victim X?" The pipeline needs structure.

An entity extraction layer reads each OCR output and produces structured JSON: people with names, aliases, and roles (perpetrator, victim, witness, associate, law enforcement); locations with types (residence, island, airport, yacht); events with dates and participant lists; allegations with severity ratings and testimony sources; and associations between individuals.

The extraction prompt instructs the model to preserve exact quotes from the source document alongside each extracted entity. Every claim in the graph traces back to specific language in a specific document.

The system tries multiple AI providers in sequence — a local model first, then cloud APIs as fallback — so that extraction continues even when one provider fails or rate-limits. A schema validator catches malformed output: enum values the model invented, pipe-separated fields where it couldn't choose one answer, hallucinated placeholder names like "Full Name as written" echoed back from the prompt template.

## Building the Graph

A Neo4j graphDB stores the extracted entities as nodes and their relationships as edges. The data model has sixteen node types (Person, Location, Organization, Event, Allegation, Document, Aircraft, Vessel, and more) and a dozen relationship types (MENTIONED_IN, PARTICIPATED_IN, ASSOCIATED_WITH, VICTIM_OF, TRAVELED_VIA, and others).

Before ingestion, an alias resolver collapses name variants. "J. Epstein," "Jeffrey Epstein," "Epstein, Jeffrey," and "Jeffrey E. Epstein" all merge into a single Person node. Without this step, the graph would fragment into disconnected clusters that share a subject but not a name.

Every write uses MERGE, not CREATE. The pipeline can run repeatedly on the same data and produce the same graph. This idempotency matters because the extraction models sometimes need rerunning with better prompts or updated models, the project can take on collaborators, and the graph must absorb corrections without duplicating nodes.

## Vector Embeddings for Semantic Search

After ingestion, the pipeline generates vector embeddings for key node types — events, allegations, people, and locations. These vectors enable semantic search: a query about "underage girls transported to private islands" finds relevant allegations even when the documents use different language.

This layer turns the graph from a lookup tool into a discovery tool. You can ask questions the documents never explicitly answer, and the system finds relevant evidence by meaning rather than keyword.

## What the Graph Will Reveal

The power of a knowledge graph is not in any single document. It will be in the intersections. I remember my brother hypothesizing in 2005 that the only chance at privacy in the future will be to have none-- to publish so much data about yourself that any one event is like finidng a needle in a needle stack. That's what the DoJ has done here. But, I believe the combination of knowledge graphs with Vision+Language AI models traversing them has ended that comfort. Now, every new datum is a thread that makes up a tapestry that patterns your deepest secrets and instead of 'overwhelming' this attempt to hide only increases the resolution.

When the same person appears in a flight log, a deposition transcript, and a victim's testimony, the graph draws those three documents together. When a location appears in financial records and in a witness statement describing abuse, the graph connects them. When an associate appears in one document and a government official appears in another, and both documents reference the same event, the graph shows that too.

These are connections that exist in the evidence already. They were always there. The graph does not fabricate them. It makes them visible.

The pattern that emerges — wealthy individuals connected to Epstein through flights, properties, financial transactions, and social events, alongside allegations of abuse at those same locations and times — is documented across thousands of pages. The graph collapses those thousands of pages into a queryable structure where a single Cypher query can surface what would take a researcher months of manual cross-referencing.

## Why This Matters

The Epstein case produced an unprecedented volume of evidence implicating powerful people in serious crimes. Much of that evidence was released in forms that discourage analysis. Thousands of scanned PDFs, inconsistent redactions, no cross-referencing, no index.

The justice system processed a fraction of this evidence. Epstein died in custody. Ghislaine Maxwell was convicted, but the network she and Epstein built — the associates, the enablers, the people who visited the island and looked the other way — remains largely unexamined by any court. We cannot move forward being seen as a legitimate country until we become legitimate through a willing and honest investigation into what was done, how the presence of wealth affects weak leadership and easily influenced minds.

A knowledge graph does not replace a courtroom. But it does something courts have failed to do: it holds the entire body of evidence in a single structure where patterns cannot hide. When "J. Doe" in one document is the same person as "John Doe" in another, and both documents place him at the same location on the same date as a reported assault, that pattern is no longer buried across two filing cabinets in two different jurisdictions. It is one query away.

The code is open. The documents are public. The connections are real. The only question is whether anyone with authority has enough moral integrity to treat the billionare nobility as everyday individuals under the law or if sycophancy and loyalty to wealth are a prerequeisite to involvement with the US Governemnt.
