# Epstein Evidence GraphRAG

A system for extracting structured knowledge from the Epstein files using graphRAG.

## Source Files

Download the Epstein evidence PDFs from: https://github.com/yung-megafone/Epstein-Files

## Overview

This system processes PDF documents through a multi-stage pipeline:

1. **Classification** - Categorize documents by type (photograph, bank statement, legal document, etc.)
2. **OCR** - Extract text using vision models (MiniCPM-V via Ollama)
3. **Entity Extraction** - Extract structured entities using multi-provider LLM strategy
4. **Graph Ingestion** - Load entities into Neo4j with relationship mapping
5. **Embeddings** - Generate vector embeddings for semantic search
6. **Query** - Natural language queries against the evidence graph

### Prerequisites

- Python 3.11+
- LMStuidio CLI running `qwen3-VL-8b` for concurrency or
- Ollama running with `minicpm-v:8b` and `nomic-embed-text` models
- Neo4j (for graph storage)
- Optional: Gemini API key, DeepSeek API key

### Setup

```bash
cd /epstein-graphrag
uv venv
source .venv/bin/activate
uv sync
```

### Get LMStuido or Ollama Models

Options 1, 2, or 3 + an embedding model:

```powershell
lms pull lmstudio-community/Qwen3-VL-8B-GGUF
lms server start Qwen3-VL-8B-GGUF
```

```bash
lms pull lmstudio-community/Qwen3-VL-8B-MLX
lms server start lmstudio-community/Qwen3-VL-8B-MLX
```

```bash
ollama pull minicpm-v:8b
ollama pull nomic-embed-text # And you need this one.
```

To make any model changes: `src/epstein_graphrag/config.py`

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

### 1. Classify Documents

```bash
egr classify /path/to/pdf/directory
```

Parker or Coog (let's be real, no one else will get this far) please start with dataset 8 and work your way backwards. I'm going forward but alone will take ~159 days.

Outputs: `data/manifest.json`

Each classify command MERGEs to the manifest adding the new batch. So, it's resume-safe but, if you want to keep batches organized, archive each manifest set and classify a new manifest.json. or point the `egr ocr` command to a custom manifest with `--manifest custom_manifest.json`

### 2. Run OCR

```bash
egr ocr 
```

Uses the manifest to process all PDFs through the VLM (I'm using lmstudio-community/Qwen3-VL-8B-MLX). Outputs to `data/processed/`.
Key CLI Options for OCR

```bash
--ocr-provider -lmstudio # Choose ollama or lmstudio
--num-workers -w 3 # Number of parallel workers
--lm-base-url -http://localhost:1234/v1 # LM Studio base URL (LMStuido has options)
--manifest -data/manifest.json # Custom manifest path
```

### 3. Extract Entities

```bash
egr extract --num-workers 3
```

Multi-provider strategy (fallback chain):

@ToDo -  Update to replace Ollama and use LMStudio for concurrency

1. MiniCPM-V (local, via Ollama)
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

## Project Structure

```filesys
epstein-graphrag/
├── data/
│   ├── manifest.json           # Document classification manifest
│   ├── alias_table.json        # Name resolution mappings
│   ├── schema_alerts.jsonl     # Schema validation alerts
│   ├── processed/              # OCR output JSON files
│   ├── extracted/              # Entity extraction JSON files
├── src/epstein_graphrag/
│   ├── __init__.py
│   ├── cli.py                  # Command-line interface (egr)
│   ├── config.py               # Configuration management
│   ├── classify/               # Document classification
│   │   └── classifier.py
│   ├── ocr/                    # OCR pipeline
│   │   ├── marker_pipeline_ollama.py
│   │   ├── marker_pipeline.py
│   │   ├── ollama_ocr.py
│   │   ├── lmstudio_ocr.py
│   │   ├── deepseek_ocr.py
│   │   ├── forensic_prompts.py
│   │   ├── forensic_marker_processor.py
│   │   ├── quality_check.py
│   │   ├── duplicate_detector.py
│   │   └── redaction_merger.py
│   ├── extract/                # Entity extraction
│   │   ├── multi_provider_extractor.py
│   │   ├── entity_extractor.py
│   │   ├── prompts.py
│   │   └── schema_validator.py
│   ├── graph/                  # Neo4j operations
│   │   ├── schema.py
│   │   ├── ingest.py
│   │   └── dedup.py
│   ├── embeddings/             # Vector embeddings
│   │   └── embed.py
│   ├── query/                  # Query engine
│   │   ├── engine.py
│   │   ├── cypher_gen.py
│   │   ├── hybrid_search.py
│   │   └── pattern_detector.py
│   └── output/                 # Output utilities
├── tests/
├── pyproject.toml
├── uv.lock
├── AGENTS.md
├── .env.example                # Example environment configuration
└── .gitignore
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

## Maping the Epstein Files

Thousands of pages of court filings, flight logs, deposition transcripts, and photographs from the Jeffrey Epstein case sit in public document repositories. Individually, each page tells a fragment of a story. A name here, a flight date there, a location mentioned in passing. The connections between them — the patterns that reveal who knew what, who went where, and who looked the other way — remain buried in scanned PDFs that no single person could read in a lifetime.

I built a pipeline to read them all, extract every name, date, location, and allegation, and load the results into a graph database where those connections surface on their own.

## Designed to Stay Hidden

The documents released, like other "publicly available" records, have data that are obscured to protect congress' wealthy benefactors and keep the hand of the oligarchy, which is strangling democracy, hidden. Many are scanned at poor resolution, contain redactions — over names and emails of perpatrators, dates, and descriptions. Others are photographs with no searchable text at all. Government agencies released these files in formats that make systematic review difficult: thousands of individual PDFs, no index, no descriptive naming, no metadata.

This is not accidental. When you scatter evidence across 12ish separate document sets, redact key names, and release it all as images rather than text, you build a wall between the public and the truth. A determined reader might connect two documents. Connecting 2.7 million requires a machine.

## Reading the Docs

The first problem is the documents are uploaded as images on PDFs so copying text cannot be done. Eagles can't simply fly the ring to morodor. However, optical character recognition is no possible. Standard OCR tools treat every document the same — a receipt, a novel, a legal deposition all get the same processing. These documents demand more.

This pipeline uses forensic OCR prompts that tell the vision model what it is looking at. The model knows it is reading legal evidence. It knows to preserve every name exactly as written, to flag redacted regions, and to note text still visible beneath sloppy black-marker redactions. That last point matters: some redactions were done so poorly that the underlying text remains legible. The pipeline catches those.

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
