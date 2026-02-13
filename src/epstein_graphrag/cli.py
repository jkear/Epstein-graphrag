"""Command-line interface for the Epstein GraphRAG system.

Entry point: `egr` command (defined in pyproject.toml).
"""

import logging

import click
from rich.console import Console
from rich.logging import RichHandler

from epstein_graphrag.config import Config

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Epstein Evidence GraphRAG System."""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config()


@main.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.pass_context
def classify(ctx: click.Context, data_dir: str) -> None:
    """Classify all PDFs in a directory."""
    from pathlib import Path

    from epstein_graphrag.classify.classifier import classify_batch

    config = ctx.obj["config"]
    results = classify_batch(Path(data_dir), config.manifest_path)
    console.print(f"Classified {len(results)} new documents")


@main.command()
@click.option(
    "--manifest",
    type=click.Path(exists=True),
    help="Path to custom manifest file (default: data/manifest.json)",
)
@click.option(
    "--ocr-provider",
    type=click.Choice(["ollama", "lmstudio"]),
    default="ollama",
    help="OCR provider to use (default: ollama)",
)
@click.option(
    "--lm-base-url",
    type=str,
    default="http://localhost:1234/v1",
    help="LM Studio base URL (default: http://localhost:1234/v1)",
)
@click.option(
    "--num-workers",
    "-w",
    type=int,
    default=1,
    help="Number of parallel workers (default: 1, sequential processing)",
)
@click.pass_context
def ocr(ctx: click.Context, manifest: str | None, ocr_provider: str, lm_base_url: str, num_workers: int) -> None:
    """Run OCR pipeline on classified documents."""
    import json
    from pathlib import Path

    from epstein_graphrag.ocr.marker_pipeline_ollama import process_batch

    config = ctx.obj["config"]

    # Use custom manifest if provided, otherwise use default
    manifest_path = Path(manifest) if manifest else config.manifest_path

    # Log OCR provider and settings
    console.print(f"[cyan]OCR provider: {ocr_provider}[/cyan]")
    if ocr_provider == "lmstudio":
        console.print(f"[cyan]LM Studio URL: {lm_base_url}[/cyan]")
        if num_workers > 1:
            console.print("[yellow]Note: LM Studio supports concurrent processing (--num-workers {num_workers})[/yellow]")

    # Load manifest
    if not manifest_path.exists():
        console.print(f"[red]Error: Manifest not found at {manifest_path}[/red]")
        return

    manifest_data = json.loads(manifest_path.read_text())
    console.print(f"Loaded manifest: {len(manifest_data)} documents")

    # Run OCR pipeline with selected provider
    stats = process_batch(
        manifest=manifest_data,
        output_dir=config.processed_dir,
        ocr_provider=ocr_provider,
        ocr_model="minicpm-v:8b" if ocr_provider == "ollama" else None,
        lm_base_url=lm_base_url,
        num_workers=num_workers,
    )

    console.print("[green]OCR complete:[/green]")
    console.print(f"  Total: {stats['total']}")
    console.print(f"  Processed: {stats['processed']}")
    console.print(f"  Skipped: {stats['skipped']}")
    console.print(f"  Failed: {stats['failed']}")

    if stats['failed'] > 0:
        console.print("\n[yellow]Failed documents:[/yellow]")
        for doc_id in stats['failed_docs']:
            console.print(f"  - {doc_id}")


@main.command()
@click.option(
    "--num-workers",
    "-w",
    type=int,
    default=3,
    help="Number of parallel workers (default: 3)",
)
@click.pass_context
def extract(ctx: click.Context, num_workers: int) -> None:
    """Extract entities from OCR output using multi-provider strategy."""
    from epstein_graphrag.extract.multi_provider_extractor import extract_batch_parallel

    config = ctx.obj["config"]

    # Check processed dir exists
    if not config.processed_dir.exists():
        console.print("[red]Error: No OCR output found. Run 'egr ocr' first.[/red]")
        return

    # Count OCR files
    ocr_files = list(config.processed_dir.glob("*.json"))
    ocr_files = [f for f in ocr_files if not f.name.endswith(".error.json")]
    console.print(f"Found {len(ocr_files)} OCR results to process")

    if len(ocr_files) == 0:
        console.print("[yellow]No OCR results to process.[/yellow]")
        return

    # Run multi-provider extraction
    console.print(
        f"[cyan]Starting entity extraction with {num_workers} parallel workers...[/cyan]"
    )
    console.print(
        "[dim]Providers: MiniCPM-V via Ollama (local) → DeepSeek API → Gemini 2.5 Flash[/dim]\n"
    )

    stats = extract_batch_parallel(
        processed_dir=config.processed_dir,
        extracted_dir=config.extracted_dir,
        ollama_model="minicpm-v:8b",
        ollama_host="http://localhost:11434",
        deepseek_api_key=config.deepseek_api_key,
        gemini_api_key=config.gemini_api_key,
        num_workers=num_workers,
        resume=True,
    )

    console.print("\n[green]Extraction complete:[/green]")
    console.print(f"  Total: {stats['total']}")
    console.print(f"  Processed: {stats['processed']}")
    console.print(f"  Skipped: {stats['skipped']}")
    console.print(f"  Failed: {stats['failed']}")

    # Provider stats
    if stats['processed'] > 0:
        console.print("\n[cyan]Provider usage:[/cyan]")
        console.print(f"  GLM (local): {stats['provider_stats']['glm']}")
        console.print(f"  DeepSeek API: {stats['provider_stats']['deepseek']}")
        console.print(f"  Gemini API: {stats['provider_stats']['gemini']}")

    # Timing stats
    if stats['processed'] > 0:
        avg_time = stats['total_time'] / stats['processed']
        console.print("\n[cyan]Performance:[/cyan]")
        console.print(f"  Total time: {stats['total_time']:.1f}s")
        console.print(f"  Avg per doc: {avg_time:.1f}s")

    if stats["failed"] > 0:
        console.print("\n[yellow]Failed documents:[/yellow]")
        for doc_id in stats["failed_docs"]:
            console.print(f"  - {doc_id}")



@main.command()
@click.pass_context
def ingest(ctx: click.Context) -> None:
    """Ingest extracted entities into Neo4j.

    Reads all extraction JSONs from data/extracted/ and MERGE-ingests
    them into the graph. Skips .error.json files. Idempotent — safe
    to run repeatedly.
    """
    import json

    from epstein_graphrag.graph.dedup import AliasResolver
    from epstein_graphrag.graph.ingest import GraphIngestor

    config = ctx.obj["config"]

    # Find extraction files
    if not config.extracted_dir.exists():
        console.print("[red]Error: No extraction output found. Run 'egr extract' first.[/red]")
        return

    extraction_files = sorted(
        f for f in config.extracted_dir.glob("*.json")
        if not f.name.endswith(".error.json")
    )

    if not extraction_files:
        console.print("[yellow]No extraction files to ingest.[/yellow]")
        return

    console.print(f"Found {len(extraction_files)} extraction files to ingest")

    # Load all extractions
    extractions = []
    for fpath in extraction_files:
        try:
            extractions.append(json.loads(fpath.read_text()))
        except json.JSONDecodeError as e:
            console.print(f"[red]Skipping {fpath.name}: invalid JSON ({e})[/red]")

    # Ingest
    alias_resolver = AliasResolver(config)
    console.print(f"Loaded {len(alias_resolver.alias_table)} aliases")

    ingestor = GraphIngestor(config, alias_resolver=alias_resolver)
    try:
        result = ingestor.ingest_batch(extractions)
    finally:
        ingestor.close()

    # Print results
    console.print("\n[green]Ingestion complete:[/green]")
    console.print(f"  Total documents:  {result['total']}")
    console.print(f"  Succeeded:        {result['succeeded']}")
    console.print(f"  Failed:           {result['failed']}")
    console.print(f"  Nodes merged:     {result['total_nodes_merged']}")
    console.print(f"  Rels merged:      {result['total_relationships_merged']}")

    if result["errors"]:
        console.print("\n[yellow]Errors:[/yellow]")
        for err in result["errors"]:
            console.print(f"  - {err['doc_id']}: {err['error']}")


@main.command()
@click.option(
    "--label", "-l",
    type=click.Choice(["all", "Event", "Allegation", "Person", "Location"],
                       case_sensitive=True),
    default="all",
    help="Which node type to embed (default: all).",
)
@click.pass_context
def embed(ctx: click.Context, label: str) -> None:
    """Generate 768d embeddings for graph nodes via nomic-embed-text.

    Reads nodes from Neo4j where the embedding property is NULL,
    generates vectors via Ollama, and writes them back. Resume-safe.
    """
    from epstein_graphrag.embeddings.embed import (
        EMBEDDING_TARGETS,
        NodeEmbedder,
    )

    config = ctx.obj["config"]

    embedder = NodeEmbedder(config)
    try:
        if label == "all":
            console.print("[cyan]Generating embeddings for all node types...[/cyan]")
            result = embedder.embed_all()
        else:
            # Find the matching target
            target = next(
                (t for t in EMBEDDING_TARGETS if t.label == label), None
            )
            if not target:
                console.print(f"[red]Unknown label: {label}[/red]")
                return
            console.print(f"[cyan]Generating embeddings for {label} nodes...[/cyan]")
            single_result = embedder.embed_target(target)
            result = {
                "targets": [single_result],
                "total_nodes": single_result["total"],
                "total_embedded": single_result["embedded"],
                "total_skipped": single_result["skipped"],
            }
    finally:
        embedder.close()

    # Print results
    console.print("\n[green]Embedding complete:[/green]")
    for t in result["targets"]:
        status = "[green]done[/green]" if t["embedded"] > 0 else "[dim]none[/dim]"
        console.print(f"  {t['label']:15s}  {t['embedded']:3d} embedded  "
                       f"{t['skipped']:3d} skipped  {status}")

    console.print(f"\n  Total: {result['total_embedded']} vectors generated")


@main.command()
@click.argument("question")
@click.pass_context
def query(ctx: click.Context, question: str) -> None:
    """Query the evidence graph."""
    console.print("[yellow]Query engine not yet implemented — see Task 8[/yellow]")


@main.command()
@click.option("--create/--no-create", default=True, help="Create missing constraints and indexes.")
@click.option("--verify/--no-verify", default=True, help="Verify schema after creation.")
@click.pass_context
def schema(ctx: click.Context, create: bool, verify: bool) -> None:
    """Create and verify the Neo4j schema.

    Creates constraints, vector indexes, and fulltext indexes.
    Idempotent — safe to run repeatedly.
    """
    from epstein_graphrag.graph.schema import create_schema, verify_schema

    config = ctx.obj["config"]

    if create:
        console.print("[cyan]Creating schema...[/cyan]")
        stats = create_schema(config)
        console.print(f"  Constraints: {stats['constraints_created']} created, "
                       f"{stats['constraints_existing']} existing")
        console.print(f"  Vector indexes: {stats['vector_indexes_created']} created, "
                       f"{stats['vector_indexes_existing']} existing")
        console.print(f"  Fulltext indexes: {stats['fulltext_indexes_created']} created, "
                       f"{stats['fulltext_indexes_existing']} existing")

    if verify:
        console.print("\n[cyan]Verifying schema...[/cyan]")
        result = verify_schema(config)
        console.print(f"  Constraints:     {result['constraints']}")
        console.print(f"  Vector indexes:  {result['vector_indexes']}")
        console.print(f"  Fulltext indexes: {result['fulltext_indexes']}")

        # Check expected counts
        ok = (
            result["constraints"] >= 18
            and result["vector_indexes"] >= 6
            and result["fulltext_indexes"] >= 4
        )
        if ok:
            console.print("\n[green]Schema OK[/green]")
        else:
            console.print(
                "\n[red]Schema incomplete — "
                "some constraints or indexes are missing[/red]"
            )


if __name__ == "__main__":
    main()
