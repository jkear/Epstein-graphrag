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
@click.pass_context
def ocr(ctx: click.Context) -> None:
    """Run OCR pipeline on classified documents."""
    import json
    from pathlib import Path

    from epstein_graphrag.ocr.marker_pipeline import process_batch

    config = ctx.obj["config"]

    # Load manifest
    if not config.manifest_path.exists():
        console.print("[red]Error: manifest.json not found. Run 'egr classify' first.[/red]")
        return

    manifest = json.loads(config.manifest_path.read_text())
    console.print(f"Loaded manifest: {len(manifest)} documents")

    # Run OCR pipeline
    stats = process_batch(
        manifest=manifest,
        output_dir=config.processed_dir,
        gemini_api_key=config.gemini_api_key,
    )

    console.print(f"[green]OCR complete:[/green]")
    console.print(f"  Total: {stats['total']}")
    console.print(f"  Processed: {stats['processed']}")
    console.print(f"  Skipped: {stats['skipped']}")
    console.print(f"  Failed: {stats['failed']}")
    
    if stats['failed'] > 0:
        console.print(f"\n[yellow]Failed documents:[/yellow]")
        for doc_id in stats['failed_docs']:
            console.print(f"  - {doc_id}")


@main.command()
@click.pass_context
def extract(ctx: click.Context) -> None:
    """Extract entities from OCR output."""
    from epstein_graphrag.extract.entity_extractor import extract_batch

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

    # Run extraction
    console.print("[cyan]Starting entity extraction...[/cyan]")
    stats = extract_batch(
        processed_dir=config.processed_dir,
        extracted_dir=config.extracted_dir,
        gemini_api_key=config.gemini_api_key,
    )

    console.print(f"\n[green]Extraction complete:[/green]")
    console.print(f"  Total: {stats['total']}")
    console.print(f"  Processed: {stats['processed']}")
    console.print(f"  Skipped: {stats['skipped']}")
    console.print(f"  Failed: {stats['failed']}")

    if stats["failed"] > 0:
        console.print(f"\n[yellow]Failed documents:[/yellow]")
        for doc_id in stats["failed_docs"]:
            console.print(f"  - {doc_id}")



@main.command()
@click.pass_context
def ingest(ctx: click.Context) -> None:
    """Ingest extracted entities into Neo4j."""
    console.print("[yellow]Graph ingestion not yet implemented — see Task 6[/yellow]")


@main.command()
@click.pass_context
def embed(ctx: click.Context) -> None:
    """Generate embeddings for graph nodes."""
    console.print("[yellow]Embedding generation not yet implemented — see Task 7[/yellow]")


@main.command()
@click.argument("question")
@click.pass_context
def query(ctx: click.Context, question: str) -> None:
    """Query the evidence graph."""
    console.print("[yellow]Query engine not yet implemented — see Task 8[/yellow]")


@main.command()
@click.pass_context
def schema(ctx: click.Context) -> None:
    """Verify the Neo4j schema."""
    console.print("[yellow]Schema verification not yet implemented — see Task 1[/yellow]")


if __name__ == "__main__":
    main()
