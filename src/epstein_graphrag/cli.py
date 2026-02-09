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
    """Run OCR pipeline on classified text documents."""
    console.print("[yellow]OCR pipeline not yet implemented — see Task 4[/yellow]")


@main.command()
@click.pass_context
def extract(ctx: click.Context) -> None:
    """Extract entities from OCR output."""
    console.print("[yellow]Entity extraction not yet implemented — see Task 5[/yellow]")


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
