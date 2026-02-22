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
@click.option(
    "--num-workers",
    "-w",
    type=int,
    default=None,
    help="Number of parallel workers (default: CPU count)",
)
@click.pass_context
def classify(ctx: click.Context, data_dir: str, num_workers: int | None) -> None:
    """Classify all PDFs in a directory."""
    from pathlib import Path
    import multiprocessing as mp

    from epstein_graphrag.classify.classifier import classify_batch

    config = ctx.obj["config"]
    
    workers = num_workers or mp.cpu_count()
    console.print(f"[cyan]Classifying with {workers} parallel workers[/cyan]")
    
    results = classify_batch(Path(data_dir), config.manifest_path, num_workers=workers)
    console.print(f"[green]Classified {len(results)} new documents[/green]")


@main.command()
@click.option(
    "--manifest",
    type=click.Path(exists=True),
    help="Path to custom manifest file (default: data/manifest.json)",
)
@click.option(
    "--ocr-provider",
    type=click.Choice(["ollama", "lmstudio", "vllm"]),
    default="ollama",
    help="OCR provider: ollama, lmstudio, or vllm (fastest)",
)
@click.option(
    "--lm-base-url",
    type=str,
    default="http://localhost:1234/v1",
    help="LM Studio base URL (default: http://localhost:1234/v1)",
)
@click.option(
    "--vllm-url",
    type=str,
    default="http://localhost:8000/v1",
    help="vLLM server URL (default: http://localhost:8000/v1)",
)
@click.option(
    "--ocr-model",
    type=str,
    default=None,
    help="Vision model (default: minicpm-v:8b for Ollama, Qwen2-VL-7B for vLLM)",
)
@click.option(
    "--num-workers",
    "-w",
    type=int,
    default=1,
    help="Number of parallel workers (default: 1, sequential processing)",
)
@click.pass_context
def ocr(ctx: click.Context, manifest: str | None, ocr_provider: str, lm_base_url: str, vllm_url: str, ocr_model: str | None, num_workers: int) -> None:
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
    elif ocr_provider == "vllm":
        console.print(f"[cyan]vLLM URL: {vllm_url}[/cyan]")
    
    if num_workers > 1:
        console.print(f"[cyan]Workers: {num_workers}[/cyan]")

    # Load manifest
    if not manifest_path.exists():
        console.print(f"[red]Error: Manifest not found at {manifest_path}[/red]")
        return

    manifest_data = json.loads(manifest_path.read_text())
    
    # Handle both formats: dict of docs or {"documents": [...]}
    if isinstance(manifest_data, dict) and "documents" in manifest_data:
        docs = manifest_data["documents"]
        console.print(f"Loaded manifest: {len(docs)} documents")
    else:
        docs = manifest_data
        console.print(f"Loaded manifest: {len(docs)} documents")

    # Default model per provider if not specified
    if ocr_model is None:
        if ocr_provider == "ollama":
            ocr_model = "minicpm-v:8b"
        elif ocr_provider == "vllm":
            ocr_model = "Qwen/Qwen2-VL-7B-Instruct"

    # Select base URL based on provider
    base_url = vllm_url if ocr_provider == "vllm" else lm_base_url

    stats = process_batch(
        manifest=manifest_data,
        output_dir=config.processed_dir,
        ocr_provider=ocr_provider,
        ocr_model=ocr_model,
        lm_base_url=base_url,
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


@main.command(name="vision-ocr")
@click.option(
    "--manifest",
    type=click.Path(exists=True),
    help="Path to manifest file (default: data/manifest.json)",
)
@click.option(
    "--ocr-provider",
    type=click.Choice(["lmstudio", "ollama", "gemini"]),
    default="lmstudio",
    help="OCR provider (default: lmstudio)",
)
@click.option(
    "--lm-base-url",
    type=str,
    default="http://localhost:1234/v1",
    help="LM Studio base URL (default: http://localhost:1234/v1)",
)
@click.option(
    "--ocr-model",
    type=str,
    default=None,
    help="Vision model name/ID (auto-detect for LM Studio)",
)
@click.option(
    "--num-workers", "-w",
    type=int,
    default=1,
    help="Number of parallel workers (default: 1)",
)
@click.option(
    "--dpi",
    type=int,
    default=300,
    help="DPI for PDF-to-image conversion (default: 300, try 150 for speed)",
)
@click.option(
    "--no-resume",
    is_flag=True,
    help="Reprocess existing files",
)
@click.option(
    "--no-marker-fallback",
    is_flag=True,
    help="Disable Marker fallback (vision-only, fastest mode)",
)
@click.pass_context
def vision_ocr(
    ctx: click.Context,
    manifest: str | None,
    ocr_provider: str,
    lm_base_url: str,
    ocr_model: str | None,
    num_workers: int,
    dpi: int,
    no_resume: bool,
    no_marker_fallback: bool,
) -> None:
    """Vision-first OCR: photograph analysis first, Marker fallback for text docs.

    Faster than 'egr ocr' for image-heavy corpora. Handles mixed/photograph
    docs correctly by routing to visual analysis instead of text-only OCR.

    \b
    Examples:
        egr vision-ocr                          # Full manifest, LM Studio
        egr vision-ocr --manifest data/triage/split_0_image.json -w 2
        egr vision-ocr --ocr-provider gemini    # Use Gemini API (fast + cheap)
        egr vision-ocr --dpi 150 --no-marker-fallback  # Maximum speed
    """
    from pathlib import Path

    from epstein_graphrag.ocr.fast_ocr_runner import run_batch

    config = ctx.obj["config"]
    manifest_path = Path(manifest) if manifest else config.manifest_path

    if not manifest_path.exists():
        console.print(f"[red]Error: Manifest not found at {manifest_path}[/red]")
        return

    console.print(f"[cyan]OCR provider: {ocr_provider}[/cyan]")
    console.print(f"[cyan]DPI: {dpi}[/cyan]")
    console.print(f"[cyan]Workers: {num_workers}[/cyan]")
    console.print(f"[cyan]Marker fallback: {not no_marker_fallback}[/cyan]")

    stats = run_batch(
        manifest_path=manifest_path,
        output_dir=config.processed_dir,
        ocr_provider=ocr_provider,
        ocr_model=ocr_model,
        num_workers=num_workers,
        lm_base_url=lm_base_url,
        resume=not no_resume,
        marker_fallback=not no_marker_fallback,
        dpi=dpi,
    )

    console.print("\n[green]Vision-first OCR complete:[/green]")
    console.print(f"  Total:     {stats['total']}")
    console.print(f"  Processed: {stats['processed']}")
    console.print(f"  Skipped:   {stats['skipped']}")
    console.print(f"  Failed:    {stats['failed']}")

    if stats['failed'] > 0:
        console.print("\n[yellow]Failed documents:[/yellow]")
        for doc_id in stats['failed_docs'][:20]:
            console.print(f"  - {doc_id}")
        if len(stats['failed_docs']) > 20:
            console.print(f"  ... and {len(stats['failed_docs']) - 20} more")


@main.command()
@click.option(
    "--manifest",
    type=click.Path(exists=True),
    help="Path to manifest file (default: data/manifest.json)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="data/triage",
    help="Output directory for triage results (default: data/triage)",
)
@click.option(
    "--splits",
    type=int,
    default=4,
    help="Number of splits for parallel processing (default: 4)",
)
@click.option(
    "--remaining/--all",
    default=True,
    help="Only include unprocessed docs (default: --remaining)",
)
@click.option(
    "--processed-dir",
    type=click.Path(),
    default="data/processed",
    help="Directory to check for already-processed files (default: data/processed)",
)
@click.pass_context
def split(ctx: click.Context, manifest: str | None, output_dir: str, splits: int, remaining: bool, processed_dir: str) -> None:
    """Split manifest for parallel OCR processing.

    By default, filters out already-processed docs before splitting.
    Use --all to include everything.

    \b
    Examples:
        egr split                          # 4-way split of remaining docs
        egr split --all                    # 4-way split of ALL docs
        egr split --splits 2 --remaining   # 2-way split of unprocessed only
        egr split --manifest custom.json --splits 2
    """
    import json
    from pathlib import Path

    from epstein_graphrag.ocr.fast_triage import triage_manifest

    config = ctx.obj["config"]
    manifest_path = Path(manifest) if manifest else config.manifest_path
    out_path = Path(output_dir)

    if not manifest_path.exists():
        console.print(f"[red]Error: Manifest not found at {manifest_path}[/red]")
        return

    # Filter to remaining-only if requested
    if remaining:
        proc_path = Path(processed_dir)
        full_manifest = json.loads(manifest_path.read_text())
        already_done = set()
        if proc_path.exists():
            for f in proc_path.iterdir():
                if f.suffix == ".json" and not f.name.endswith(".error.json"):
                    already_done.add(f.stem)

        remaining_manifest = {k: v for k, v in full_manifest.items() if k not in already_done}
        console.print(f"[cyan]Filtering: {len(full_manifest)} total, {len(already_done)} done, [bold]{len(remaining_manifest)} remaining[/bold][/cyan]")

        if not remaining_manifest:
            console.print("[green]All documents already processed![/green]")
            return

        # Write filtered manifest
        out_path.mkdir(parents=True, exist_ok=True)
        filtered_path = out_path / "remaining_manifest.json"
        filtered_path.write_text(json.dumps(remaining_manifest, indent=2))
        manifest_path = filtered_path

    console.print(f"[cyan]Triaging and splitting {manifest_path.name} into {splits} chunks...[/cyan]")

    stats = triage_manifest(manifest_path, out_path, splits)

    console.print("\n[green]Triage complete:[/green]")
    console.print(f"  Total documents:    {stats['total']}")
    console.print(f"  Text (Marker OK):   {stats['text']}")
    console.print(f"  Image (vision OCR): {stats['image']}")

    if splits > 1:
        console.print(f"\n[cyan]Run in separate terminals:[/cyan]")
        for i in range(splits):
            console.print(f"  egr vision-ocr --manifest {out_path}/split_{i}_image.json -w 2")
        console.print(f"\n[cyan]Text docs (slower, uses Marker):[/cyan]")
        for i in range(splits):
            console.print(f"  egr vision-ocr --manifest {out_path}/split_{i}_text.json -w 2")


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
