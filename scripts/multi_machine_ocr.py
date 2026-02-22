#!/usr/bin/env python3
"""
Multi-machine OCR coordinator.

Splits work between multiple vLLM servers (e.g., RTX 3080 Ti + M4 Mac).
Each machine processes a portion of the manifest in parallel.

Usage:
    python scripts/multi_machine_ocr.py \
        --servers http://pc:8000/v1,http://mac:8001/v1 \
        --manifest data/manifest.json
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epstein_graphrag.ocr.vllm_ocr import extract_text_from_pdf_vllm, check_vllm_available
from epstein_graphrag.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Server:
    url: str
    name: str
    available: bool = False
    processed: int = 0
    failed: int = 0


def check_servers(server_urls: list[str]) -> list[Server]:
    """Check which servers are available."""
    servers = []
    for i, url in enumerate(server_urls):
        name = f"server_{i}"
        available = check_vllm_available(url)
        servers.append(Server(url=url, name=name, available=available))
        status = "✓ available" if available else "✗ unavailable"
        logger.info(f"{name} ({url}): {status}")
    return [s for s in servers if s.available]


def process_document(
    doc: dict,
    server: Server,
    output_dir: Path,
    model: str,
) -> dict[str, Any]:
    """Process a single document on a specific server."""
    doc_id = doc.get("doc_id", doc.get("file_name", "unknown"))
    pdf_path = Path(doc.get("path", doc.get("file_path", "")))
    
    output_path = output_dir / f"{doc_id}.json"
    
    # Skip if already processed
    if output_path.exists():
        return {"doc_id": doc_id, "status": "skipped", "server": server.name}
    
    try:
        text, metadata = extract_text_from_pdf_vllm(
            pdf_path=pdf_path,
            model=model,
            base_url=server.url,
            document_type=doc.get("doc_type", "general"),
        )
        
        result = {
            "doc_id": doc_id,
            "source_path": str(pdf_path),
            "text": text,
            "metadata": metadata,
            "server": server.name,
        }
        
        output_path.write_text(json.dumps(result, indent=2))
        server.processed += 1
        
        return {"doc_id": doc_id, "status": "success", "server": server.name}
        
    except Exception as e:
        server.failed += 1
        error_path = output_dir / f"{doc_id}.error.json"
        error_path.write_text(json.dumps({
            "doc_id": doc_id,
            "error": str(e),
            "server": server.name,
        }, indent=2))
        
        return {"doc_id": doc_id, "status": "failed", "error": str(e), "server": server.name}


def main():
    parser = argparse.ArgumentParser(description="Multi-machine OCR coordinator")
    parser.add_argument(
        "--servers",
        type=str,
        required=True,
        help="Comma-separated list of vLLM server URLs (e.g., http://pc:8000/v1,http://mac:8001/v1)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.json"),
        help="Path to manifest file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Vision model name",
    )
    parser.add_argument(
        "--workers-per-server",
        type=int,
        default=3,
        help="Concurrent workers per server",
    )
    
    args = parser.parse_args()
    
    # Parse server URLs
    server_urls = [url.strip() for url in args.servers.split(",")]
    
    logger.info(f"Checking {len(server_urls)} servers...")
    servers = check_servers(server_urls)
    
    if not servers:
        logger.error("No servers available!")
        return 1
    
    logger.info(f"{len(servers)} servers ready")
    
    # Load manifest
    if not args.manifest.exists():
        logger.error(f"Manifest not found: {args.manifest}")
        return 1
    
    manifest = json.loads(args.manifest.read_text())
    documents = manifest.get("documents", [])
    
    if not documents:
        logger.error("No documents in manifest")
        return 1
    
    logger.info(f"Processing {len(documents)} documents across {len(servers)} servers")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Distribute work across servers (round-robin)
    total_workers = len(servers) * args.workers_per_server
    
    start_time = time.time()
    processed = 0
    failed = 0
    skipped = 0
    
    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        futures = {}
        
        for i, doc in enumerate(documents):
            # Round-robin server selection
            server = servers[i % len(servers)]
            
            future = executor.submit(
                process_document,
                doc,
                server,
                args.output_dir,
                args.model,
            )
            futures[future] = doc
        
        for future in as_completed(futures):
            result = future.result()
            
            if result["status"] == "success":
                processed += 1
            elif result["status"] == "skipped":
                skipped += 1
            else:
                failed += 1
            
            total = processed + failed + skipped
            if total % 100 == 0:
                elapsed = time.time() - start_time
                rate = total / elapsed if elapsed > 0 else 0
                remaining = len(documents) - total
                eta_hours = remaining / rate / 3600 if rate > 0 else 0
                
                logger.info(
                    f"Progress: {total}/{len(documents)} "
                    f"({processed} ok, {skipped} skip, {failed} fail) "
                    f"| {rate:.1f} docs/sec | ETA: {eta_hours:.1f}h"
                )
    
    elapsed = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total documents: {len(documents)}")
    logger.info(f"Processed: {processed}")
    logger.info(f"Skipped (already done): {skipped}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Time: {elapsed/3600:.1f} hours")
    logger.info(f"Rate: {len(documents)/elapsed:.1f} docs/sec")
    logger.info("")
    logger.info("Per-server stats:")
    for server in servers:
        logger.info(f"  {server.name}: {server.processed} processed, {server.failed} failed")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
