#!/usr/bin/env python3
"""Split manifest.json into chunks for distributed processing.

Usage:
    python deploy/split_manifest.py --chunks 20
    python deploy/split_manifest.py --chunks 50 --output-dir /path/to/output
"""

import json
import argparse
from pathlib import Path


def split_manifest(manifest_path: Path, num_chunks: int, output_dir: Path) -> list[Path]:
    """Split a manifest file into N chunks for distributed processing.
    
    Args:
        manifest_path: Path to the source manifest.json
        num_chunks: Number of chunks to create
        output_dir: Directory to write chunk files
        
    Returns:
        List of paths to created chunk files
    """
    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    documents = manifest.get("documents", [])
    total_docs = len(documents)
    
    if total_docs == 0:
        raise ValueError("Manifest contains no documents")
    
    print(f"Splitting {total_docs:,} documents into {num_chunks} chunks...")
    
    # Calculate chunk size
    chunk_size = total_docs // num_chunks
    remainder = total_docs % num_chunks
    
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = []
    
    start_idx = 0
    for i in range(num_chunks):
        # Distribute remainder across first chunks
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
        
        chunk_docs = documents[start_idx:end_idx]
        chunk_manifest = {
            "chunk_id": i,
            "total_chunks": num_chunks,
            "doc_range": f"{start_idx}-{end_idx}",
            "documents": chunk_docs
        }
        
        chunk_path = output_dir / f"manifest_chunk_{i:03d}.json"
        with open(chunk_path, "w") as f:
            json.dump(chunk_manifest, f, indent=2)
        
        chunk_paths.append(chunk_path)
        print(f"  Chunk {i:3d}: {len(chunk_docs):,} documents -> {chunk_path.name}")
        
        start_idx = end_idx
    
    print(f"\nCreated {num_chunks} chunk files in {output_dir}/")
    print(f"Average chunk size: {total_docs // num_chunks:,} documents")
    
    return chunk_paths


def main():
    parser = argparse.ArgumentParser(description="Split manifest for distributed OCR processing")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.json"),
        help="Path to source manifest (default: data/manifest.json)"
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=20,
        help="Number of chunks to create (default: 20)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manifest_chunks"),
        help="Output directory for chunk files (default: data/manifest_chunks)"
    )
    
    args = parser.parse_args()
    
    if not args.manifest.exists():
        print(f"Error: Manifest not found at {args.manifest}")
        print("Run 'egr classify /path/to/pdfs' first to create the manifest.")
        return 1
    
    split_manifest(args.manifest, args.chunks, args.output_dir)
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Upload chunk files to accessible URLs (GitHub Gist, S3, etc.)")
    print("2. Deploy workers on Vast.ai, each pointing to a different chunk")
    print("3. Collect /data/processed/ from each worker when done")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())
