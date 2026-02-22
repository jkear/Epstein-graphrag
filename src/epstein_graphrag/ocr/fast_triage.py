"""Fast pre-triage: classify PDFs as text-extractable vs image-only.

Runs ~100x faster than Marker by using pdfplumber font detection.
Image-only PDFs get routed straight to vision OCR, skipping Marker.

Usage:
    # Triage and split manifest for parallel processing
    python -m epstein_graphrag.ocr.fast_triage \
        --manifest data/manifest.json \
        --output-dir data/triage \
        --splits 4

    # Then run each split in a separate terminal:
    egr ocr --manifest data/triage/split_0_image.json --ocr-provider lmstudio
    egr ocr --manifest data/triage/split_1_text.json --ocr-provider lmstudio
    # etc.
"""

import json
import logging
import math
import sys
from pathlib import Path

import pdfplumber
from tqdm import tqdm

logger = logging.getLogger(__name__)


def triage_pdf(pdf_path: Path) -> dict:
    """Fast triage: detect whether a PDF has real extractable text.

    Checks:
    - Embedded text content length (pdfplumber)
    - Whether fonts are present (indicates text layer vs rasterized)
    - Page count

    Returns dict with triage metadata.
    """
    result = {
        "has_text_layer": False,
        "text_char_count": 0,
        "has_fonts": False,
        "page_count": 0,
        "triage": "image",  # default: assume image
    }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            result["page_count"] = len(pdf.pages)

            total_text = ""
            fonts_found = False

            for page in pdf.pages:
                # Check for fonts (real text layer indicator)
                if page.chars:
                    fonts_found = True

                text = page.extract_text() or ""
                total_text += text

            text_len = len(total_text.strip())
            result["text_char_count"] = text_len
            result["has_fonts"] = fonts_found
            result["has_text_layer"] = fonts_found and text_len > 50

            # Triage decision
            if fonts_found and text_len > 200:
                result["triage"] = "text"  # Marker will likely succeed
            elif fonts_found and text_len > 50:
                result["triage"] = "text_weak"  # Marker might work
            else:
                result["triage"] = "image"  # Skip Marker, go straight to vision

    except Exception as e:
        logger.warning(f"Triage failed for {pdf_path.name}: {e}")
        result["triage"] = "image"  # On failure, assume image

    return result


def triage_manifest(manifest_path: Path, output_dir: Path, splits: int = 1) -> dict:
    """Triage all documents in a manifest and optionally split for parallel runs.

    Creates:
    - triage_report.json: Full triage results
    - manifest_text.json: Docs that should use Marker (text-extractable)
    - manifest_image.json: Docs that skip Marker (image-only → vision OCR)
    - split_N_text.json / split_N_image.json: Split manifests for parallel terminals

    Returns summary stats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    logger.info(f"Loaded manifest: {len(manifest)} documents")

    # Run triage
    triage_results = {}
    text_docs = {}
    image_docs = {}

    for doc_id, info in tqdm(manifest.items(), desc="Triaging PDFs"):
        path_str = info.get("file_path") or info.get("path")
        if not path_str:
            continue

        pdf_path = Path(path_str)
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            continue

        triage = triage_pdf(pdf_path)
        triage_results[doc_id] = triage

        if triage["triage"] in ("text", "text_weak"):
            text_docs[doc_id] = info
        else:
            image_docs[doc_id] = info

    # Save triage report
    report_path = output_dir / "triage_report.json"
    report_path.write_text(json.dumps(triage_results, indent=2))

    # Save separated manifests
    text_manifest_path = output_dir / "manifest_text.json"
    text_manifest_path.write_text(json.dumps(text_docs, indent=2))

    image_manifest_path = output_dir / "manifest_image.json"
    image_manifest_path.write_text(json.dumps(image_docs, indent=2))

    # Split for parallel processing
    if splits > 1:
        _split_manifest(text_docs, "text", splits, output_dir)
        _split_manifest(image_docs, "image", splits, output_dir)

    stats = {
        "total": len(manifest),
        "text": len(text_docs),
        "image": len(image_docs),
        "skipped": len(manifest) - len(text_docs) - len(image_docs),
        "splits": splits,
    }

    logger.info(
        f"Triage complete: {stats['text']} text, {stats['image']} image, "
        f"{stats['skipped']} skipped"
    )
    logger.info(f"Output written to {output_dir}")

    return stats


def _split_manifest(docs: dict, label: str, n_splits: int, output_dir: Path) -> None:
    """Split a manifest dict into N roughly equal chunks."""
    items = list(docs.items())
    chunk_size = math.ceil(len(items) / n_splits)

    for i in range(n_splits):
        chunk = dict(items[i * chunk_size : (i + 1) * chunk_size])
        if not chunk:
            continue
        split_path = output_dir / f"split_{i}_{label}.json"
        split_path.write_text(json.dumps(chunk, indent=2))
        logger.info(f"  {split_path.name}: {len(chunk)} docs")


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Fast PDF triage for parallel OCR")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.json"),
        help="Input manifest file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/triage"),
        help="Output directory for triage results",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=1,
        help="Number of splits for parallel processing (default: 1)",
    )

    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Error: manifest not found at {args.manifest}", file=sys.stderr)
        sys.exit(1)

    stats = triage_manifest(args.manifest, args.output_dir, args.splits)

    print(f"\n--- Triage Summary ---")
    print(f"Total documents:  {stats['total']}")
    print(f"Text (Marker OK): {stats['text']}")
    print(f"Image (skip Marker): {stats['image']}")
    print(f"Skipped (no path):   {stats['skipped']}")

    if args.splits > 1:
        print(f"\nSplit into {args.splits} chunks.")
        print(f"Run in separate terminals:")
        for i in range(args.splits):
            print(f"  Terminal {i+1}: egr ocr --manifest {args.output_dir}/split_{i}_image.json --ocr-provider lmstudio -w 2")
        print(f"\nText docs (Marker-first):")
        for i in range(args.splits):
            print(f"  Terminal {i+1}: egr ocr --manifest {args.output_dir}/split_{i}_text.json --ocr-provider lmstudio -w 2")


if __name__ == "__main__":
    main()
