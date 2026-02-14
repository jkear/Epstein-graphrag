"""Multi-provider entity extraction with parallel processing.

Uses multiple LLM providers in parallel to maximize speed and avoid rate limits:
- MiniCPM-V via Ollama - Local model (proven 100% OCR success)
- DeepSeek API - Fast cloud
- Gemini 2.5 Flash - Free tier
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import ollama
from tqdm import tqdm

from epstein_graphrag.extract.prompts import TEXT_ENTITY_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Structured entity extraction result."""

    doc_id: str
    doc_type: str
    people: list
    locations: list
    organizations: list
    events: list
    allegations: list
    associations: list
    identity_documents: list  # NEW: Passports, IDs, birth certificates, etc.
    communications: list  # NEW: Emails, phone calls, letters, etc.
    legal_documents: list  # NEW: Subpoenas, depositions, warrants, etc.
    transactions: list  # NEW: Financial transactions when present
    physical_evidence: list  # NEW: Documents as objects, devices, recordings
    redacted_entities: list  # NEW: Track redactions for potential de-redaction
    objects_of_interest: list  # LEGACY: For photo analysis
    provider: str  # Which LLM provider was used
    processing_time: float
    raw_llm_response: str

    @staticmethod
    def from_parsed_json(
        doc_id: str,
        doc_type: str,
        parsed: dict,
        provider: str,
        processing_time: float,
        raw_response: str,
    ) -> "ExtractionResult":
        """Create ExtractionResult from parsed JSON with defaults for new fields."""
        return ExtractionResult(
            doc_id=doc_id,
            doc_type=doc_type,
            people=parsed.get("people", []),
            locations=parsed.get("locations", []),
            organizations=parsed.get("organizations", []),
            events=parsed.get("events", []),
            allegations=parsed.get("allegations", []),
            associations=parsed.get("associations", []),
            identity_documents=parsed.get("identity_documents", []),
            communications=parsed.get("communications", []),
            legal_documents=parsed.get("legal_documents", []),
            transactions=parsed.get("transactions", []),
            physical_evidence=parsed.get("physical_evidence", []),
            redacted_entities=parsed.get("redacted_entities", []),
            objects_of_interest=parsed.get("objects_of_interest", []),
            provider=provider,
            processing_time=processing_time,
            raw_llm_response=raw_response,
        )


class MultiProviderExtractor:
    """Manages multiple LLM providers for parallel entity extraction."""

    def __init__(
        self,
        ollama_model: str = "minicpm-v:8b",
        ollama_host: str = "http://localhost:11434",
        deepseek_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        mlx_model_path: Optional[str] = None,
        num_workers: int = 3,
    ):
        """Initialize multi-provider extractor.

        Args:
            ollama_model: Ollama model to use (default: minicpm-v:8b)
            ollama_host: Ollama server URL
            deepseek_api_key: DeepSeek API key (optional)
            gemini_api_key: Gemini API key (optional)
            mlx_model_path: Path to MLX model (e.g., Qwen3-VL) (optional)
            num_workers: Number of parallel workers (one per provider)
        """
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self.deepseek_api_key = deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.mlx_model_path = mlx_model_path
        self.num_workers = num_workers

        # Test Ollama connection
        logger.info(f"Testing Ollama connection to {ollama_host} with model {ollama_model}")
        try:
            ollama.list()
            logger.info("Ollama connection successful")
        except Exception as e:
            logger.warning(f"Ollama connection test failed: {e}")

        # Initialize API clients lazily
        self._deepseek_client = None
        self._gemini_client = None
        self._mlx_model = None
        self._mlx_tokenizer = None

    @property
    def deepseek_client(self):
        """Lazy-load DeepSeek client."""
        if self._deepseek_client is None and self.deepseek_api_key:
            from openai import OpenAI

            self._deepseek_client = OpenAI(
                api_key=self.deepseek_api_key,
                base_url="https://api.deepseek.com",
            )
        return self._deepseek_client

    @property
    def gemini_client(self):
        """Lazy-load Gemini client."""
        if self._gemini_client is None and self.gemini_api_key:
            from google import genai

            self._gemini_client = genai.Client(api_key=self.gemini_api_key)
        return self._gemini_client

    def load_mlx_model(self):
        """Lazy-load MLX model (Qwen3-VL).

        Returns:
            Tuple of (model, tokenizer) or (None, None) if model path not set
        """
        if self._mlx_model is None and self.mlx_model_path:
            try:
                from mlx_lm import load

                logger.info(f"Loading MLX model from {self.mlx_model_path}")
                self._mlx_model, self._mlx_tokenizer = load(self.mlx_model_path)
                logger.info("MLX model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load MLX model: {e}")
                return None, None

        return self._mlx_model, self._mlx_tokenizer

    def extract_with_ollama(self, doc_id: str, doc_type: str, text: str) -> ExtractionResult:
        """Extract entities using Ollama (MiniCPM-V).

        Args:
            doc_id: Document ID
            doc_type: Document type
            text: OCR text

        Returns:
            ExtractionResult with extracted entities
        """
        start_time = time.time()

        # Truncate text to fit context (8K for speed)
        truncated_text = text[:8000]

        # Build prompt
        prompt = TEXT_ENTITY_EXTRACTION_PROMPT.format(
            doc_id=doc_id,
            doc_type=doc_type,
            text=truncated_text,
        )

        # Add JSON instruction
        prompt += "\n\nRespond ONLY with valid JSON. Do not include any other text."

        try:
            # Generate response using Ollama
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 2048},
            )

            response_text = response["message"]["content"].strip()

            # Parse JSON from response
            # Sometimes models add extra text, so extract JSON block
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
                response_text = response_text[json_start:json_end].strip()

            parsed = json.loads(response_text)

            processing_time = time.time() - start_time

            return ExtractionResult.from_parsed_json(
                doc_id=doc_id,
                doc_type=doc_type,
                parsed=parsed,
                provider="ollama-minicpm-v",
                processing_time=processing_time,
                raw_response=response_text,
            )

        except json.JSONDecodeError as e:
            logger.error(f"{doc_id} (Ollama): JSON parse error: {e}")
            raise ValueError(f"Ollama returned invalid JSON: {e}")
        except Exception as e:
            logger.error(f"{doc_id} (Ollama): Extraction failed: {e}")
            raise

    def extract_with_mlx(self, doc_id: str, doc_type: str, text: str) -> ExtractionResult:
        """Extract entities using MLX-loaded model (e.g., Qwen3-VL).

        Args:
            doc_id: Document ID
            doc_type: Document type
            text: OCR text

        Returns:
            ExtractionResult with extracted entities
        """
        model, tokenizer = self.load_mlx_model()

        if model is None or tokenizer is None:
            raise ValueError("MLX model not loaded. Set mlx_model_path in constructor.")

        start_time = time.time()

        # Truncate text to fit context (8K for speed)
        truncated_text = text[:8000]

        # Build prompt
        prompt = TEXT_ENTITY_EXTRACTION_PROMPT.format(
            doc_id=doc_id,
            doc_type=doc_type,
            text=truncated_text,
        )

        # Add JSON instruction
        prompt += "\n\nRespond ONLY with valid JSON. Do not include any other text."

        try:
            from mlx_lm import generate

            # Generate response using MLX
            # Note: mlx-lm doesn't support temperature parameter in generate()
            # It uses a sampler instead, but we'll use default settings
            response_text = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=2048,
                verbose=False,
            )

            # Clean up response
            response_text = response_text.strip()

            # Parse JSON from response
            # Sometimes models add extra text, so extract JSON block
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            # Try to find JSON object boundaries if no code blocks
            if "{" in response_text:
                # Find the first { and try to parse from there
                json_start = response_text.find("{")
                # Try to find matching closing brace by counting braces
                brace_count = 0
                json_end = json_start
                for i in range(json_start, len(response_text)):
                    if response_text[i] == "{":
                        brace_count += 1
                    elif response_text[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break

                if json_end > json_start:
                    response_text = response_text[json_start:json_end]

            parsed = json.loads(response_text)

            processing_time = time.time() - start_time

            # Extract model name from path for provider string
            model_name = Path(self.mlx_model_path).name if self.mlx_model_path else "mlx"

            return ExtractionResult.from_parsed_json(
                doc_id=doc_id,
                doc_type=doc_type,
                parsed=parsed,
                provider=f"mlx-{model_name}",
                processing_time=processing_time,
                raw_response=response_text,
            )

        except json.JSONDecodeError as e:
            logger.error(f"{doc_id} (MLX): JSON parse error: {e}")
            raise ValueError(f"MLX model returned invalid JSON: {e}")
        except Exception as e:
            logger.error(f"{doc_id} (MLX): Extraction failed: {e}")
            raise

    def extract_with_deepseek(self, doc_id: str, doc_type: str, text: str) -> ExtractionResult:
        """Extract entities using DeepSeek API.

        Args:
            doc_id: Document ID
            doc_type: Document type
            text: OCR text

        Returns:
            ExtractionResult with extracted entities
        """
        if not self.deepseek_client:
            raise ValueError("DeepSeek API key not configured")

        start_time = time.time()
        truncated_text = text[:8000]

        prompt = TEXT_ENTITY_EXTRACTION_PROMPT.format(
            doc_id=doc_id,
            doc_type=doc_type,
            text=truncated_text,
        )

        try:
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a forensic document analyst. "
                            "Respond only with valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            raw_response = response.choices[0].message.content
            parsed = json.loads(raw_response)
            processing_time = time.time() - start_time

            return ExtractionResult.from_parsed_json(
                doc_id=doc_id,
                doc_type=doc_type,
                parsed=parsed,
                provider="deepseek-chat",
                processing_time=processing_time,
                raw_response=raw_response,
            )

        except Exception as e:
            logger.error(f"{doc_id} (DeepSeek): Extraction failed: {e}")
            raise

    def extract_with_gemini(self, doc_id: str, doc_type: str, text: str) -> ExtractionResult:
        """Extract entities using Gemini 2.5 Flash API.

        Args:
            doc_id: Document ID
            doc_type: Document type
            text: OCR text

        Returns:
            ExtractionResult with extracted entities
        """
        if not self.gemini_client:
            raise ValueError("Gemini API key not configured")

        start_time = time.time()
        truncated_text = text[:8000]

        prompt = TEXT_ENTITY_EXTRACTION_PROMPT.format(
            doc_id=doc_id,
            doc_type=doc_type,
            text=truncated_text,
        )

        try:
            from google.genai import types

            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )

            raw_response = response.text
            parsed = json.loads(raw_response)
            processing_time = time.time() - start_time

            return ExtractionResult.from_parsed_json(
                doc_id=doc_id,
                doc_type=doc_type,
                parsed=parsed,
                provider="gemini-2.5-flash",
                processing_time=processing_time,
                raw_response=raw_response,
            )

        except Exception as e:
            logger.error(f"{doc_id} (Gemini): Extraction failed: {e}")
            raise

    def extract(self, doc_id: str, doc_type: str, text: str) -> ExtractionResult:
        """Extract entities from a document using the first available provider.

        Tries providers in order: Ollama -> DeepSeek -> Gemini

        Args:
            doc_id: Document ID
            doc_type: Document type
            text: OCR text

        Returns:
            ExtractionResult from first successful provider
        """
        # Try Ollama first (local, always available)
        try:
            return self.extract_with_ollama(doc_id, doc_type, text)
        except Exception as e:
            logger.warning(f"{doc_id}: Ollama extraction failed: {e}")

        # Try DeepSeek if available
        if self.deepseek_api_key:
            try:
                return self.extract_with_deepseek(doc_id, doc_type, text)
            except Exception as e:
                logger.warning(f"{doc_id}: DeepSeek extraction failed: {e}")

        # Try Gemini as fallback
        if self.gemini_api_key:
            try:
                return self.extract_with_gemini(doc_id, doc_type, text)
            except Exception as e:
                logger.error(f"{doc_id}: Gemini extraction failed: {e}")
                raise

        raise RuntimeError(f"{doc_id}: All extraction providers failed")


def extract_batch_parallel(
    processed_dir: Path,
    extracted_dir: Path,
    ollama_model: str = "minicpm-v:8b",
    ollama_host: str = "http://localhost:11434",
    deepseek_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    num_workers: int = 3,
    resume: bool = True,
) -> dict:
    """Extract entities from all documents using parallel workers.

    Each worker gets documents from a queue and processes them with the
    multi-provider extractor.

    Args:
        processed_dir: Directory with OCR output JSONs
        extracted_dir: Directory to save extraction results
        ollama_model: Ollama model to use for local extraction
        ollama_host: Ollama server URL
        deepseek_api_key: Optional DeepSeek API key
        gemini_api_key: Optional Gemini API key
        num_workers: Number of parallel workers
        resume: Skip already-extracted documents

    Returns:
        Stats dict with counts and provider usage
    """
    # Ensure output directory exists
    extracted_dir.mkdir(parents=True, exist_ok=True)

    # Find all OCR JSONs
    json_files = list(processed_dir.glob("*.json"))
    json_files = [f for f in json_files if not f.name.endswith(".error.json")]

    # Filter out already-extracted documents if resuming
    if resume:
        to_process = [f for f in json_files if not (extracted_dir / f"{f.stem}.json").exists()]
    else:
        to_process = json_files

    stats = {
        "total": len(json_files),
        "processed": 0,
        "skipped": len(json_files) - len(to_process),
        "failed": 0,
        "failed_docs": [],
        "provider_stats": {"ollama": 0, "deepseek": 0, "gemini": 0},
        "total_time": 0.0,
    }

    if len(to_process) == 0:
        logger.info("All documents already extracted")
        return stats

    # Initialize extractor
    extractor = MultiProviderExtractor(
        ollama_model=ollama_model,
        ollama_host=ollama_host,
        deepseek_api_key=deepseek_api_key,
        gemini_api_key=gemini_api_key,
        num_workers=num_workers,
    )

    def process_document(json_file: Path) -> tuple[str, Optional[ExtractionResult], Optional[str]]:
        """Process a single document.

        Returns:
            (doc_id, result, error)
        """
        doc_id = json_file.stem
        try:
            # Load OCR result
            data = json.loads(json_file.read_text())
            text = data.get("text", "")
            doc_type = data.get("track", "text")  # Use "track" field from OCR output

            # Extract entities
            result = extractor.extract(doc_id, doc_type, text)

            # Write result
            output_path = extracted_dir / f"{doc_id}.json"
            output_path.write_text(json.dumps(asdict(result), indent=2))

            return (doc_id, result, None)

        except Exception as e:
            error_msg = str(e)
            error_path = extracted_dir / f"{doc_id}.error.json"
            error_path.write_text(json.dumps({"doc_id": doc_id, "error": error_msg}, indent=2))
            return (doc_id, None, error_msg)

    # Process documents in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_document, json_file): json_file for json_file in to_process
        }

        with tqdm(total=len(to_process), desc="Extracting entities") as pbar:
            for future in as_completed(futures):
                doc_id, result, error = future.result()

                if error:
                    stats["failed"] += 1
                    stats["failed_docs"].append(doc_id)
                    logger.error(f"{doc_id}: Extraction failed: {error}")
                else:
                    stats["processed"] += 1
                    provider = result.provider
                    stats["provider_stats"][provider] = stats["provider_stats"].get(provider, 0) + 1
                    stats["total_time"] += result.processing_time
                    logger.info(f"{doc_id}: Success ({provider}, {result.processing_time:.1f}s)")

                pbar.update(1)

    # Log summary
    logger.info("\nExtraction complete:")
    logger.info(f"  Total: {stats['total']}")
    logger.info(f"  Processed: {stats['processed']}")
    logger.info(f"  Skipped: {stats['skipped']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info("\nProvider usage:")
    for provider, count in stats["provider_stats"].items():
        if count > 0:
            logger.info(f"  {provider}: {count}")

    return stats
