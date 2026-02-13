"""Forensic-aware Marker LLM processor for legal document analysis.

This processor injects forensic extraction context into Marker's block-level
LLM processing, ensuring the OCR model is aware of the evidentiary value
of the content while processing each block.

The processor extends Marker's BaseLLMSimpleBlockProcessor to add domain-
specific prompts for forensic document analysis in legal cases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import markdown2
from pydantic import BaseModel

# Type-only imports for proper type hints (Pylance needs these outside try/except)
if TYPE_CHECKING:
    from marker.processors.llm import BaseLLMSimpleBlockProcessor, PromptData
    from marker.schema import BlockTypes
    from marker.schema.document import Document

# Make Marker imports optional (for Python 3.14 compatibility)
try:
    from marker.processors.llm import BaseLLMSimpleBlockProcessor, PromptData
    from marker.schema import BlockTypes
    from marker.schema.document import Document

    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False
    # Create placeholder types for runtime when Marker is not available
    # Use underscore suffix to avoid shadowing TYPE_CHECKING imports
    PromptData_ = dict  # type: ignore
    BaseLLMSimpleBlockProcessor_ = object  # type: ignore
    Document_ = object  # type: ignore
    BlockTypes_ = object  # type: ignore


# Forensic context constants
FORENSIC_CONTEXT = """FORENSIC DOCUMENT ANALYSIS CONTEXT:
- Case: Jeffrey Epstein investigation
- Purpose: Extract evidence to corroborate victim testimony
- Goal: Preserve all evidentiary information with high accuracy

ENTITY TYPES TO PRESERVE:
- People: names, aliases, roles (perpetrator, victim, witness, associate, legal, law_enforcement)
- Locations: residences, islands, airports, offices, schools, hotels, vehicles, yachts, properties
- Organizations: companies, foundations, schools,
government agencies, legal firms, financial institutions
- Events: flights, meetings, transactions,
phone calls, visits, assaults, arrests, testimony, court hearings
- Dates: specific dates, approximate dates, time periods
- Identifiers: document numbers, account numbers, phone numbers, addresses, passport numbers

CRITICAL INVESTIGATION CONTEXT:
1. CODE WORDS AND EUPHEMISMS: This investigation
involves evidence where criminal activity (including child exploitation)
is discussed using code words, euphemisms, or seemingly
    innocent phrases.
   - Examples: "cheese pizza" (C.P. = child pornography), "pizza", "hot dogs", "walnut sauce",
   "pasta", "jerky" and other food terms
   - Code words often appear OUT OF CONTEXT compared to normal sentence structure
   - PRESERVE ALL instances of suspicious language, euphemisms, or odd phrases
   - Note any language that feels forced, unnatural, or strategically placed
   - Food terms appearing in non-food contexts are HIGHLY SUSPICIOUS

2. DISTURBING CONTENT: Some documents may contain explicit references to abuse, exploitation, or
illegal activities.
   - DO NOT censor, skip, or sanitize any content regardless of how disturbing
   - This is the damning evidence we need to uncover
   - Preserve all text exactly as written

3. [REDACTED] regions: Note surrounding context carefully -
redactions often hide the most damning evidence

4. SUPERFICIAL REDACTIONS (CRITICAL):
Many redactions are POORLY DONE - text may still be visible beneath black marker
    - If you can see text beneath a redaction,
    extract BOTH: note the redaction AND the visible text
    - Poorly redacted text reveals exactly what
    they tried to hide - this is DAMNING EVIDENCE
    - Look for faint text, ghostly outlines, or text showing through thin markers

SPECIAL HANDLING:
- Identity documents: Extract ALL visible information including document numbers, dates, names
- Names: Transcribe exactly as written, including titles and suffixes
- Relationships: Note associations between people (employer, friend, associate, etc.)
- Tables: Ensure all columns and rows are accurately transcribed
- Forms: Preserve all field labels and values"""


# Block-specific forensic instructions
BLOCK_INSTRUCTIONS = {
    "Text": "Focus on preserving names, dates, locations, phone numbers, and "
    "relationships. Watch for code words, euphemisms, or suspicious language "
    "that appears OUT OF CONTEXT (e.g., food terms in non-food conversations).",
    "SectionHeader": "Preserve document titles, section headings, page numbers, and any "
    "identifying information.",
    "Form": "CRITICAL: Preserve all form fields, labels, and filled values. "
    "Forms often contain key evidentiary information. Note any suspicious "
    "entries or code words.",
    "Table": "CRITICAL: Ensure all table cells are accurately transcribed. "
    "Tables often contain flight logs, call records, financial data. Watch for "
    "suspicious patterns or code words in table contents.",
    "TableOfContents": "Preserve document structure and section references for navigation.",
    "List": "Preserve all list items - these may contain names, dates, or other "
    "evidence. Watch for suspicious entries or code words.",
}


class ForensicSchema(BaseModel):
    """Schema for forensic block correction response."""

    corrected_markdown: str
    forensic_notes: str = ""
    entities_detected: str = ""


def _get_block_instruction(block_type_str: str) -> str:
    """Get block-specific forensic extraction instructions."""
    # Extract just the type name from the full BlockType string
    if "BlockTypes." in block_type_str:
        type_name = block_type_str.split(".")[-1]
    else:
        type_name = block_type_str

    return BLOCK_INSTRUCTIONS.get(type_name, "")


def _get_forensic_prompt(extracted_text: str, block_type_str: str) -> str:
    """Generate forensic-aware OCR prompt for a block."""
    block_instruction = _get_block_instruction(block_type_str)

    prompt = f"""You are a forensic document analyst extracting evidence from legal documents.

{FORENSIC_CONTEXT}

{block_instruction}

TEXT EXTRACTION RULES:
- Preserve original formatting, layout, and structure
- Include ALL text content - do not omit anything
- Do NOT self-censor or omit any visible information
- For [REDACTED] markers, preserve them and note surrounding context
- For tables/forms, ensure all cells are accurately transcribed
- Maintain reading order

**CRITICAL: CODE WORDS AND EUPHEMISMS**
- This investigation involves evidence where criminal activity is discussed using code words
- Examples: "cheese pizza" (C.P. = child pornography), "pizza", "hot dogs", "walnut sauce", "pasta"
- Code words appear OUT OF CONTEXT - food terms in non-food conversations
- PRESERVE ALL suspicious language, euphemisms, or odd phrases exactly as written
- DO NOT sanitize or omit any content regardless of how disturbing - this is the damning evidence

**Instructions:**
1. Carefully examine the provided block image
2. Analyze the existing text representation
3. Generate the corrected markdown representation of the content
4. Note any forensic observations (names, dates, locations identified)

**Formatting should be in markdown:**
- * for italics, ** for bold, and ` for inline code
- Use <sup>...</sup> for superscripts
- Headers should be formatted with #, with one # for the largest header
- Lists should be formatted with either - or 1. for unordered and ordered lists
- Tables should be formatted with markdown tables, with the headers bolded

**Input:**
```text
{extracted_text}
```

**Output Format:**
Respond with a JSON object containing:
- corrected_markdown: The corrected markdown representation
- forensic_notes: Any notable observations (names, dates, locations detected)
- entities_detected: Brief list of key entity types found (people, locations, dates)

If the text extraction is already accurate and complete, return it unchanged
with a note "No corrections needed" in forensic_notes.
"""

    return prompt


class ForensicBlockProcessor(
    BaseLLMSimpleBlockProcessor if MARKER_AVAILABLE else object  # type: ignore
):
    """Forensic-aware processor for text blocks in legal documents.

    This processor extends Marker's default LLM processing with domain-specific
    context for forensic document analysis. It processes Text, SectionHeader,
    Form, Table, and List blocks with awareness of the evidentiary value.

    The processor injects forensic context into the LLM prompt and captures
    forensic observations in the block metadata.

    Attributes:
        block_types: Tuple of BlockTypes to process with forensic context
        forensic_context_enabled: Whether to inject forensic analysis context
    """

    def __init__(self, llm_service=None, config=None, forensic_context_enabled: bool = True):
        """Initialize the forensic block processor.

        Args:
            llm_service: Marker LLM service instance
            config: Processor configuration
            forensic_context_enabled: Whether to enable forensic context injection
        """
        if not MARKER_AVAILABLE:
            raise ImportError("Marker is not available. Cannot use ForensicBlockProcessor.")

        super().__init__(llm_service, config)
        self.forensic_context_enabled = forensic_context_enabled

    # Define block types to process - override in subclass if needed
    @property
    def block_types(self):
        if MARKER_AVAILABLE:
            return (
                BlockTypes.Text,  # type: ignore[attr-defined]
                BlockTypes.SectionHeader,  # type: ignore[attr-defined]
                BlockTypes.Form,  # type: ignore[attr-defined]
                BlockTypes.Table,  # type: ignore[attr-defined]
                BlockTypes.List,  # type: ignore[attr-defined]
            )
        else:
            # Return simple tuple of type names when Marker is not available
            return ("Text", "SectionHeader", "Form", "Table", "List")

    def block_prompts(self, document: Document) -> List[PromptData]:
        """Generate prompts for all blocks to be processed.

        Args:
            document: The Marker Document object

        Returns:
            List of PromptData dictionaries with prompt, image, block, schema, page
        """
        if not self.forensic_context_enabled:
            # Fall back to standard processing without forensic context
            return self._standard_block_prompts(document)

        prompt_data = []
        for block in self.inference_blocks(document):
            text = block["block"].raw_text(document)
            block_type_str = str(block["block"].id.block_type)

            # Generate forensic-aware prompt
            prompt = _get_forensic_prompt(text, block_type_str)
            image = self.extract_image(document, block["block"])

            # Add forensic context to additional_data
            additional_data = {
                "forensic_context": True,
                "block_type": block_type_str,
                "case_context": "Jeffrey Epstein investigation",
            }

            prompt_data.append(
                {
                    "prompt": prompt,
                    "image": image,
                    "block": block["block"],
                    "schema": ForensicSchema,
                    "page": block["page"],
                    "additional_data": additional_data,
                }
            )

        return prompt_data

    def _standard_block_prompts(self, document: Document) -> List[PromptData]:
        """Standard block prompts without forensic context (fallback).

        This is used when forensic_context_enabled is False, providing
        standard high-quality OCR without domain-specific context.
        """
        prompt_data = []
        for block in self.inference_blocks(document):
            text = block["block"].raw_text(document)

            prompt = f"""You are a text correction expert specializing in accurately
reproducing text from images. You will receive an image of a text block and the text
that can be extracted from the image.
Your task is to generate markdown to properly represent the content of the image.

Formatting should be in markdown, with the following rules:
- * for italics, ** for bold, and ` for inline code
- Headers should be formatted with #, with one # for the largest header
- Lists should be formatted with either - or 1. for unordered and ordered lists
- Tables should be formatted with markdown tables, with the headers bolded

**Input:**
```text
{text}
```
"""

            image = self.extract_image(document, block["block"])

            prompt_data.append(
                {
                    "prompt": prompt,
                    "image": image,
                    "block": block["block"],
                    "schema": ForensicSchema,
                    "page": block["page"],
                    "additional_data": {"forensic_context": False},
                }
            )

        return prompt_data

    def rewrite_block(self, response: dict, prompt_data: PromptData, document: Document):
        """Rewrite a block based on LLM response.

        Args:
            response: The LLM response dict with corrected_markdown
            prompt_data: The original prompt data
            document: The Marker Document object
        """
        block = prompt_data["block"]
        text = block.raw_text(document)

        if not response or "corrected_markdown" not in response:
            block.update_metadata(llm_error_count=1)
            return

        corrected_markdown = response["corrected_markdown"]

        # The original text is okay
        if "no corrections needed" in corrected_markdown.lower():
            return

        # Potentially a partial response - too short compared to original
        if len(corrected_markdown) < len(text) * 0.5:
            block.update_metadata(llm_error_count=1)
            return

        # Clean up markdown formatting
        corrected_markdown = corrected_markdown.strip().lstrip("```markdown").rstrip("```").strip()

        # Convert LLM markdown to HTML
        try:
            block.html = markdown2.markdown(corrected_markdown, extras=["tables"])
        except Exception as e:
            # Fall back to original text if markdown conversion fails
            block.update_metadata(llm_error_count=1, markdown_error=str(e))
            return

        # Store forensic analysis in block metadata
        if self.forensic_context_enabled:
            forensic_notes = response.get("forensic_notes", "")
            entities_detected = response.get("entities_detected", "")

            block.update_metadata(
                forensic_notes=forensic_notes,
                entities_detected=entities_detected,
                forensic_processed=True,
            )


# Factory function for easy processor creation
def create_forensic_processor(
    llm_service, forensic_context_enabled: bool = True
) -> ForensicBlockProcessor:
    """Factory function to create a forensic block processor.

    Args:
        llm_service: Marker LLM service instance
        forensic_context_enabled: Whether to enable forensic context injection

    Returns:
        Configured ForensicBlockProcessor instance

    Raises:
        ImportError: If Marker is not available
    """
    if not MARKER_AVAILABLE:
        raise ImportError("Marker is not available. Cannot create ForensicBlockProcessor.")

    return ForensicBlockProcessor(
        llm_service=llm_service,
        forensic_context_enabled=forensic_context_enabled,
    )
