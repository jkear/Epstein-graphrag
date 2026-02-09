"""Extraction prompt templates.

All prompts used for entity extraction are defined here.
Prompts contain no logic — only text templates.
"""

ENTITY_EXTRACTION_SYSTEM = """You are an expert legal document analyst specializing in evidence
analysis. Extract all entities and relationships from the provided document text.

Return valid JSON with the following structure:
{
  "persons": [{"name": "...", "role": "...", "aliases": [...]}],
  "organizations": [{"name": "...", "type": "..."}],
  "locations": [{"name": "...", "type": "...", "address": "..."}],
  "events": [{"description": "...", "date": "...", "location": "...", "participants": [...]}],
  "allegations": [{"description": "...", "victim": "...", "accused": "...", "date": "..."}],
  "relationships": [{"source": "...", "target": "...", "type": "...", "context": "..."}]
}

Be thorough. Extract every person, place, date, and allegation mentioned.
Preserve exact names as they appear in the document.
"""

ENTITY_EXTRACTION_USER = """Analyze the following document and extract all entities and
relationships. Document ID: {doc_id}

---
{text}
---

Extract all persons, organizations, locations, events, allegations, and relationships.
Return valid JSON only."""

VISUAL_ANALYSIS_PROMPT = """Analyze this photograph from the Epstein case evidence files.
Describe:
1. Scene description (location type, setting, indoor/outdoor)
2. People visible (number, apparent roles, any identifiable features)
3. Objects of interest (documents, devices, furnishings)
4. Any visible text (signs, labels, documents)
5. Estimated time period based on visual cues

Return JSON:
{{
  "scene_description": "...",
  "people_count": 0,
  "people_descriptions": ["..."],
  "objects_detected": ["..."],
  "visible_text": ["..."],
  "estimated_period": "...",
  "evidence_relevance": "high|medium|low",
  "notes": "..."
}}"""
