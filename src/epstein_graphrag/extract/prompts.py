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
Provide a forensic analysis including:

1. Scene description: Describe the location, setting, indoor/outdoor, room type
2. Objects detected: List all significant objects, furnishings, devices, documents visible
3. Anomalies noted: Unusual features, security measures, modifications, or concerning elements
4. Faces detected: Number and description of people visible (DO NOT attempt facial recognition)
5. Visible text: Any readable text on signs, labels, documents
6. Estimated time period: Based on visual cues (furnishings, technology, etc.)

Return ONLY valid JSON in this exact format:
{
  "scene_description": "...",
  "objects_detected": ["object1", "object2", ...],
  "anomalies_noted": ["anomaly1", "anomaly2", ...],
  "faces_detected": [],
  "visible_text": ["text1", "text2", ...],
  "estimated_period": "...",
  "evidence_relevance": "high|medium|low",
  "analysis_notes": "Additional context or observations"
}"""
