"""Extraction prompt templates.

All prompts used for entity extraction are defined here.
Prompts contain no logic — only text templates.
"""

TEXT_ENTITY_EXTRACTION_PROMPT = """You are a forensic document analyst extracting structured evidence from legal documents related to the Jeffrey Epstein case.

DOCUMENT ID: {doc_id}
DOCUMENT TYPE: {doc_type}
DOCUMENT TEXT:
---
{text}
---

Extract ALL entities and relationships from this document. Be thorough — every name, date, location, and event matters.

Respond in this exact JSON format:
{{
  "people": [
    {{
      "name": "Full Name as written",
      "aliases": ["any other names used for this person in the document"],
      "role": "perpetrator|victim|witness|associate|legal|law_enforcement|unknown",
      "context": "One sentence explaining why this person appears in this document",
      "excerpt": "Exact quote from the document mentioning this person"
    }}
  ],
  "locations": [
    {{
      "name": "Location name",
      "address": "Full address if available, otherwise empty string",
      "location_type": "residence|island|airport|office|school|hotel|vehicle|unknown",
      "context": "Why this location is mentioned",
      "excerpt": "Exact quote mentioning this location"
    }}
  ],
  "organizations": [
    {{
      "name": "Organization name",
      "org_type": "company|foundation|school|government|legal|financial|unknown",
      "context": "Why this organization is mentioned",
      "excerpt": "Exact quote"
    }}
  ],
  "events": [
    {{
      "event_type": "flight|meeting|transaction|phone_call|visit|assault|arrest|testimony|unknown",
      "date": "YYYY-MM-DD if available, otherwise approximate or empty string",
      "description": "What happened",
      "participants": ["Person Name 1", "Person Name 2"],
      "location": "Location name if mentioned",
      "excerpt": "Exact quote describing this event"
    }}
  ],
  "allegations": [
    {{
      "description": "What is being alleged",
      "accused": ["Person Name"],
      "victims": ["Person Name if mentioned"],
      "severity": "critical|severe|moderate|minor",
      "status": "confirmed_by_court|alleged_under_oath|alleged|disputed|rumored",
      "excerpt": "Exact quote supporting this allegation"
    }}
  ],
  "associations": [
    {{
      "person_a": "Person Name",
      "person_b": "Person Name",
      "nature": "employer|friend|associate|co-conspirator|victim-perpetrator|legal|familial|unknown",
      "timeframe": "Date range or description if available",
      "excerpt": "Exact quote establishing this connection"
    }}
  ]
}}

RULES:
1. Extract EVERY person mentioned, even in passing. Every name matters.
2. Include exact quotes (excerpts) from the document for EVERY extracted entity.
3. If a person's role is unclear, use "unknown" — do not guess.
4. If a date is approximate, prefix with "~" (e.g., "~1999-06").
5. For allegations, distinguish between court-confirmed facts and unverified claims.
6. Extract associations between people even if the nature is unclear.
7. Do NOT hallucinate entities that are not in the document text.
8. If the document contains no extractable entities, return empty arrays for all fields.
"""

PHOTO_ENTITY_EXTRACTION_PROMPT = """You are a forensic evidence analyst reviewing the analysis of a photograph from the Jeffrey Epstein case.

DOCUMENT ID: {doc_id}
VISION ANALYSIS:
---
Scene: {scene_description}
Objects: {objects_detected}
Anomalies: {anomalies_noted}
Faces: {faces_detected}
Visible Text: {visible_text}
Estimated Period: {estimated_period}
Evidence Relevance: {evidence_relevance}
---

Based on this analysis, extract any identifiable entities:

{{
  "locations": [
    {{
      "name": "Best guess at location name based on visual clues",
      "location_type": "Type of space shown",
      "context": "What this location appears to be based on the photograph",
      "visual_clues": "What in the photo suggests this identification"
    }}
  ],
  "objects_of_interest": [
    {{
      "description": "Object description",
      "evidence_relevance": "Why this object might be significant",
      "location_in_frame": "Where in the photo this appears"
    }}
  ],
  "potential_allegations_supported": [
    {{
      "description": "What allegation this visual evidence might support",
      "visual_basis": "What in the photo supports this",
      "confidence": "high|medium|low"
    }}
  ]
}}

RULES:
1. Only extract what the visual analysis actually describes. Do not infer beyond the evidence.
2. If faces are detected, note them but do NOT attempt to identify anyone.
3. Focus on anomalies — these are the most evidentiary elements.
4. Note any objects that could corroborate victim testimony.
"""

# Legacy visual analysis prompt (used in OCR pipeline)
VISUAL_ANALYSIS_PROMPT = """Analyze this photograph from the Epstein case evidence files.
Provide a forensic analysis including:

1. Scene description: Describe the location, setting, indoor/outdoor, room type
2. Objects detected: List all significant objects, furnishings, devices, documents visible
3. Anomalies noted: Unusual features, security measures, modifications, or concerning elements
4. Faces detected: Number and description of people visible (DO NOT attempt facial recognition)
5. Visible text: Any readable text on signs, labels, documents
6. Estimated time period: Based on visual cues (furnishings, technology, etc.)

Return ONLY valid JSON in this exact format:
{{
  "scene_description": "...",
  "objects_detected": ["object1", "object2", ...],
  "anomalies_noted": ["anomaly1", "anomaly2", ...],
  "faces_detected": [],
  "visible_text": ["text1", "text2", ...],
  "estimated_period": "...",
  "evidence_relevance": "high|medium|low",
  "analysis_notes": "Additional context or observations"
}}"""
