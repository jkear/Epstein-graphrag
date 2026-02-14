"""Schema validation for entity extraction results.

Validates that extracted entities conform to expected types and values.
Logs alerts for unexpected entity types or malformed data.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from epstein_graphrag.extract.multi_provider_extractor import ExtractionResult

logger = logging.getLogger(__name__)

# Allowed values for enum-like fields
ALLOWED_VALUES = {
    "person_role": {
        "perpetrator",
        "victim",
        "witness",
        "associate",
        "legal",
        "law_enforcement",
        "unknown",
    },
    "location_type": {
        "residence",
        "island",
        "airport",
        "office",
        "school",
        "hotel",
        "vehicle",
        "yacht",
        "property",
        "unknown",
    },
    "org_type": {
        "company",
        "foundation",
        "school",
        "government",
        "legal",
        "financial",
        "unknown",
    },
    "event_type": {
        "flight",
        "meeting",
        "transaction",
        "phone_call",
        "visit",
        "assault",
        "arrest",
        "testimony",
        "court_hearing",
        "deposition",
        "travel",
        "unknown",
    },
    "allegation_severity": {"critical", "severe", "moderate", "minor"},
    "allegation_status": {
        "confirmed_by_court",
        "alleged_under_oath",
        "alleged",
        "disputed",
        "rumored",
    },
    "association_nature": {
        "employer",
        "friend",
        "associate",
        "co-conspirator",
        "victim-perpetrator",
        "legal",
        "familial",
        "unknown",
    },
    "identity_document_type": {
        "passport",
        "drivers_license",
        "birth_certificate",
        "ssn_card",
        "visa",
        "id_card",
        "unknown",
    },
    "communication_type": {
        "email",
        "phone_call",
        "text_message",
        "letter",
        "fax",
        "unknown",
    },
    "legal_document_type": {
        "subpoena",
        "deposition",
        "warrant",
        "affidavit",
        "court_order",
        "motion",
        "indictment",
        "plea_agreement",
        "testimony",
        "unknown",
    },
    "transaction_type": {
        "wire_transfer",
        "check",
        "cash",
        "credit_card",
        "payment",
        "unknown",
    },
    "physical_evidence_type": {
        "document",
        "photograph",
        "recording",
        "device",
        "vehicle",
        "clothing",
        "other",
    },
    "redaction_type": {
        "name",
        "address",
        "phone",
        "email",
        "account_number",
        "identifier",
        "unknown",
    },
    "confidence": {"high", "medium", "low"},
}


class SchemaValidator:
    """Validates extraction results against expected schema."""

    def __init__(self, alert_file: Path | None = None):
        """Initialize validator.

        Args:
            alert_file: Path to write schema violation alerts (default: data/schema_alerts.jsonl)
        """
        self.alert_file = alert_file or Path("data/schema_alerts.jsonl")
        self.alert_file.parent.mkdir(parents=True, exist_ok=True)

    def validate(self, result: ExtractionResult) -> tuple[bool, list[str]]:
        """Validate extraction result against schema.

        Args:
            result: ExtractionResult to validate

        Returns:
            (is_valid, violations): True if valid, list of violation messages
        """
        violations = []

        # Validate people
        for person in result.people:
            if "role" in person:
                role_val = person["role"]
                # Handle pipe-separated multi-roles (legacy from old extractions)
                roles = [r.strip() for r in role_val.split("|")]
                for role in roles:
                    if role not in ALLOWED_VALUES["person_role"]:
                        violations.append(
                            f"Invalid person.role: '{role}' in doc {result.doc_id}"
                        )

        # Validate locations
        for loc in result.locations:
            if "location_type" in loc:
                loc_type = loc["location_type"]
                # Handle pipe-separated multi-types
                types = [t.strip() for t in loc_type.split("|")]
                for t in types:
                    if t not in ALLOWED_VALUES["location_type"]:
                        violations.append(
                            f"Invalid location.location_type: '{t}' in doc {result.doc_id}"
                        )

        # Validate organizations
        for org in result.organizations:
            if "org_type" in org:
                org_type = org["org_type"]
                types = [t.strip() for t in org_type.split("|")]
                for t in types:
                    if t not in ALLOWED_VALUES["org_type"]:
                        violations.append(
                            f"Invalid organization.org_type: '{t}' in doc {result.doc_id}"
                        )

        # Validate events
        for event in result.events:
            if "event_type" in event:
                event_type = event["event_type"]
                types = [t.strip() for t in event_type.split("|")]
                for t in types:
                    if t not in ALLOWED_VALUES["event_type"]:
                        violations.append(
                            f"Invalid event.event_type: '{t}' in doc {result.doc_id}"
                        )

        # Validate allegations
        for allegation in result.allegations:
            if "severity" in allegation:
                severity = allegation["severity"]
                severities = [s.strip() for s in severity.split("|")]
                for s in severities:
                    if s and s not in ALLOWED_VALUES["allegation_severity"]:
                        violations.append(
                            f"Invalid allegation.severity: '{s}' in doc {result.doc_id}"
                        )

            if "status" in allegation:
                status = allegation["status"]
                statuses = [s.strip() for s in status.split("|")]
                for s in statuses:
                    if s not in ALLOWED_VALUES["allegation_status"]:
                        violations.append(
                            f"Invalid allegation.status: '{s}' in doc {result.doc_id}"
                        )

        # Validate associations
        for assoc in result.associations:
            if "nature" in assoc:
                nature = assoc["nature"]
                if nature not in ALLOWED_VALUES["association_nature"]:
                    violations.append(
                        f"Invalid association.nature: '{nature}' in doc {result.doc_id}"
                    )

        # Validate identity documents
        for doc in result.identity_documents:
            if "document_type" in doc:
                doc_type = doc["document_type"]
                if doc_type not in ALLOWED_VALUES["identity_document_type"]:
                    violations.append(
                        f"Invalid identity_document.document_type: '{doc_type}' "
                        f"in doc {result.doc_id}"
                    )

        # Validate communications
        for comm in result.communications:
            if "communication_type" in comm:
                comm_type = comm["communication_type"]
                if comm_type not in ALLOWED_VALUES["communication_type"]:
                    violations.append(
                        f"Invalid communication.communication_type: '{comm_type}' "
                        f"in doc {result.doc_id}"
                    )

        # Validate legal documents
        for legal_doc in result.legal_documents:
            if "document_type" in legal_doc:
                doc_type = legal_doc["document_type"]
                if doc_type not in ALLOWED_VALUES["legal_document_type"]:
                    violations.append(
                        f"Invalid legal_document.document_type: '{doc_type}' in doc {result.doc_id}"
                    )

        # Validate transactions
        for txn in result.transactions:
            if "transaction_type" in txn:
                txn_type = txn["transaction_type"]
                if txn_type not in ALLOWED_VALUES["transaction_type"]:
                    violations.append(
                        f"Invalid transaction.transaction_type: '{txn_type}' in doc {result.doc_id}"
                    )

        # Validate physical evidence
        for evidence in result.physical_evidence:
            if "evidence_type" in evidence:
                ev_type = evidence["evidence_type"]
                if ev_type not in ALLOWED_VALUES["physical_evidence_type"]:
                    violations.append(
                        f"Invalid physical_evidence.evidence_type: '{ev_type}' "
                        f"in doc {result.doc_id}"
                    )

        # Validate redacted entities
        for redacted in result.redacted_entities:
            if "redaction_type" in redacted:
                red_type = redacted["redaction_type"]
                if red_type not in ALLOWED_VALUES["redaction_type"]:
                    violations.append(
                        f"Invalid redacted_entity.redaction_type: '{red_type}' "
                        f"in doc {result.doc_id}"
                    )

            if "confidence" in redacted:
                conf = redacted["confidence"]
                if conf not in ALLOWED_VALUES["confidence"]:
                    violations.append(
                        f"Invalid redacted_entity.confidence: '{conf}' in doc {result.doc_id}"
                    )

        # Log violations if any
        if violations:
            self._log_alert(result.doc_id, violations, result.provider)
            logger.warning(
                f"{result.doc_id}: {len(violations)} schema violations detected"
            )

        return len(violations) == 0, violations

    def _log_alert(self, doc_id: str, violations: list[str], provider: str):
        """Log schema violation alert to JSONL file.

        Args:
            doc_id: Document ID with violations
            violations: List of violation messages
            provider: LLM provider that generated the extraction
        """
        alert = {
            "timestamp": datetime.now().isoformat(),
            "doc_id": doc_id,
            "provider": provider,
            "violations": violations,
        }

        with open(self.alert_file, "a") as f:
            f.write(json.dumps(alert) + "\n")

        logger.info(f"Schema alert logged to {self.alert_file}")

    def get_alert_summary(self) -> dict[str, Any]:
        """Get summary of all schema alerts.

        Returns:
            Summary dict with total_alerts, by_doc_id, by_violation_type
        """
        if not self.alert_file.exists():
            return {"total_alerts": 0, "by_doc_id": {}, "by_violation_type": {}}

        alerts = []
        with open(self.alert_file) as f:
            for line in f:
                if line.strip():
                    alerts.append(json.loads(line))

        by_doc = {}
        by_violation = {}

        for alert in alerts:
            doc_id = alert["doc_id"]
            by_doc[doc_id] = by_doc.get(doc_id, 0) + 1

            for violation in alert["violations"]:
                # Extract violation type (e.g., "Invalid person.role")
                if ":" in violation:
                    v_type = violation.split(":")[0].strip()
                    by_violation[v_type] = by_violation.get(v_type, 0) + 1

        return {
            "total_alerts": len(alerts),
            "by_doc_id": by_doc,
            "by_violation_type": by_violation,
        }
