import re
from typing import List, Dict

class ContradictionDetector:
    """Minimal stub for contradiction detection."""

    def __init__(self):
        pass

    async def find_contradictions(self, case_id: str) -> List[Dict]:
        """Return a list with a sample contradiction."""
        return [
            {
                "id": "contradiction_1",
                "description": "Sample contradiction",
                "severity": "low",
                "confidence": 0.5,
            }
        ]

    def _analyze_date_conflicts(self, documents: Dict[str, str]) -> List[str]:
        """Detect conflicts if multiple documents mention different dates."""
        dates = {}
        date_pattern = r"\b\d{4}-\d{2}-\d{2}\b|\b\w+ \d{1,2}, \d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
        for name, text in documents.items():
            match = re.search(date_pattern, text)
            if match:
                dates.setdefault(match.group(), []).append(name)
        # Conflict if same context has more than one date
        if len(dates) > 1:
            return ["date_conflict"]
        return []
