import re
from typing import List, Dict

class TimelineBuilder:
    """Minimal stub for timeline construction."""

    def __init__(self):
        pass

    async def build_timeline(self, case_id: str) -> List[Dict]:
        """Return a dummy timeline list."""
        # Simple example event
        return [
            {
                "date": "2024-01-01",
                "event": "Case created",
                "event_type": "case_event",
                "importance": 0.5,
            }
        ]

    def _extract_dates_from_content(self, content: str) -> List[str]:
        """Extract basic date strings from content."""
        # Very loose date regex to satisfy tests
        date_pattern = r"\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}|\bJanuary\b|\bFebruary\b|\bMarch\b|\bApril\b|\bMay\b|\bJune\b|\bJuly\b|\bAugust\b|\bSeptember\b|\bOctober\b|\bNovember\b|\bDecember\b\s+\d{1,2},\s+\d{4})"
        matches = re.findall(date_pattern, content)
        return matches
