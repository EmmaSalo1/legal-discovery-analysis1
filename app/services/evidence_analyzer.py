from typing import List, Dict

class EvidenceAnalyzer:
    """Minimal stub for evidence analysis."""

    def __init__(self):
        pass

    async def analyze_evidence(self, case_id: str) -> Dict:
        """Return dummy analysis results."""
        return {
            "key_evidence": [],
            "evidence_timeline": [],
            "credibility_assessment": {}
        }

    def _detect_financial_patterns(self, documents: List[str]) -> List[str]:
        """Simple financial pattern detection used in tests."""
        patterns = []
        for doc in documents:
            if any(word in doc.lower() for word in ["$", "payment", "fee", "paid"]):
                patterns.append(doc)
        return patterns
