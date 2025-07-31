import pytest
import asyncio
from app.services.contradiction_detector import ContradictionDetector

class TestContradictionDetector:
    @pytest.fixture
    def contradiction_detector(self):
        return ContradictionDetector()
    
    def test_find_contradictions(self, contradiction_detector):
        """Test contradiction detection functionality"""
        result = asyncio.run(contradiction_detector.find_contradictions("test_case_001"))
        
        assert isinstance(result, list)
        if len(result) > 0:
            contradiction = result[0]
            assert "id" in contradiction
            assert "description" in contradiction
            assert "severity" in contradiction
            assert "confidence" in contradiction
            assert contradiction["severity"] in ["low", "medium", "high", "critical"]
            assert 0 <= contradiction["confidence"] <= 1

    def test_analyze_date_conflicts(self, contradiction_detector):
        """Test date conflict analysis"""
        documents = {
            "doc1": "Meeting scheduled for January 15, 2024",
            "doc2": "The meeting was on January 16, 2024"
        }
        
        conflicts = contradiction_detector._analyze_date_conflicts(documents)
        assert isinstance(conflicts, list)
