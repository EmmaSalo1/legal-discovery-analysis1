import pytest
import asyncio
from app.services.evidence_analyzer import EvidenceAnalyzer

class TestEvidenceAnalyzer:
    @pytest.fixture
    def evidence_analyzer(self):
        return EvidenceAnalyzer()
    
    @pytest.fixture
    def sample_case_data(self):
        return {
            "case_id": "test_case_001",
            "documents": [
                {
                    "id": "doc1",
                    "content": "Contract signed on January 15, 2024 for $50,000",
                    "type": "contract"
                },
                {
                    "id": "doc2", 
                    "content": "Payment was late by 30 days causing breach",
                    "type": "email"
                }
            ]
        }
    
    def test_analyze_evidence(self, evidence_analyzer, sample_case_data):
        """Test evidence analysis functionality"""
        result = asyncio.run(evidence_analyzer.analyze_evidence(sample_case_data["case_id"]))
        
        assert "key_evidence" in result
        assert "evidence_timeline" in result
        assert "credibility_assessment" in result
    
    def test_financial_pattern_detection(self, evidence_analyzer):
        """Test detection of financial patterns"""
        financial_docs = [
            "Payment of $10,000 due January 1",
            "Payment of $15,000 received January 15", 
            "Late fee of $500 assessed February 1"
        ]
        
        patterns = evidence_analyzer._detect_financial_patterns(financial_docs)
        assert len(patterns) > 0
        assert any("payment" in pattern.lower() for pattern in patterns)

if __name__ == "__main__":
    pytest.main([__file__])