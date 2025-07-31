import pytest
import asyncio
from app.services.privilege_scanner import PrivilegeScanner

class TestPrivilegeScanner:
    @pytest.fixture
    def privilege_scanner(self):
        return PrivilegeScanner()
    
    def test_scan_for_privilege(self, privilege_scanner):
        """Test privilege scanning functionality"""
        result = asyncio.run(privilege_scanner.scan_for_privilege("test_case_001"))
        
        assert "potentially_privileged_documents" in result
        assert "privilege_statistics" in result
        assert "recommendations" in result
        assert isinstance(result["potentially_privileged_documents"], list)
    
    def test_scan_document_content(self, privilege_scanner):
        """Test individual document privilege scanning"""
        privileged_content = "This is attorney-client privileged communication regarding legal advice"
        regular_content = "This is a regular business email about project status"
        
        privileged_result = privilege_scanner._scan_document_content(privileged_content, "email.txt")
        regular_result = privilege_scanner._scan_document_content(regular_content, "business.txt")
        
        assert privileged_result["confidence"] > regular_result["confidence"]
        assert privileged_result["privilege_type"] in ["attorney_client", "work_product", None]
