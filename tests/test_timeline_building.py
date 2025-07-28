import pytest
import asyncio
from app.services.timeline_builder import TimelineBuilder

class TestTimelineBuilder:
    @pytest.fixture
    def timeline_builder(self):
        return TimelineBuilder()
    
    @pytest.mark.asyncio
    async def test_build_timeline(self, timeline_builder):
        """Test timeline building functionality"""
        result = await timeline_builder.build_timeline("test_case_001")
        
        assert isinstance(result, list)
        if len(result) > 0:
            event = result[0]
            assert "date" in event
            assert "event" in event
            assert "event_type" in event
            assert "importance" in event
            assert 0 <= event["importance"] <= 1
    
    def test_extract_dates_from_content(self, timeline_builder):
        """Test date extraction from content"""
        content = "Contract signed on January 15, 2024. Payment due 2/1/2024. Meeting on 2024-02-15."
        
        dates = timeline_builder._extract_dates_from_content(content)
        assert isinstance(dates, list)
        assert len(dates) > 0

if __name__ == "__main__":
    pytest.main([__file__])