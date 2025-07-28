import re
from typing import List, Dict

class PrivilegeScanner:
    def __init__(self):
        # Attorney-client privilege indicators
        self.privilege_patterns = [
            r'\b(?:attorney|lawyer|counsel|legal counsel)\b',
            r'\battorney-client\b',
            r'\blegal advice\b',
            r'\bconfidential\s+(?:and\s+)?privileged\b',
            r'\bprivileged\s+(?:and\s+)?confidential\b',
            r'\bprivileged\s+communication\b',
            r'\battorney\s+work\s+product\b',
            r'\blaw\s+firm\b',
        ]
        
    def scan_text(self, text: str) -> List[Dict]:
        """Scan text for privilege indicators"""
        flags = []
        text_lower = text.lower()
        
        for pattern in self.privilege_patterns:
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            for match in matches:
                flags.append({
                    'type': 'attorney_client_privilege',
                    'pattern': pattern,
                    'match': match.group(),
                    'position': match.span(),
                    'confidence': 0.8,
                    'context': self._extract_context(text, match.span())
                })
        
        return flags
    
    def _extract_context(self, text: str, position: tuple, context_length: int = 100) -> str:
        """Extract context around a privilege match"""
        start, end = position
        context_start = max(0, start - context_length)
        context_end = min(len(text), end + context_length)
        
        context = text[context_start:context_end]
        
        if context_start > 0:
            context = "..." + context
        if context_end < len(text):
            context = context + "..."
        
        return context.strip()