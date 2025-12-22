"""
Workflow Analysis Agent using Pydantic AI
Analyzes workflow structure and content
"""

from typing import Dict, Any
from dataclasses import dataclass
from config_loader import ConfigLoader, N8NVersionDetector


@dataclass
class AnalysisDependencies:
    """Dependencies for workflow analysis"""
    config: ConfigLoader


class WorkflowAnalysisAgent:
    """Agent for analyzing N8N workflows"""
    
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.version_detector = N8NVersionDetector(config)
    
    async def analyze(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow and return comprehensive information"""
        version = self.version_detector.detect_version(workflow_data)
        is_supported = self.version_detector.is_version_supported(version)
        
        nodes = workflow_data.get('nodes', [])
        
        # Count sticky notes
        sticky_note_types = self.config.get_sticky_note_types()
        sticky_notes = [
            n for n in nodes
            if n.get('type', '') in sticky_note_types
        ]
        
        # Check for RTL content
        has_rtl_content = self._check_rtl_content(workflow_data)
        
        # Check RTL positioning
        is_rtl_positioned = self._check_rtl_positioning(workflow_data)
        
        return {
            'version': version or 'unknown',
            'is_supported': is_supported,
            'total_nodes': len(nodes),
            'sticky_notes_count': len(sticky_notes),
            'has_rtl_content': has_rtl_content,
            'is_rtl_positioned': is_rtl_positioned
        }
    
    def _check_rtl_content(self, workflow_data: Dict[str, Any]) -> bool:
        """Check if workflow contains RTL text content"""
        import unicodedata
        
        # Check sticky notes
        sticky_note_types = self.config.get_sticky_note_types()
        nodes = workflow_data.get('nodes', [])
        
        for node in nodes:
            if node.get('type') in sticky_note_types:
                content = node.get('parameters', {}).get('content', '')
                if self._has_rtl_chars(content):
                    return True
            
            # Check node names
            node_name = node.get('name', '')
            if self._has_rtl_chars(node_name):
                return True
        
        return False
    
    def _has_rtl_chars(self, text: str) -> bool:
        """Check if text contains RTL characters"""
        import unicodedata
        
        if not text:
            return False
        
        rtl_chars = 0
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return False
        
        for char in text:
            if unicodedata.bidirectional(char) in ['R', 'AL']:
                rtl_chars += 1
        
        threshold = self.config.get('translation.rtl_detection_threshold', 0.3)
        return (rtl_chars / total_chars) > threshold if total_chars > 0 else False
    
    def _check_rtl_positioning(self, workflow_data: Dict[str, Any]) -> bool:
        """Check if workflow nodes are positioned in RTL layout"""
        nodes = workflow_data.get('nodes', [])
        if len(nodes) < 2:
            return False
        
        positions = []
        for node in nodes:
            if 'position' in node and len(node['position']) >= 2:
                positions.append(node['position'][0])
        
        if len(positions) < 2:
            return False
        
        # Simple heuristic: check if positions follow RTL pattern
        sorted_positions = sorted(positions, reverse=True)
        return positions == sorted_positions[:len(positions)]

