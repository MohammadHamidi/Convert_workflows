"""
Workflow Processor Service
Handles workflow operations using existing logic
"""

import copy
import unicodedata
from typing import Dict, List, Any, Optional
from config_loader import ConfigLoader, N8NVersionDetector, WorkflowValidator


class WorkflowProcessor:
    """Service for processing N8N workflows"""
    
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.version_detector = N8NVersionDetector(config)
        self.validator = WorkflowValidator(config)
        
        # Load translation configuration
        trans_config = self.config.get_translation_config()
        self.rtl_threshold = trans_config.get('rtl_detection_threshold', 0.3)
    
    def is_sticky_note(self, node: Dict[str, Any]) -> bool:
        """Check if node is a sticky note"""
        node_type = node.get('type', '')
        sticky_note_types = self.config.get_sticky_note_types()
        return node_type in sticky_note_types
    
    def get_workflow_info(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive workflow information"""
        version = self.version_detector.detect_version(workflow_data)
        is_supported = self.version_detector.is_version_supported(version)
        
        nodes = workflow_data.get('nodes', [])
        sticky_notes = [n for n in nodes if self.is_sticky_note(n)]
        
        return {
            'version': version or 'unknown',
            'is_supported': is_supported,
            'total_nodes': len(nodes),
            'sticky_notes_count': len(sticky_notes),
            'has_rtl_content': self._is_workflow_rtl(workflow_data),
            'is_rtl_positioned': self._is_workflow_already_rtl_positioned(workflow_data)
        }
    
    def _is_workflow_rtl(self, workflow_data: Dict) -> bool:
        """Check if workflow appears to be already in RTL format"""
        try:
            sticky_notes = self.extract_sticky_notes(workflow_data)
            if sticky_notes:
                for note in sticky_notes:
                    if note['content'] and self._detect_rtl_content(note['content']):
                        return True
            
            nodes = workflow_data.get('nodes', [])
            for node in nodes:
                node_name = node.get('name', '')
                if node_name and self._detect_rtl_content(node_name):
                    return True
            
            return False
        except Exception:
            return False
    
    def _is_workflow_already_rtl_positioned(self, workflow_data: Dict) -> bool:
        """Check if workflow nodes are already positioned in RTL layout"""
        try:
            nodes = workflow_data.get('nodes', [])
            if len(nodes) < 2:
                return False
            
            positions = []
            for node in nodes:
                if 'position' in node and len(node['position']) >= 2:
                    positions.append(node['position'][0])
            
            if len(positions) < 2:
                return False
            
            sorted_positions = sorted(positions, reverse=True)
            return positions == sorted_positions[:len(positions)]
        except Exception:
            return False
    
    def _detect_rtl_content(self, text: str) -> bool:
        """Detect if text contains RTL characters"""
        if not text:
            return False
        
        rtl_chars = 0
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return False
        
        for char in text:
            if unicodedata.bidirectional(char) in ['R', 'AL']:
                rtl_chars += 1
        
        return (rtl_chars / total_chars) > self.rtl_threshold if total_chars > 0 else False
    
    def extract_sticky_notes(self, workflow_data: Dict) -> List[Dict]:
        """Extract all sticky notes from workflow"""
        sticky_notes = []
        
        if 'nodes' in workflow_data:
            for i, node in enumerate(workflow_data['nodes']):
                if self.is_sticky_note(node):
                    sticky_notes.append({
                        'id': node.get('id'),
                        'name': node.get('name', f'Sticky Note {i+1}'),
                        'content': node.get('parameters', {}).get('content', ''),
                        'node_index': i,
                        'node_type': node.get('type')
                    })
        
        return sticky_notes
    
    def replace_notes_in_workflow(
        self,
        workflow_data: Dict,
        translated_notes: List[Dict]
    ) -> Dict:
        """Replace original notes with translated ones"""
        updated_workflow = copy.deepcopy(workflow_data)
        
        for translated_note in translated_notes:
            node_index = translated_note['node_index']
            if node_index < len(updated_workflow['nodes']):
                if 'parameters' not in updated_workflow['nodes'][node_index]:
                    updated_workflow['nodes'][node_index]['parameters'] = {}
                updated_workflow['nodes'][node_index]['parameters']['content'] = \
                    translated_note['translated_content']
        
        return updated_workflow
    
    def extract_node_names(self, workflow_data: Dict) -> List[Dict]:
        """Extract all node names from workflow"""
        node_names = []
        
        if 'nodes' in workflow_data:
            for i, node in enumerate(workflow_data['nodes']):
                node_name = node.get('name', '')
                if node_name and not self._detect_rtl_content(node_name):
                    node_names.append({
                        'node_index': i,
                        'original_name': node_name,
                        'node_type': node.get('type', 'unknown'),
                        'node_id': node.get('id', '')
                    })
        
        return node_names
    
    def replace_node_names_in_workflow(
        self,
        workflow_data: Dict,
        translated_names: List[Dict]
    ) -> Dict:
        """Replace original node names with translated ones"""
        updated_workflow = copy.deepcopy(workflow_data)
        
        # Create name mapping
        name_mapping = {}
        for translated_name in translated_names:
            name_mapping[translated_name['original_name']] = translated_name['translated_name']
        
        # Update node names
        for translated_name in translated_names:
            node_index = translated_name['node_index']
            if node_index < len(updated_workflow['nodes']):
                updated_workflow['nodes'][node_index]['name'] = translated_name['translated_name']
        
        # Update references in sticky notes
        sticky_notes = self.extract_sticky_notes(updated_workflow)
        for note in sticky_notes:
            content = note['content']
            for old_name, new_name in name_mapping.items():
                content = content.replace(old_name, new_name)
            
            node_index = note['node_index']
            if node_index < len(updated_workflow['nodes']):
                if 'parameters' not in updated_workflow['nodes'][node_index]:
                    updated_workflow['nodes'][node_index]['parameters'] = {}
                updated_workflow['nodes'][node_index]['parameters']['content'] = content
        
        return updated_workflow
    
    def convert_ltr_to_rtl(
        self,
        workflow_json: Any,
        canvas_width: Optional[float] = None
    ) -> Optional[Dict]:
        """Convert LTR workflow to RTL by mirroring node positions"""
        try:
            # Parse JSON if string
            if isinstance(workflow_json, str):
                import json
                workflow_data = json.loads(workflow_json)
            else:
                workflow_data = workflow_json
            
            # Validate workflow
            is_valid, messages, fixed_workflow = self.validator.validate_and_fix(workflow_data)
            
            # Create deep copy
            converted_workflow = copy.deepcopy(fixed_workflow)
            
            # Load layout configuration
            layout_config = self.config.get_layout_config()
            canvas_buffer = layout_config.get('canvas_width_buffer', 0.1)
            default_sticky_width = layout_config.get('default_sticky_width', 300)
            
            # Extract X coordinates
            x_coordinates = []
            for node in converted_workflow.get('nodes', []):
                if 'position' in node and len(node['position']) >= 2:
                    x_coordinates.append(node['position'][0])
                    
                    # Consider sticky note width
                    if self.is_sticky_note(node):
                        width = node.get('parameters', {}).get('width', default_sticky_width)
                        if width:
                            x_coordinates.append(node['position'][0] + width)
            
            if not x_coordinates:
                return converted_workflow
            
            min_x = min(x_coordinates)
            max_x = max(x_coordinates)
            
            # Calculate canvas width
            if canvas_width is None:
                canvas_width = max_x + abs(max_x - min_x) * canvas_buffer
            
            # Mirror positions
            for node in converted_workflow.get('nodes', []):
                if 'position' in node and len(node['position']) >= 2:
                    original_x, y = node['position'][0], node['position'][1]
                    
                    if self.is_sticky_note(node):
                        width = node.get('parameters', {}).get('width', default_sticky_width)
                        new_x = canvas_width - original_x - width
                    else:
                        new_x = canvas_width - original_x
                    
                    node['position'] = [new_x, y]
            
            return converted_workflow
            
        except Exception as e:
            print(f"Error converting workflow: {e}")
            return None

