"""
RTL Conversion Agent
Handles LTR to RTL layout conversion
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import copy
from config_loader import ConfigLoader


@dataclass
class RTLDependencies:
    """Dependencies for RTL conversion"""
    config: ConfigLoader
    canvas_width: Optional[float] = None


class RTLConversionAgent:
    """Agent for converting workflows from LTR to RTL"""
    
    def __init__(self, config: ConfigLoader):
        self.config = config
    
    def _is_sticky_note(self, node: Dict[str, Any]) -> bool:
        """Check if node is a sticky note"""
        node_type = node.get('type', '')
        sticky_note_types = self.config.get_sticky_note_types()
        return node_type in sticky_note_types
    
    async def convert(
        self,
        workflow_data: Dict[str, Any],
        canvas_width: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Convert workflow from LTR to RTL layout"""
        try:
            # Create deep copy
            converted_workflow = copy.deepcopy(workflow_data)
            
            # Load layout configuration
            layout_config = self.config.get_layout_config()
            canvas_buffer = layout_config.get('canvas_width_buffer', 0.1)
            default_sticky_width = layout_config.get('default_sticky_width', 300)
            
            # Extract X coordinates
            x_coordinates = []
            nodes = converted_workflow.get('nodes', [])
            
            for node in nodes:
                if 'position' in node and len(node['position']) >= 2:
                    x_coordinates.append(node['position'][0])
                    
                    # Consider sticky note width
                    if self._is_sticky_note(node):
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
            for node in nodes:
                if 'position' in node and len(node['position']) >= 2:
                    original_x, y = node['position'][0], node['position'][1]
                    
                    if self._is_sticky_note(node):
                        width = node.get('parameters', {}).get('width', default_sticky_width)
                        new_x = canvas_width - original_x - width
                    else:
                        new_x = canvas_width - original_x
                    
                    node['position'] = [new_x, y]
            
            return converted_workflow
            
        except Exception as e:
            print(f"Error converting workflow: {e}")
            return None

