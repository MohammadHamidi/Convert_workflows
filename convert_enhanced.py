"""
Enhanced N8N Workflow Converter with Dynamic Configuration
Version: 2.0
- Dynamic configuration support
- N8N version detection and validation
- Cross-version compatibility
- Robust error handling
"""

import streamlit as st
import json
import requests
import re
import copy
import unicodedata
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Import configuration and validation modules
from config_loader import ConfigLoader, N8NVersionDetector, WorkflowValidator


class AvalAITranslator:
    """Enhanced translator with dynamic configuration"""

    def __init__(self, api_key: str, model: str = None, config: ConfigLoader = None):
        """
        Initialize translator with dynamic configuration

        Args:
            api_key: AvalAI API key
            model: Model to use (optional, uses config default if not specified)
            config: Configuration loader instance
        """
        self.config = config or ConfigLoader()
        self.api_key = api_key
        self.model = model or self.config.get('api.default_model', 'gpt-4o-mini')

        # Load API configuration
        api_config = self.config.get_api_config()
        self.base_url = api_config.get('base_url', 'https://api.avalai.ir/v1')
        self.endpoint = api_config.get('endpoint', '/chat/completions')
        self.timeout = api_config.get('timeout', 30)
        self.default_temperature = api_config.get('default_temperature', 0.3)
        self.max_tokens = api_config.get('max_tokens', 2000)

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available models from configuration"""
        return self.config.get_models()

    def translate_text(self, text: str, target_language: str = "فارسی",
                      temperature: float = None, max_tokens: int = None) -> str:
        """
        Translate text using AvalAI API with dynamic configuration

        Args:
            text: Text to translate
            target_language: Target language
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Translated text
        """
        if not text or not text.strip():
            return text

        prompt = f"""Please translate the following text to {target_language} with high accuracy and natural flow.
        Maintain the technical terminology appropriately and ensure the translation sounds professional and native.

        Text to translate:
        {text}

        Please provide only the translated text without any additional explanations."""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature or self.default_temperature,
            "max_tokens": max_tokens or self.max_tokens
        }

        try:
            response = requests.post(
                f"{self.base_url}{self.endpoint}",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            translated_text = result['choices'][0]['message']['content'].strip()

            # Remove any quotes or extra formatting
            translated_text = re.sub(r'^["\']|["\']$', '', translated_text)

            return translated_text

        except requests.exceptions.Timeout:
            st.error(f"Translation timeout after {self.timeout} seconds")
            return text
        except requests.exceptions.RequestException as e:
            st.error(f"Translation API error: {str(e)}")
            return text
        except KeyError as e:
            st.error(f"Unexpected API response format: {str(e)}")
            return text
        except Exception as e:
            st.error(f"Translation error: {str(e)}")
            return text


class N8NWorkflowProcessor:
    """Enhanced workflow processor with version detection and validation"""

    def __init__(self, translator: AvalAITranslator = None, config: ConfigLoader = None):
        """
        Initialize workflow processor

        Args:
            translator: Translator instance
            config: Configuration loader instance
        """
        self.translator = translator
        self.config = config or ConfigLoader()
        self.version_detector = N8NVersionDetector(self.config)
        self.validator = WorkflowValidator(self.config)

        # Load translation configuration
        trans_config = self.config.get_translation_config()
        self.rtl_threshold = trans_config.get('rtl_detection_threshold', 0.3)

    def detect_rtl_content(self, text: str) -> bool:
        """
        Detect if text contains RTL characters using dynamic threshold

        Args:
            text: Text to check

        Returns:
            True if RTL content detected
        """
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

    def is_sticky_note(self, node: Dict[str, Any]) -> bool:
        """
        Check if node is a sticky note (supports multiple versions)

        Args:
            node: Node data

        Returns:
            True if node is a sticky note
        """
        node_type = node.get('type', '')
        sticky_note_types = self.config.get_sticky_note_types()
        return node_type in sticky_note_types

    def validate_workflow(self, workflow_data: Dict[str, Any],
                         auto_fix: bool = True) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate workflow structure with optional auto-fix

        Args:
            workflow_data: Workflow JSON data
            auto_fix: Automatically fix common issues

        Returns:
            Tuple of (is_valid, messages, fixed_workflow)
        """
        if auto_fix:
            fixed_workflow, warnings = self.validator.validate_and_fix(workflow_data)
            return True, warnings, fixed_workflow
        else:
            is_valid, errors = self.validator.validate_workflow(workflow_data)
            return is_valid, errors, workflow_data

    def get_workflow_info(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive workflow information

        Args:
            workflow_data: Workflow JSON data

        Returns:
            Dictionary with workflow information
        """
        version = self.version_detector.detect_version(workflow_data)
        is_supported = self.version_detector.is_version_supported(version)

        nodes = workflow_data.get('nodes', [])
        sticky_notes = [n for n in nodes if self.is_sticky_note(n)]

        return {
            'version': version,
            'is_supported': is_supported,
            'total_nodes': len(nodes),
            'sticky_notes_count': len(sticky_notes),
            'has_rtl_content': self.is_workflow_rtl(workflow_data),
            'is_rtl_positioned': self.is_workflow_already_rtl_positioned(workflow_data)
        }

    def is_workflow_rtl(self, workflow_data: Dict) -> bool:
        """Check if workflow appears to be already in RTL format"""
        try:
            # Check sticky notes for RTL content
            sticky_notes = self.extract_sticky_notes(workflow_data)
            if sticky_notes:
                for note in sticky_notes:
                    if note['content'] and self.detect_rtl_content(note['content']):
                        return True

            # Check node names for RTL content
            nodes = workflow_data.get('nodes', [])
            for node in nodes:
                node_name = node.get('name', '')
                if node_name and self.detect_rtl_content(node_name):
                    return True

            return False
        except Exception:
            return False

    def is_workflow_translated(self, workflow_data: Dict, target_language: str = "فارسی") -> bool:
        """Check if workflow appears to be already translated"""
        if target_language == "فارسی":
            return self.is_workflow_rtl(workflow_data)
        return False

    def is_workflow_already_rtl_positioned(self, workflow_data: Dict) -> bool:
        """Check if workflow nodes are already positioned in RTL layout"""
        try:
            nodes = workflow_data.get('nodes', [])
            if len(nodes) < 2:
                return False

            # Get positions
            positions = []
            for node in nodes:
                if 'position' in node and len(node['position']) >= 2:
                    positions.append(node['position'][0])

            if len(positions) < 2:
                return False

            # Simple heuristic: check if positions follow RTL pattern
            sorted_positions = sorted(positions, reverse=True)
            return positions == sorted_positions[:len(positions)]

        except Exception:
            return False

    def extract_node_names(self, workflow_data: Dict) -> List[Dict]:
        """Extract all node names from workflow"""
        node_names = []

        if 'nodes' in workflow_data:
            for i, node in enumerate(workflow_data['nodes']):
                node_name = node.get('name', '')
                if node_name and not self.detect_rtl_content(node_name):
                    node_names.append({
                        'node_index': i,
                        'original_name': node_name,
                        'node_type': node.get('type', 'unknown'),
                        'node_id': node.get('id', '')
                    })

        return node_names

    def translate_node_names(self, node_names: List[Dict],
                           target_language: str = "فارسی") -> List[Dict]:
        """Translate node names to target language"""
        if not self.translator:
            raise ValueError("Translator not initialized")

        translated_names = []

        for i, node_info in enumerate(node_names):
            if node_info['original_name'].strip():
                st.write(f"🔄 Translating node {i+1}/{len(node_names)}: {node_info['original_name']}")

                translated_name = self.translator.translate_text(
                    node_info['original_name'],
                    target_language
                )

                translated_names.append({
                    **node_info,
                    'translated_name': translated_name
                })
            else:
                translated_names.append({
                    **node_info,
                    'translated_name': node_info['original_name']
                })

        return translated_names

    def replace_node_names_in_workflow(self, workflow_data: Dict,
                                      translated_names: List[Dict]) -> Dict:
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

    def extract_sticky_notes(self, workflow_data: Dict) -> List[Dict]:
        """Extract all sticky notes from workflow (version-aware)"""
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

    def translate_sticky_notes(self, sticky_notes: List[Dict],
                              target_language: str) -> List[Dict]:
        """Translate all sticky notes content"""
        if not self.translator:
            raise ValueError("Translator not initialized")

        translated_notes = []

        for i, note in enumerate(sticky_notes):
            if note['content'].strip():
                progress_text = f"🔄 Translating note {i+1}/{len(sticky_notes)}: {note['name'][:30]}..."
                st.write(progress_text)

                translated_content = self.translator.translate_text(
                    note['content'],
                    target_language
                )

                translated_notes.append({
                    **note,
                    'translated_content': translated_content
                })
            else:
                translated_notes.append({
                    **note,
                    'translated_content': note['content']
                })

        return translated_notes

    def replace_notes_in_workflow(self, workflow_data: Dict,
                                 translated_notes: List[Dict]) -> Dict:
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

    def convert_ltr_to_rtl(self, workflow_json: Any, canvas_width: float = None) -> Optional[Dict]:
        """
        Convert LTR workflow to RTL by mirroring node positions (version-aware)

        Args:
            workflow_json: Workflow data (dict or JSON string)
            canvas_width: Canvas width (auto-calculated if not provided)

        Returns:
            Converted workflow data or None on error
        """
        try:
            # Parse JSON if string
            if isinstance(workflow_json, str):
                workflow_data = json.loads(workflow_json)
            else:
                workflow_data = workflow_json

            # Validate workflow
            is_valid, messages, fixed_workflow = self.validate_workflow(workflow_data, auto_fix=True)
            if messages:
                for msg in messages:
                    st.info(f"Auto-fix: {msg}")

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
            regular_nodes = 0
            sticky_notes_count = 0

            for node in converted_workflow.get('nodes', []):
                if 'position' in node and len(node['position']) >= 2:
                    original_x, y = node['position'][0], node['position'][1]

                    if self.is_sticky_note(node):
                        sticky_notes_count += 1
                        width = node.get('parameters', {}).get('width', default_sticky_width)
                        new_x = canvas_width - original_x - width
                    else:
                        regular_nodes += 1
                        new_x = canvas_width - original_x

                    node['position'] = [new_x, y]

            # Store stats
            st.session_state['conversion_stats'] = {
                'regular_nodes': regular_nodes,
                'sticky_notes': sticky_notes_count,
                'canvas_width': canvas_width
            }

            return converted_workflow

        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Error converting workflow: {str(e)}")
            return None


# Utility Functions

def validate_json(json_string: str) -> Tuple[bool, str]:
    """Validate JSON string"""
    try:
        json.loads(json_string)
        return True, "Valid JSON"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"


def calculate_estimated_cost(text_length: int, model_key: str, config: ConfigLoader) -> float:
    """
    Calculate estimated translation cost using configuration

    Args:
        text_length: Length of text to translate
        model_key: Model identifier
        config: Configuration loader

    Returns:
        Estimated cost in dollars
    """
    model_info = config.get_model_info(model_key)

    if not model_info:
        return 0.0

    # Get token ratio from config
    trans_config = config.get_translation_config()
    token_ratio = trans_config.get('token_to_char_ratio', 4)

    estimated_tokens = text_length / token_ratio
    cost = (estimated_tokens / 1_000_000) * model_info.get('input_cost', 0)

    return cost


def create_model_selector(key_prefix: str, config: ConfigLoader,
                         default_model: str = None) -> str:
    """
    Create model selector widget with dynamic model list

    Args:
        key_prefix: Unique key prefix for widget
        config: Configuration loader
        default_model: Default model to select

    Returns:
        Selected model key
    """
    models = config.get_models()

    if not default_model:
        default_model = config.get('api.default_model', 'gpt-4o-mini')

    # Create model options
    model_options = []
    model_keys = []

    for key, model_info in models.items():
        input_cost = model_info.get('input_cost', 0)
        output_cost = model_info.get('output_cost', 0)
        name = model_info.get('name', key)
        desc = model_info.get('description', '')

        label = f"{name} - ${input_cost:.4f}/${output_cost:.4f} - {desc}"

        model_options.append(label)
        model_keys.append(key)

    # Find default index
    try:
        default_index = model_keys.index(default_model)
    except ValueError:
        default_index = 0

    # Create selector
    selected_index = st.selectbox(
        "Select AI Model",
        range(len(model_options)),
        format_func=lambda i: model_options[i],
        index=default_index,
        key=f"{key_prefix}_model_selector"
    )

    return model_keys[selected_index]


# Initialize configuration at module level
try:
    GLOBAL_CONFIG = ConfigLoader()
except Exception as e:
    st.error(f"Failed to load configuration: {e}")
    GLOBAL_CONFIG = None
