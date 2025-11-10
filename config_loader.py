"""
Configuration Loader for N8N Workflow Converter
Handles loading and managing configuration from config.json
"""

import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path


class ConfigLoader:
    """Loads and provides access to application configuration"""

    def __init__(self, config_path: str = "config.json"):
        """
        Initialize configuration loader

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise Exception(f"Error loading configuration: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key_path: Path to config value (e.g., 'api.base_url')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self.config.get('api', {})

    def get_n8n_config(self) -> Dict[str, Any]:
        """Get N8N configuration"""
        return self.config.get('n8n', {})

    def get_translation_config(self) -> Dict[str, Any]:
        """Get translation configuration"""
        return self.config.get('translation', {})

    def get_layout_config(self) -> Dict[str, Any]:
        """Get layout configuration"""
        return self.config.get('layout', {})

    def get_models(self) -> Dict[str, Dict[str, Any]]:
        """Get available models configuration"""
        return self.config.get('models', {})

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        return self.config.get('models', {}).get(model_id)

    def get_supported_n8n_versions(self) -> List[str]:
        """Get list of supported N8N versions"""
        return self.config.get('n8n', {}).get('supported_versions', [])

    def get_sticky_note_types(self) -> List[str]:
        """Get list of possible sticky note node types"""
        return self.config.get('n8n', {}).get('node_types', {}).get('sticky_note', [])

    def get_required_fields(self, category: str) -> List[str]:
        """
        Get required fields for validation

        Args:
            category: Category name ('workflow', 'node', 'sticky_note')

        Returns:
            List of required field names
        """
        return self.config.get('n8n', {}).get('required_fields', {}).get(category, [])

    def reload(self):
        """Reload configuration from file"""
        self.config = self._load_config()


class N8NVersionDetector:
    """Detects and validates N8N workflow versions"""

    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize version detector

        Args:
            config_loader: Configuration loader instance
        """
        self.config = config_loader

    def detect_version(self, workflow_data: Dict[str, Any]) -> Optional[str]:
        """
        Detect N8N version from workflow data

        Args:
            workflow_data: N8N workflow JSON data

        Returns:
            Version string or None if not detected
        """
        # Try to find version in metadata
        if 'meta' in workflow_data:
            if 'instanceId' in workflow_data['meta']:
                # Newer versions include instanceId
                return "1.x"

        # Check for version field in various locations
        if 'version' in workflow_data:
            return workflow_data['version']

        if 'versionId' in workflow_data:
            return workflow_data['versionId']

        # Try to infer from node structure
        if 'nodes' in workflow_data and len(workflow_data['nodes']) > 0:
            first_node = workflow_data['nodes'][0]

            # Check node structure patterns
            if 'typeVersion' in first_node:
                # Newer versions have typeVersion
                return "1.x"

            # Check node type format
            node_type = first_node.get('type', '')
            if node_type.startswith('@n8n/'):
                return "1.x"
            elif node_type.startswith('n8n-nodes-base.'):
                return "0.x"

        # Default to latest supported version
        return "1.x"

    def is_version_supported(self, version: Optional[str]) -> bool:
        """
        Check if detected version is supported

        Args:
            version: Version string

        Returns:
            True if supported, False otherwise
        """
        if version is None:
            return True  # Assume supported if cannot detect

        supported_versions = self.config.get_supported_n8n_versions()

        # Check exact match or pattern match
        for supported in supported_versions:
            if version == supported or version.startswith(supported.replace('.x', '')):
                return True

        return False

    def get_sticky_note_type(self, workflow_data: Dict[str, Any]) -> str:
        """
        Get the appropriate sticky note type for the workflow version

        Args:
            workflow_data: N8N workflow JSON data

        Returns:
            Sticky note node type string
        """
        version = self.detect_version(workflow_data)
        sticky_note_types = self.config.get_sticky_note_types()

        if not sticky_note_types:
            # Fallback to default
            return 'n8n-nodes-base.stickyNote'

        # For version 1.x, prefer @n8n/ prefix
        if version and version.startswith('1'):
            for note_type in sticky_note_types:
                if note_type.startswith('@n8n/'):
                    return note_type

        # Return first (default) type
        return sticky_note_types[0]


class WorkflowValidator:
    """Validates N8N workflow structure"""

    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize workflow validator

        Args:
            config_loader: Configuration loader instance
        """
        self.config = config_loader

    def validate_workflow(self, workflow_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate workflow structure

        Args:
            workflow_data: N8N workflow JSON data

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check required workflow fields
        required_workflow_fields = self.config.get_required_fields('workflow')
        for field in required_workflow_fields:
            if field not in workflow_data:
                errors.append(f"Missing required workflow field: {field}")

        # Validate nodes if present
        if 'nodes' in workflow_data:
            if not isinstance(workflow_data['nodes'], list):
                errors.append("'nodes' must be a list")
            else:
                for i, node in enumerate(workflow_data['nodes']):
                    node_errors = self._validate_node(node, i)
                    errors.extend(node_errors)

        return len(errors) == 0, errors

    def _validate_node(self, node: Dict[str, Any], index: int) -> List[str]:
        """
        Validate individual node structure

        Args:
            node: Node data
            index: Node index in workflow

        Returns:
            List of error messages
        """
        errors = []

        # Check required node fields
        required_node_fields = self.config.get_required_fields('node')
        for field in required_node_fields:
            if field not in node:
                errors.append(f"Node {index}: Missing required field '{field}'")

        # Validate position if present
        if 'position' in node:
            if not isinstance(node['position'], list):
                errors.append(f"Node {index}: 'position' must be a list")
            elif len(node['position']) < 2:
                errors.append(f"Node {index}: 'position' must have at least 2 elements [x, y]")

        # Validate sticky note specific fields
        sticky_note_types = self.config.get_sticky_note_types()
        if node.get('type') in sticky_note_types:
            if 'parameters' not in node:
                errors.append(f"Node {index} (sticky note): Missing 'parameters' field")
            elif not isinstance(node['parameters'], dict):
                errors.append(f"Node {index} (sticky note): 'parameters' must be an object")

        return errors

    def validate_and_fix(self, workflow_data: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        """
        Validate and attempt to fix common issues

        Args:
            workflow_data: N8N workflow JSON data

        Returns:
            Tuple of (fixed workflow data, list of warnings)
        """
        import copy
        fixed_workflow = copy.deepcopy(workflow_data)
        warnings = []

        # Ensure nodes array exists
        if 'nodes' not in fixed_workflow:
            fixed_workflow['nodes'] = []
            warnings.append("Added missing 'nodes' array")

        # Fix node issues
        for i, node in enumerate(fixed_workflow.get('nodes', [])):
            # Ensure position exists and is valid
            if 'position' not in node:
                node['position'] = [0, 0]
                warnings.append(f"Node {i}: Added default position [0, 0]")
            elif not isinstance(node['position'], list) or len(node['position']) < 2:
                node['position'] = [0, 0]
                warnings.append(f"Node {i}: Fixed invalid position to [0, 0]")

            # Ensure sticky notes have parameters
            sticky_note_types = self.config.get_sticky_note_types()
            if node.get('type') in sticky_note_types:
                if 'parameters' not in node:
                    node['parameters'] = {'content': ''}
                    warnings.append(f"Node {i} (sticky note): Added missing parameters")
                elif 'content' not in node['parameters']:
                    node['parameters']['content'] = ''
                    warnings.append(f"Node {i} (sticky note): Added missing content field")

        return fixed_workflow, warnings
