"""
Translation Agent using Pydantic AI
Handles translation of sticky notes and node names
"""

import httpx
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from config_loader import ConfigLoader


@dataclass
class TranslationDependencies:
    """Dependencies for translation agent"""
    api_key: str
    base_url: str
    model: str
    config: ConfigLoader
    target_language: str = "فارسی"


class TranslationAgent:
    """Pydantic AI agent for translating workflow content"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        config: ConfigLoader,
        target_language: str = "فارسی"
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.config = config
        self.target_language = target_language
        self.agent = None
        self.http_client = None
    
    async def initialize(self, http_client: Optional[httpx.AsyncClient] = None):
        """Initialize the HTTP client"""
        if http_client is None:
            self.http_client = httpx.AsyncClient()
        else:
            self.http_client = http_client
    
    
    async def _translate_direct(self, text: str, target_language: str) -> str:
        """Direct translation using OpenAI API"""
        import json
        
        if not self.http_client:
            self.http_client = httpx.AsyncClient()
        
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
            "temperature": self.config.get('api.default_temperature', 0.3),
            "max_tokens": self.config.get('api.max_tokens', 2000)
        }
        
        try:
            response = await self.http_client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload,
                timeout=self.config.get('api.timeout', 30)
            )
            response.raise_for_status()
            
            result = response.json()
            translated_text = result['choices'][0]['message']['content'].strip()
            
            # Remove quotes if present
            if translated_text.startswith('"') and translated_text.endswith('"'):
                translated_text = translated_text[1:-1]
            if translated_text.startswith("'") and translated_text.endswith("'"):
                translated_text = translated_text[1:-1]
            
            return translated_text
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original on error
    
    async def translate_text(self, text: str, target_language: Optional[str] = None) -> str:
        """Translate a single text"""
        target = target_language or self.target_language
        return await self._translate_direct(text, target)
    
    async def translate_sticky_notes(
        self,
        sticky_notes: List[Dict[str, Any]],
        target_language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Translate all sticky notes"""
        target = target_language or self.target_language
        translated_notes = []
        
        for note in sticky_notes:
            if note.get('content', '').strip():
                translated_content = await self.translate_text(note['content'], target)
                translated_notes.append({
                    **note,
                    'translated_content': translated_content
                })
            else:
                translated_notes.append({
                    **note,
                    'translated_content': note.get('content', '')
                })
        
        return translated_notes
    
    async def translate_node_names(
        self,
        node_names: List[Dict[str, Any]],
        target_language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Translate node names"""
        target = target_language or self.target_language
        translated_names = []
        
        for node_info in node_names:
            if node_info.get('original_name', '').strip():
                translated_name = await self.translate_text(
                    node_info['original_name'],
                    target
                )
                translated_names.append({
                    **node_info,
                    'translated_name': translated_name
                })
            else:
                translated_names.append({
                    **node_info,
                    'translated_name': node_info.get('original_name', '')
                })
        
        return translated_names

