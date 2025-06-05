import streamlit as st
import json
import requests
import re
import copy
from typing import Dict, List, Any
import io

class AvalAITranslator:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.avalai.ir/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    @staticmethod
    def get_available_models():
        """Get list of available AvalAI models with their specifications"""
        return {
            # OpenAI Models - Based on your table
            "gpt-4.1-nano": {
                "name": "GPT-4.1 Nano",
                "provider": "OPENAI",
                "input_cost": 0.1,
                "output_cost": 0.4,
                "max_tokens": 32768,
                "context": 1047576,
                "description": "💰 Ultra cost-effective - Best for simple translations",
                "recommended_for": "budget"
            },
            "gpt-4o-mini": {
                "name": "GPT-4o Mini", 
                "provider": "OPENAI",
                "input_cost": 0.15,
                "output_cost": 0.6,
                "max_tokens": 16384,
                "context": 128000,
                "description": "🔥 Best balance - Recommended for most translations",
                "recommended_for": "balanced"
            },
            "gpt-4.1-mini": {
                "name": "GPT-4.1 Mini",
                "provider": "OPENAI",
                "input_cost": 0.4,
                "output_cost": 1.6,
                "max_tokens": 32768,
                "context": 1047576,
                "description": "💎 Enhanced performance - Good for complex content",
                "recommended_for": "quality"
            },
            "o1-mini": {
                "name": "O1 Mini",
                "provider": "OPENAI",
                "input_cost": 1.1,
                "output_cost": 4.4,
                "max_tokens": 65536,
                "context": 128000,
                "description": "🧠 Reasoning model - Best for analytical content",
                "recommended_for": "reasoning"
            },
            "gpt-4o": {
                "name": "GPT-4o",
                "provider": "OPENAI",
                "input_cost": 2.5,
                "output_cost": 10,
                "max_tokens": 16384,
                "context": 128000,
                "description": "⭐ Premium quality - Highest accuracy translations",
                "recommended_for": "premium"
            },
            "gpt-4.1": {
                "name": "GPT-4.1",
                "provider": "OPENAI",
                "input_cost": 28,
                "output_cost": 32,
                "max_tokens": 32768,
                "context": 1047576,
                "description": "🚀 Latest technology - Advanced reasoning",
                "recommended_for": "advanced"
            },
            "gpt-4.5-preview": {
                "name": "GPT-4.5 Preview",
                "provider": "OPENAI",
                "input_cost": 75,
                "output_cost": 150,
                "max_tokens": 16384,
                "context": 128000,
                "description": "🔬 Experimental - Cutting-edge capabilities",
                "recommended_for": "experimental"
            },
            "gpt-4": {
                "name": "GPT-4",
                "provider": "OPENAI",
                "input_cost": 30,
                "output_cost": 60,
                "max_tokens": 4096,
                "context": 8192,
                "description": "🏆 Classic reliability - Proven performance",
                "recommended_for": "reliable"
            },
            "gpt-4-turbo": {
                "name": "GPT-4 Turbo",
                "provider": "OPENAI",
                "input_cost": 10,
                "output_cost": 30,
                "max_tokens": 4096,
                "context": 128000,
                "description": "⚡ Fast processing - Quick results",
                "recommended_for": "speed"
            },
            "gpt-3.5-turbo": {
                "name": "GPT-3.5 Turbo",
                "provider": "OPENAI",
                "input_cost": 0.5,
                "output_cost": 1.5,
                "max_tokens": 4097,
                "context": 16385,
                "description": "💡 Budget option - Basic translations",
                "recommended_for": "basic"
            },
            "o1-preview": {
                "name": "O1 Preview",
                "provider": "OPENAI",
                "input_cost": 15,
                "output_cost": 60,
                "max_tokens": 32768,
                "context": 128000,
                "description": "🧠 Advanced reasoning - Complex analysis",
                "recommended_for": "analysis"
            }
        }
    
    def translate_text(self, text: str, target_language: str = "فارسی") -> str:
        """Translate text using AvalAI API"""
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
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            translated_text = result['choices'][0]['message']['content'].strip()
            
            # Remove any quotes or extra formatting that might be added
            translated_text = re.sub(r'^["\']|["\']$', '', translated_text)
            
            return translated_text
            
        except requests.exceptions.RequestException as e:
            st.error(f"Translation API error: {str(e)}")
            return text  # Return original text if translation fails
        except Exception as e:
            st.error(f"Translation error: {str(e)}")
            return text

class N8NWorkflowProcessor:
    def __init__(self, translator: AvalAITranslator = None):
        self.translator = translator
    
    def detect_rtl_content(self, text: str) -> bool:
        """Detect if text contains RTL characters (Arabic, Persian, Hebrew)"""
        import unicodedata
        rtl_chars = 0
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return False
        
        for char in text:
            if unicodedata.bidirectional(char) in ['R', 'AL']:
                rtl_chars += 1
        
        # If more than 30% of alphabetic characters are RTL, consider it RTL content
        return (rtl_chars / total_chars) > 0.3 if total_chars > 0 else False
    
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
        except:
            return False
    
    def is_workflow_translated(self, workflow_data: Dict, target_language: str = "فارسی") -> bool:
        """Check if workflow appears to be already translated to target language"""
        if target_language == "فارسی":
            return self.is_workflow_rtl(workflow_data)
        
        # For other languages, check for typical patterns
        # This is a basic implementation - could be enhanced
        return False
    
    def is_workflow_already_rtl_positioned(self, workflow_data: Dict) -> bool:
        """Check if workflow nodes are already positioned in RTL layout"""
        try:
            nodes = workflow_data.get('nodes', [])
            if len(nodes) < 2:
                return False
            
            # Get positions and check if they follow RTL pattern
            positions = []
            for node in nodes:
                if 'position' in node and len(node['position']) >= 2:
                    positions.append(node['position'][0])
            
            if len(positions) < 2:
                return False
            
            # Simple heuristic: if most nodes are positioned in decreasing X order
            # (considering connection flow), it might be RTL
            # This is a basic check - could be enhanced with connection analysis
            sorted_positions = sorted(positions, reverse=True)
            return positions == sorted_positions[:len(positions)]
            
        except:
            return False
    
    def extract_node_names(self, workflow_data: Dict) -> List[Dict]:
        """Extract all node names from N8N workflow"""
        node_names = []
        
        if 'nodes' in workflow_data:
            for i, node in enumerate(workflow_data['nodes']):
                node_name = node.get('name', '')
                if node_name and not self.detect_rtl_content(node_name):
                    node_names.append({
                        'node_index': i,
                        'original_name': node_name,
                        'node_type': node.get('type', 'unknown')
                    })
        
        return node_names
    
    def translate_node_names(self, node_names: List[Dict], target_language: str = "فارسی") -> List[Dict]:
        """Translate node names to target language"""
        if not self.translator:
            raise ValueError("Translator not initialized")
        
        translated_names = []
        
        for i, node_info in enumerate(node_names):
            if node_info['original_name'].strip():
                st.write(f"🔄 Translating node {i+1}/{len(node_names)}: {node_info['original_name']}")
                
                # Create a more specific prompt for node names
                prompt = f"""Please translate the following N8N workflow node name to {target_language}. 
                Keep it concise, professional, and suitable for a technical workflow context.
                
                Node name to translate: {node_info['original_name']}
                Node type: {node_info['node_type']}
                
                Provide only the translated name without explanations."""
                
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
    
    def replace_node_names_in_workflow(self, workflow_data: Dict, translated_names: List[Dict]) -> Dict:
        """Replace original node names with translated ones in the workflow"""
        updated_workflow = copy.deepcopy(workflow_data)
        
        # Create a mapping of old names to new names
        name_mapping = {}
        for translated_name in translated_names:
            name_mapping[translated_name['original_name']] = translated_name['translated_name']
        
        # Update node names
        for translated_name in translated_names:
            node_index = translated_name['node_index']
            if node_index < len(updated_workflow['nodes']):
                updated_workflow['nodes'][node_index]['name'] = translated_name['translated_name']
        
        # Update references to node names in sticky notes
        sticky_notes = self.extract_sticky_notes(updated_workflow)
        for note in sticky_notes:
            content = note['content']
            for old_name, new_name in name_mapping.items():
                # Replace node name references in content
                content = content.replace(old_name, new_name)
            
            # Update the sticky note content
            node_index = note['node_index']
            if node_index < len(updated_workflow['nodes']):
                if 'parameters' not in updated_workflow['nodes'][node_index]:
                    updated_workflow['nodes'][node_index]['parameters'] = {}
                updated_workflow['nodes'][node_index]['parameters']['content'] = content
        
        return updated_workflow
    
    def extract_sticky_notes(self, workflow_data: Dict) -> List[Dict]:
        """Extract all sticky notes from N8N workflow"""
        sticky_notes = []
        
        if 'nodes' in workflow_data:
            for i, node in enumerate(workflow_data['nodes']):
                if node.get('type') == 'n8n-nodes-base.stickyNote':
                    sticky_notes.append({
                        'id': node.get('id'),
                        'name': node.get('name', f'Sticky Note {i+1}'),
                        'content': node.get('parameters', {}).get('content', ''),
                        'node_index': i
                    })
        
        return sticky_notes
    
    def translate_sticky_notes(self, sticky_notes: List[Dict], target_language: str) -> List[Dict]:
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
    
    def replace_notes_in_workflow(self, workflow_data: Dict, translated_notes: List[Dict]) -> Dict:
        """Replace original notes with translated ones in the workflow"""
        updated_workflow = copy.deepcopy(workflow_data)
        
        for translated_note in translated_notes:
            node_index = translated_note['node_index']
            if node_index < len(updated_workflow['nodes']):
                if 'parameters' not in updated_workflow['nodes'][node_index]:
                    updated_workflow['nodes'][node_index]['parameters'] = {}
                updated_workflow['nodes'][node_index]['parameters']['content'] = translated_note['translated_content']
        
        return updated_workflow

    def convert_ltr_to_rtl(self, workflow_json, canvas_width=None):
        """Convert LTR workflow to RTL by mirroring node positions including sticky notes"""
        try:
            # Parse JSON if it's a string
            if isinstance(workflow_json, str):
                workflow_data = json.loads(workflow_json)
            else:
                workflow_data = workflow_json
            
            # Create a deep copy to avoid modifying original
            converted_workflow = copy.deepcopy(workflow_data)
            
            # Extract all X coordinates to determine canvas boundaries
            x_coordinates = []
            for node in converted_workflow.get('nodes', []):
                if 'position' in node and len(node['position']) >= 2:
                    x_coordinates.append(node['position'][0])
                    
                    # For sticky notes, also consider their width
                    if node.get('type') == 'n8n-nodes-base.stickyNote':
                        width = node.get('parameters', {}).get('width', 0)
                        if width:
                            x_coordinates.append(node['position'][0] + width)
            
            if not x_coordinates:
                return converted_workflow
            
            min_x = min(x_coordinates)
            max_x = max(x_coordinates)
            
            # Calculate canvas width if not provided
            if canvas_width is None:
                canvas_width = max_x + abs(max_x - min_x) * 0.1
            
            # Mirror each node position
            regular_nodes = 0
            sticky_notes = 0
            
            for node in converted_workflow.get('nodes', []):
                if 'position' in node and len(node['position']) >= 2:
                    original_x, y = node['position'][0], node['position'][1]
                    
                    # For sticky notes, account for their width when mirroring
                    if node.get('type') == 'n8n-nodes-base.stickyNote':
                        sticky_notes += 1
                        width = node.get('parameters', {}).get('width', 0)
                        new_x = canvas_width - original_x - width
                    else:
                        regular_nodes += 1
                        new_x = canvas_width - original_x
                    
                    # Update the position
                    node['position'] = [new_x, y]
            
            # Store conversion stats
            st.session_state['conversion_stats'] = {
                'regular_nodes': regular_nodes,
                'sticky_notes': sticky_notes,
                'canvas_width': canvas_width
            }
            
            return converted_workflow
        
        except Exception as e:
            st.error(f"Error converting workflow: {str(e)}")
            return None

def validate_json(json_string):
    """Validate if the input is valid JSON"""
    try:
        json.loads(json_string)
        return True, "Valid JSON"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"

def calculate_estimated_cost(text_length: int, model_key: str) -> float:
    """Calculate estimated translation cost"""
    models = AvalAITranslator.get_available_models()
    model_info = models.get(model_key, {})
    
    if not model_info:
        return 0.0
    
    estimated_tokens = text_length / 4  # Rough estimate: 4 chars = 1 token
    cost = (estimated_tokens / 1_000_000) * model_info['input_cost']
    
    return cost

def create_model_selector(key_prefix: str, default_model: str = "gpt-4o-mini"):
    """Create a model selector widget"""
    models = AvalAITranslator.get_available_models()
    
    # Create model options
    model_options = []
    model_keys = []
    
    for key, model_info in models.items():
        label = f"{model_info['name']} - ${model_info['input_cost']:.2f}/${model_info['output_cost']:.2f}"
        if model_info['recommended_for'] == 'balanced':
            label = f"⭐ {label} (Recommended)"
        model_options.append(label)
        model_keys.append(key)
    
    # Find default index
    default_index = 0
    try:
        default_index = model_keys.index(default_model)
    except ValueError:
        pass
    
    selected_index = st.selectbox(
        "Choose Translation Model",
        range(len(model_options)),
        format_func=lambda x: model_options[x],
        index=default_index,
        key=f"{key_prefix}_model_select"
    )
    
    selected_model_key = model_keys[selected_index]
    selected_model_info = models[selected_model_key]
    
    return selected_model_key, selected_model_info

def translator_tab():
    """Translation functionality tab"""
    st.header("🌐 Workflow Translation")
    st.markdown("*Translate N8N workflow sticky notes using AvalAI*")
    
    # Configuration section
    col_config, col_main = st.columns([1, 2])
    
    with col_config:
        st.subheader("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input(
            "AvalAI API Key",
            type="password",
            help="Enter your AvalAI API key from https://avalai.ir",
            key="translation_api_key"
        )
        
        # Language selection
        target_language = st.selectbox(
            "Target Language",
            ["فارسی", "English", "العربية", "Español", "Français", "Deutsch"],
            index=0,
            key="translation_language"
        )
        
        # Model selection
        st.markdown("### 🤖 Model Selection")
        selected_model_key, selected_model_info = create_model_selector("translation")
        
        # Display model info
        with st.expander("📊 Model Details", expanded=False):
            st.markdown(f"**{selected_model_info['description']}**")
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Input Cost", f"${selected_model_info['input_cost']}/1M")
                st.metric("Context", f"{selected_model_info['context']:,}")
                
            with col_info2:
                st.metric("Output Cost", f"${selected_model_info['output_cost']}/1M")
                st.metric("Max Tokens", f"{selected_model_info['max_tokens']:,}")
        
        # Quick recommendations
        st.markdown("### 💡 Quick Select")
        models = AvalAITranslator.get_available_models()
        
        col_quick1, col_quick2 = st.columns(2)
        with col_quick1:
            if st.button("💰 Cheapest", key="trans_cheapest"):
                cheapest = min(models.items(), key=lambda x: x[1]['input_cost'])
                st.session_state['translation_model_select'] = list(models.keys()).index(cheapest[0])
                st.rerun()
            
            if st.button("⚖️ Balanced", key="trans_balanced"):
                for i, (key, info) in enumerate(models.items()):
                    if info['recommended_for'] == 'balanced':
                        st.session_state['translation_model_select'] = i
                        st.rerun()
                        break
        
        with col_quick2:
            if st.button("⭐ Premium", key="trans_premium"):
                for i, (key, info) in enumerate(models.items()):
                    if info['recommended_for'] == 'premium':
                        st.session_state['translation_model_select'] = i
                        st.rerun()
                        break
            
            if st.button("🧠 Reasoning", key="trans_reasoning"):
                for i, (key, info) in enumerate(models.items()):
                    if info['recommended_for'] == 'reasoning':
                        st.session_state['translation_model_select'] = i
                        st.rerun()
                        break
    
    with col_main:
        st.subheader("📥 Input Workflow")
        
        workflow_input = st.text_area(
            "Paste your N8N workflow JSON here:",
            height=300,
            placeholder="Paste your complete N8N workflow JSON here...",
            key="translation_input"
        )
        
        # Check if workflow is already translated
        if workflow_input.strip():
            try:
                workflow_data = json.loads(workflow_input)
                processor = N8NWorkflowProcessor()
                
                if processor.is_workflow_translated(workflow_data, target_language):
                    st.warning("⚠️ This workflow appears to already be translated to the target language!")
                    
            except:
                pass
        
        # Action buttons
        col_analyze, col_translate = st.columns(2)
        
        with col_analyze:
            if st.button("🔍 Analyze Workflow", type="secondary", key="analyze_btn"):
                if workflow_input.strip():
                    try:
                        workflow_data = json.loads(workflow_input)
                        processor = N8NWorkflowProcessor()
                        sticky_notes = processor.extract_sticky_notes(workflow_data)
                        
                        # Check if already translated
                        if processor.is_workflow_translated(workflow_data, target_language):
                            st.warning("⚠️ This workflow appears to already be translated!")
                            return
                        
                        st.success(f"✅ Found {len(sticky_notes)} sticky notes")
                        
                        if sticky_notes:
                            # Calculate cost estimation
                            total_chars = sum(len(note['content']) for note in sticky_notes)
                            estimated_cost = calculate_estimated_cost(total_chars, selected_model_key)
                            
                            st.info(f"💰 Estimated cost: ${estimated_cost:.4f}")
                            st.info(f"📊 Total characters: {total_chars:,}")
                            
                            # Show sticky notes
                            st.subheader("📝 Found Sticky Notes:")
                            for i, note in enumerate(sticky_notes, 1):
                                with st.expander(f"Note {i}: {note['name'][:40]}..."):
                                    st.text_area(
                                        "Content:",
                                        note['content'],
                                        height=100,
                                        disabled=True,
                                        key=f"note_preview_{i}"
                                    )
                        else:
                            st.info("No sticky notes found in this workflow")
                            
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Invalid JSON: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Error analyzing workflow: {str(e)}")
                else:
                    st.warning("Please paste your workflow JSON first")
        
        with col_translate:
            if st.button("🚀 Translate Workflow", type="primary", key="translate_btn"):
                if not api_key:
                    st.error("❌ Please enter your AvalAI API key")
                    return
                
                if not workflow_input.strip():
                    st.error("❌ Please paste your workflow JSON")
                    return
                
                try:
                    # Parse workflow
                    workflow_data = json.loads(workflow_input)
                    
                    # Initialize components
                    translator = AvalAITranslator(api_key, selected_model_key)
                    processor = N8NWorkflowProcessor(translator)
                    
                    # Check if already translated
                    if processor.is_workflow_translated(workflow_data, target_language):
                        st.warning("⚠️ This workflow appears to already be translated!")
                        return
                    
                    # Extract sticky notes
                    sticky_notes = processor.extract_sticky_notes(workflow_data)
                    
                    if not sticky_notes:
                        st.warning("⚠️ No sticky notes found to translate")
                        return
                    
                    # Show processing info
                    total_chars = sum(len(note['content']) for note in sticky_notes)
                    estimated_cost = calculate_estimated_cost(total_chars, selected_model_key)
                    
                    st.info(f"🤖 Using: {selected_model_info['name']}")
                    st.info(f"🌐 Target: {target_language}")
                    st.info(f"💰 Estimated cost: ${estimated_cost:.4f}")
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Translate notes
                    status_text.text("🔄 Translating sticky notes...")
                    with st.container():
                        translated_notes = processor.translate_sticky_notes(sticky_notes, target_language)
                    progress_bar.progress(0.7)
                    
                    # Replace in workflow
                    status_text.text("🔧 Updating workflow...")
                    updated_workflow = processor.replace_notes_in_workflow(workflow_data, translated_notes)
                    progress_bar.progress(1.0)
                    
                    status_text.text("✅ Translation completed!")
                    
                    # Store result
                    json_str = json.dumps(updated_workflow, ensure_ascii=False, indent=2)
                    st.session_state['translated_workflow'] = json_str
                    st.session_state['translation_model_used'] = selected_model_info['name']
                    
                    st.success(f"🎉 Successfully translated {len(translated_notes)} sticky notes!")
                    
                    # Preview results
                    st.subheader("🔍 Translation Preview:")
                    for i, note in enumerate(translated_notes[:3], 1):
                        with st.expander(f"Translated Note {i}: {note['name']}"):
                            col_orig, col_trans = st.columns(2)
                            with col_orig:
                                st.text_area("Original:", note['content'][:200], height=100, disabled=True, key=f"orig_{i}")
                            with col_trans:
                                st.text_area("Translated:", note['translated_content'][:200], height=100, disabled=True, key=f"trans_{i}")
                    
                except json.JSONDecodeError as e:
                    st.error(f"❌ Invalid JSON format: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Translation error: {str(e)}")
        
        # Download section
        if 'translated_workflow' in st.session_state:
            st.subheader("📥 Download Results")
            
            col_download, col_info = st.columns([2, 1])
            
            with col_download:
                st.download_button(
                    label="📥 Download Translated Workflow",
                    data=st.session_state['translated_workflow'],
                    file_name=f"translated_workflow_{target_language.lower()}_{selected_model_key}.json",
                    mime="application/json",
                    key="download_translated"
                )
            
            with col_info:
                file_size = len(st.session_state['translated_workflow'].encode('utf-8'))
                st.metric("File Size", f"{file_size:,} bytes")
                if 'translation_model_used' in st.session_state:
                    st.metric("Model Used", st.session_state['translation_model_used'])

def converter_tab():
    """LTR to RTL conversion functionality tab"""
    st.header("🔄 LTR → RTL Converter")
    st.markdown("*Convert your N8N workflows from Left-to-Right to Right-to-Left orientation*")
    
    # Create two columns for input and output
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Input: LTR Workflow JSON")
        
        ltr_json = st.text_area(
            "Paste your LTR workflow JSON here:",
            height=400,
            placeholder='{\n  "name": "My Workflow",\n  "nodes": [\n    ...\n  ]\n}',
            key="ltr_input"
        )
        
        # Check if workflow is already RTL
        if ltr_json.strip():
            try:
                workflow_data = json.loads(ltr_json)
                processor = N8NWorkflowProcessor()
                
                if processor.is_workflow_already_rtl_positioned(workflow_data):
                    st.warning("⚠️ This workflow appears to already be in RTL layout!")
                    
            except:
                pass
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            canvas_width = st.number_input(
                "Canvas Width (0 = auto-calculate)",
                min_value=0,
                value=0,
                step=50,
                help="Specify custom canvas width for mirroring",
                key="canvas_width_input"
            )
            
            show_preview = st.checkbox("Show position preview", value=True, key="show_preview")
    
    with col2:
        st.subheader("📤 Output: RTL Workflow JSON")
        
        if st.button("🔄 Convert to RTL", type="primary", key="convert_btn"):
            if ltr_json.strip():
                is_valid, validation_message = validate_json(ltr_json)
                
                if is_valid:
                    try:
                        workflow_data = json.loads(ltr_json)
                        processor = N8NWorkflowProcessor()
                        
                        # Check if already RTL
                        if processor.is_workflow_already_rtl_positioned(workflow_data):
                            st.warning("⚠️ This workflow appears to already be in RTL layout!")
                            return
                        
                        canvas_w = canvas_width if canvas_width > 0 else None
                        rtl_workflow = processor.convert_ltr_to_rtl(ltr_json, canvas_w)
                        
                        if rtl_workflow:
                            st.session_state['rtl_result'] = json.dumps(rtl_workflow, indent=2, ensure_ascii=False)
                            
                            stats = st.session_state.get('conversion_stats', {})
                            st.success(f"✅ Converted {stats.get('regular_nodes', 0)} nodes and {stats.get('sticky_notes', 0)} sticky notes to RTL!")
                        else:
                            st.error("❌ Conversion failed!")
                    except Exception as e:
                        st.error(f"❌ Error during conversion: {str(e)}")
                else:
                    st.error(f"❌ {validation_message}")
            else:
                st.warning("⚠️ Please paste your LTR workflow JSON first!")
        
        # Display result
        if 'rtl_result' in st.session_state:
            st.text_area(
                "Converted RTL workflow JSON:",
                value=st.session_state['rtl_result'],
                height=300,
                key="rtl_output"
            )
            
            st.download_button(
                label="💾 Download RTL Workflow",
                data=st.session_state['rtl_result'],
                file_name="rtl_workflow.json",
                mime="application/json",
                key="download_rtl"
            )
    
    # Position preview
    if ('rtl_result' in st.session_state and 
        ltr_json.strip() and 
        show_preview):
        
        st.subheader("📊 Position Preview")
        
        try:
            ltr_data = json.loads(ltr_json)
            rtl_data = json.loads(st.session_state['rtl_result'])
            
            col_ltr, col_rtl = st.columns(2)
            
            with col_ltr:
                st.markdown("**LTR Positions:**")
                for node in ltr_data.get('nodes', [])[:10]:  # Show first 10
                    if 'position' in node:
                        name = node.get('name', 'Unknown')[:20]
                        pos = node['position']
                        node_type = "📝" if node.get('type') == 'n8n-nodes-base.stickyNote' else "🔧"
                        st.text(f"{node_type} {name}: [{pos[0]}, {pos[1]}]")
            
            with col_rtl:
                st.markdown("**RTL Positions:**")
                for node in rtl_data.get('nodes', [])[:10]:  # Show first 10
                    if 'position' in node:
                        name = node.get('name', 'Unknown')[:20]
                        pos = node['position']
                        node_type = "📝" if node.get('type') == 'n8n-nodes-base.stickyNote' else "🔧"
                        st.text(f"{node_type} {name}: [{pos[0]}, {pos[1]}]")
                        
        except Exception as e:
            st.error(f"Error showing preview: {str(e)}")

def node_renamer_tab():
    """Node name translation functionality tab"""
    st.header("🏷️ Node Name Translation")
    st.markdown("*Translate N8N workflow node names to Persian for better readability*")
    
    # Configuration section
    col_config, col_main = st.columns([1, 2])
    
    with col_config:
        st.subheader("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input(
            "AvalAI API Key",
            type="password",
            help="Enter your AvalAI API key from https://avalai.ir",
            key="node_renamer_api_key"
        )
        
        # Model selection
        st.markdown("### 🤖 Model Selection")
        selected_model_key, selected_model_info = create_model_selector("node_renamer")
        
        # Display model info
        with st.expander("📊 Model Details", expanded=False):
            st.markdown(f"**{selected_model_info['description']}**")
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Input Cost", f"${selected_model_info['input_cost']}/1M")
                st.metric("Context", f"{selected_model_info['context']:,}")
                
            with col_info2:
                st.metric("Output Cost", f"${selected_model_info['output_cost']}/1M")
                st.metric("Max Tokens", f"{selected_model_info['max_tokens']:,}")
        
        st.markdown("### 📋 What This Does:")
        st.markdown("""
        - Translates node names to Persian
        - Updates node name references in sticky notes
        - Preserves all functionality and connections
        - Makes workflows more readable in Persian
        """)
        
        st.markdown("### 💡 Examples:")
        st.markdown("""
        - `HTTP Request` → `درخواست HTTP`
        - `Set Variable` → `تنظیم متغیر`
        - `Send Email` → `ارسال ایمیل`
        - `Webhook Trigger` → `محرک وب‌هوک`
        """)
    
    with col_main:
        st.subheader("📥 Input Workflow")
        
        workflow_input = st.text_area(
            "Paste your N8N workflow JSON here:",
            height=300,
            placeholder="Paste your N8N workflow JSON here...",
            key="node_renamer_input"
        )
        
        # Action buttons
        col_analyze, col_translate = st.columns(2)
        
        with col_analyze:
            if st.button("🔍 Analyze Node Names", type="secondary", key="analyze_nodes_btn"):
                if workflow_input.strip():
                    try:
                        workflow_data = json.loads(workflow_input)
                        processor = N8NWorkflowProcessor()
                        node_names = processor.extract_node_names(workflow_data)
                        
                        st.success(f"✅ Found {len(node_names)} node names to translate")
                        
                        if node_names:
                            # Calculate cost estimation
                            total_chars = sum(len(node['original_name']) for node in node_names)
                            estimated_cost = calculate_estimated_cost(total_chars, selected_model_key)
                            
                            st.info(f"💰 Estimated cost: ${estimated_cost:.4f}")
                            st.info(f"📊 Total characters: {total_chars:,}")
                            
                            # Show node names
                            st.subheader("🔧 Node Names to Translate:")
                            for i, node in enumerate(node_names, 1):
                                with st.expander(f"Node {i}: {node['original_name']}"):
                                    st.text(f"Type: {node['node_type']}")
                                    st.text(f"Name: {node['original_name']}")
                        else:
                            st.info("No translatable node names found in this workflow")
                            
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Invalid JSON: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Error analyzing workflow: {str(e)}")
                else:
                    st.warning("Please paste your workflow JSON first")
        
        with col_translate:
            if st.button("🏷️ Translate Node Names", type="primary", key="translate_nodes_btn"):
                if not api_key:
                    st.error("❌ Please enter your AvalAI API key")
                    return
                
                if not workflow_input.strip():
                    st.error("❌ Please paste your workflow JSON")
                    return
                
                try:
                    # Parse workflow
                    workflow_data = json.loads(workflow_input)
                    
                    # Initialize components
                    translator = AvalAITranslator(api_key, selected_model_key)
                    processor = N8NWorkflowProcessor(translator)
                    
                    # Extract node names
                    node_names = processor.extract_node_names(workflow_data)
                    
                    if not node_names:
                        st.warning("⚠️ No translatable node names found")
                        return
                    
                    # Show processing info
                    total_chars = sum(len(node['original_name']) for node in node_names)
                    estimated_cost = calculate_estimated_cost(total_chars, selected_model_key)
                    
                    st.info(f"🤖 Using: {selected_model_info['name']}")
                    st.info(f"🏷️ Translating: {len(node_names)} node names")
                    st.info(f"💰 Estimated cost: ${estimated_cost:.4f}")
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Translate node names
                    status_text.text("🔄 Translating node names...")
                    with st.container():
                        translated_node_names = processor.translate_node_names(node_names, "فارسی")
                    progress_bar.progress(0.7)
                    
                    # Replace in workflow
                    status_text.text("🔧 Updating workflow...")
                    updated_workflow = processor.replace_node_names_in_workflow(workflow_data, translated_node_names)
                    progress_bar.progress(1.0)
                    
                    status_text.text("✅ Translation completed!")
                    
                    # Store result
                    json_str = json.dumps(updated_workflow, ensure_ascii=False, indent=2)
                    st.session_state['translated_nodes_workflow'] = json_str
                    st.session_state['node_translation_model_used'] = selected_model_info['name']
                    
                    st.success(f"🎉 Successfully translated {len(translated_node_names)} node names!")
                    
                    # Preview results
                    st.subheader("🔍 Translation Preview:")
                    for i, node in enumerate(translated_node_names[:5], 1):
                        with st.expander(f"Translated Node {i}"):
                            col_orig, col_trans = st.columns(2)
                            with col_orig:
                                st.text_area("Original:", node['original_name'], height=50, disabled=True, key=f"node_orig_{i}")
                            with col_trans:
                                st.text_area("Translated:", node['translated_name'], height=50, disabled=True, key=f"node_trans_{i}")
                    
                except json.JSONDecodeError as e:
                    st.error(f"❌ Invalid JSON format: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Translation error: {str(e)}")
        
        # Download section
        if 'translated_nodes_workflow' in st.session_state:
            st.subheader("📥 Download Results")
            
            col_download, col_info = st.columns([2, 1])
            
            with col_download:
                st.download_button(
                    label="📥 Download Workflow with Translated Names",
                    data=st.session_state['translated_nodes_workflow'],
                    file_name=f"persian_node_names_{selected_model_key}.json",
                    mime="application/json",
                    key="download_translated_nodes"
                )
            
            with col_info:
                file_size = len(st.session_state['translated_nodes_workflow'].encode('utf-8'))
                st.metric("File Size", f"{file_size:,} bytes")
                if 'node_translation_model_used' in st.session_state:
                    st.metric("Model Used", st.session_state['node_translation_model_used'])

def combined_tab():
    """Combined functionality: Translation + RTL conversion + Node renaming"""
    st.header("🔄🌐🏷️ Complete Workflow Localization")
    st.markdown("*Full Persian localization: Translate notes + Convert to RTL + Rename nodes*")
    
    # Configuration
    with st.expander("⚙️ Configuration", expanded=True):
        col_config1, col_config2, col_config3 = st.columns(3)
        
        with col_config1:
            api_key = st.text_input(
                "AvalAI API Key",
                type="password",
                help="Enter your AvalAI API key",
                key="combined_api_key"
            )
            
            target_language = st.selectbox(
                "Target Language",
                ["فارسی", "English", "العربية", "Español", "Français", "Deutsch"],
                index=0,
                key="combined_language"
            )
        
        with col_config2:
            selected_model_key, selected_model_info = create_model_selector("combined")
            
            canvas_width = st.number_input(
                "Canvas Width (0 = auto)",
                min_value=0,
                value=0,
                step=50,
                key="combined_canvas_width"
            )
        
        with col_config3:
            st.markdown("**Processing Options:**")
            
            translate_notes = st.checkbox(
                "🌐 Translate Sticky Notes",
                value=True,
                key="combined_translate_notes"
            )
            
            translate_node_names = st.checkbox(
                "🏷️ Translate Node Names",
                value=True,
                key="combined_translate_nodes"
            )
            
            convert_to_rtl = st.checkbox(
                "🔄 Convert to RTL Layout",
                value=True,
                key="combined_convert_rtl"
            )
            
            st.markdown("**Selected Model:**")
            st.info(f"🤖 {selected_model_info['name']}")
            st.info(f"💰 ${selected_model_info['input_cost']}/{selected_model_info['output_cost']}")
    
    # Input section
    st.subheader("📥 Input Workflow")
    workflow_input = st.text_area(
        "Paste your N8N workflow JSON here:",
        height=300,
        placeholder="Paste your N8N workflow JSON here for complete localization...",
        key="combined_input"
    )
    
    # Check workflow status
    if workflow_input.strip():
        try:
            workflow_data = json.loads(workflow_input)
            processor = N8NWorkflowProcessor()
            
            col_status1, col_status2, col_status3 = st.columns(3)
            
            with col_status1:
                if processor.is_workflow_translated(workflow_data, target_language):
                    st.warning("⚠️ Notes appear translated")
                else:
                    st.success("✅ Notes ready for translation")
            
            with col_status2:
                if processor.is_workflow_rtl(workflow_data):
                    st.warning("⚠️ Node names appear Persian")
                else:
                    st.success("✅ Node names ready for translation")
                    
            with col_status3:
                if processor.is_workflow_already_rtl_positioned(workflow_data):
                    st.warning("⚠️ Layout appears RTL")
                else:
                    st.success("✅ Layout ready for RTL conversion")
                    
        except:
            st.error("❌ Invalid JSON format")
    
    # Process section
    col_process, col_info = st.columns([2, 1])
    
    with col_process:
        if st.button("🚀 Complete Localization", type="primary", key="combined_btn"):
            if not api_key:
                st.error("❌ Please enter your AvalAI API key")
                return
            
            if not workflow_input.strip():
                st.error("❌ Please paste your workflow JSON")
                return
            
            if not any([translate_notes, translate_node_names, convert_to_rtl]):
                st.error("❌ Please select at least one processing option")
                return
            
            try:
                # Parse workflow
                workflow_data = json.loads(workflow_input)
                
                # Initialize components
                translator = AvalAITranslator(api_key, selected_model_key)
                processor = N8NWorkflowProcessor(translator)
                
                current_workflow = workflow_data
                total_steps = sum([translate_notes, translate_node_names, convert_to_rtl])
                current_step = 0
                
                # Step 1: Translate Sticky Notes
                if translate_notes:
                    current_step += 1
                    st.subheader(f"🌐 Step {current_step}/{total_steps}: Translating Sticky Notes")
                    
                    sticky_notes = processor.extract_sticky_notes(current_workflow)
                    
                    if sticky_notes:
                        if processor.is_workflow_translated(current_workflow, target_language):
                            st.warning("⚠️ Sticky notes appear already translated, skipping...")
                        else:
                            total_chars = sum(len(note['content']) for note in sticky_notes)
                            estimated_cost = calculate_estimated_cost(total_chars, selected_model_key)
                            st.info(f"🤖 Processing {len(sticky_notes)} notes | 💰 Cost: ${estimated_cost:.4f}")
                            
                            progress_bar = st.progress(0)
                            
                            with st.container():
                                translated_notes = processor.translate_sticky_notes(sticky_notes, target_language)
                            
                            current_workflow = processor.replace_notes_in_workflow(current_workflow, translated_notes)
                            progress_bar.progress(current_step / total_steps)
                            st.success(f"✅ Translated {len(translated_notes)} sticky notes")
                    else:
                        st.info("ℹ️ No sticky notes found, skipping translation")
                
                # Step 2: Translate Node Names
                if translate_node_names:
                    current_step += 1
                    st.subheader(f"🏷️ Step {current_step}/{total_steps}: Translating Node Names")
                    
                    node_names = processor.extract_node_names(current_workflow)
                    
                    if node_names:
                        if processor.is_workflow_rtl(current_workflow):
                            st.warning("⚠️ Node names appear already translated, skipping...")
                        else:
                            total_chars = sum(len(node['original_name']) for node in node_names)
                            estimated_cost = calculate_estimated_cost(total_chars, selected_model_key)
                            st.info(f"🤖 Processing {len(node_names)} node names | 💰 Cost: ${estimated_cost:.4f}")
                            
                            progress_bar = st.progress(current_step / total_steps)
                            
                            with st.container():
                                translated_node_names = processor.translate_node_names(node_names, "فارسی")
                            
                            current_workflow = processor.replace_node_names_in_workflow(current_workflow, translated_node_names)
                            progress_bar.progress(current_step / total_steps)
                            st.success(f"✅ Translated {len(translated_node_names)} node names")
                    else:
                        st.info("ℹ️ No translatable node names found, skipping")
                
                # Step 3: RTL Conversion
                if convert_to_rtl:
                    current_step += 1
                    st.subheader(f"🔄 Step {current_step}/{total_steps}: Converting to RTL Layout")
                    
                    if processor.is_workflow_already_rtl_positioned(current_workflow):
                        st.warning("⚠️ Workflow appears already in RTL layout, skipping...")
                    else:
                        canvas_w = canvas_width if canvas_width > 0 else None
                        final_workflow = processor.convert_ltr_to_rtl(current_workflow, canvas_w)
                        
                        if final_workflow:
                            current_workflow = final_workflow
                            progress_bar = st.progress(1.0)
                            
                            # Show final stats
                            stats = st.session_state.get('conversion_stats', {})
                            st.success(f"✅ Converted {stats.get('regular_nodes', 0)} nodes + {stats.get('sticky_notes', 0)} sticky notes to RTL")
                        else:
                            st.error("❌ RTL conversion failed!")
                            return
                
                # Store final result
                json_str = json.dumps(current_workflow, indent=2, ensure_ascii=False)
                st.session_state['combined_result'] = json_str
                st.session_state['combined_model_used'] = selected_model_info['name']
                st.session_state['combined_steps_completed'] = [
                    "🌐 Sticky Notes" if translate_notes else "",
                    "🏷️ Node Names" if translate_node_names else "",
                    "🔄 RTL Layout" if convert_to_rtl else ""
                ]
                
                st.success("🎉 Complete localization finished!")
                    
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON: {str(e)}")
            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")
    
    with col_info:
        if workflow_input.strip():
            try:
                workflow_data = json.loads(workflow_input)
                processor = N8NWorkflowProcessor()
                sticky_notes = processor.extract_sticky_notes(workflow_data)
                node_names = processor.extract_node_names(workflow_data)
                total_nodes = len(workflow_data.get('nodes', []))
                
                st.metric("Total Nodes", total_nodes)
                st.metric("Sticky Notes", len(sticky_notes))
                st.metric("Node Names", len(node_names))
                
                # Show cost estimate for selected operations
                total_cost = 0
                if translate_notes and sticky_notes:
                    notes_chars = sum(len(note['content']) for note in sticky_notes)
                    total_cost += calculate_estimated_cost(notes_chars, selected_model_key)
                
                if translate_node_names and node_names:
                    names_chars = sum(len(node['original_name']) for node in node_names)
                    total_cost += calculate_estimated_cost(names_chars, selected_model_key)
                
                if total_cost > 0:
                    st.metric("Est. Total Cost", f"${total_cost:.4f}")
                
            except:
                st.metric("Status", "Invalid JSON")
    
    # Results section
    if 'combined_result' in st.session_state:
        st.subheader("📥 Download Results")
        
        col_download, col_preview = st.columns([1, 1])
        
        with col_download:
            # Create descriptive filename
            steps_completed = [step for step in st.session_state.get('combined_steps_completed', []) if step]
            filename_parts = []
            if "🌐 Sticky Notes" in str(steps_completed):
                filename_parts.append("translated")
            if "🏷️ Node Names" in str(steps_completed):
                filename_parts.append("persian_names")
            if "🔄 RTL Layout" in str(steps_completed):
                filename_parts.append("rtl")
            
            filename = f"localized_workflow_{'_'.join(filename_parts)}_{selected_model_key}.json"
            
            st.download_button(
                label="📥 Download Localized Workflow",
                data=st.session_state['combined_result'],
                file_name=filename,
                mime="application/json",
                key="download_combined"
            )
            
            file_size = len(st.session_state['combined_result'].encode('utf-8'))
            st.info(f"📊 Size: {file_size:,} bytes")
            if 'combined_model_used' in st.session_state:
                st.info(f"🤖 Model: {st.session_state['combined_model_used']}")
            
            # Show completed steps
            if 'combined_steps_completed' in st.session_state:
                completed_steps = [step for step in st.session_state['combined_steps_completed'] if step]
                if completed_steps:
                    st.success(f"✅ Completed: {' + '.join(completed_steps)}")
        
        with col_preview:
            if st.button("👁️ Preview Result", key="preview_btn"):
                with st.expander("🔍 Workflow Preview", expanded=True):
                    preview_text = st.session_state['combined_result']
                    if len(preview_text) > 1000:
                        preview_text = preview_text[:1000] + "..."
                    st.code(preview_text, language="json")

def main():
    st.set_page_config(
        page_title="N8N Workflow Tools",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔧 N8N Workflow Tools")
    st.markdown("*Complete toolkit for N8N workflow translation and RTL conversion*")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 Translation", 
        "🔄 LTR → RTL", 
        "🏷️ Node Names",
        "🔄🌐🏷️ Complete"
    ])
    
    with tab1:
        translator_tab()
    
    with tab2:
        converter_tab()
    
    with tab3:
        node_renamer_tab()
    
    with tab4:
        combined_tab()
    
    # Sidebar
    with st.sidebar:
        st.header("📖 Guide & Tools")
        
        # Model comparison
        st.markdown("### 🤖 Model Comparison")
        models = AvalAITranslator.get_available_models()
        
        # Show top 5 most cost-effective models
        sorted_models = sorted(models.items(), key=lambda x: x[1]['input_cost'])
        
        for model_key, model_info in sorted_models[:5]:
            st.markdown(f"**{model_info['name']}**")
            st.markdown(f"💰 ${model_info['input_cost']}/{model_info['output_cost']} | {model_info['description']}")
            st.markdown("---")
        
        # Cost calculator
        st.markdown("### 💰 Cost Calculator")
        
        calc_chars = st.number_input(
            "Characters to translate:",
            min_value=0,
            value=1000,
            step=100
        )
        
        calc_model = st.selectbox(
            "Model for calculation:",
            options=list(models.keys()),
            format_func=lambda x: models[x]['name'],
            index=1  # Default to gpt-4o-mini
        )
        
        if calc_chars > 0:
            calc_cost = calculate_estimated_cost(calc_chars, calc_model)
            st.metric("Estimated Cost", f"${calc_cost:.4f}")
        
        # Instructions
        st.markdown("### 📋 Instructions")
        
        with st.expander("🌐 Sticky Notes Translation", expanded=False):
            st.markdown("""
            1. Enter AvalAI API key
            2. Choose model & language
            3. Paste workflow JSON
            4. Click 'Analyze' to preview
            5. Click 'Translate' to process
            6. Download result
            """)
        
        with st.expander("🔄 LTR → RTL Conversion", expanded=False):
            st.markdown("""
            1. Paste LTR workflow JSON
            2. Set canvas width (optional)
            3. Click 'Convert to RTL'
            4. Review position preview
            5. Download RTL workflow
            """)
        
        with st.expander("🏷️ Node Name Translation", expanded=False):
            st.markdown("""
            1. Configure API key & model
            2. Paste workflow JSON
            3. Click 'Analyze Node Names'
            4. Click 'Translate Node Names'
            5. Download result with Persian names
            """)
        
        with st.expander("🔄🌐🏷️ Complete Localization", expanded=False):
            st.markdown("""
            1. Configure API key & model
            2. Select desired operations
            3. Paste workflow JSON
            4. Click 'Complete Localization'
            5. Download fully localized workflow
            """)
        
        # Tips
        st.markdown("### 💡 Pro Tips")
        st.markdown("""
        - **gpt-4.1-nano**: Ultra cheap for basic translations
        - **gpt-4o-mini**: Best balance (recommended)
        - **gpt-4o**: Premium quality for important work
        - Always backup original workflows
        - Test translations before production use
        - Use complete localization for Persian workflows
        """)
        
        # Feature highlights
        st.markdown("### ✨ New Features")
        st.markdown("""
        - **Smart Detection**: Automatically detects already translated/RTL content
        - **Node Name Translation**: Persian node names for better readability
        - **Complete Localization**: All-in-one Persian workflow conversion
        - **Cost Optimization**: Real-time cost calculation
        - **Enhanced Safety**: Prevents duplicate processing
        """)
        
        # Links
        st.markdown("### 🔗 Useful Links")
        st.markdown("""
        - [AvalAI Dashboard](https://avalai.ir) - Get API key
        - [N8N Documentation](https://docs.n8n.io) - Workflow help
        - [GitHub Issues](https://github.com) - Report bugs
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p><strong>🚀 N8N Workflow Tools v2.0</strong></p>
            <p>Built with ❤️ using Streamlit & AvalAI</p>
            <p>🌐 Translation • 🔄 RTL Conversion • 🏷️ Node Names • 🤖 AI Models • 💰 Cost Optimization</p>
            <p><em>Complete solution for multilingual workflow localization</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()