# N8N Workflow Tools v2.0 Enhanced

Complete toolkit for N8N workflow translation, RTL conversion, and localization with dynamic configuration and cross-version support.

## 🚀 Features

### Core Capabilities
- **🌐 Sticky Notes Translation**: Translate sticky notes using AvalAI API (OpenAI models)
- **🔄 LTR → RTL Conversion**: Mirror workflow layouts for right-to-left languages
- **🏷️ Node Name Translation**: Translate node names for better readability
- **🔄🌐🏷️ Complete Localization**: All-in-one workflow localization

### Enhanced Features v2.0
- **🔍 Version Detection**: Automatically detects N8N workflow version (0.x, 1.x)
- **🛠️ Dynamic Configuration**: Customizable settings via `config.json`
- **✅ Auto-validation**: Validates and auto-fixes workflow structure issues
- **🌐 Multi-version Support**: Compatible with N8N 0.x and 1.x workflows
- **🧠 Smart Detection**: Detects already translated/RTL content to prevent duplicate processing
- **💰 Cost Optimization**: Real-time translation cost calculation
- **🔒 Enhanced Safety**: Comprehensive error handling and fallback mechanisms

## 📋 Requirements

```bash
pip install streamlit requests
```

## 🎯 Quick Start

1. **Get API Key**: Sign up at [AvalAI](https://avalai.ir) to get your API key

2. **Run the application**:
   ```bash
   streamlit run convert_final_ver.py
   ```

3. **Configure** (Optional): Customize settings in `config.json`

## 📁 Project Structure

```
Convert_workflows/
├── convert_final_ver.py      # Main application with UI
├── convert_enhanced.py        # Enhanced backend (standalone version)
├── config_loader.py           # Configuration loader and validators
├── config.json                # Configuration file
└── README.md                  # This file
```

## ⚙️ Configuration

### Configuration File: `config.json`

The application uses a dynamic configuration system that allows you to customize behavior without modifying code.

#### API Configuration
```json
{
  "api": {
    "base_url": "https://api.avalai.ir/v1",
    "endpoint": "/chat/completions",
    "timeout": 30,
    "default_model": "gpt-4o-mini",
    "default_temperature": 0.3,
    "max_tokens": 2000
  }
}
```

#### N8N Version Support
```json
{
  "n8n": {
    "supported_versions": ["0.x", "1.x"],
    "node_types": {
      "sticky_note": [
        "n8n-nodes-base.stickyNote",
        "@n8n/n8n-nodes-base.stickyNote",
        "n8n-nodes-base.sticky"
      ]
    }
  }
}
```

#### Translation Settings
```json
{
  "translation": {
    "rtl_detection_threshold": 0.3,
    "token_to_char_ratio": 4,
    "supported_languages": {
      "فارسی": "fa",
      "English": "en",
      "العربية": "ar",
      "Español": "es",
      "Français": "fr",
      "Deutsch": "de"
    }
  }
}
```

#### Layout Configuration
```json
{
  "layout": {
    "canvas_width_buffer": 0.1,
    "preview_node_limit": 10,
    "default_sticky_width": 300
  }
}
```

#### Model Configuration
Models are configured with pricing and specifications:
```json
{
  "models": {
    "gpt-4o-mini": {
      "name": "GPT-4o Mini",
      "description": "مدل کم‌هزینه و سریع - مناسب برای ترجمه‌های ساده",
      "input_cost": 0.00015,
      "output_cost": 0.0006,
      "context_window": 128000
    }
  }
}
```

## 🔧 Usage Guide

### 1. Sticky Notes Translation

**Purpose**: Translate sticky note content in your N8N workflows

**Steps**:
1. Enter your AvalAI API key
2. Select target language (default: Persian)
3. Choose AI model (recommended: gpt-4o-mini for balance)
4. Paste your N8N workflow JSON
5. Click "🔍 Analyze Workflow" to preview
6. Click "🌐 Translate Workflow" to process
7. Download the translated workflow

**Features**:
- Automatic detection of already translated content
- Real-time cost estimation
- Support for multiple languages
- Batch processing of multiple sticky notes

### 2. LTR → RTL Conversion

**Purpose**: Mirror workflow layout for right-to-left languages

**Steps**:
1. Paste your LTR workflow JSON
2. (Optional) Set custom canvas width
3. Click "🔄 Convert to RTL"
4. Review position changes in the preview
5. Download RTL workflow

**Features**:
- Automatic canvas width calculation
- Sticky note width consideration
- Position preview with before/after comparison
- Handles both regular nodes and sticky notes

### 3. Node Name Translation

**Purpose**: Translate node names for better readability

**Steps**:
1. Enter API key and select model
2. Choose target language
3. Paste workflow JSON
4. Click "📊 Analyze Node Names"
5. Click "🏷️ Translate Node Names"
6. Download workflow with translated names

**Features**:
- Filters already translated node names
- Updates references in sticky notes
- Preserves node connections
- Shows translation progress

### 4. Complete Localization

**Purpose**: All-in-one workflow localization (recommended for Persian)

**Steps**:
1. Configure API key and model
2. Select operations to perform:
   - 🌐 Translate sticky notes
   - 🏷️ Translate node names
   - 🔄 Convert to RTL layout
3. Paste workflow JSON
4. Click "🔄🌐🏷️ Complete Localization"
5. Download fully localized workflow

**Features**:
- One-click complete localization
- Configurable steps
- Comprehensive cost estimation
- Progress tracking for each step

## 🔍 Version Detection

The enhanced version automatically detects N8N workflow versions:

### Detection Methods
1. **Metadata inspection**: Checks for `meta.instanceId` (v1.x)
2. **Version fields**: Looks for `version` or `versionId`
3. **Node structure**: Analyzes node `typeVersion` and type prefixes
4. **Type prefix patterns**:
   - `@n8n/` prefix → v1.x
   - `n8n-nodes-base.` prefix → v0.x

### Supported Versions
- **N8N 0.x**: Full support with legacy node types
- **N8N 1.x**: Full support with new node type format

## ✅ Workflow Validation

The application automatically validates and fixes common workflow issues:

### Validation Checks
- Required workflow fields (`nodes` array)
- Node structure (id, name, type, position)
- Position array format `[x, y]`
- Sticky note parameters
- Parameter structure integrity

### Auto-fix Features
- Adds missing `nodes` array
- Fixes invalid positions (sets to `[0, 0]`)
- Adds missing sticky note parameters
- Creates default content fields

## 💰 Cost Optimization

### Real-time Cost Calculation
The application calculates estimated costs before translation:

```
Estimated Cost = (Characters / Token Ratio) / 1,000,000 × Model Input Cost
```

### Model Recommendations
- **💰 Budget**: gpt-4.1-nano, gpt-4o-mini
- **⚖️ Balanced**: gpt-4o-mini (recommended)
- **⭐ Premium**: gpt-4o, claude-3-5-sonnet
- **🧠 Reasoning**: o1-mini, o1-preview

### Cost-Saving Tips
1. Use cheaper models for simple sticky notes
2. Use premium models only for complex technical content
3. Batch multiple workflows to optimize API calls
4. Review translations with cheaper models first

## 🛡️ Error Handling

### Comprehensive Error Management
- **Network Timeouts**: Configurable timeout with retry suggestions
- **API Errors**: Clear error messages with troubleshooting tips
- **JSON Parsing**: Validation before processing
- **Workflow Structure**: Auto-fix common issues
- **Translation Failures**: Graceful fallback to original text

### Safety Features
- **Duplicate Detection**: Prevents re-translating already translated content
- **Position Validation**: Ensures valid node positions
- **Data Integrity**: Deep copying prevents original data modification
- **Fallback Mechanisms**: Returns original data on critical errors

## 🔌 API Integration

### AvalAI API
This tool integrates with [AvalAI](https://avalai.ir), providing access to multiple AI models:

**Supported Models**:
- OpenAI: GPT-4, GPT-4 Turbo, GPT-4o, GPT-3.5 Turbo, o1-series
- Claude: Claude 3 Opus, Sonnet, Haiku, Claude 3.5 Sonnet
- Gemini: Gemini 1.5 Pro, Flash

**API Configuration**:
```python
from config_loader import ConfigLoader

config = ConfigLoader()
api_config = config.get_api_config()
# Use api_config for custom implementations
```

## 🧪 Testing

### Manual Testing Checklist
- [ ] Test with N8N 0.x workflow
- [ ] Test with N8N 1.x workflow
- [ ] Test sticky note translation
- [ ] Test node name translation
- [ ] Test LTR → RTL conversion
- [ ] Test complete localization
- [ ] Test with empty workflow
- [ ] Test with malformed JSON
- [ ] Test with already translated workflow
- [ ] Test with invalid API key

### Configuration Testing
```python
from config_loader import ConfigLoader, N8NVersionDetector, WorkflowValidator

# Test configuration loading
config = ConfigLoader('config.json')
print(config.get_supported_n8n_versions())

# Test version detection
detector = N8NVersionDetector(config)
workflow_data = {...}  # Your workflow JSON
version = detector.detect_version(workflow_data)
print(f"Detected version: {version}")

# Test validation
validator = WorkflowValidator(config)
is_valid, errors = validator.validate_workflow(workflow_data)
print(f"Valid: {is_valid}, Errors: {errors}")
```

## 📊 Model Comparison

| Model | Input Cost | Output Cost | Context | Best For |
|-------|------------|-------------|---------|----------|
| gpt-4.1-nano | $0.10/1M | $0.40/1M | 1M | Ultra budget |
| gpt-4o-mini | $0.15/1M | $0.60/1M | 128K | **Recommended** |
| gpt-4o | $2.50/1M | $10.00/1M | 128K | Premium quality |
| claude-3-5-sonnet | $3.00/1M | $15.00/1M | 200K | High accuracy |
| gemini-1.5-flash | $0.075/1M | $0.30/1M | 1M | Speed |

*Prices in USD per 1 million tokens*

## 🐛 Troubleshooting

### Common Issues

**1. Configuration not loading**
```
Error: Configuration file not found: config.json
```
**Solution**: Ensure `config.json` is in the same directory as the script

**2. Import error**
```
Warning: Configuration modules not found. Running in legacy mode.
```
**Solution**: Ensure `config_loader.py` is present. Legacy mode still works but without enhanced features.

**3. API timeout**
```
Translation timeout after 30 seconds
```
**Solution**: Increase timeout in `config.json`:
```json
{"api": {"timeout": 60}}
```

**4. Version not detected**
```
Detected version: unknown
```
**Solution**: Version detection fallback is safe. The tool will use default node types.

**5. Invalid JSON**
```
Invalid JSON format: ...
```
**Solution**: Validate your JSON using a JSON validator before pasting

## 🔄 Migration Guide

### From v1.0 to v2.0

**Breaking Changes**: None! v2.0 is fully backward compatible.

**New Features**:
1. Configuration file support (optional)
2. Version detection (automatic)
3. Auto-validation (automatic)

**Migration Steps**:
1. Keep using `convert_final_ver.py` as before
2. (Optional) Add `config.json` for customization
3. Enjoy enhanced features automatically

## 🤝 Contributing

### Adding New Models
Edit `config.json` to add new models:
```json
{
  "models": {
    "your-model-id": {
      "name": "Model Name",
      "description": "Description",
      "input_cost": 0.001,
      "output_cost": 0.002,
      "context_window": 128000
    }
  }
}
```

### Adding New N8N Node Types
Edit `config.json` to support new sticky note types:
```json
{
  "n8n": {
    "node_types": {
      "sticky_note": [
        "existing-type",
        "new-sticky-note-type"
      ]
    }
  }
}
```

### Adding New Languages
Edit `config.json`:
```json
{
  "translation": {
    "supported_languages": {
      "NewLanguage": "lang_code"
    }
  }
}
```

## 📚 Advanced Usage

### Using as a Library

```python
from config_loader import ConfigLoader
from convert_enhanced import AvalAITranslator, N8NWorkflowProcessor

# Initialize with configuration
config = ConfigLoader('config.json')
translator = AvalAITranslator(api_key="your-key", config=config)
processor = N8NWorkflowProcessor(translator, config)

# Get workflow info
workflow_data = {...}  # Your workflow JSON
info = processor.get_workflow_info(workflow_data)
print(f"Version: {info['version']}")
print(f"Nodes: {info['total_nodes']}")
print(f"Sticky notes: {info['sticky_notes_count']}")

# Translate sticky notes
sticky_notes = processor.extract_sticky_notes(workflow_data)
translated = processor.translate_sticky_notes(sticky_notes, "فارسی")
updated_workflow = processor.replace_notes_in_workflow(workflow_data, translated)

# Convert to RTL
rtl_workflow = processor.convert_ltr_to_rtl(updated_workflow)
```

### Custom Configuration

Create a custom configuration file:
```python
import json

custom_config = {
    "api": {
        "timeout": 60,
        "default_model": "gpt-4o"
    },
    "translation": {
        "rtl_detection_threshold": 0.4
    }
}

with open('custom_config.json', 'w') as f:
    json.dump(custom_config, f, indent=2)

# Use custom config
config = ConfigLoader('custom_config.json')
```

## 📞 Support

- **Documentation**: This README
- **Issues**: Report bugs or request features
- **API Support**: [AvalAI Dashboard](https://avalai.ir)
- **N8N Help**: [N8N Documentation](https://docs.n8n.io)

## 📄 License

This project is provided as-is for N8N workflow localization purposes.

## 🙏 Acknowledgments

- **Streamlit**: For the amazing UI framework
- **AvalAI**: For providing access to multiple AI models
- **N8N**: For the excellent workflow automation platform

---

## 🚀 What's New in v2.0

### Major Enhancements
✅ Dynamic configuration system
✅ N8N version detection (0.x, 1.x)
✅ Automatic workflow validation
✅ Cross-version node type support
✅ Enhanced error handling
✅ Configurable timeouts and thresholds
✅ Auto-fix for common issues
✅ Better cost estimation
✅ Improved user feedback

### Performance Improvements
- Faster validation with caching
- Optimized API calls
- Better error recovery
- Reduced memory usage

### Developer Experience
- Modular architecture
- Standalone library (`convert_enhanced.py`)
- Configuration API
- Comprehensive validation
- Better code documentation

---

**Built with ❤️ for the N8N community**

*Complete solution for multilingual workflow localization*
