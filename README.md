# N8N Workflow Converter v3.0 - Enhanced with Pydantic AI

Complete toolkit for N8N workflow translation, RTL conversion, and localization with **Pydantic AI agents** and modern web interface.

## 🚀 What's New in v3.0

- **🎨 Modern Web Interface**: Beautiful HTML/CSS/JS frontend with FastAPI backend
- **🤖 Pydantic AI Agents**: Enhanced capabilities using Pydantic AI framework
- **⚡ FastAPI Backend**: High-performance async API
- **🔧 Modular Architecture**: Clean separation of agents, services, and API

## 📋 Features

### Core Capabilities
- **🌐 Sticky Notes Translation**: Translate sticky notes using AvalAI API (OpenAI models)
- **🔄 LTR → RTL Conversion**: Mirror workflow layouts for right-to-left languages
- **🏷️ Node Name Translation**: Translate node names for better readability
- **🔄🌐🏷️ Complete Localization**: All-in-one workflow localization

### Enhanced Features
- **🔍 Version Detection**: Automatically detects N8N workflow version (0.x, 1.x)
- **🛠️ Dynamic Configuration**: Customizable settings via `config.json`
- **✅ Auto-validation**: Validates and auto-fixes workflow structure issues
- **🌐 Multi-version Support**: Compatible with N8N 0.x and 1.x workflows
- **🧠 Smart Detection**: Detects already translated/RTL content to prevent duplicate processing
- **💰 Cost Optimization**: Real-time translation cost calculation
- **🔒 Enhanced Safety**: Comprehensive error handling and fallback mechanisms

## 📦 Installation

### Option 1: Docker (Recommended)

1. **Build and run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

2. **Set environment variables (optional)**:
   Create a `.env` file:
   ```env
   AVALAI_API_KEY=your_api_key_here
   PORT=8000
   ```

3. **Access the application**:
   ```
   http://localhost:8000
   ```

4. **View logs**:
   ```bash
   docker-compose logs -f
   ```

5. **Stop the container**:
   ```bash
   docker-compose down
   ```

See [README_DOCKER.md](README_DOCKER.md) for detailed Docker instructions.

### Option 2: Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variable (optional)**:
   ```bash
   export AVALAI_API_KEY=your_api_key_here
   ```

3. **Run the server**:
   ```bash
   python main.py
   # Or with uvicorn:
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Open in browser**:
   ```
   http://localhost:8000
   ```

## 🏗️ Project Structure

```
Convert_workflows/
├── main.py                 # FastAPI application
├── config_loader.py        # Configuration loader
├── config.json             # Configuration file
├── requirements.txt        # Python dependencies
├── agents/                 # Pydantic AI agents
│   ├── translation_agent.py
│   ├── workflow_agent.py
│   └── rtl_agent.py
├── services/               # Business logic services
│   └── workflow_processor.py
└── static/                 # Frontend assets
    ├── index.html
    ├── styles.css
    └── app.js
```

## 🎯 API Endpoints

### Workflow Analysis
```http
POST /api/workflow/analyze
Content-Type: application/json

{
  "workflow_json": "..."
}
```

### Translation
```http
POST /api/workflow/translate
Content-Type: application/json

{
  "workflow_json": "...",
  "api_key": "your_key",
  "model": "gpt-4o-mini",
  "target_language": "فارسی"
}
```

### RTL Conversion
```http
POST /api/workflow/convert-rtl
Content-Type: application/json

{
  "workflow_json": "...",
  "canvas_width": 0
}
```

### Node Name Translation
```http
POST /api/workflow/translate-nodes
Content-Type: application/json

{
  "workflow_json": "...",
  "api_key": "your_key",
  "model": "gpt-4o-mini",
  "target_language": "فارسی"
}
```

### Complete Localization
```http
POST /api/workflow/complete-localization
Content-Type: application/json

{
  "workflow_json": "...",
  "api_key": "your_key",
  "model": "gpt-4o-mini",
  "target_language": "فارسی",
  "translate_notes": true,
  "translate_node_names": true,
  "convert_to_rtl": true,
  "canvas_width": 0
}
```

## 🤖 Pydantic AI Agents

The project uses Pydantic AI agents for enhanced processing:

### TranslationAgent
- Handles translation of sticky notes and node names
- Uses OpenAI-compatible API (AvalAI)
- Supports multiple languages

### WorkflowAnalysisAgent
- Analyzes workflow structure
- Detects N8N version
- Identifies RTL content and positioning

### RTLConversionAgent
- Converts LTR layouts to RTL
- Handles canvas width calculation
- Mirrors node positions intelligently

## ⚙️ Configuration

See `config.json` for detailed configuration options including:
- API settings
- Model configurations
- Translation settings
- Layout parameters

## 🔧 Development

### Running in Development Mode
```bash
uvicorn main:app --reload
```

### Testing API
```bash
# Health check
curl http://localhost:8000/health

# Get models
curl http://localhost:8000/api/models
```

## 📚 Migration from v2.0

v3.0 is a complete rewrite with:
- FastAPI instead of Streamlit
- Pydantic AI agents instead of direct API calls
- Modern web frontend instead of Streamlit UI
- Better separation of concerns

The core functionality remains the same, but the architecture is more scalable and maintainable.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is provided as-is for N8N workflow localization purposes.

---

**Built with ❤️ using FastAPI, Pydantic AI, and modern web technologies**
