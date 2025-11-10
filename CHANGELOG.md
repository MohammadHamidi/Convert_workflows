# Changelog

All notable changes to the N8N Workflow Tools project.

## [2.0.0] - 2025-11-10

### 🚀 Major Enhancements

#### Dynamic Configuration System
- Added `config.json` for centralized configuration management
- Created `config_loader.py` module for configuration handling
- All hardcoded values now configurable (API URLs, timeouts, thresholds, models)
- Backward compatible - works with or without config file

#### N8N Version Detection
- Automatic detection of N8N workflow versions (0.x, 1.x)
- Multiple detection strategies:
  - Metadata inspection
  - Version field parsing
  - Node structure analysis
  - Type prefix pattern matching
- Version compatibility validation

#### Cross-Version Node Type Support
- Dynamic node type registry
- Support for multiple sticky note type formats:
  - `n8n-nodes-base.stickyNote` (v0.x)
  - `@n8n/n8n-nodes-base.stickyNote` (v1.x)
  - `n8n-nodes-base.sticky` (legacy)
- Easy addition of new node types via configuration

#### Workflow Validation & Auto-Fix
- Comprehensive workflow structure validation
- Automatic fixing of common issues:
  - Missing position fields
  - Invalid position arrays
  - Missing parameters
  - Malformed sticky note structure
- Detailed validation error reporting

### 🛠️ Improvements

#### Enhanced Error Handling
- Specific error messages for different failure scenarios
- Timeout handling with configurable duration
- Network error recovery
- JSON parsing error details
- Graceful fallback mechanisms

#### Better User Feedback
- Version information display in UI
- Configuration status indicators
- Auto-fix notifications
- Detailed progress tracking
- Enhanced error messages with emoji indicators

#### Code Quality
- Modular architecture with separate concerns
- Type hints throughout codebase
- Comprehensive documentation
- Standalone library version (`convert_enhanced.py`)
- Better code organization

### 📚 Documentation

#### New Documentation
- Comprehensive README with all features
- Configuration guide
- Usage examples
- Troubleshooting section
- API integration guide
- Migration guide from v1.0

#### Code Documentation
- Detailed docstrings for all functions
- Inline comments for complex logic
- Type annotations
- Usage examples in docstrings

### 🔧 Configuration Features

#### Configurable Settings
- API endpoints and timeouts
- Model specifications and pricing
- RTL detection threshold
- Token-to-character ratio
- Canvas width buffer
- Default sticky note width
- Supported languages
- N8N version support

#### Model Management
- 13 pre-configured AI models
- Easy addition of new models
- Dynamic model selection
- Cost information per model
- Context window specifications

### 🎯 New Features

#### Workflow Information
- `get_workflow_info()` method provides:
  - Detected version
  - Version support status
  - Total node count
  - Sticky note count
  - RTL content detection
  - RTL positioning detection

#### Enhanced Validation
- Required field validation
- Structure integrity checks
- Auto-fix with warning messages
- Graceful error recovery

### 📦 New Files

1. **config.json** - Dynamic configuration file
2. **config_loader.py** - Configuration management module
   - `ConfigLoader` class
   - `N8NVersionDetector` class
   - `WorkflowValidator` class
3. **convert_enhanced.py** - Standalone enhanced version
4. **README.md** - Comprehensive documentation
5. **requirements.txt** - Python dependencies
6. **CHANGELOG.md** - This file

### 🔄 Modified Files

1. **convert_final_ver.py**
   - Integrated configuration system
   - Added version detection
   - Enhanced validation
   - Improved error handling
   - Better user feedback
   - Backward compatibility maintained

### ✅ Testing

- Configuration loading tested
- Version detection tested (v0.x, v1.x)
- Workflow validation tested
- Auto-fix functionality tested
- Syntax validation passed

### 🐛 Bug Fixes

- Fixed hardcoded sticky note type (now supports multiple versions)
- Improved RTL detection with configurable threshold
- Better handling of missing workflow fields
- Enhanced timeout handling
- Fixed position validation

### 🔒 Security & Safety

- Validation before processing
- Safe fallback mechanisms
- No modification of original data
- Comprehensive error catching
- Input sanitization

### ⚡ Performance

- Efficient configuration caching
- Optimized validation logic
- Reduced redundant operations
- Better memory management

### 🎨 UI Improvements

- Version detection status banner
- Enhanced feature list in sidebar
- Better error formatting
- Improved footer with feature highlights
- Configuration mode indicators

## [1.0.0] - Previous Version

### Initial Release
- Sticky notes translation
- LTR to RTL conversion
- Node name translation
- Complete localization
- Multiple AI model support
- Cost estimation
- Smart duplicate detection

---

## Upgrade Notes

### From v1.0 to v2.0

**No Breaking Changes** - v2.0 is fully backward compatible.

#### What You Get Automatically
1. ✅ Version detection (automatic)
2. ✅ Validation and auto-fix (automatic)
3. ✅ Enhanced error messages (automatic)
4. ✅ Better performance (automatic)

#### Optional Enhancements
1. Create `config.json` for custom configuration
2. Use `config_loader.py` API for advanced integrations
3. Use `convert_enhanced.py` for standalone library usage

#### Migration Steps
1. Continue using `convert_final_ver.py`
2. No code changes required
3. (Optional) Add `config.json` for customization
4. Enjoy enhanced features!

---

## Future Roadmap

### Planned Features
- [ ] Workflow connection validation
- [ ] Support for more N8N node types
- [ ] Batch workflow processing
- [ ] Web API endpoint
- [ ] Docker container
- [ ] CI/CD integration
- [ ] Plugin system
- [ ] Custom validation rules
- [ ] Translation memory
- [ ] Workflow templates

### Under Consideration
- [ ] GUI application
- [ ] N8N plugin integration
- [ ] Real-time collaboration
- [ ] Version control integration
- [ ] Cloud deployment
- [ ] Multi-user support
- [ ] Advanced analytics
- [ ] Custom AI model integration

---

**Note**: All features in v2.0 are production-ready and tested.
