"""
FastAPI Backend for N8N Workflow Converter
Enhanced with Pydantic AI Agents
"""

import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json

from config_loader import ConfigLoader
from agents.translation_agent import TranslationAgent
from agents.workflow_agent import WorkflowAnalysisAgent
from agents.rtl_agent import RTLConversionAgent
from services.workflow_processor import WorkflowProcessor

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="N8N Workflow Converter API",
    description="Enhanced N8N workflow translation and RTL conversion with Pydantic AI agents",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global configuration and agents
config = ConfigLoader()
workflow_processor = WorkflowProcessor(config)
translation_agent = None
workflow_agent = None
rtl_agent = None

# Pydantic models
class WorkflowRequest(BaseModel):
    workflow_json: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    target_language: Optional[str] = "فارسی"

class TranslationRequest(BaseModel):
    workflow_json: str
    api_key: str
    model: str = "gpt-4o-mini"
    target_language: str = "فارسی"

class RTLRequest(BaseModel):
    workflow_json: str
    canvas_width: Optional[float] = None

class NodeTranslationRequest(BaseModel):
    workflow_json: str
    api_key: str
    model: str = "gpt-4o-mini"
    target_language: str = "فارسی"

class CompleteLocalizationRequest(BaseModel):
    workflow_json: str
    api_key: str
    model: str = "gpt-4o-mini"
    target_language: str = "فارسی"
    translate_notes: bool = True
    translate_node_names: bool = True
    convert_to_rtl: bool = True
    canvas_width: Optional[float] = None

class WorkflowInfoResponse(BaseModel):
    version: str
    is_supported: bool
    total_nodes: int
    sticky_notes_count: int
    has_rtl_content: bool
    is_rtl_positioned: bool

@app.on_event("startup")
async def startup():
    """Initialize agents on startup"""
    global translation_agent, workflow_agent, rtl_agent
    
    try:
        # Initialize agents
        api_key = os.getenv("AVALAI_API_KEY", "")
        base_url = config.get('api.base_url', 'https://api.avalai.ir/v1')
        default_model = config.get('api.default_model', 'gpt-4o-mini')
        
        translation_agent = TranslationAgent(
            api_key=api_key,
            base_url=base_url,
            model=default_model,
            config=config
        )
        
        workflow_agent = WorkflowAnalysisAgent(config=config)
        rtl_agent = RTLConversionAgent(config=config)
        
        logger.info("All agents initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing agents: {e}")
        # Continue without agents - they'll be initialized per-request if needed

@app.get("/")
async def root():
    """Serve the frontend"""
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "n8n-workflow-converter",
        "version": "3.0.0"
    }

@app.get("/api/models")
async def get_models():
    """Get available AI models"""
    models = config.get_models()
    return {
        "models": models,
        "default": config.get('api.default_model', 'gpt-4o-mini')
    }

@app.post("/api/workflow/analyze")
async def analyze_workflow(request: WorkflowRequest):
    """Analyze workflow structure and content"""
    try:
        workflow_data = json.loads(request.workflow_json)
        
        # Use workflow agent for analysis
        if workflow_agent:
            info = await workflow_agent.analyze(workflow_data)
        else:
            info = workflow_processor.get_workflow_info(workflow_data)
        
        return WorkflowInfoResponse(**info)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Error analyzing workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/translate")
async def translate_workflow(request: TranslationRequest):
    """Translate sticky notes in workflow"""
    try:
        workflow_data = json.loads(request.workflow_json)
        
        # Initialize agent with provided API key if not already initialized
        agent = translation_agent
        if not agent or request.api_key:
            agent = TranslationAgent(
                api_key=request.api_key,
                base_url=config.get('api.base_url', 'https://api.avalai.ir/v1'),
                model=request.model,
                config=config
            )
            await agent.initialize()
        
        # Extract and translate sticky notes
        sticky_notes = workflow_processor.extract_sticky_notes(workflow_data)
        translated_notes = await agent.translate_sticky_notes(
            sticky_notes,
            request.target_language
        )
        
        # Replace notes in workflow
        updated_workflow = workflow_processor.replace_notes_in_workflow(
            workflow_data,
            translated_notes
        )
        
        return {
            "success": True,
            "workflow": updated_workflow,
            "translated_count": len(translated_notes)
        }
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Error translating workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/convert-rtl")
async def convert_rtl(request: RTLRequest):
    """Convert workflow from LTR to RTL layout"""
    try:
        workflow_data = json.loads(request.workflow_json)
        
        # Use RTL agent if available
        if rtl_agent:
            converted = await rtl_agent.convert(workflow_data, request.canvas_width)
        else:
            converted = workflow_processor.convert_ltr_to_rtl(
                workflow_data,
                request.canvas_width
            )
        
        if not converted:
            raise HTTPException(status_code=400, detail="RTL conversion failed")
        
        return {
            "success": True,
            "workflow": converted
        }
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Error converting to RTL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/translate-nodes")
async def translate_node_names(request: NodeTranslationRequest):
    """Translate node names in workflow"""
    try:
        workflow_data = json.loads(request.workflow_json)
        
        # Initialize agent with provided API key
        agent = TranslationAgent(
            api_key=request.api_key,
            base_url=config.get('api.base_url', 'https://api.avalai.ir/v1'),
            model=request.model,
            config=config
        )
        await agent.initialize()
        
        # Extract and translate node names
        node_names = workflow_processor.extract_node_names(workflow_data)
        translated_names = await agent.translate_node_names(
            node_names,
            request.target_language
        )
        
        # Replace node names in workflow
        updated_workflow = workflow_processor.replace_node_names_in_workflow(
            workflow_data,
            translated_names
        )
        
        return {
            "success": True,
            "workflow": updated_workflow,
            "translated_count": len(translated_names)
        }
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Error translating node names: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/complete-localization")
async def complete_localization(request: CompleteLocalizationRequest):
    """Complete workflow localization (translate + RTL + node names)"""
    try:
        workflow_data = json.loads(request.workflow_json)
        current_workflow = workflow_data
        
        # Initialize agent
        agent = TranslationAgent(
            api_key=request.api_key,
            base_url=config.get('api.base_url', 'https://api.avalai.ir/v1'),
            model=request.model,
            config=config
        )
        await agent.initialize()
        
        results = {
            "translate_notes": False,
            "translate_nodes": False,
            "convert_rtl": False
        }
        
        # Step 1: Translate sticky notes
        if request.translate_notes:
            sticky_notes = workflow_processor.extract_sticky_notes(current_workflow)
            if sticky_notes:
                translated_notes = await agent.translate_sticky_notes(
                    sticky_notes,
                    request.target_language
                )
                current_workflow = workflow_processor.replace_notes_in_workflow(
                    current_workflow,
                    translated_notes
                )
                results["translate_notes"] = True
        
        # Step 2: Translate node names
        if request.translate_node_names:
            node_names = workflow_processor.extract_node_names(current_workflow)
            if node_names:
                translated_names = await agent.translate_node_names(
                    node_names,
                    request.target_language
                )
                current_workflow = workflow_processor.replace_node_names_in_workflow(
                    current_workflow,
                    translated_names
                )
                results["translate_nodes"] = True
        
        # Step 3: Convert to RTL
        if request.convert_to_rtl:
            if rtl_agent:
                current_workflow = await rtl_agent.convert(
                    current_workflow,
                    request.canvas_width
                )
            else:
                current_workflow = workflow_processor.convert_ltr_to_rtl(
                    current_workflow,
                    request.canvas_width
                )
            results["convert_rtl"] = True
        
        return {
            "success": True,
            "workflow": current_workflow,
            "results": results
        }
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Error in complete localization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

