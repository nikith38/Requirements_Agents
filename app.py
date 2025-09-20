from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import os
from datetime import datetime

from workflow import RequirementGatheringWorkflow
from models import WorkflowState
from config import load_config

app = FastAPI(
    title="Requirements Analysis API",
    description="API for analyzing and processing software requirements using AI agents",
    version="1.0.0"
)

class RequirementsInput(BaseModel):
    requirements_text: str
    project_name: Optional[str] = "Untitled Project"
    stakeholder_context: Optional[str] = None

class RequirementsResponse(BaseModel):
    success: bool
    message: str
    result_file: Optional[str] = None
    processing_time: Optional[float] = None
    quality_score: Optional[float] = None
    errors: Optional[list] = None
    warnings: Optional[list] = None

@app.get("/")
async def root():
    return {
        "message": "Requirements Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Analyze requirements text",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/analyze", response_model=RequirementsResponse)
async def analyze_requirements(input_data: RequirementsInput):
    """
    Analyze requirements text and return processed results.
    
    Args:
        input_data: Requirements input containing text and optional metadata
        
    Returns:
        RequirementsResponse with analysis results and file location
    """
    try:
        # Load configuration
        config = load_config()
        
        # Initialize workflow
        workflow = RequirementsWorkflow(config)
        
        # Create initial state
        initial_state = WorkflowState(
            raw_input=input_data.requirements_text,
            current_stage="intake",
            iteration_count=0,
            errors=[],
            warnings=[]
        )
        
        # Run workflow
        start_time = datetime.now()
        result = workflow.run(initial_state)
        end_time = datetime.now()
        
        # Handle result conversion if needed
        if isinstance(result, dict):
            result = WorkflowState(**result)
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"requirements_result_{timestamp}.json"
        output_path = os.path.join("output", output_filename)
        
        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)
        
        # Prepare output data
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "raw_input": input_data.requirements_text,
            "project_name": input_data.project_name,
            "stakeholder_context": input_data.stakeholder_context,
            "current_stage": result.current_stage,
            "iteration_count": result.iteration_count,
            "errors": result.errors,
            "warnings": result.warnings,
            "processing_time": {
                "total_seconds": processing_time,
                "intake": getattr(result, 'intake_time', 0),
                "analysis": getattr(result, 'analysis_time', 0),
                "ambiguity": getattr(result, 'ambiguity_time', 0),
                "stakeholder": getattr(result, 'stakeholder_time', 0),
                "validation": getattr(result, 'validation_time', 0),
                "documentation": getattr(result, 'documentation_time', 0)
            },
            "quality_metrics": {
                "overall_score": getattr(result, 'quality_score', 0.0)
            }
        }
        
        # Add workflow outputs if they exist
        if hasattr(result, 'intake_output') and result.intake_output:
            output_data["intake_output"] = result.intake_output.model_dump(mode='json')
        
        if hasattr(result, 'analysis_output') and result.analysis_output:
            output_data["analysis_output"] = result.analysis_output.model_dump(mode='json')
            
        if hasattr(result, 'ambiguity_output') and result.ambiguity_output:
            output_data["ambiguity_output"] = result.ambiguity_output.model_dump(mode='json')
            
        if hasattr(result, 'stakeholder_output') and result.stakeholder_output:
            output_data["stakeholder_output"] = result.stakeholder_output.model_dump(mode='json')
            
        if hasattr(result, 'validation_output') and result.validation_output:
            output_data["validation_output"] = result.validation_output.model_dump(mode='json')
            
        if hasattr(result, 'documentation_output') and result.documentation_output:
            output_data["documentation_output"] = result.documentation_output.model_dump(mode='json')
        
        # Save results to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Extract quality score for response
        quality_score = getattr(result, 'quality_score', 0.0)
        
        return RequirementsResponse(
            success=True,
            message="Requirements analysis completed successfully",
            result_file=output_path,
            processing_time=processing_time,
            quality_score=quality_score,
            errors=result.errors,
            warnings=result.warnings
        )
        
    except Exception as e:
        return RequirementsResponse(
            success=False,
            message=f"Error during requirements analysis: {str(e)}",
            errors=[str(e)]
        )

@app.get("/results/{filename}")
async def get_results(filename: str):
    """
    Retrieve analysis results by filename.
    
    Args:
        filename: Name of the result file
        
    Returns:
        JSON content of the analysis results
    """
    try:
        file_path = os.path.join("output", filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Result file not found")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        return results
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Result file not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON in result file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading result file: {str(e)}")

@app.get("/results")
async def list_results():
    """
    List all available result files.
    
    Returns:
        List of available result files with metadata
    """
    try:
        output_dir = "output"
        if not os.path.exists(output_dir):
            return {"files": []}
        
        files = []
        for filename in os.listdir(output_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(output_dir, filename)
                stat = os.stat(file_path)
                files.append({
                    "filename": filename,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        # Sort by creation time, newest first
        files.sort(key=lambda x: x['created'], reverse=True)
        
        return {"files": files}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing result files: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)