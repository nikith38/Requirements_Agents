"""LangGraph workflow implementation for multi-agent requirement gathering system."""

from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import logging
from datetime import datetime

from models import WorkflowState, WorkflowConfig, AgentConfig
from agents import AgentFactory, BaseAgent


class RequirementGatheringWorkflow:
    """Main workflow orchestrator for the multi-agent requirement gathering system."""
    
    def __init__(self, config: WorkflowConfig = None):
        self.config = config or WorkflowConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize agents
        self.agents = AgentFactory.create_all_agents(self.config.agent_config)
        
        # Initialize workflow graph
        self.graph = None
        self.memory = MemorySaver()
        
        self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        # Create the state graph
        workflow = StateGraph(WorkflowState)
        
        # Add agent nodes
        workflow.add_node("intake", self._intake_node)
        workflow.add_node("analysis", self._analysis_node)
        workflow.add_node("ambiguity_detection", self._ambiguity_detection_node)
        workflow.add_node("stakeholder_simulation", self._stakeholder_simulation_node)
        workflow.add_node("validation", self._validation_node)
        workflow.add_node("refinement", self._refinement_node)
        workflow.add_node("documentation", self._documentation_node)
        workflow.add_node("error_handler", self._error_handler_node)
        workflow.add_node("quality_assurance", self._quality_assurance_node)
        
        # Define the workflow edges
        self._add_workflow_edges(workflow)
        
        # Set entry point
        workflow.set_entry_point("intake")
        
        # Compile the graph
        self.graph = workflow.compile(checkpointer=self.memory)
        
        self.logger.info("Workflow graph built successfully")
    
    def _add_workflow_edges(self, workflow: StateGraph):
        """Add edges to define the workflow flow."""
        
        # Sequential flow: intake -> analysis -> ambiguity_detection
        workflow.add_edge("intake", "analysis")
        workflow.add_edge("analysis", "ambiguity_detection")
        
        # Conditional flow from ambiguity detection
        workflow.add_conditional_edges(
            "ambiguity_detection",
            self._should_simulate_stakeholders,
            {
                "stakeholder_simulation": "stakeholder_simulation",
                "validation": "validation"
            }
        )
        
        # From stakeholder simulation to validation
        workflow.add_edge("stakeholder_simulation", "validation")
        
        # Conditional flow from validation
        workflow.add_conditional_edges(
            "validation",
            self._should_refine_requirements,
            {
                "refinement": "refinement",
                "documentation": "documentation",
                "error_handler": "error_handler"
            }
        )
        
        # Refinement loop logic
        workflow.add_conditional_edges(
            "refinement",
            self._should_iterate_refinement,
            {
                "validation": "validation",
                "documentation": "documentation",
                "error_handler": "error_handler"
            }
        )
        
        # Documentation to quality assurance
        workflow.add_edge("documentation", "quality_assurance")
        
        # Quality assurance conditional flow
        workflow.add_conditional_edges(
            "quality_assurance",
            self._should_pass_quality_check,
            {
                "end": END,
                "refinement": "refinement",
                "error_handler": "error_handler"
            }
        )
        
        # Error handler flows
        workflow.add_conditional_edges(
            "error_handler",
            self._handle_error_recovery,
            {
                "retry": "intake",
                "end": END
            }
        )
    
    # Node implementations
    def _intake_node(self, state: WorkflowState) -> WorkflowState:
        """Process raw input through intake agent."""
        try:
            # Validate state object
            if isinstance(state, dict):
                state = WorkflowState(**state)
            
            self.logger.info("Executing intake node")
            return self.agents["intake"].process(state)
        except Exception as e:
            if hasattr(state, 'errors'):
                state.errors.append(f"Intake node error: {str(e)}")
            state.current_stage = "error_handler"
            return state
    
    def _analysis_node(self, state: WorkflowState) -> WorkflowState:
        """Analyze requirements through analysis agent."""
        try:
            # Validate state object
            if isinstance(state, dict):
                state = WorkflowState(**state)
            
            self.logger.info("Executing analysis node")
            return self.agents["analysis"].process(state)
        except Exception as e:
            if hasattr(state, 'errors'):
                state.errors.append(f"Analysis node error: {str(e)}")
            state.current_stage = "error_handler"
            return state
    
    def _ambiguity_detection_node(self, state: WorkflowState) -> WorkflowState:
        """Detect ambiguities through ambiguity detection agent."""
        try:
            # Validate state object
            if isinstance(state, dict):
                state = WorkflowState(**state)
            
            self.logger.info("Executing ambiguity detection node")
            return self.agents["ambiguity_detection"].process(state)
        except Exception as e:
            if hasattr(state, 'errors'):
                state.errors.append(f"Ambiguity detection node error: {str(e)}")
            state.current_stage = "error_handler"
            return state
    
    def _stakeholder_simulation_node(self, state: WorkflowState) -> WorkflowState:
        """Simulate stakeholder perspectives."""
        try:
            # Validate state object
            if isinstance(state, dict):
                state = WorkflowState(**state)
            
            self.logger.info("Executing stakeholder simulation node")
            return self.agents["stakeholder_simulation"].process(state)
        except Exception as e:
            if hasattr(state, 'errors'):
                state.errors.append(f"Stakeholder simulation node error: {str(e)}")
            state.current_stage = "error_handler"
            return state
    
    def _validation_node(self, state: WorkflowState) -> WorkflowState:
        """Validate requirements through validation agent."""
        try:
            # Validate state object
            if isinstance(state, dict):
                state = WorkflowState(**state)
            
            self.logger.info("Executing validation node")
            return self.agents["validation"].process(state)
        except Exception as e:
            if hasattr(state, 'errors'):
                state.errors.append(f"Validation node error: {str(e)}")
            state.current_stage = "error_handler"
            return state
    
    def _refinement_node(self, state: WorkflowState) -> WorkflowState:
        """Refine requirements through refinement agent."""
        try:
            # Validate state object
            if isinstance(state, dict):
                state = WorkflowState(**state)
            
            self.logger.info("Executing refinement node")
            return self.agents["refinement"].process(state)
        except Exception as e:
            if hasattr(state, 'errors'):
                state.errors.append(f"Refinement node error: {str(e)}")
            state.current_stage = "error_handler"
            return state
    
    def _documentation_node(self, state: WorkflowState) -> WorkflowState:
        """Generate documentation through documentation agent."""
        try:
            # Validate state object
            if isinstance(state, dict):
                state = WorkflowState(**state)
            
            self.logger.info("Executing documentation node")
            return self.agents["documentation"].process(state)
        except Exception as e:
            if hasattr(state, 'errors'):
                state.errors.append(f"Documentation node error: {str(e)}")
            state.current_stage = "error_handler"
            return state
    
    def _error_handler_node(self, state: WorkflowState) -> WorkflowState:
        """Handle errors - simplified for single-try workflow."""
        try:
            # Ensure we have a proper WorkflowState object
            if isinstance(state, dict):
                self.logger.error("Received dict instead of WorkflowState object")
                # Convert dict back to WorkflowState if needed
                state = WorkflowState(**state)
            
            self.logger.error(f"Error handler activated. Errors: {state.errors}")
            
            # In single-try workflow, log errors and proceed to completion
            state.current_stage = "completed"
            state.warnings.append("Errors encountered, proceeding to completion in single-try mode")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error handler failed: {e}")
            state.current_stage = "completed"
            if hasattr(state, 'errors'):
                state.errors.append(f"Error handler failed: {str(e)}")
            return state
    

    
    def _quality_assurance_node(self, state: WorkflowState) -> WorkflowState:
        """Perform quality assurance checks."""
        try:
            self.logger.info("Executing quality assurance node")
            
            # Handle case where state might be a dict (LangGraph internal conversion)
            if isinstance(state, dict):
                state = WorkflowState(**state)
            
            # Basic quality checks
            quality_score = self._calculate_quality_score(state)
            state.quality_metrics["overall_quality"] = quality_score
            
            # In single-try workflow, always complete after quality check
            state.current_stage = "completed"
            state.warnings.append(f"Workflow completed with quality score: {quality_score:.2f}")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Quality assurance error: {str(e)}")
            if isinstance(state, dict):
                state = WorkflowState(**state)
            state.errors.append(f"Quality assurance error: {str(e)}")
            state.current_stage = "completed"
            return state
    
    # Decision functions for conditional edges
    def _should_simulate_stakeholders(self, state: WorkflowState) -> str:
        """Determine if stakeholder simulation is needed."""
        if not state.ambiguity_output:
            return "validation"  # Proceed to validation instead of error handler
        
        # Check ambiguity score
        if state.ambiguity_output.overall_ambiguity_score > self.config.ambiguity_threshold:
            return "stakeholder_simulation"
        
        return "validation"
    
    def _should_refine_requirements(self, state: WorkflowState) -> str:
        """Determine if requirements need refinement."""
        if not state.validation_output:
            return "documentation"  # Proceed to documentation instead of error handler
        
        # Always proceed to documentation regardless of validation score
        # This creates a single-try workflow
        return "documentation"
    
    def _should_iterate_refinement(self, state: WorkflowState) -> str:
        """Determine if another refinement iteration is needed."""
        # In single-try workflow, always proceed to documentation after refinement
        return "documentation"
    
    def _should_pass_quality_check(self, state: WorkflowState) -> str:
        """Determine if quality assurance passed."""
        # In single-try workflow, always end after quality check
        return "end"
    

    
    def _handle_error_recovery(self, state: WorkflowState) -> str:
        """Handle error recovery strategy - simplified for single-try workflow."""
        # In single-try workflow, always end after error handling
        return "end"
    
    def _calculate_quality_score(self, state: WorkflowState) -> float:
        """Calculate overall quality score for the workflow."""
        scores = []
        
        # Validation score
        if state.validation_output:
            scores.append(state.validation_output.overall_validation_score)
        
        # Ambiguity score (inverted - lower ambiguity is better)
        if state.ambiguity_output:
            scores.append(1.0 - state.ambiguity_output.overall_ambiguity_score)
        
        # Refinement confidence
        if state.refinement_output:
            avg_confidence = sum(
                req.refinement_confidence for req in state.refinement_output.refined_requirements
            ) / len(state.refinement_output.refined_requirements)
            scores.append(avg_confidence)
        
        # Documentation completeness
        if state.documentation_output:
            scores.append(state.documentation_output.document_metadata.document_completeness)
        
        # Error penalty
        error_penalty = min(len(state.errors) * 0.1, 0.5)
        
        if scores:
            base_score = sum(scores) / len(scores)
            return max(0.0, base_score - error_penalty)
        
        return 0.0
    
    # Public interface methods
    def run(self, user_input: str, thread_id: str = None) -> WorkflowState:
        """Run the complete workflow on user input."""
        try:
            self.logger.info(f"Starting workflow for input: {user_input[:100]}...")
            
            # Initialize state
            initial_state = WorkflowState(
                raw_input=user_input,
                current_stage="intake"
            )
            
            # Configure thread
            config = {"configurable": {"thread_id": thread_id or "default"}}
            
            # Run the workflow
            result = self.graph.invoke(initial_state, config=config)
            
            # Handle case where LangGraph returns a dict instead of WorkflowState
            if isinstance(result, dict):
                result = WorkflowState(**result)
            
            self.logger.info(f"Workflow completed with stage: {result.current_stage}")
            return result
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            error_state = WorkflowState(
                raw_input=user_input,
                current_stage="error",
                errors=[f"Workflow execution failed: {str(e)}"]
            )
            return error_state
    
    def run_async(self, user_input: str, thread_id: str = None):
        """Run the workflow asynchronously (returns generator for streaming)."""
        try:
            initial_state = WorkflowState(
                raw_input=user_input,
                current_stage="intake"
            )
            
            config = {"configurable": {"thread_id": thread_id or "default"}}
            
            # Stream the workflow execution
            for step in self.graph.stream(initial_state, config=config):
                yield step
                
        except Exception as e:
            self.logger.error(f"Async workflow execution failed: {str(e)}")
            yield {
                "error": WorkflowState(
                    raw_input=user_input,
                    current_stage="error",
                    errors=[f"Async workflow execution failed: {str(e)}"]
                )
            }
    
    def get_workflow_status(self, thread_id: str = "default") -> Dict[str, Any]:
        """Get the current status of a workflow thread."""
        try:
            # Get the latest state from memory
            config = {"configurable": {"thread_id": thread_id}}
            state = self.graph.get_state(config)
            
            return {
                "current_stage": state.values.get("current_stage", "unknown"),
                "iteration_count": state.values.get("iteration_count", 0),
                "errors": state.values.get("errors", []),
                "warnings": state.values.get("warnings", []),
                "processing_time": state.values.get("processing_time", {}),
                "quality_metrics": state.values.get("quality_metrics", {})
            }
        except Exception as e:
            return {"error": f"Failed to get workflow status: {str(e)}"}
    
    def reset_workflow(self, thread_id: str = "default"):
        """Reset a workflow thread."""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            # Clear the thread's memory
            # Note: MemorySaver doesn't have a direct clear method, 
            # so we'd need to implement this based on the specific checkpointer
            self.logger.info(f"Workflow thread {thread_id} reset requested")
        except Exception as e:
            self.logger.error(f"Failed to reset workflow thread {thread_id}: {str(e)}")


# Convenience function for quick workflow execution
def run_requirement_gathering(user_input: str, config: WorkflowConfig = None) -> WorkflowState:
    """Convenience function to run the requirement gathering workflow."""
    workflow = RequirementGatheringWorkflow(config)
    return workflow.run(user_input)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    sample_input = """
    We need a web application that allows users to register, login, and manage their profiles.
    The system should be secure, fast, and user-friendly. Users should be able to upload profile pictures
    and update their information. The application should work on mobile devices and support multiple languages.
    """
    
    try:
        # Create and run workflow
        workflow = RequirementGatheringWorkflow()
        result = workflow.run(sample_input)
        
        print(f"Workflow completed with stage: {result.current_stage}")
        print(f"Errors: {result.errors}")
        print(f"Warnings: {result.warnings}")
        print(f"Processing times: {result.processing_time}")
        
        if result.documentation_output:
            print("\nDocumentation generated successfully!")
        
    except Exception as e:
        print(f"Workflow execution failed: {str(e)}")