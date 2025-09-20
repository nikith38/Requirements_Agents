"""Agent implementations for multi-agent requirement gathering system."""

from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import logging
from datetime import datetime
import time

from models import (
    IntakeOutput, AnalysisOutput, AmbiguityOutput, StakeholderOutput,
    ValidationOutput, RefinementOutput, DocumentationOutput, WorkflowState,
    AgentConfig
)
from system_prompts import SYSTEM_PROMPTS


class BaseAgent:
    """Base class for all agents in the system."""
    
    def __init__(self, agent_type: str, config: AgentConfig = None):
        self.agent_type = agent_type
        self.config = config or AgentConfig()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Initialize OpenAI model with LangChain
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout
        )
        
        # Get system prompt
        self.system_prompt = SYSTEM_PROMPTS.get(agent_type, "")
        
    def _create_chain(self, output_parser: PydanticOutputParser, user_input: str = "{input}"):
        """Create a LangChain processing chain."""
        # Get format instructions and escape curly braces to prevent template variable conflicts
        format_instructions = output_parser.get_format_instructions()
        # Escape curly braces that are not meant to be template variables
        format_instructions = format_instructions.replace('{', '{{').replace('}', '}}')
        # But keep the {input} variable as a single brace for template substitution if it exists
        format_instructions = format_instructions.replace('{{input}}', '{input}')
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt + "\n\n" + format_instructions),
            ("human", user_input)
        ])
        
        chain = (
            {"input": RunnablePassthrough()}
            | prompt
            | self.llm
            | output_parser
        )
        
        return chain
    
    def _log_processing_time(self, stage: str, start_time: float, state: WorkflowState):
        """Log processing time for a stage."""
        processing_time = time.time() - start_time
        state.processing_time[stage] = processing_time
        self.logger.info(f"{stage} completed in {processing_time:.2f} seconds")
    
    def _handle_error(self, error: Exception, state: WorkflowState, stage: str):
        """Handle errors during processing."""
        error_msg = f"Error in {stage}: {str(error)}"
        self.logger.error(error_msg)
        state.errors.append(error_msg)
        return None


class IntakeAgent(BaseAgent):
    """Agent responsible for parsing and structuring raw input."""
    
    def __init__(self, config: AgentConfig = None):
        super().__init__("intake", config)
        self.output_parser = PydanticOutputParser(pydantic_object=IntakeOutput)
        self.chain = self._create_chain(self.output_parser)
    
    def process(self, state: WorkflowState) -> WorkflowState:
        """Process raw input and extract structured requirements."""
        start_time = time.time()
        
        try:
            self.logger.info("Starting intake processing")
            
            # Process the raw input
            result = self.chain.invoke(state.raw_input)
            
            # Update state
            state.intake_output = result
            state.current_stage = "analysis"
            
            self._log_processing_time("intake", start_time, state)
            self.logger.info(f"Extracted {len(result.parsed_requirements)} requirements")
            
            return state
            
        except Exception as e:
            return self._handle_error(e, state, "intake") or state


class AnalysisAgent(BaseAgent):
    """Agent responsible for analyzing requirements relationships and quality."""
    
    def __init__(self, config: AgentConfig = None):
        super().__init__("analysis", config)
        self.output_parser = PydanticOutputParser(pydantic_object=AnalysisOutput)
        self.chain = self._create_chain(
            self.output_parser,
            "Analyze the following requirements:\n{input}"
        )
    
    def process(self, state: WorkflowState) -> WorkflowState:
        """Analyze requirements for completeness, dependencies, and conflicts."""
        start_time = time.time()
        
        try:
            self.logger.info("Starting analysis processing")
            
            if not state.intake_output:
                raise ValueError("No intake output available for analysis")
            
            # Prepare input for analysis
            requirements_text = "\n".join([
                f"ID: {req.id}\nType: {req.type}\nDescription: {req.description}\nPriority: {req.priority}\n"
                for req in state.intake_output.parsed_requirements
            ])
            
            # Process the requirements
            result = self.chain.invoke(requirements_text)
            
            # Update state
            state.analysis_output = result
            state.current_stage = "ambiguity_detection"
            
            self._log_processing_time("analysis", start_time, state)
            self.logger.info(f"Analyzed {len(result.analyzed_requirements)} requirements")
            
            return state
            
        except Exception as e:
            return self._handle_error(e, state, "analysis") or state


class AmbiguityDetectionAgent(BaseAgent):
    """Agent responsible for detecting ambiguities and generating clarification questions."""
    
    def __init__(self, config: AgentConfig = None):
        super().__init__("ambiguity_detection", config)
        self.output_parser = PydanticOutputParser(pydantic_object=AmbiguityOutput)
        self.chain = self._create_chain(
            self.output_parser,
            "Detect ambiguities in the following analyzed requirements:\n{input}"
        )
    
    def process(self, state: WorkflowState) -> WorkflowState:
        """Detect ambiguities and generate clarification questions."""
        start_time = time.time()
        
        try:
            self.logger.info("Starting ambiguity detection")
            
            if not state.analysis_output:
                raise ValueError("No analysis output available for ambiguity detection")
            
            # Prepare input for ambiguity detection
            requirements_text = "\n".join([
                f"ID: {req.id}\nOriginal: {req.original_requirement}\n"
                f"Completeness Score: {req.completeness_score}\n"
                f"Specificity Score: {req.specificity_score}\n"
                f"Sub-requirements: {[sub.description for sub in req.decomposed_requirements]}\n"
                for req in state.analysis_output.analyzed_requirements
            ])
            
            # Process for ambiguities
            result = self.chain.invoke(requirements_text)
            
            # Update state
            state.ambiguity_output = result
            
            # Determine next stage based on ambiguity level
            if result.overall_ambiguity_score > 0.3:  # High ambiguity threshold
                state.current_stage = "stakeholder_simulation"
            else:
                state.current_stage = "validation"
            
            self._log_processing_time("ambiguity_detection", start_time, state)
            self.logger.info(f"Detected ambiguities in {len(result.critical_ambiguities)} requirements")
            
            return state
            
        except Exception as e:
            return self._handle_error(e, state, "ambiguity_detection") or state


class StakeholderSimulationAgent(BaseAgent):
    """Agent responsible for simulating stakeholder perspectives and identifying conflicts."""
    
    def __init__(self, config: AgentConfig = None):
        super().__init__("stakeholder_simulation", config)
        self.output_parser = PydanticOutputParser(pydantic_object=StakeholderOutput)
        self.chain = self._create_chain(
            self.output_parser,
            "Simulate stakeholder feedback for the following requirements with ambiguities:\n{input}"
        )
    
    def process(self, state: WorkflowState) -> WorkflowState:
        """Simulate stakeholder perspectives and identify conflicts."""
        start_time = time.time()
        
        try:
            self.logger.info("Starting stakeholder simulation")
            
            if not state.ambiguity_output:
                raise ValueError("No ambiguity output available for stakeholder simulation")
            
            # Prepare input combining analysis and ambiguity data
            combined_data = []
            for analysis_req in state.analysis_output.analyzed_requirements:
                # Find corresponding ambiguity analysis
                ambiguity_data = next(
                    (amb for amb in state.ambiguity_output.ambiguity_analysis 
                     if amb.requirement_id == analysis_req.id),
                    None
                )
                
                req_text = f"ID: {analysis_req.id}\n"
                req_text += f"Requirement: {analysis_req.original_requirement}\n"
                req_text += f"Completeness: {analysis_req.completeness_score}\n"
                
                if ambiguity_data:
                    req_text += f"Ambiguity Score: {ambiguity_data.ambiguity_score}\n"
                    req_text += f"Ambiguous Terms: {ambiguity_data.ambiguous_terms}\n"
                    req_text += f"Missing Info: {ambiguity_data.missing_information}\n"
                
                combined_data.append(req_text)
            
            input_text = "\n\n".join(combined_data)
            
            # Process stakeholder simulation
            result = self.chain.invoke(input_text)
            
            # Update state
            state.stakeholder_output = result
            state.current_stage = "validation"
            
            self._log_processing_time("stakeholder_simulation", start_time, state)
            self.logger.info(f"Simulated feedback for {len(result.stakeholder_feedback)} requirements")
            
            return state
            
        except Exception as e:
            return self._handle_error(e, state, "stakeholder_simulation") or state


class ValidationAgent(BaseAgent):
    """Agent responsible for validating requirements against business and technical criteria."""
    
    def __init__(self, config: AgentConfig = None):
        super().__init__("validation", config)
        self.output_parser = PydanticOutputParser(pydantic_object=ValidationOutput)
        self.chain = self._create_chain(
            self.output_parser,
            "Validate the following requirements against business and technical criteria:\n{input}"
        )
    
    def process(self, state: WorkflowState) -> WorkflowState:
        """Validate requirements for feasibility and alignment."""
        start_time = time.time()
        
        try:
            self.logger.info("Starting validation")
            
            if not state.analysis_output:
                raise ValueError("No analysis output available for validation")
            
            # Prepare comprehensive input for validation
            validation_data = []
            
            for req in state.analysis_output.analyzed_requirements:
                req_data = f"ID: {req.id}\n"
                req_data += f"Requirement: {req.original_requirement}\n"
                req_data += f"Sub-requirements: {[sub.description for sub in req.decomposed_requirements]}\n"
                req_data += f"Dependencies: {req.dependencies}\n"
                req_data += f"Conflicts: {req.conflicts}\n"
                req_data += f"Quality Scores: Completeness={req.completeness_score}, "
                req_data += f"Specificity={req.specificity_score}, Testability={req.testability_score}\n"
                
                # Add stakeholder feedback if available
                if state.stakeholder_output:
                    stakeholder_feedback = [
                        feedback for feedback in state.stakeholder_output.stakeholder_feedback
                        if feedback.requirement_id == req.id
                    ]
                    if stakeholder_feedback:
                        req_data += f"Stakeholder Concerns: {[f.concerns for f in stakeholder_feedback]}\n"
                        req_data += f"Approval Status: {[f.approval_status for f in stakeholder_feedback]}\n"
                
                validation_data.append(req_data)
            
            input_text = "\n\n".join(validation_data)
            
            # Process validation
            result = self.chain.invoke(input_text)
            
            # Update state
            state.validation_output = result
            
            # Determine next stage based on validation results
            if result.overall_validation_score < 0.7:  # Low validation threshold
                state.current_stage = "refinement"
            else:
                state.current_stage = "documentation"
            
            self._log_processing_time("validation", start_time, state)
            self.logger.info(f"Validated {len(result.validation_results)} requirements")
            self.logger.info(f"Overall validation score: {result.overall_validation_score}")
            
            return state
            
        except Exception as e:
            return self._handle_error(e, state, "validation") or state


class RefinementAgent(BaseAgent):
    """Agent responsible for refining and improving requirements quality."""
    
    def __init__(self, config: AgentConfig = None):
        super().__init__("refinement", config)
        self.output_parser = PydanticOutputParser(pydantic_object=RefinementOutput)
        self.chain = self._create_chain(
            self.output_parser,
            "Refine and improve the following requirements based on validation feedback:\n{input}"
        )
    
    def process(self, state: WorkflowState) -> WorkflowState:
        """Refine requirements based on analysis and validation feedback."""
        start_time = time.time()
        
        try:
            self.logger.info("Starting refinement")
            
            if not state.validation_output:
                raise ValueError("No validation output available for refinement")
            
            # Prepare comprehensive refinement input
            refinement_data = []
            
            for validation_result in state.validation_output.validation_results:
                req_id = validation_result.requirement_id
                
                # Find original requirement
                original_req = next(
                    (req for req in state.analysis_output.analyzed_requirements 
                     if req.id == req_id),
                    None
                )
                
                if original_req:
                    ref_data = f"ID: {req_id}\n"
                    ref_data += f"Original: {original_req.original_requirement}\n"
                    ref_data += f"Validation Status: {validation_result.validation_status}\n"
                    ref_data += f"Validation Notes: {validation_result.validation_notes}\n"
                    ref_data += f"Risks: {validation_result.risks}\n"
                    ref_data += f"Business Alignment: {validation_result.business_alignment_score}\n"
                    ref_data += f"Technical Feasibility: {validation_result.technical_feasibility_score}\n"
                    
                    # Add ambiguity information if available
                    if state.ambiguity_output:
                        ambiguity_info = next(
                            (amb for amb in state.ambiguity_output.ambiguity_analysis 
                             if amb.requirement_id == req_id),
                            None
                        )
                        if ambiguity_info:
                            ref_data += f"Ambiguous Terms: {ambiguity_info.ambiguous_terms}\n"
                            ref_data += f"Suggested Improvements: {ambiguity_info.suggested_improvements}\n"
                    
                    refinement_data.append(ref_data)
            
            input_text = "\n\n".join(refinement_data)
            
            # Process refinement
            result = self.chain.invoke(input_text)
            
            # Update state
            state.refinement_output = result
            state.current_stage = "documentation"
            state.iteration_count += 1
            
            self._log_processing_time("refinement", start_time, state)
            self.logger.info(f"Refined {len(result.refined_requirements)} requirements")
            
            return state
            
        except Exception as e:
            return self._handle_error(e, state, "refinement") or state


class DocumentationAgent(BaseAgent):
    """Agent responsible for creating comprehensive requirements documentation."""
    
    def __init__(self, config: AgentConfig = None):
        super().__init__("documentation", config)
        self.output_parser = PydanticOutputParser(pydantic_object=DocumentationOutput)
        self.chain = self._create_chain(
            self.output_parser,
            "Create comprehensive documentation for the following processed requirements:\n{input}"
        )
    
    def process(self, state: WorkflowState) -> WorkflowState:
        """Generate comprehensive requirements documentation."""
        start_time = time.time()
        
        try:
            self.logger.info("Starting documentation generation")
            
            # Prepare comprehensive documentation input
            doc_data = []
            
            # Add project overview
            doc_data.append(f"Original Input: {state.raw_input}")
            doc_data.append(f"Processing Stages Completed: {state.current_stage}")
            doc_data.append(f"Total Iterations: {state.iteration_count}")
            
            # Add intake information
            if state.intake_output:
                doc_data.append(f"\nTotal Requirements Extracted: {len(state.intake_output.parsed_requirements)}")
                for req in state.intake_output.parsed_requirements:
                    doc_data.append(f"- {req.id}: {req.description} (Type: {req.type}, Priority: {req.priority})")
            
            # Add analysis results
            if state.analysis_output:
                doc_data.append(f"\nAnalysis Summary:")
                doc_data.append(f"- Total Conflicts: {state.analysis_output.analysis_summary.total_conflicts}")
                doc_data.append(f"- Dependencies: {state.analysis_output.analysis_summary.dependency_count}")
                doc_data.append(f"- Completeness Issues: {state.analysis_output.analysis_summary.completeness_issues}")
            
            # Add validation results
            if state.validation_output:
                doc_data.append(f"\nValidation Results:")
                doc_data.append(f"- Overall Score: {state.validation_output.overall_validation_score}")
                doc_data.append(f"- Approved: {len(state.validation_output.approved_requirements)}")
                doc_data.append(f"- Rejected: {len(state.validation_output.rejected_requirements)}")
            
            # Add refined requirements if available
            if state.refinement_output:
                doc_data.append(f"\nRefined Requirements:")
                for refined in state.refinement_output.refined_requirements:
                    doc_data.append(f"- {refined.id}: {refined.refined_text}")
                    doc_data.append(f"  Acceptance Criteria: {refined.acceptance_criteria}")
            
            # Add processing metrics
            if state.processing_time:
                doc_data.append(f"\nProcessing Times:")
                for stage, time_taken in state.processing_time.items():
                    doc_data.append(f"- {stage}: {time_taken:.2f}s")
            
            input_text = "\n".join(doc_data)
            
            # Generate documentation
            result = self.chain.invoke(input_text)
            
            # Update state
            state.documentation_output = result
            state.current_stage = "completed"
            
            self._log_processing_time("documentation", start_time, state)
            self.logger.info("Documentation generation completed")
            
            return state
            
        except Exception as e:
            return self._handle_error(e, state, "documentation") or state


# Agent factory for easy instantiation
class AgentFactory:
    """Factory class for creating agent instances."""
    
    @staticmethod
    def create_agent(agent_type: str, config: AgentConfig = None):
        """Create an agent instance based on type."""
        agents = {
            "intake": IntakeAgent,
            "analysis": AnalysisAgent,
            "ambiguity_detection": AmbiguityDetectionAgent,
            "stakeholder_simulation": StakeholderSimulationAgent,
            "validation": ValidationAgent,
            "refinement": RefinementAgent,
            "documentation": DocumentationAgent
        }
        
        if agent_type not in agents:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        return agents[agent_type](config)
    
    @staticmethod
    def create_all_agents(config: AgentConfig = None) -> Dict[str, BaseAgent]:
        """Create all agent instances."""
        agent_types = [
            "intake", "analysis", "ambiguity_detection", 
            "stakeholder_simulation", "validation", "refinement", "documentation"
        ]
        
        return {
            agent_type: AgentFactory.create_agent(agent_type, config)
            for agent_type in agent_types
        }