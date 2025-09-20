"""Pydantic models for multi-agent requirement gathering system."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Any
from enum import Enum
from datetime import datetime


class RequirementType(str, Enum):
    """Types of requirements."""
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    CONSTRAINT = "constraint"
    ASSUMPTION = "assumption"


class Priority(str, Enum):
    """Priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ValidationStatus(str, Enum):
    """Validation status options."""
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_REVISION = "needs_revision"


class ApprovalStatus(str, Enum):
    """Approval status options."""
    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"
    REJECTED = "rejected"


class ComplexityLevel(str, Enum):
    """Implementation complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ComplianceStatus(str, Enum):
    """Compliance status options."""
    COMPLIANT = "compliant"
    NEEDS_REVIEW = "needs_review"
    NON_COMPLIANT = "non_compliant"


class StakeholderType(str, Enum):
    """Types of stakeholders."""
    BUSINESS = "business"
    TECHNICAL = "technical"
    USER = "user"
    PROJECT_MANAGER = "pm"
    QUALITY_ASSURANCE = "qa"


# Intake Agent Models
class ParsedRequirement(BaseModel):
    """Individual parsed requirement from intake agent."""
    id: str = Field(description="Unique requirement identifier")
    type: RequirementType = Field(description="Type of requirement")
    description: str = Field(description="Clear requirement statement")
    source: str = Field(description="Original text reference")
    stakeholders: List[str] = Field(description="List of relevant stakeholders")
    priority: Priority = Field(description="Requirement priority")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in parsing")
    extracted_at: datetime = Field(default_factory=datetime.now, description="Extraction timestamp")


class IntakeMetadata(BaseModel):
    """Metadata from intake processing."""
    total_requirements: int = Field(description="Total number of requirements extracted")
    processing_notes: str = Field(description="Important observations during processing")
    input_length: int = Field(description="Length of original input")
    processing_time: float = Field(description="Time taken to process input")


class IntakeOutput(BaseModel):
    """Output from intake agent."""
    parsed_requirements: List[ParsedRequirement]
    metadata: IntakeMetadata


# Analysis Agent Models
class SubRequirement(BaseModel):
    """Decomposed sub-requirement."""
    sub_id: str = Field(description="Sub-requirement identifier")
    description: str = Field(description="Atomic requirement description")
    acceptance_criteria: List[str] = Field(description="Testable criteria")
    estimated_complexity: ComplexityLevel = Field(description="Implementation complexity")


class AnalyzedRequirement(BaseModel):
    """Analyzed requirement with decomposition and relationships."""
    id: str = Field(description="Requirement identifier")
    original_requirement: str = Field(description="Original requirement text")
    decomposed_requirements: List[SubRequirement]
    dependencies: List[str] = Field(description="Dependent requirement IDs")
    conflicts: List[str] = Field(description="Conflicting requirement IDs")
    completeness_score: float = Field(ge=0.0, le=1.0, description="How complete the requirement is")
    specificity_score: float = Field(ge=0.0, le=1.0, description="How specific the requirement is")
    testability_score: float = Field(ge=0.0, le=1.0, description="How testable the requirement is")


class AnalysisSummary(BaseModel):
    """Summary of analysis results."""
    total_conflicts: int = Field(description="Number of conflicts found")
    dependency_count: int = Field(description="Total number of dependencies")
    completeness_issues: List[str] = Field(description="List of completeness issues")
    circular_dependencies: List[List[str]] = Field(description="Circular dependency chains")
    orphaned_requirements: List[str] = Field(description="Requirements with no dependencies")


class AnalysisOutput(BaseModel):
    """Output from analysis agent."""
    analyzed_requirements: List[AnalyzedRequirement]
    analysis_summary: AnalysisSummary


# Ambiguity Detection Models
class ClarificationQuestion(BaseModel):
    """Question for clarifying ambiguous requirements."""
    question: str = Field(description="Specific clarification question")
    target_stakeholder: StakeholderType = Field(description="Who should answer")
    urgency: Priority = Field(description="Question urgency")
    context: str = Field(description="Context for the question")


class AmbiguityAnalysis(BaseModel):
    """Analysis of ambiguities in a requirement."""
    requirement_id: str
    ambiguity_score: float = Field(ge=0.0, le=1.0, description="Overall ambiguity score")
    ambiguous_terms: List[str] = Field(description="List of unclear terms")
    missing_information: List[str] = Field(description="What information is missing")
    clarification_questions: List[ClarificationQuestion]
    suggested_improvements: List[str] = Field(description="Specific improvement suggestions")
    vague_quantifiers: List[str] = Field(description="Vague quantifiers found")
    undefined_terms: List[str] = Field(description="Undefined technical terms")


class AmbiguityOutput(BaseModel):
    """Output from ambiguity detection agent."""
    ambiguity_analysis: List[AmbiguityAnalysis]
    overall_ambiguity_score: float = Field(ge=0.0, le=1.0, description="Overall ambiguity across all requirements")
    critical_ambiguities: List[str] = Field(description="High-priority ambiguous requirements")
    total_questions_generated: int = Field(description="Total clarification questions generated")


# Stakeholder Simulation Models
class StakeholderFeedback(BaseModel):
    """Feedback from a specific stakeholder perspective."""
    requirement_id: str
    stakeholder_type: StakeholderType = Field(description="Type of stakeholder providing feedback")
    concerns: List[str] = Field(description="Stakeholder concerns")
    questions: List[str] = Field(description="Questions from stakeholder")
    suggestions: List[str] = Field(description="Improvement suggestions")
    approval_status: ApprovalStatus = Field(description="Approval decision")
    rationale: str = Field(description="Explanation of position")
    business_impact: Optional[str] = Field(description="Impact on business objectives")
    technical_feasibility: Optional[str] = Field(description="Technical feasibility assessment")


class ConflictAnalysis(BaseModel):
    """Analysis of conflicts between requirements."""
    conflicting_requirements: List[str] = Field(description="IDs of conflicting requirements")
    stakeholders_involved: List[StakeholderType] = Field(description="Stakeholders with conflicting views")
    conflict_description: str = Field(description="Nature of the conflict")
    proposed_resolution: str = Field(description="Suggested compromise or solution")
    impact_assessment: str = Field(description="Impact of the conflict")
    resolution_priority: Priority = Field(description="Priority for resolving conflict")


class StakeholderOutput(BaseModel):
    """Output from stakeholder simulation agent."""
    stakeholder_feedback: List[StakeholderFeedback]
    conflict_analysis: List[ConflictAnalysis]
    consensus_requirements: List[str] = Field(description="Requirements with stakeholder consensus")
    disputed_requirements: List[str] = Field(description="Requirements under dispute")


# Validation Agent Models
class ValidationResult(BaseModel):
    """Validation result for a single requirement."""
    requirement_id: str
    business_alignment_score: float = Field(ge=0.0, le=1.0, description="Alignment with business goals")
    technical_feasibility_score: float = Field(ge=0.0, le=1.0, description="Technical implementation feasibility")
    compliance_status: ComplianceStatus = Field(description="Regulatory compliance status")
    implementation_complexity: ComplexityLevel = Field(description="Implementation difficulty")
    estimated_effort: str = Field(description="Estimated development effort")
    risks: List[str] = Field(description="Identified implementation risks")
    validation_status: ValidationStatus = Field(description="Overall validation result")
    validation_notes: str = Field(description="Detailed validation feedback")
    resource_requirements: List[str] = Field(description="Required resources for implementation")
    timeline_impact: str = Field(description="Impact on project timeline")


class ValidationOutput(BaseModel):
    """Output from validation agent."""
    validation_results: List[ValidationResult]
    overall_validation_score: float = Field(ge=0.0, le=1.0, description="Overall validation score")
    critical_issues: List[str] = Field(description="Blocking validation issues")
    recommendations: List[str] = Field(description="Improvement recommendations")
    approved_requirements: List[str] = Field(description="Requirements that passed validation")
    rejected_requirements: List[str] = Field(description="Requirements that failed validation")


# Refinement Agent Models
class RefinedRequirement(BaseModel):
    """Refined and improved requirement."""
    id: str
    original_text: str = Field(description="Original requirement text")
    refined_text: str = Field(description="Improved requirement text")
    changes_made: List[str] = Field(description="Specific changes applied")
    acceptance_criteria: List[str] = Field(description="Clear, testable acceptance criteria")
    definition_of_done: List[str] = Field(description="Completion criteria")
    test_scenarios: List[str] = Field(description="Example test cases")
    refinement_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in refinements")
    quality_improvements: List[str] = Field(description="Quality improvements made")
    measurable_criteria: List[str] = Field(description="Quantifiable success metrics")


class RefinementSummary(BaseModel):
    """Summary of refinement process."""
    total_changes: int = Field(description="Total number of changes made")
    issues_resolved: List[str] = Field(description="Issues that were resolved")
    remaining_issues: List[str] = Field(description="Issues that still need attention")
    quality_score_improvement: float = Field(description="Improvement in overall quality score")


class RefinementOutput(BaseModel):
    """Output from refinement agent."""
    refined_requirements: List[RefinedRequirement]
    refinement_summary: RefinementSummary


# Documentation Agent Models
class TraceabilityItem(BaseModel):
    """Traceability matrix item."""
    requirement_id: str
    business_objective: str = Field(description="Linked business objective")
    test_cases: List[str] = Field(description="Associated test cases")
    implementation_tasks: List[str] = Field(description="Development tasks")
    acceptance_criteria: List[str] = Field(description="Acceptance criteria")
    stakeholder_approval: Dict[str, str] = Field(description="Stakeholder approvals")


class RequirementsDocument(BaseModel):
    """Structured requirements document."""
    functional_requirements: List[Dict[str, Any]] = Field(description="Functional requirements")
    non_functional_requirements: List[Dict[str, Any]] = Field(description="Non-functional requirements")
    constraints: List[Dict[str, Any]] = Field(description="System constraints")
    assumptions: List[Dict[str, Any]] = Field(description="Project assumptions")
    glossary: Dict[str, str] = Field(description="Term definitions")


class DocumentationPackage(BaseModel):
    """Complete documentation package."""
    executive_summary: str = Field(description="High-level project overview")
    requirements_document: RequirementsDocument
    traceability_matrix: List[TraceabilityItem]
    acceptance_criteria_document: Dict[str, List[str]] = Field(description="Detailed acceptance criteria")
    test_plan_outline: List[str] = Field(description="High-level test plan")
    implementation_roadmap: List[str] = Field(description="Implementation phases")
    risk_assessment: List[str] = Field(description="Project risks")
    appendices: Dict[str, Any] = Field(description="Additional documentation")


class DocumentMetadata(BaseModel):
    """Documentation metadata."""
    version: str = Field(description="Document version")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    total_requirements: int = Field(description="Total number of requirements")
    document_completeness: float = Field(ge=0.0, le=1.0, description="Completeness score")
    review_status: str = Field(description="Document review status")
    approvers: List[str] = Field(description="List of approvers")


class DocumentationOutput(BaseModel):
    """Output from documentation agent."""
    documentation: DocumentationPackage
    document_metadata: DocumentMetadata


# Workflow State Models
class WorkflowState(BaseModel):
    """State object for the workflow."""
    raw_input: str = Field(description="Original user input")
    current_stage: str = Field(description="Current workflow stage")
    intake_output: Optional[IntakeOutput] = None
    analysis_output: Optional[AnalysisOutput] = None
    ambiguity_output: Optional[AmbiguityOutput] = None
    stakeholder_output: Optional[StakeholderOutput] = None
    validation_output: Optional[ValidationOutput] = None
    refinement_output: Optional[RefinementOutput] = None
    documentation_output: Optional[DocumentationOutput] = None
    iteration_count: int = Field(default=0, description="Number of refinement iterations")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Warnings generated")
    processing_time: Dict[str, float] = Field(default_factory=dict, description="Time spent in each stage")
    quality_metrics: Dict[str, float] = Field(default_factory=dict, description="Quality metrics")


# Configuration Models
class AgentConfig(BaseModel):
    """Configuration for individual agents."""
    model_name: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens per response")
    timeout: int = Field(default=60, description="Request timeout in seconds")


class WorkflowConfig(BaseModel):
    """Configuration for the entire workflow."""
    max_iterations: int = Field(default=3, description="Maximum refinement iterations")
    ambiguity_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Ambiguity threshold for stakeholder simulation")
    validation_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Validation threshold for approval")
    enable_human_escalation: bool = Field(default=False, description="Human escalation disabled - not implemented")
    parallel_processing: bool = Field(default=False, description="Enable parallel agent processing where possible")
    agent_config: AgentConfig = Field(default_factory=AgentConfig, description="Default agent configuration")