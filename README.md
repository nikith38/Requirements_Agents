# Multi-Agent Requirements Gathering System

A sophisticated AI-powered system that uses multiple specialized agents to gather, analyze, and refine software requirements through an intelligent workflow built with LangChain and LangGraph.

## 🚀 Features

- **Multi-Agent Architecture**: 7 specialized agents working in coordination
- **Structured Output**: Pydantic models ensure consistent, validated outputs
- **LangChain Integration**: Latest LangChain framework with OpenAI models
- **LangGraph Workflow**: Intelligent routing and decision-making
- **Configurable**: Extensive configuration options via YAML and environment variables
- **Extensible**: Easy to add new agents or modify workflows
- **Production Ready**: Comprehensive error handling, logging, and validation

## 🏗️ Architecture

### Agent Workflow

```
User Input → Intake Agent → Analysis Agent → Stakeholder Simulation
                ↓              ↓                    ↓
         Technical Analysis → Validation → Quality Assurance
                ↓              ↓              ↓
         Risk Assessment → Prioritization → Final Output
                           ↓
                    Documentation
```

### Core Agents

1. **Intake Agent**: Initial requirement capture and basic validation
2. **Analysis Agent**: Deep analysis of requirements and identification of gaps
3. **Ambiguity Detection Agent**: Identifies unclear or ambiguous requirements
4. **Stakeholder Simulation Agent**: Simulates different stakeholder perspectives
5. **Validation Agent**: Validates completeness and consistency
6. **Refinement Agent**: Improves and refines requirements quality
7. **Documentation Agent**: Creates comprehensive documentation

## 📦 Installation

### Prerequisites

- Python 3.8+
- OpenAI API key

### Setup

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   # Create .env file from template
   python config.py --create-env .env.template
   cp .env.template .env
   
   # Edit .env and add your OpenAI API key
   OPENAI_API_KEY=your_api_key_here
   ```

4. **Create configuration file** (optional):
   ```bash
   python config.py --create-config config.yaml
   ```

5. **Validate environment**:
   ```bash
   python config.py --validate
   ```

## 🚀 Quick Start

### Basic Usage

```python
from workflow import RequirementGatheringWorkflow
from models import WorkflowState

# Initialize workflow
workflow = RequirementGatheringWorkflow()

# Create initial state
initial_state = WorkflowState(
    user_input="I need a web application for managing customer orders",
    requirements=[],
    current_agent="intake",
    iteration_count=0
)

# Run workflow
result = workflow.run(initial_state)
print(result.final_output)
```

### Command Line Interface

```bash
# Interactive mode
python main.py --interactive

# Process from file
python main.py --input requirements.txt --output results.json

# Stream processing
python main.py --input requirements.txt --stream

# Generate sample input
python main.py --create-sample
```

### Application Class

```python
from main import RequirementGatheringApp

# Initialize app
app = RequirementGatheringApp()

# Process requirements
result = app.process_requirements(
    "Build a mobile app for food delivery"
)

# Stream processing with callback
def on_update(state):
    print(f"Current agent: {state.current_agent}")
    print(f"Progress: {len(state.requirements)} requirements")

app.process_requirements_stream(
    "Build a mobile app for food delivery",
    callback=on_update
)
```

## 📋 Input Format

The system accepts various input formats:

### Simple Text
```
I need a web application for managing customer orders with payment processing.
```

### Structured Input (JSON)
```json
{
  "description": "Customer order management system",
  "context": {
    "business_domain": "E-commerce",
    "target_users": ["customers", "administrators"],
    "constraints": ["must integrate with existing payment system"]
  },
  "initial_requirements": [
    "User registration and authentication",
    "Product catalog management",
    "Shopping cart functionality"
  ]
}
```

### File Input
```bash
# Create sample input file
python main.py --create-sample

# Process from file
python main.py --input sample_input.json
```

## 📤 Output Format

The system produces comprehensive structured output:

```json
{
  "final_requirements": [
    {
      "id": "REQ-001",
      "title": "User Authentication",
      "description": "System shall provide secure user authentication",
      "type": "functional",
      "priority": "high",
      "acceptance_criteria": [...],
      "technical_notes": [...]
    }
  ],
  "metadata": {
    "processing_time": 45.2,
    "agents_involved": ["intake", "analysis", "validation"],
    "iteration_count": 2,
    "confidence_score": 0.92
  },
  "documentation": {
    "executive_summary": "...",
    "technical_architecture": "...",
    "implementation_roadmap": "..."
  },
  "risks_and_recommendations": [...]
}
```

## ⚙️ Configuration

### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_ORGANIZATION=your_org_id  # Optional
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional

# Agent Configuration
AGENT_MODEL_NAME=gpt-4o-mini
AGENT_TEMPERATURE=0.1
AGENT_MAX_TOKENS=4000

# Workflow Configuration
WORKFLOW_MAX_ITERATIONS=3
WORKFLOW_AMBIGUITY_THRESHOLD=0.3
WORKFLOW_VALIDATION_THRESHOLD=0.7

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Output
OUTPUT_DIRECTORY=output
```

### Configuration File (config.yaml)

```yaml
workflow:
  max_iterations: 3
  ambiguity_threshold: 0.3
  validation_threshold: 0.7
  enable_human_escalation: true

agent:
  model_name: "gpt-4o-mini"
  temperature: 0.1
  max_tokens: null
  timeout: 60

logging:
  level: "INFO"
  file: "logs/app.log"

output:
  directory: "output"
  format: "json"
  include_metadata: true
```

## 🔧 Advanced Usage

### Custom Agent Implementation

```python
from agents import BaseAgent
from models import WorkflowState

class CustomAgent(BaseAgent):
    def process(self, state: WorkflowState) -> WorkflowState:
        # Custom processing logic
        result = self.llm_chain.invoke({
            "input": state.user_input,
            "requirements": state.requirements
        })
        
        # Update state
        state.requirements.extend(result.new_requirements)
        return state

# Register custom agent
from agents import AgentFactory
AgentFactory.register_agent("custom", CustomAgent)
```

### Workflow Customization

```python
from workflow import RequirementGatheringWorkflow
from langgraph.graph import StateGraph

class CustomWorkflow(RequirementGatheringWorkflow):
    def _build_graph(self) -> StateGraph:
        graph = super()._build_graph()
        
        # Add custom node
        graph.add_node("custom_analysis", self._custom_analysis_node)
        graph.add_edge("analysis", "custom_analysis")
        graph.add_edge("custom_analysis", "validation")
        
        return graph
    
    def _custom_analysis_node(self, state: WorkflowState) -> WorkflowState:
        # Custom analysis logic
        return state
```

### Batch Processing

```python
from main import RequirementGatheringApp
import json

app = RequirementGatheringApp()

# Process multiple requirements
inputs = [
    "Build a CRM system",
    "Create a mobile banking app",
    "Develop an inventory management system"
]

results = []
for req_input in inputs:
    result = app.process_requirements(req_input)
    results.append(result)
    
    # Save individual result
    with open(f"output/result_{len(results)}.json", "w") as f:
        json.dump(result, f, indent=2)

# Save batch results
with open("output/batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## 🧪 Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock

# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

### Example Test

```python
import pytest
from workflow import RequirementGatheringWorkflow
from models import WorkflowState

def test_workflow_basic_flow():
    workflow = RequirementGatheringWorkflow()
    
    initial_state = WorkflowState(
        user_input="Simple web app",
        requirements=[],
        current_agent="intake"
    )
    
    result = workflow.run(initial_state)
    
    assert result.final_output is not None
    assert len(result.requirements) > 0
    assert result.metadata["confidence_score"] > 0.5
```

## 📊 Monitoring and Logging

### Log Configuration

```python
import logging
from config import setup_logging, load_config

# Load configuration
config = load_config()

# Setup logging
setup_logging(config["logging"])

# Use logger
logger = logging.getLogger(__name__)
logger.info("Application started")
```

### Performance Monitoring

```python
from main import RequirementGatheringApp
import time

app = RequirementGatheringApp()

start_time = time.time()
result = app.process_requirements("Build a web app")
processing_time = time.time() - start_time

print(f"Processing time: {processing_time:.2f} seconds")
print(f"Confidence score: {result['metadata']['confidence_score']}")
print(f"Requirements generated: {len(result['final_requirements'])}")
```

## 🚨 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   ```
   Error: OpenAI API key not found
   Solution: Set OPENAI_API_KEY environment variable
   ```

2. **Module Import Error**
   ```
   Error: No module named 'langchain'
   Solution: pip install -r requirements.txt
   ```

3. **Configuration File Not Found**
   ```
   Warning: Config file config.yaml not found
   Solution: python config.py --create-config config.yaml
   ```

4. **Output Directory Permission Error**
   ```
   Error: Permission denied writing to output directory
   Solution: Check directory permissions or set OUTPUT_DIRECTORY
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py --interactive

# Or in code
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

### Validation

```bash
# Validate environment
python config.py --validate

# Show current configuration
python config.py --show-config config.yaml
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run tests and ensure they pass
6. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-mock black flake8

# Format code
black .

# Lint code
flake8 .

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the excellent framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) for workflow orchestration
- [Pydantic](https://pydantic.dev/) for data validation
- [OpenAI](https://openai.com/) for the powerful language models

## 📞 Support

For questions, issues, or contributions:

1. Check the [troubleshooting section](#-troubleshooting)
2. Search existing issues
3. Create a new issue with detailed information
4. Include logs and configuration when reporting bugs

---

**Happy Requirements Gathering! 🎯**