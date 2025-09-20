"""Main application entry point for the multi-agent requirement gathering system."""

import os
import sys
import logging
import argparse
from typing import Optional, Dict, Any
from datetime import datetime
import json
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import WorkflowConfig, AgentConfig, WorkflowState
from workflow import RequirementGatheringWorkflow, run_requirement_gathering
from config import load_config, setup_logging, validate_environment


class RequirementGatheringApp:
    """Main application class for the requirement gathering system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the application with configuration."""
        self.config_path = config_path or "config.yaml"
        self.config = None
        self.workflow = None
        self.logger = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the application components."""
        try:
            # Load configuration
            self.config = load_config(self.config_path)
            
            # Setup logging
            setup_logging(self.config.get("logging", {}))
            self.logger = logging.getLogger(self.__class__.__name__)
            
            # Validate environment
            validate_environment()
            
            # Initialize workflow
            workflow_config = WorkflowConfig(**self.config.get("workflow", {}))
            agent_config = AgentConfig(**self.config.get("agent", {}))
            workflow_config.agent_config = agent_config
            
            self.workflow = RequirementGatheringWorkflow(workflow_config)
            
            self.logger.info("Application initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize application: {str(e)}")
            sys.exit(1)
    
    def process_requirements(self, input_text: str, thread_id: Optional[str] = None) -> WorkflowState:
        """Process requirements using the workflow."""
        try:
            self.logger.info(f"Processing requirements for thread: {thread_id or 'default'}")
            
            # Run the workflow
            result = self.workflow.run(input_text, thread_id)
            
            self.logger.info(f"Requirements processing completed: {result.current_stage}")
            return result
            
        except Exception as e:
            self.logger.error(f"Requirements processing failed: {str(e)}")
            raise
    
    def process_requirements_stream(self, input_text: str, thread_id: Optional[str] = None):
        """Process requirements with streaming output."""
        try:
            self.logger.info(f"Starting streaming requirements processing for thread: {thread_id or 'default'}")
            
            for step in self.workflow.run_async(input_text, thread_id):
                yield step
                
        except Exception as e:
            self.logger.error(f"Streaming requirements processing failed: {str(e)}")
            yield {"error": str(e)}
    
    def get_status(self, thread_id: str = "default") -> Dict[str, Any]:
        """Get workflow status for a thread."""
        return self.workflow.get_workflow_status(thread_id)
    
    def reset_thread(self, thread_id: str = "default"):
        """Reset a workflow thread."""
        self.workflow.reset_workflow(thread_id)
        self.logger.info(f"Thread {thread_id} reset")
    
    def save_results(self, result: WorkflowState, output_path: str):
        """Save workflow results to file."""
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert result to dictionary for JSON serialization
            result_dict = {
                "timestamp": datetime.now().isoformat(),
                "raw_input": result.raw_input,
                "current_stage": result.current_stage,
                "iteration_count": result.iteration_count,
                "errors": result.errors,
                "warnings": result.warnings,
                "processing_time": result.processing_time,
                "quality_metrics": result.quality_metrics
            }
            
            # Add outputs if available (using mode='json' for proper datetime serialization)
            if result.intake_output:
                result_dict["intake_output"] = result.intake_output.model_dump(mode='json')
            
            if result.analysis_output:
                result_dict["analysis_output"] = result.analysis_output.model_dump(mode='json')
            
            if result.ambiguity_output:
                result_dict["ambiguity_output"] = result.ambiguity_output.model_dump(mode='json')
            
            if result.stakeholder_output:
                result_dict["stakeholder_output"] = result.stakeholder_output.model_dump(mode='json')
            
            if result.validation_output:
                result_dict["validation_output"] = result.validation_output.model_dump(mode='json')
            
            if result.refinement_output:
                result_dict["refinement_output"] = result.refinement_output.model_dump(mode='json')
            
            if result.documentation_output:
                result_dict["documentation_output"] = result.documentation_output.model_dump(mode='json')
            
            # Save to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise


def create_sample_input_file(file_path: str):
    """Create a sample input file for testing."""
    sample_content = """
# Sample Requirements Input

We need to build a comprehensive e-commerce platform that will serve both B2B and B2C customers.

## Core Features Needed:

1. **User Management**
   - User registration and authentication
   - Profile management
   - Role-based access control (customers, vendors, admins)
   - Social media login integration

2. **Product Catalog**
   - Product listing and search functionality
   - Category and tag management
   - Product reviews and ratings
   - Inventory management
   - Multi-vendor support

3. **Shopping Experience**
   - Shopping cart functionality
   - Wishlist management
   - Product comparison
   - Recommendation engine
   - Mobile-responsive design

4. **Order Management**
   - Checkout process
   - Multiple payment methods (credit cards, PayPal, digital wallets)
   - Order tracking
   - Return and refund management
   - Invoice generation

5. **Business Intelligence**
   - Sales analytics and reporting
   - Customer behavior tracking
   - Inventory analytics
   - Performance dashboards

## Non-Functional Requirements:

- The system should handle 10,000 concurrent users
- Page load times should be under 2 seconds
- 99.9% uptime availability
- GDPR and PCI DSS compliance
- Multi-language support (English, Spanish, French)
- Mobile-first responsive design
- SEO optimization

## Technical Constraints:

- Must integrate with existing ERP system
- Should use cloud infrastructure (AWS preferred)
- Must support API integrations for third-party services
- Database should be scalable and support high availability

## Business Goals:

- Increase online sales by 40% within first year
- Reduce customer acquisition cost by 25%
- Improve customer satisfaction scores to 4.5/5
- Support expansion to 3 new markets within 18 months

The project timeline is 8 months with a budget of $500,000. The system needs to be launched before the holiday shopping season.
    """
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(sample_content.strip())
    
    print(f"Sample input file created: {file_path}")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Requirements Gathering System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input "We need a user management system"
  python main.py --file requirements.txt --output results.json
  python main.py --create-sample sample_input.txt
  python main.py --interactive
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        help="Direct input text for requirements processing"
    )
    
    parser.add_argument(
        "--file", "-f",
        help="Input file containing requirements text"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="output/requirements_result.json",
        help="Output file path for results (default: output/requirements_result.json)"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Configuration file path (default: config.yaml)"
    )
    
    parser.add_argument(
        "--thread-id", "-t",
        help="Thread ID for workflow tracking"
    )
    
    parser.add_argument(
        "--stream", "-s",
        action="store_true",
        help="Enable streaming output"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--create-sample",
        help="Create a sample input file at the specified path"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Handle sample file creation
    if args.create_sample:
        create_sample_input_file(args.create_sample)
        return
    
    # Setup basic logging if verbose
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    try:
        # Initialize application
        app = RequirementGatheringApp(args.config)
        
        # Interactive mode
        if args.interactive:
            run_interactive_mode(app)
            return
        
        # Get input text
        input_text = None
        if args.input:
            input_text = args.input
        elif args.file:
            if not os.path.exists(args.file):
                print(f"Error: Input file '{args.file}' not found")
                sys.exit(1)
            with open(args.file, 'r', encoding='utf-8') as f:
                input_text = f.read()
        else:
            print("Error: Either --input or --file must be provided")
            parser.print_help()
            sys.exit(1)
        
        # Process requirements
        if args.stream:
            print("Starting streaming requirements processing...")
            for step in app.process_requirements_stream(input_text, args.thread_id):
                print(f"Step: {step}")
        else:
            print("Processing requirements...")
            result = app.process_requirements(input_text, args.thread_id)
            
            # Save results
            app.save_results(result, args.output)
            
            # Print summary
            print(f"\nProcessing completed!")
            print(f"Final stage: {result.current_stage}")
            print(f"Iterations: {result.iteration_count}")
            print(f"Errors: {len(result.errors)}")
            print(f"Warnings: {len(result.warnings)}")
            print(f"Results saved to: {args.output}")
            
            if result.errors:
                print(f"\nErrors encountered:")
                for error in result.errors:
                    print(f"  - {error}")
            
            if result.warnings:
                print(f"\nWarnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_interactive_mode(app: RequirementGatheringApp):
    """Run the application in interactive mode."""
    print("\n=== Multi-Agent Requirements Gathering System ===")
    print("Interactive Mode - Type 'help' for commands, 'quit' to exit\n")
    
    thread_id = "interactive"
    
    while True:
        try:
            command = input("req-agent> ").strip()
            
            if not command:
                continue
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            elif command.lower() == 'help':
                print_help()
                continue
            
            elif command.lower() == 'status':
                status = app.get_status(thread_id)
                print(f"Status: {json.dumps(status, indent=2)}")
                continue
            
            elif command.lower() == 'reset':
                app.reset_thread(thread_id)
                print("Thread reset successfully")
                continue
            
            elif command.lower().startswith('save '):
                output_path = command[5:].strip()
                if not output_path:
                    print("Error: Please specify output path")
                    continue
                # This would need the last result - simplified for demo
                print(f"Save functionality would save to: {output_path}")
                continue
            
            # Process as requirements input
            print("Processing requirements...")
            result = app.process_requirements(command, thread_id)
            
            print(f"\nProcessing completed!")
            print(f"Stage: {result.current_stage}")
            print(f"Errors: {len(result.errors)}")
            print(f"Warnings: {len(result.warnings)}")
            
            if result.documentation_output:
                print("\nDocumentation generated successfully!")
                print(f"Executive Summary: {result.documentation_output.documentation.executive_summary[:200]}...")
            
        except KeyboardInterrupt:
            print("\nUse 'quit' to exit")
        except Exception as e:
            print(f"Error: {str(e)}")


def print_help():
    """Print help information for interactive mode."""
    help_text = """
Available commands:
  help          - Show this help message
  status        - Show current workflow status
  reset         - Reset the current thread
  save <path>   - Save results to specified path
  quit/exit/q   - Exit the application
  
  Any other input will be processed as requirements text.
  
Example:
  req-agent> We need a user authentication system with login and registration
    """
    print(help_text)


if __name__ == "__main__":
    main()