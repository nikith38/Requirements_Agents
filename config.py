"""Configuration management for the multi-agent requirement gathering system."""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv


# Default configuration
DEFAULT_CONFIG = {
    "workflow": {
        "max_iterations": 3,
        "ambiguity_threshold": 0.3,
        "validation_threshold": 0.7,
        "enable_human_escalation": False,
        "parallel_processing": False
    },
    "agent": {
        "model_name": "gpt-4o-mini",
        "temperature": 0.1,
        "max_tokens": None,
        "timeout": 60
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": None,
        "max_bytes": 10485760,  # 10MB
        "backup_count": 5
    },
    "openai": {
        "api_key": None,  # Will be loaded from environment
        "organization": None,
        "base_url": None
    },
    "output": {
        "directory": "output",
        "format": "json",
        "include_metadata": True,
        "compress": False
    },
    "performance": {
        "cache_enabled": True,
        "cache_ttl": 3600,  # 1 hour
        "max_concurrent_requests": 5,
        "request_timeout": 30
    }
}


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from file with fallback to defaults."""
    # Load environment variables
    load_dotenv(override=True)
    
    config = DEFAULT_CONFIG.copy()
    
    # Try to load from file
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    config = merge_configs(config, file_config)
        except Exception as e:
            print(f"Warning: Failed to load config file {config_path}: {str(e)}")
            print("Using default configuration")
    else:
        print(f"Config file {config_path} not found, using defaults")
    
    # Override with environment variables
    config = apply_environment_overrides(config)
    
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def apply_environment_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration."""
    # OpenAI configuration
    if os.getenv("OPENAI_API_KEY"):
        config["openai"]["api_key"] = os.getenv("OPENAI_API_KEY")
    
    if os.getenv("OPENAI_ORGANIZATION"):
        config["openai"]["organization"] = os.getenv("OPENAI_ORGANIZATION")
    
    if os.getenv("OPENAI_BASE_URL"):
        config["openai"]["base_url"] = os.getenv("OPENAI_BASE_URL")
    
    # Agent configuration
    if os.getenv("AGENT_MODEL_NAME"):
        config["agent"]["model_name"] = os.getenv("AGENT_MODEL_NAME")
    
    if os.getenv("AGENT_TEMPERATURE"):
        try:
            config["agent"]["temperature"] = float(os.getenv("AGENT_TEMPERATURE"))
        except ValueError:
            pass
    
    if os.getenv("AGENT_MAX_TOKENS"):
        try:
            config["agent"]["max_tokens"] = int(os.getenv("AGENT_MAX_TOKENS"))
        except ValueError:
            pass
    
    # Workflow configuration
    if os.getenv("WORKFLOW_MAX_ITERATIONS"):
        try:
            config["workflow"]["max_iterations"] = int(os.getenv("WORKFLOW_MAX_ITERATIONS"))
        except ValueError:
            pass
    
    if os.getenv("WORKFLOW_AMBIGUITY_THRESHOLD"):
        try:
            config["workflow"]["ambiguity_threshold"] = float(os.getenv("WORKFLOW_AMBIGUITY_THRESHOLD"))
        except ValueError:
            pass
    
    if os.getenv("WORKFLOW_VALIDATION_THRESHOLD"):
        try:
            config["workflow"]["validation_threshold"] = float(os.getenv("WORKFLOW_VALIDATION_THRESHOLD"))
        except ValueError:
            pass
    
    # Logging configuration
    if os.getenv("LOG_LEVEL"):
        config["logging"]["level"] = os.getenv("LOG_LEVEL").upper()
    
    if os.getenv("LOG_FILE"):
        config["logging"]["file"] = os.getenv("LOG_FILE")
    
    # Output configuration
    if os.getenv("OUTPUT_DIRECTORY"):
        config["output"]["directory"] = os.getenv("OUTPUT_DIRECTORY")
    
    return config


def setup_logging(logging_config: Dict[str, Any]):
    """Setup logging configuration."""
    level = getattr(logging, logging_config.get("level", "INFO").upper())
    format_str = logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_str,
        force=True
    )
    
    # Setup file logging if specified
    log_file = logging_config.get("file")
    if log_file:
        try:
            from logging.handlers import RotatingFileHandler
            
            # Create log directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Setup rotating file handler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=logging_config.get("max_bytes", 10485760),
                backupCount=logging_config.get("backup_count", 5)
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(format_str))
            
            # Add to root logger
            logging.getLogger().addHandler(file_handler)
            
        except Exception as e:
            print(f"Warning: Failed to setup file logging: {str(e)}")


def validate_environment() -> bool:
    """Validate that the environment is properly configured."""
    errors = []
    warnings = []
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY environment variable is not set")
    
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        errors.append(f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check required packages
    required_packages = [
        "langchain",
        "langgraph", 
        "langchain_openai",
        "pydantic",
        "openai",
        "yaml",
        "python-dotenv"
    ]
    
    # Map install names to import names for packages that differ
    import_name_map = {
        "python-dotenv": "dotenv",
        "langchain_openai": "langchain_openai",
        "langgraph": "langgraph"
    }
    
    missing_packages = []
    for package in required_packages:
        import_name = import_name_map.get(package, package.replace("-", "_"))
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        errors.append(f"Missing required packages: {', '.join(missing_packages)}")
        errors.append("Run: pip install -r requirements.txt")
    
    # Check output directory permissions
    output_dir = os.getenv("OUTPUT_DIRECTORY", "output")
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        test_file = Path(output_dir) / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        warnings.append(f"Output directory may not be writable: {str(e)}")
    
    # Print validation results
    if errors:
        print("\nEnvironment Validation Errors:")
        for error in errors:
            print(f"  ❌ {error}")
    
    if warnings:
        print("\nEnvironment Validation Warnings:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    
    if not errors and not warnings:
        print("✅ Environment validation passed")
    
    if errors:
        raise EnvironmentError("Environment validation failed. Please fix the errors above.")
    
    return True


def create_default_config_file(config_path: str = "config.yaml"):
    """Create a default configuration file."""
    config_content = """
# Multi-Agent Requirements Gathering System Configuration

# Workflow Configuration
workflow:
  max_iterations: 3                    # Maximum refinement iterations
  ambiguity_threshold: 0.3             # Threshold for triggering stakeholder simulation
  validation_threshold: 0.7            # Threshold for passing validation
  enable_human_escalation: false       # Human escalation disabled - not implemented
  parallel_processing: false           # Enable parallel agent processing (experimental)

# Agent Configuration
agent:
  model_name: "gpt-4o-mini"            # OpenAI model to use
  temperature: 0.1                     # Model temperature (0.0-2.0)
  max_tokens: null                     # Maximum tokens per response (null for model default)
  timeout: 60                          # Request timeout in seconds

# OpenAI Configuration (can be overridden by environment variables)
openai:
  api_key: null                        # Set via OPENAI_API_KEY environment variable
  organization: null                   # Set via OPENAI_ORGANIZATION environment variable
  base_url: null                       # Set via OPENAI_BASE_URL environment variable

# Logging Configuration
logging:
  level: "INFO"                        # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: null                           # Log file path (null for console only)
  max_bytes: 10485760                  # Max log file size (10MB)
  backup_count: 5                      # Number of backup log files

# Output Configuration
output:
  directory: "output"                  # Output directory for results
  format: "json"                       # Output format (json, yaml)
  include_metadata: true               # Include processing metadata
  compress: false                      # Compress output files

# Performance Configuration
performance:
  cache_enabled: true                  # Enable response caching
  cache_ttl: 3600                      # Cache time-to-live in seconds
  max_concurrent_requests: 5           # Maximum concurrent API requests
  request_timeout: 30                  # Individual request timeout
    """
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content.strip())
    
    print(f"Default configuration file created: {config_path}")


def create_env_template(env_path: str = ".env.template"):
    """Create a template .env file."""
    env_content = """
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_ORGANIZATION=your_org_id_here
# OPENAI_BASE_URL=https://api.openai.com/v1

# Agent Configuration
# AGENT_MODEL_NAME=gpt-4o-mini
# AGENT_TEMPERATURE=0.1
# AGENT_MAX_TOKENS=4000

# Workflow Configuration
# WORKFLOW_MAX_ITERATIONS=3
# WORKFLOW_AMBIGUITY_THRESHOLD=0.3
# WORKFLOW_VALIDATION_THRESHOLD=0.7

# Logging Configuration
# LOG_LEVEL=INFO
# LOG_FILE=logs/app.log

# Output Configuration
# OUTPUT_DIRECTORY=output
    """
    
    with open(env_path, 'w', encoding='utf-8') as f:
        f.write(env_content.strip())
    
    print(f"Environment template file created: {env_path}")
    print("Copy this to .env and fill in your actual values")


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get a configuration value using dot notation (e.g., 'agent.model_name')."""
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def set_config_value(config: Dict[str, Any], key_path: str, value: Any):
    """Set a configuration value using dot notation."""
    keys = key_path.split('.')
    current = config
    
    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the final value
    current[keys[-1]] = value


def print_config_summary(config: Dict[str, Any]):
    """Print a summary of the current configuration."""
    print("\n=== Configuration Summary ===")
    
    # Agent settings
    print(f"Agent Model: {get_config_value(config, 'agent.model_name')}")
    print(f"Temperature: {get_config_value(config, 'agent.temperature')}")
    print(f"Max Tokens: {get_config_value(config, 'agent.max_tokens') or 'Default'}")
    
    # Workflow settings
    print(f"Max Iterations: {get_config_value(config, 'workflow.max_iterations')}")
    print(f"Ambiguity Threshold: {get_config_value(config, 'workflow.ambiguity_threshold')}")
    print(f"Validation Threshold: {get_config_value(config, 'workflow.validation_threshold')}")
    
    # Output settings
    print(f"Output Directory: {get_config_value(config, 'output.directory')}")
    print(f"Log Level: {get_config_value(config, 'logging.level')}")
    
    # API settings
    api_key = get_config_value(config, 'openai.api_key')
    if api_key:
        print(f"OpenAI API Key: {'*' * (len(api_key) - 8) + api_key[-8:]}")
    else:
        print("OpenAI API Key: Not configured")
    
    print("=" * 30)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration management utilities")
    parser.add_argument("--create-config", help="Create default config file")
    parser.add_argument("--create-env", help="Create .env template file")
    parser.add_argument("--validate", action="store_true", help="Validate environment")
    parser.add_argument("--show-config", help="Show configuration from file")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_default_config_file(args.create_config)
    elif args.create_env:
        create_env_template(args.create_env)
    elif args.validate:
        try:
            validate_environment()
            print("Environment validation successful!")
        except Exception as e:
            print(f"Environment validation failed: {str(e)}")
    elif args.show_config:
        config = load_config(args.show_config)
        print_config_summary(config)
    else:
        parser.print_help()