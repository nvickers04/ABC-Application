# src/utils.py
# Purpose: Shared utility functions for the AI Portfolio Manager, starting with YAML loading and prompt templating.
# This enables beautiful integration (e.g., fresh YAML pulls in agents) and traceability (e.g., error logging for audits).
# Structural Reasoning: Centralizes common ops to avoid duplication across agents; backs funding with robust error-handling (e.g., defaults on YAML fails, preserving 10% ROI floor).
# Ties to code-skeleton.md: Implements load_yaml and prompt_loader as base for agent inits; async-ready for scalability.
# For legacy wealth: Ensures configs load dynamically without crashes, protecting capital via safe defaults.

import os
import yaml
import logging
from string import Template  # Use Python's built-in Template instead of langchain_core

# Setup logging for traceability (output to console/file for audits)
# Logging configured centrally in logging_config.py
logger = logging.getLogger(__name__)

def load_yaml(file_path: str) -> dict:
    """
    Loads a YAML file safely, with defaults on failure for robustness.
    Args:
        file_path (str): Path to YAML (e.g., '../config/risk-constraints.yaml').
    Returns:
        dict: Parsed YAML or safe defaults.
    Reasoning: Fresh loads ensure up-to-date constraints (e.g., max_drawdown=0.05); logs errors for audits, preserving alpha via defaults.
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        logger.info(f"Loaded YAML from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"YAML file not found: {file_path}. Using safe defaults.")
        return {'constraints': {'max_drawdown': 0.05, 'pop_floor': 0.60}}  # Safe defaults to prevent crashes.
    except yaml.YAMLError as e:
        logger.error(f"YAML parse error in {file_path}: {e}. Using safe defaults.")
        return {'constraints': {'max_drawdown': 0.05, 'pop_floor': 0.60}}

def load_prompt_template(role: str, base_path: str = '../config/base_prompt.txt', role_path: str = None) -> str:
    """
    Loads and formats a prompt template for an agent.
    Args:
        role (str): Agent role (e.g., 'risk').
        base_path (str): Path to shared base_prompt.txt.
        role_path (str): Optional path to role-specific prompt (e.g., '../risk-agent-prompt.md').
    Returns:
        str: Formatted prompt string.
    Reasoning: Combines base + role-specific for consistency (e.g., "You are a well-informed Risk Agent..."); ties to behaviors for self-improving decisions.
    """
    try:
        with open(base_path, 'r', encoding='utf-8') as base_file:
            base_prompt = base_file.read()

        # Use Python's built-in Template for simple string substitution
        base_template = Template(base_prompt)
        base_formatted = base_template.safe_substitute(
            agent_role=role.capitalize(),
            goal_weights='profit:0.60, time:0.20, risk:0.20',
            roi_targets='10-20% monthly',
            sd_value='1.1'
        )

        full_prompt = base_formatted
        if role_path:
            with open(role_path, 'r', encoding='utf-8') as role_file:
                role_prompt = role_file.read()
            # Replace {base_prompt} with the formatted base
            full_prompt = role_prompt.replace('{base_prompt}', base_formatted)
            # No need for second substitute since role has no other placeholders
            formatted = full_prompt
        else:
            formatted = base_formatted
        logger.info(f"Loaded and formatted prompt for role: {role}")
        return formatted
    except FileNotFoundError as e:
        logger.error(f"Prompt file not found: {e}. Using minimal default.")
        return f"You are a {role} Agent. Pursue max profitability with discipline."  # Fallback to avoid crashes.

# Example standalone test (run python src/utils.py to verify)
if __name__ == "__main__":
    # Test YAML load
    risk_data = load_yaml('../config/risk-constraints.yaml')
    print("Loaded Risk Constraints:", risk_data)
    
    # Test prompt load
    risk_prompt = load_prompt_template('risk', base_path='../config/base_prompt.txt', role_path='../agents/risk-agent-complete.md')
    print("Formatted Risk Prompt Snippet:", risk_prompt[:200] + "...")  # Truncated for brevity.