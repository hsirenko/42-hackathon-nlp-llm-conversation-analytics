#!/usr/bin/env python3
"""
Unit Tests Script for LLM Benchmarks

This script orchestrates the evaluation of LLM models across three core tasks:
1. Spam Classification
2. Conversation Clustering
3. Topic Labeling

Usage:
    python run_unit_tests.py --model MODEL_NAME --community COMMUNITY_NAME [--prompt PROMPT_FILE]
"""

import argparse
import logging
from pathlib import Path
import sys
import json
from typing import Dict, Any

# Import evaluation modules
from spam_metrics import evaluate_spam_detection
from conversation_metrics import evaluate_conversation_detection
from evaluate_topics import evaluate_topic_labels

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_prompt(prompt_path: Path) -> str:
    """Load the prompt template from a file."""
    try:
        with open(prompt_path, 'r') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading prompt file: {e}")
        raise

def run_evaluations(
    model: str,
    community: str,
    prompt_path: Path
) -> Dict[str, Any]:
    """
    Run all three evaluations and collect results.
    
    Args:
        model: Name of the model being evaluated
        community: Name of the community dataset to use
        prompt_path: Path to the prompt template file
    
    Returns:
        Dictionary containing results from all evaluations
    """
    results = {}
    community_path = Path(community)
    
    try:
        # 1. Spam Classification
        logger.info("Running spam classification evaluation...")
        evaluate_spam_detection(str(community_path))
        results['spam_metrics'] = f"metrics_spam_detection_{community_path.name}.csv"
        
        # 2. Conversation Detection
        logger.info("Running conversation detection evaluation...")
        evaluate_conversation_detection(str(community_path))
        results['conversation_metrics'] = f"metrics_conversation_detection_{community_path.name}.csv"
        
        # 3. Topic Labeling
        logger.info("Running topic labeling evaluation...")
        evaluate_topic_labels(str(community_path))
        results['topic_metrics'] = f"metrics_topic_labels_{community_path.name}.csv"
        
        return results
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description='Run unit tests for LLM benchmarks across all core tasks.'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Name of the model to evaluate (e.g., claude35, gpt4)'
    )
    parser.add_argument(
        '--community',
        required=True,
        help='Path to the community dataset directory'
    )
    parser.add_argument(
        '--prompt',
        default='conversation_detection_prompt.txt',
        help='Path to the prompt template file (default: conversation_detection_prompt.txt)'
    )
    
    args = parser.parse_args()
    
    try:
        # Validate paths
        community_path = Path(args.community)
        prompt_path = Path(args.prompt)
        
        if not community_path.exists():
            raise ValueError(f"Community directory does not exist: {args.community}")
        if not prompt_path.exists():
            raise ValueError(f"Prompt file does not exist: {args.prompt}")
            
        # Run evaluations
        results = run_evaluations(
            model=args.model,
            community=args.community,
            prompt_path=prompt_path
        )
        
        # Output results summary
        print("\nEvaluation Results Summary:")
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        logger.error(f"Error running unit tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 