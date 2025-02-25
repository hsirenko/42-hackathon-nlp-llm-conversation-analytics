"""
LLM-based inference modules for conversation detection.
This package contains different LLM implementations for detecting and labeling conversations.
"""

from .gpt4 import GPT4ConversationDetector

__all__ = ['GPT4ConversationDetector'] 