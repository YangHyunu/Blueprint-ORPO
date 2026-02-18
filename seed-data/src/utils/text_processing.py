"""Utility functions for text processing and answer extraction."""

import re
import json
from typing import Optional


def extract_answer_num(text: str) -> Optional[int]:
    """
    Extract answer number from text containing reasoning and JSON.
    
    Args:
        text: Text containing reasoning and answer in format {"Answer": "3"}
        
    Returns:
        Extracted answer number or None if not found
    """
    try:
        # Search for JSON pattern with Answer field
        match = re.search(r'\{"Answer":\s*"(\d+)"\}', text)
        if match:
            return int(match.group(1))

        # Backup: Search for full JSON object
        match_json = re.search(r'\{.*?\}', text, re.DOTALL)
        if match_json:
            data = json.loads(match_json.group(0))
            return int(data.get("Answer"))

        return None
    except Exception:
        return None


def format_problem_prompt(paragraph: str, question: str, choices: list) -> str:
    """
    Format problem into a prompt for the LLM.
    
    Args:
        paragraph: Problem context/passage
        question: Question text
        choices: List of answer choices
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""<제시문>
{paragraph}
<질문>
{question}
"""
    for k, choice in enumerate(choices):
        prompt += f"{k+1}. {choice}\n"
    
    return prompt
