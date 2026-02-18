"""LLM client wrapper for API calls."""

import os
from dotenv import load_dotenv
load_dotenv()
from typing import Dict, Any, List, Optional
from openai import OpenAI

import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper for OpenAI-compatible API client."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 200.0):
        """
        Initialize LLM client.
        
        Args:
            base_url: Base URL for the API
            api_key: API key (if None, read from environment)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        
        if api_key is None:
            api_key = os.getenv("OLLAMA_API_KEY")
            
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        
    def call(
        self,
        messages: List[Dict[str, str]],
        model_name: str,
        max_tokens: int = 16382,
        temperature: float = 0.7,
        top_p: float = 0.95,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Call LLM with given messages and parameters.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model_name: Name of the model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            extra_body: Extra parameters for the request body
            **kwargs: Additional parameters
            
        Returns:
            Model response text
        """
        params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            **kwargs
        }
        
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **params,
                extra_body=extra_body if extra_body else None
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM Call Error: {e}")
            return ""
