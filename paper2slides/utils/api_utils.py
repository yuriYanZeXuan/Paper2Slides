"""
API Utilities for Paper2Slides

Provides unified API key loading and client configuration compatible with
PosterGen2 environment settings (GEMINI_TEXT_KEY, RUNWAY_API_KEY, etc.)
"""

import os
import sys
import json
import logging
import requests
from pathlib import Path
from typing import Optional, Any, Dict, List, Union

# Configure logging
logger = logging.getLogger(__name__)

def load_env_api_key() -> str:
    """
    Load API key from environment variables with fallback support.
    
    Priority:
    1. RAG_LLM_API_KEY (Project specific - Highest Priority)
    2. GEMINI_TEXT_KEY
    3. RUNWAY_API_KEY
    4. OPENAI_API_KEY
    """
    try:
        # Try loading from project root .env if it exists
        from dotenv import load_dotenv
        
        # Go up 3 levels from utils/api_utils.py to reach project root
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
        else:
            # Fallback to standard load_dotenv search
            load_dotenv()
    except Exception:
        pass

    api_key = (
        os.getenv("RAG_LLM_API_KEY")
        or os.getenv("GEMINI_TEXT_KEY")
        or os.getenv("RUNWAY_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    return api_key.strip()

def get_api_base_url() -> Optional[str]:
    """
    Get API base URL from environment variables.
    
    Priority:
    1. RAG_LLM_BASE_URL (Project specific - Highest Priority)
    2. OPENAI_BASE_URL
    3. RUNWAY_API_BASE
    4. Default fallback (if needed, otherwise None)
    """
    return (
        os.getenv("RAG_LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("RUNWAY_API_BASE")
    )

class CustomHTTPClient:
    """
    A wrapper that mimics OpenAI client structure but uses raw HTTP requests
    to support custom endpoints (like Runway/Nano Banana) compatible with PosterGen2 logic.
    """
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.chat = self.Chat(self)
        self.embeddings = self.Embeddings(self)

    class Chat:
        def __init__(self, client):
            self.client = client
            self.completions = self.Completions(client)

        class Completions:
            def __init__(self, client):
                self.client = client

            def create(self, model: str, messages: List[Dict], **kwargs) -> Any:
                # Handle Gemini native calls separately via HTTP if needed, 
                # but here we focus on Chat Completions format via HTTP
                
                # Construct endpoint
                # If using Nano Banana/Runway, base_url might need specific handling
                # Default expectation: base_url ends with /v1 or similar, so we append /chat/completions
                if "/openai" in self.client.base_url and "runway" in self.client.base_url:
                     # Specialized runway path construction if needed, similar to APIconn_test.py
                     # But usually standard OpenAI compatible endpoints work with /chat/completions
                     url = f"{self.client.base_url}/chat/completions?api-version=2024-12-01-preview"
                else:
                     url = f"{self.client.base_url}/chat/completions"

                headers = {
                    "api-key": self.client.api_key, # Use api-key header for internal gateways
                    "Content-Type": "application/json",
                }
                
                # Add Authorization header as fallback for standard OpenAI
                if "openai.com" in self.client.base_url or "api.openai.com" in url:
                    headers["Authorization"] = f"Bearer {self.client.api_key}"
                    if "api-key" in headers: del headers["api-key"]

                payload = {
                    "model": model,
                    "messages": messages,
                    **kwargs
                }
                
                # Remove extra_body if present (used for image generation modalities in ImageGenerator)
                # Raw HTTP doesn't usually need it unless the backend specifically requires it
                if "extra_body" in payload:
                     # For image generation, we might need specific handling, 
                     # but for text chat, extra_body might be ignored or cause issues
                     pass

                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=120)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Wrap response in a simple object structure to mimic OpenAI SDK object
                    class Message:
                        def __init__(self, content): self.content = content
                    
                    class Choice:
                        def __init__(self, message_content):
                            self.message = Message(message_content)

                    class Response:
                        def __init__(self, choices_data):
                            self.choices = []
                            for c in choices_data:
                                content = c.get("message", {}).get("content", "")
                                self.choices.append(Choice(content))
                    
                    return Response(data.get("choices", []))

                except Exception as e:
                    logger.error(f"Custom HTTP Chat Completion failed: {e}")
                    raise

    class Embeddings:
        def __init__(self, client):
            self.client = client

        def create(self, input: Union[str, List[str]], model: str, **kwargs) -> Any:
            url = f"{self.client.base_url}/embeddings"
             # Runway/Azure specific version param
            if "runway" in self.client.base_url:
                url += "?api-version=2024-12-01-preview"

            headers = {
                "api-key": self.client.api_key,
                "Content-Type": "application/json",
            }
            if "openai.com" in self.client.base_url:
                headers["Authorization"] = f"Bearer {self.client.api_key}"
                if "api-key" in headers: del headers["api-key"]

            payload = {
                "model": model,
                "input": input,
                **kwargs
            }

            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                
                # Wrap response
                class EmbeddingData:
                    def __init__(self, embedding): self.embedding = embedding

                class Response:
                    def __init__(self, data_list):
                        self.data = [EmbeddingData(d["embedding"]) for d in data_list]

                return Response(data.get("data", []))

            except Exception as e:
                logger.error(f"Custom HTTP Embeddings failed: {e}")
                raise

def get_openai_client(
    api_key: Optional[str] = None, 
    base_url: Optional[str] = None
):
    """
    Get configured OpenAI client using standardized env loading.
    
    IMPORTANT: Returns a CustomHTTPClient if detecting internal/runway endpoints
    to ensure compatibility with 'api-key' header requirements.
    Otherwise returns standard OpenAI SDK client.
    """
    final_api_key = api_key or load_env_api_key()
    final_base_url = base_url or get_api_base_url()
    
    if not final_api_key:
        raise ValueError("No API key found in environment variables")

    # Check if we should use Custom HTTP Client (for Nano Banana/Runway compatibility)
    # Logic: if base_url is set and looks like an internal endpoint or specifically requested
    use_custom_http = False
    
    if final_base_url:
        if "runway" in final_base_url or "nano" in final_base_url or "devops" in final_base_url:
            use_custom_http = True
            
    if use_custom_http:
        logger.info(f"Using CustomHTTPClient for compatibility with: {final_base_url}")
        return CustomHTTPClient(api_key=final_api_key, base_url=final_base_url)
    
    # Fallback to standard OpenAI SDK
    from openai import OpenAI
    return OpenAI(
        api_key=final_api_key,
        base_url=final_base_url
    )
