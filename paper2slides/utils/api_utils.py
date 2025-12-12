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
from urllib.parse import urlparse, urlunparse
from pathlib import Path
from typing import Optional, Any, Dict, List, Union

# Configure logging
logger = logging.getLogger(__name__)

def _normalize_openai_compatible_base_url(base_url: Optional[str]) -> Optional[str]:
    """Normalize OpenAI-compatible base_url.

    Most OpenAI-compatible servers expect base_url ending with `/v1`.
    If the given url doesn't contain `/v1`, append it.
    """
    if not base_url:
        return base_url
    base_url = base_url.strip()
    if not base_url:
        return base_url

    # Runway/devops gateways commonly expose OpenAI-compatible endpoints under `/openai/*`
    # and may NOT support `/openai/v1/*`. If user provides `/openai/v1`, strip the `/v1`.
    parsed = urlparse(base_url)
    path = (parsed.path or "").rstrip("/")
    parts = [p for p in path.split("/") if p]
    for i in range(len(parts) - 1):
        if parts[i] == "openai" and parts[i + 1] == "v1":
            new_parts = parts[: i + 1] + parts[i + 2 :]
            new_path = "/" + "/".join(new_parts) if new_parts else ""
            normalized = urlunparse(parsed._replace(path=new_path))
            return normalized.rstrip("/")

    # Already contains /v1 somewhere in path -> keep
    if "/v1" in path.split("/"):
        return base_url.rstrip("/")

    # Special-case: some gateways expose OpenAI-compatible endpoints under `/openai/*`
    # (e.g. `/openai/chat/completions`), and appending `/v1` would break routing.
    if "/openai" in path.split("/"):
        return base_url.rstrip("/")

    # Append /v1 to path
    new_path = (path + "/v1") if path else "/v1"
    normalized = urlunparse(parsed._replace(path=new_path))
    return normalized.rstrip("/")

def load_env_api_key(key_type: str = "text") -> str:
    """
    Load API key from environment variables with fallback support.
    
    Args:
        key_type: "text" for RAG/LLM tasks, "image" for Image Generation.

    Priority (Text/RAG):
    1. RAG_LLM_API_KEY
    2. GEMINI_TEXT_KEY
    3. RUNWAY_API_KEY
    4. OPENAI_API_KEY
    
    Priority (Image):
    1. IMAGE_GEN_API_KEY
    2. GEMINI_IMAGE_API_KEY
    3. RUNWAY_API_KEY
    4. OPENAI_API_KEY
    """
    try:
        from dotenv import load_dotenv
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
        else:
            load_dotenv()
    except Exception:
        pass

    if key_type == "image":
        return (
            os.getenv("IMAGE_GEN_API_KEY")
            or os.getenv("GEMINI_IMAGE_API_KEY")
            or os.getenv("RUNWAY_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()
    else:
        return (
            os.getenv("RAG_LLM_API_KEY")
            or os.getenv("GEMINI_TEXT_KEY")
            or os.getenv("RUNWAY_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()

def get_api_base_url(key_type: str = "text") -> Optional[str]:
    """
    Get API base URL from environment variables.
    """
    if key_type == "image":
        return (
            os.getenv("IMAGE_GEN_BASE_URL")
            or os.getenv("RAG_LLM_BASE_URL") # Fallback to text URL if not set
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("RUNWAY_API_BASE")
        )
    else:
        return (
            os.getenv("RAG_LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("RUNWAY_API_BASE")
        )

class CustomHTTPClient:
    """
    A wrapper that mimics OpenAI client structure but uses raw HTTP requests.
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
                # Determine endpoint based on model type or URL pattern
                url = f"{self.client.base_url}/chat/completions"
                
                # Special handling for internal gateways if needed
                if "runway" in self.client.base_url or "devops" in self.client.base_url:
                     if "?" not in url:
                         url += "?api-version=2024-12-01-preview"

                headers = {
                    "api-key": self.client.api_key,
                    "Content-Type": "application/json",
                }
                
                payload = {
                    "model": model,
                    "messages": messages,
                    **kwargs
                }
                # Remove extra_body if present
                if "extra_body" in payload:
                     del payload["extra_body"]

                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=120)
                    response.raise_for_status()
                    data = response.json()
                    
                    class Message:
                        def __init__(self, content): self.content = content
                    class Choice:
                        def __init__(self, message_content): self.message = Message(message_content)
                    class Response:
                        def __init__(self, choices_data):
                            self.choices = [Choice(c.get("message", {}).get("content", "")) for c in choices_data]
                    
                    return Response(data.get("choices", []))

                except Exception as e:
                    logger.error(f"Custom HTTP Chat Completion failed: {e}")
                    raise

    class Embeddings:
        def __init__(self, client):
            self.client = client

        def create(self, input: Union[str, List[str]], model: str, **kwargs) -> Any:
            url = f"{self.client.base_url}/embeddings"
            if "runway" in self.client.base_url or "devops" in self.client.base_url:
                url += "?api-version=2024-12-01-preview"

            headers = {
                "api-key": self.client.api_key,
                "Content-Type": "application/json",
            }
            
            payload = {"model": model, "input": input, **kwargs}

            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                
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
    base_url: Optional[str] = None,
    key_type: str = "text"
):
    """
    Get configured OpenAI client.
    Args:
        key_type: "text" (default) or "image" to select appropriate env vars if api_key not provided.
    """
    final_api_key = api_key or load_env_api_key(key_type)
    final_base_url = base_url or get_api_base_url(key_type)
    final_base_url = _normalize_openai_compatible_base_url(final_base_url)
    
    if not final_api_key:
        raise ValueError(f"No API key found for {key_type}")

    use_custom_http = False
    if final_base_url and ("runway" in final_base_url or "nano" in final_base_url or "devops" in final_base_url):
        use_custom_http = True
            
    if use_custom_http:
        logger.info(f"Using CustomHTTPClient for {key_type} (URL: {final_base_url})")
        return CustomHTTPClient(api_key=final_api_key, base_url=final_base_url)
    
    from openai import OpenAI
    return OpenAI(api_key=final_api_key, base_url=final_base_url)
