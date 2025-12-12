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

            def _looks_like_azure_openai(self, base_url: str) -> bool:
                """
                Heuristically detect Azure OpenAI style endpoints that require api-version.
                Examples:
                - https://{resource}.openai.azure.com/openai/deployments/{deployment}
                - https://{resource}.openai.azure.com/openai
                """
                try:
                    parsed = urlparse(base_url)
                except Exception:
                    return False
                host = (parsed.netloc or "").lower()
                path = (parsed.path or "").lower()
                if host.endswith(".openai.azure.com"):
                    return True
                if "/openai/deployments" in path:
                    return True
                return False

            def _with_api_version(self, url: str, api_version: str) -> str:
                if "api-version=" in url:
                    return url
                return f"{url}{'&' if '?' in url else '?'}api-version={api_version}"

            def create(self, model: str, messages: List[Dict], **kwargs) -> Any:
                # Determine endpoint based on model type or URL pattern
                url = f"{self.client.base_url}/chat/completions"
                
                headers = {
                    "api-key": self.client.api_key,
                    "Authorization": f"Bearer {self.client.api_key}",
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

                api_version = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview"
                request_urls: List[str] = [url]
                # Only attach api-version for Azure OpenAI-like endpoints.
                if self._looks_like_azure_openai(self.client.base_url):
                    request_urls = [self._with_api_version(url, api_version)]

                last_exc: Exception | None = None
                for req_url in request_urls:
                    try:
                        response = requests.post(req_url, headers=headers, json=payload, timeout=120)
                        # If api-version was accidentally added to a non-Azure gateway, retry without it on 404.
                        if response.status_code == 404 and "api-version=" in req_url and req_url != url:
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
                        last_exc = e
                        continue

                logger.error(f"Custom HTTP Chat Completion failed: {last_exc}")
                raise last_exc

    class Embeddings:
        def __init__(self, client):
            self.client = client

        def create(self, input: Union[str, List[str]], model: str, **kwargs) -> Any:
            url = f"{self.client.base_url}/embeddings"

            headers = {
                "api-key": self.client.api_key,
                "Authorization": f"Bearer {self.client.api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {"model": model, "input": input, **kwargs}

            try:
                # Same api-version handling policy as chat completions: only for Azure OpenAI-like endpoints,
                # and retry without api-version on 404.
                api_version = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview"
                parsed = urlparse(self.client.base_url)
                is_azure_like = (parsed.netloc or "").lower().endswith(".openai.azure.com") or ("/openai/deployments" in (parsed.path or "").lower())
                req_url = f"{url}{'&' if '?' in url else '?'}api-version={api_version}" if is_azure_like else url

                response = requests.post(req_url, headers=headers, json=payload, timeout=60)
                if response.status_code == 404 and "api-version=" in req_url and req_url != url:
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
