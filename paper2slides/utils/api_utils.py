"""
API Utilities for Paper2Slides

LLM (text): 使用本地 gemini_proxy 提供的 OpenAI 兼容接口 (PosterGen3/gemini_proxy.py)，
不再从环境变量加载 API key。

Image: 仍支持通过 load_env_api_key(key_type="image") / get_api_base_url(key_type="image") 配置。
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

# 本地 gemini_proxy 默认地址 (PosterGen3/gemini_proxy.py 默认端口 51958)
# 可通过环境变量 P2S_LLM_PROXY_URL 覆盖，例如不同主机/端口
DEFAULT_LLM_PROXY_URL = "http://localhost:51958/v1"


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
    key_type: str = "text",
):
    """
    获取 OpenAI 兼容客户端。

    - key_type="text" (默认, LLM): 使用本地 gemini_proxy 接口 (PosterGen3/gemini_proxy.py)，
      base_url 默认为 P2S_LLM_PROXY_URL 或 http://localhost:51958/v1，不从环境变量加载 API key；
      proxy 端用请求头或自带的 default_api_key。显式传入 api_key/base_url 时会被采用。
    - key_type="image": 仍从 load_env_api_key/get_api_base_url 读取配置（未传入时）。
    """
    from openai import OpenAI

    if key_type == "image":
        final_api_key = api_key or load_env_api_key(key_type)
        final_base_url = base_url or get_api_base_url(key_type)
        if not final_api_key:
            raise ValueError("No API key found for image (set IMAGE_GEN_API_KEY, GEMINI_IMAGE_API_KEY, etc.)")
        use_custom_http = bool(
            final_base_url
            and any(x in final_base_url for x in ("runway", "nano", "devops"))
        )
        if use_custom_http:
            logger.info(f"Using CustomHTTPClient for image (URL: {final_base_url})")
            return CustomHTTPClient(api_key=final_api_key, base_url=final_base_url)
        return OpenAI(api_key=final_api_key, base_url=final_base_url)

    # LLM (text): 使用本地 gemini_proxy，不从 env 加载 key
    final_base_url = base_url or os.getenv("P2S_LLM_PROXY_URL", DEFAULT_LLM_PROXY_URL)
    final_api_key = api_key if api_key is not None else ""
    logger.info(f"Using LLM proxy for text (URL: {final_base_url})")
    return OpenAI(api_key=final_api_key, base_url=final_base_url)
