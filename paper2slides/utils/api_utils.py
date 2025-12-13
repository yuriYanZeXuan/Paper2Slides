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

# ========= 默认网关（写死，不从环境变量读取 base_url）=========
# 说明：
# - 线上/内网网关提供 OpenAI 兼容接口：{BASE}/chat/completions、{BASE}/embeddings
# - 这里的 BASE 必须包含 /openai/v1，否则会拼出 /openai/chat/completions 导致 404
DEFAULT_TEXT_BASE_URL = "https://runway.devops.rednote.life/openai/v1"
# image 侧目前主要走 Gemini 原生 endpoint（见 image_generator.py），但 fallback 的 chat/completions 仍可复用同一 base
DEFAULT_IMAGE_BASE_URL = DEFAULT_TEXT_BASE_URL

# ========= 写死的 OpenAI 兼容 endpoint（包含 api-version，禁止运行时拼接）=========
# 按你的要求：直接把 "?api-version=2024-12-01-preview" 写死在 URL 里，不从环境变量读取，也不通过格式化/拼接生成。
DEFAULT_CHAT_COMPLETIONS_URL = "https://runway.devops.rednote.life/openai/v1/chat/completions?api-version=2024-12-01-preview"

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
                # endpoint 写死（包含 api-version），不要运行时拼接
                url = DEFAULT_CHAT_COMPLETIONS_URL
                
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
    # base_url 不允许从环境变量读取（避免拼错 /openai vs /openai/v1）
    if base_url:
        final_base_url = base_url
    else:
        final_base_url = DEFAULT_IMAGE_BASE_URL if key_type == "image" else DEFAULT_TEXT_BASE_URL
    
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


def main() -> None:
    """
    直接在本文件内做连通性测试（急用）：
    - 默认模型：gpt-4o
    - 默认走写死的网关 endpoint（含 api-version）
    - API Key 仍从环境变量读取（见 load_env_api_key）
    """
    import argparse

    parser = argparse.ArgumentParser(description="Quick test for Paper2Slides LLM gateway (chat/completions)")
    parser.add_argument("--model", default="gpt-4o", help="Model name, e.g. gpt-4o")
    parser.add_argument("--prompt", default="请用一句话自我介绍。", help="User prompt")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    api_key = load_env_api_key("text")
    if not api_key:
        raise SystemExit(
            "缺少 API Key：请设置 RAG_LLM_API_KEY / GEMINI_TEXT_KEY / RUNWAY_API_KEY / OPENAI_API_KEY 之一"
        )

    # 明确使用 CustomHTTPClient（走写死 endpoint），避免不同环境下 OpenAI SDK 行为差异
    client = CustomHTTPClient(api_key=api_key, base_url=DEFAULT_TEXT_BASE_URL)

    print("[api_utils] chat_completions_url =", DEFAULT_CHAT_COMPLETIONS_URL)
    resp = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    text = (resp.choices[0].message.content if getattr(resp, "choices", None) else "") or ""
    print("[api_utils] ok\n")
    print(text)


if __name__ == "__main__":
    main()
