"""
API Utilities for Paper2Slides

Provides unified API key loading and client configuration compatible with
PosterGen2 environment settings (GEMINI_TEXT_KEY, RUNWAY_API_KEY, etc.)
"""

import os
import sys
import json
import base64
import requests
from urllib.parse import urlparse, urlunparse
from pathlib import Path
from typing import Optional, Any, Dict, List, Union


# ========= 写死的 OpenAI 兼容 endpoint（包含 api-version，禁止运行时拼接）=========
# 按你的要求：直接把 "?api-version=2024-12-01-preview" 写死在 URL 里，不从环境变量读取，也不通过格式化/拼接生成。
DEFAULT_CHAT_COMPLETIONS_URL = "https://runway.devops.rednote.life/openai/chat/completions?api-version=2024-12-01-preview"
DEFAULT_CHAT_COMPLETIONS_URL_GEMINI = "https://runway.devops.rednote.life/openai/google/v1:generateContent"

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
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        load_dotenv()

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
                # Remove unsupported parameters for runway/nano gateway
                # unsupported_keys = ["extra_body", "response_format"]
                # for key in unsupported_keys:
                #     if key in payload:
                #         del payload[key]

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


class CustomHTTPClientForOpenAI:
    """
    A wrapper that mimics OpenAI client structure but uses raw HTTP requests.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.chat = self.Chat(self)
        

    class Chat:
        def __init__(self, client):
            self.client = client
            self.completions = self.Completions(client)

        class Completions:
            def __init__(self, client):
                self.client = client

            def create(self, model: str, messages: List[Dict], **kwargs) -> Any:
                # endpoint 写死（包含 api-version），不要运行时拼接
                url = "https://runway.devops.rednote.life/openai/chat/completions?api-version=2024-12-01-preview"
                
                headers = {
                    "api-key": self.client.api_key,
                    "Content-Type": "application/json",
                }
                
                payload = {
                    "model": model,
                    "messages": messages,
                    **kwargs
                }
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

class CustomHTTPClientForGemini:
    """
    A wrapper that mimics OpenAI client structure but uses Gemini Native API.
    Ref: PosterGen2/dev/APIconn_test.py
    """
    def __init__(self, api_key: str, base_url: str = None):
        self.api_key = api_key
        # Default endpoint for Gemini Native (google/v1)
        self.endpoint = "https://runway.devops.rednote.life/openai/google/v1:generateContent"
        # If user provides a specific full endpoint (containing generateContent), use it
        if base_url and "generateContent" in base_url:
            self.endpoint = base_url
        
        self.chat = self.Chat(self)

    class Chat:
        def __init__(self, client):
            self.client = client
            self.completions = self.Completions(client)

        class Completions:
            def __init__(self, client):
                self.client = client

            def create(self, model="gemini-3-pro", messages=[], **kwargs) -> Any:
                # 1. Convert OpenAI messages to Gemini contents structure
                contents = []
                system_instruction = None

                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content")
                    
                    if role == "system":
                        # Gemini supports systemInstruction field separately
                        system_instruction = {"parts": [{"text": content}]}
                        continue
                    
                    # Map roles: user -> user, assistant -> model
                    gemini_role = "user" if role == "user" else "model"
                    parts = []
                    
                    if isinstance(content, str):
                        parts.append({"text": content})
                    elif isinstance(content, list):
                        # Handle multimodal content (text + image_url)
                        for item in content:
                            if item.get("type") == "text":
                                parts.append({"text": item.get("text")})
                            elif item.get("type") == "image_url":
                                url = item.get("image_url", {}).get("url", "")
                                if url.startswith("data:"):
                                    # Parse data URI: data:image/png;base64,....
                                    try:
                                        header, b64_data = url.split(",", 1)
                                        # header e.g. "data:image/png;base64"
                                        mime_type = header.split(":")[1].split(";")[0]
                                        parts.append({
                                            "inlineData": {
                                                "mimeType": mime_type,
                                                "data": b64_data
                                            }
                                        })
                                    except Exception as e:
                                        print(f"[GeminiClient] Error parsing image data URL: {e}")
                    
                    if parts:
                        contents.append({"role": gemini_role, "parts": parts})

                # 2. Construct Payload
                payload = {
                    "contents": contents,
                    "generationConfig": {
                        "temperature": kwargs.get("temperature", 0.6),
                        "maxOutputTokens": kwargs.get("max_tokens", 7000),
                        "topP": kwargs.get("top_p", 1),
                    }
                }
                
                if system_instruction:
                    payload["systemInstruction"] = system_instruction

                # 3. Headers
                headers = {
                    "api-key": self.client.api_key,
                    "Content-Type": "application/json",
                }

                # 4. Execute Request
                response = requests.post(self.client.endpoint, headers=headers, json=payload, timeout=120)
                
                if response.status_code < 200 or response.status_code >= 300:
                     raise RuntimeError(f"HTTP {response.status_code}: {response.text[:800]}")
                
                res_json = response.json()
                
                if "error" in res_json:
                     raise RuntimeError(f"API Error: {res_json.get('error')}")

                if "candidates" not in res_json or not res_json["candidates"]:
                     raise RuntimeError(f"No candidates returned. Full response: {res_json}")

                candidate = res_json["candidates"][0]
                # Optional: check finishReason
                # finish_reason = candidate.get("finishReason")
                
                content_parts = candidate.get("content", {}).get("parts", [])
                text_content = ""
                if content_parts:
                    text_content = content_parts[0].get("text", "")

                # 5. Wrap response to mimic OpenAI return object
                class Message:
                    def __init__(self, content): self.content = content
                class Choice:
                    def __init__(self, message_content): self.message = Message(message_content)
                class Response:
                    def __init__(self, content_str):
                        self.choices = [Choice(content_str)]
                
                return Response(text_content)


class UnifiedCustomHTTPClient:
    """
    Unified client that routes to Gemini or OpenAI implementation based on model name.
    """
    def __init__(self, api_key):
        self.openai_client = CustomHTTPClientForOpenAI(api_key)
        self.gemini_client = CustomHTTPClientForGemini(api_key)
        self.chat = self.Chat(self)

    class Chat:
        def __init__(self, client):
            self.client = client
            self.completions = self.Completions(client)
        
        class Completions:
            def __init__(self, client):
                self.client = client
            
            def create(self, model: str, **kwargs) -> Any:
                if "gemini" in model.lower():
                    return self.client.gemini_client.chat.completions.create(model=model, **kwargs)
                else:
                    return self.client.openai_client.chat.completions.create(model=model, **kwargs)


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
    final_api_key = os.getenv("RUNWAY_API_KEY")
    final_base_url = DEFAULT_CHAT_COMPLETIONS_URL

    use_custom_http = False
    if final_base_url and ("runway" in final_base_url or "nano" in final_base_url or "devops" in final_base_url):
        use_custom_http = True
            
    if use_custom_http:
        # Return Unified client to support both OpenAI and Gemini models
        return UnifiedCustomHTTPClient(api_key=final_api_key)
    
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
    parser.add_argument("--model", default="gpt-4o", help="Model name, e.g. gpt-4o, gemini-3")
    parser.add_argument("--prompt", default="请用一句话自我介绍。", help="User prompt")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    api_key = load_env_api_key("text")
    if not api_key:
        raise SystemExit(
            "缺少 API Key：请设置 RAG_LLM_API_KEY / GEMINI_TEXT_KEY / RUNWAY_API_KEY / OPENAI_API_KEY 之一"
        )

    # 明确使用 UnifiedCustomHTTPClient（支持 Gemini/OpenAI 自动切换），避免不同环境下 OpenAI SDK 行为差异
    client = UnifiedCustomHTTPClient(api_key=api_key)

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
