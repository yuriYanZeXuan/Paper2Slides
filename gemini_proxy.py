"""
Gemini to OpenAI API Proxy
Starts a FastAPI server on port 51958 that proxies OpenAI chat completions
format to Gemini Native API.
"""

import os
import time
import json
import base64
import requests
import uvicorn
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load env from parent directory if exists
load_dotenv()

app = FastAPI()

# Configuration
GEMINI_ENDPOINT = "https://runway.devops.rednote.life/openai/google/v1:generateContent"

def load_env_api_key() -> str:
    """
    Load API key from environment variables.
    """
    return (
        os.getenv("GEMINI_TEXT_KEY")
        or os.getenv("RUNWAY_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()

class GeminiClient:
    """
    Minimal Gemini Client that sends requests to the native endpoint.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = GEMINI_ENDPOINT

    def generate_content(self, model: str, messages: List[Dict], **kwargs) -> str:
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
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

        # 4. Execute Request
        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=120)
        
        if response.status_code < 200 or response.status_code >= 300:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:800]}")
        
        res_json = response.json()
        
        if "error" in res_json:
                raise RuntimeError(f"API Error: {res_json.get('error')}")

        if "candidates" not in res_json or not res_json["candidates"]:
                raise RuntimeError(f"No candidates returned. Full response: {res_json}")

        candidate = res_json["candidates"][0]
        
        content_parts = candidate.get("content", {}).get("parts", [])
        text_content = ""
        if content_parts:
            text_content = content_parts[0].get("text", "")
            
        return text_content


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    try:
        data = await request.json()
        
        # Extract fields
        model = data.get("model", "gemini-3-pro")
        messages = data.get("messages", [])
        temperature = data.get("temperature", 0.6)
        max_tokens = data.get("max_tokens", 7000)
        
        # Authorization
        auth_header = request.headers.get("Authorization")
        api_key = None
        if auth_header and "Bearer " in auth_header:
            api_key = auth_header.split("Bearer ")[1].strip()
        
        if not api_key:
            # Fallback to env
            api_key = load_env_api_key()

        # Call Gemini
        client = GeminiClient(api_key=api_key)
        
        # Run in threadpool
        import asyncio
        text_content = await asyncio.to_thread(
            client.generate_content,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Construct standard OpenAI JSON response
        resp_data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text_content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        return JSONResponse(content=resp_data)

    except Exception as e:
        print(f"[GeminiProxy] Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    print("Starting Gemini Proxy on port 51958...")
    uvicorn.run(app, host="0.0.0.0", port=51958)

