"""
Gemini to OpenAI API Proxy
Starts a FastAPI server on port 51958 that proxies OpenAI chat completions
format to Gemini Native API.

Supports:
- Standard chat completions
- Multimodal (image) inputs
- Function calling / Tool use
"""

import os
import time
import json
import uuid
import base64
import requests
import uvicorn
import re
from typing import List, Dict, Any, Optional, Tuple
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


def _convert_openai_tools_to_gemini(tools: List[Dict]) -> List[Dict]:
    """
    Convert OpenAI tools format to Gemini functionDeclarations format.
    
    OpenAI format:
    {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    
    Gemini format:
    {"functionDeclarations": [{"name": "...", "description": "...", "parameters": {...}}]}
    """
    if not tools:
        return []
    
    function_declarations = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            declaration = {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
            }
            if "parameters" in func:
                declaration["parameters"] = func["parameters"]
            function_declarations.append(declaration)
    
    if function_declarations:
        return [{"functionDeclarations": function_declarations}]
    return []


def _parse_tool_calls_from_text(text: str) -> Tuple[str, List[Dict]]:
    """
    Parse <tool_call>...</tool_call> XML format from model output.
    Returns (remaining_text, list_of_tool_calls).
    
    This handles the case where Gemini returns tool calls as text instead of
    using native function calling.
    """
    tool_calls = []
    
    # Pattern to match <tool_call>...</tool_call>
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            call_data = json.loads(match)
            tool_call = {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": call_data.get("name", ""),
                    "arguments": json.dumps(call_data.get("arguments", {}), ensure_ascii=False)
                }
            }
            tool_calls.append(tool_call)
        except json.JSONDecodeError:
            continue
    
    # Remove tool_call tags from text
    remaining_text = re.sub(pattern, '', text, flags=re.DOTALL).strip()
    
    return remaining_text, tool_calls


class GeminiClient:
    """
    Minimal Gemini Client that sends requests to the native endpoint.
    Supports function calling.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = GEMINI_ENDPOINT

    def generate_content(
        self, 
        model: str, 
        messages: List[Dict], 
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content with optional function calling support.
        
        Returns a dict with:
        - text_content: str (the text response, if any)
        - tool_calls: List[Dict] (OpenAI format tool_calls, if any)
        """
        # 1. Convert OpenAI messages to Gemini contents structure
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")
            name = msg.get("name")
            
            if role == "system":
                # Gemini supports systemInstruction field separately
                if isinstance(content, str):
                    system_instruction = {"parts": [{"text": content}]}
                continue
            
            # Handle tool/function response messages
            if role == "tool" or role == "function":
                # Convert tool response to Gemini format
                parts = [{
                    "functionResponse": {
                        "name": name or "unknown_function",
                        "response": {
                            "result": content if isinstance(content, str) else json.dumps(content)
                        }
                    }
                }]
                contents.append({"role": "user", "parts": parts})
                continue
            
            # Handle assistant messages with tool_calls
            if role == "assistant" and tool_calls:
                parts = []
                if content:
                    parts.append({"text": content})
                for tc in tool_calls:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            args = {}
                    parts.append({
                        "functionCall": {
                            "name": func.get("name", ""),
                            "args": args
                        }
                    })
                contents.append({"role": "model", "parts": parts})
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
            elif content is None and role == "assistant":
                # Assistant message with no content (pure tool call response case)
                continue
            
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
        
        # 3. Add tools if provided
        gemini_tools = _convert_openai_tools_to_gemini(tools)
        if gemini_tools:
            payload["tools"] = gemini_tools
            payload["toolConfig"] = {
                "functionCallingConfig": {
                    "mode": "AUTO"
                }
            }

        # 4. Headers
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

        # 5. Execute Request
        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=120)
        
        if response.status_code < 200 or response.status_code >= 300:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:800]}")
        
        res_json = response.json()
        
        if "error" in res_json:
            raise RuntimeError(f"API Error: {res_json.get('error')}")

        if "candidates" not in res_json or not res_json["candidates"]:
            raise RuntimeError(f"No candidates returned. Full response: {res_json}")

        candidate = res_json["candidates"][0]
        
        # 6. Parse response - handle both text and function calls
        content_parts = candidate.get("content", {}).get("parts", [])
        text_content = ""
        tool_calls = []
        
        for part in content_parts:
            if "text" in part:
                text_content += part.get("text", "")
            elif "functionCall" in part:
                # Native Gemini function call
                fc = part["functionCall"]
                tool_call = {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {}), ensure_ascii=False)
                    }
                }
                tool_calls.append(tool_call)
        
        # 7. Also parse <tool_call> XML format if present in text
        # (fallback for when model doesn't use native function calling)
        if text_content and not tool_calls:
            remaining_text, parsed_tool_calls = _parse_tool_calls_from_text(text_content)
            if parsed_tool_calls:
                tool_calls = parsed_tool_calls
                text_content = remaining_text
        
        return {
            "text_content": text_content,
            "tool_calls": tool_calls
        }


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
        tools = data.get("tools")  # OpenAI format tools
        
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
        result = await asyncio.to_thread(
            client.generate_content,
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        text_content = result.get("text_content", "")
        tool_calls = result.get("tool_calls", [])
        
        # Log for debugging
        try:
            with open("response.log", "a", encoding="utf-8") as f:
                f.write(f"--- Request at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                f.write(f"Model: {model}\n")
                f.write(f"Messages: {json.dumps(messages, ensure_ascii=False)}\n")
                f.write(f"Tools provided: {bool(tools)}\n")
                f.write(f"Response Content: {text_content}\n")
                f.write(f"Tool Calls: {json.dumps(tool_calls, ensure_ascii=False)}\n")
                f.write("-" * 50 + "\n\n")
        except Exception as log_err:
            print(f"[GeminiProxy] Failed to log response: {log_err}")
        
        # Construct standard OpenAI JSON response
        message = {
            "role": "assistant",
        }
        
        if tool_calls:
            # Function calling response
            message["content"] = text_content if text_content else None
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        else:
            # Normal text response
            message["content"] = text_content
            finish_reason = "stop"
        
        resp_data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
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
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    print("Starting Gemini Proxy on port 51958...")
    print("Supports: chat completions, multimodal inputs, function calling")
    uvicorn.run(app, host="0.0.0.0", port=51958)
