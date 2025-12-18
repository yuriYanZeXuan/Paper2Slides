"""
Gemini to OpenAI API Proxy
Starts a FastAPI server on port 51958 that proxies OpenAI chat completions
format to Gemini Native API.

Supports:
- Basic chat completions
- Function calling / Tool calling (OpenAI format <-> Gemini format)
"""

import os
import time
import json
import base64
import requests
import uvicorn
from typing import List, Dict, Any, Optional, Tuple, Union
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


def convert_openai_tools_to_gemini(tools: List[Dict]) -> List[Dict]:
    """
    Convert OpenAI tools format to Gemini function declarations.
    
    OpenAI format:
    [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
    
    Gemini format:
    [{"functionDeclarations": [{"name": "...", "description": "...", "parameters": {...}}]}]
    """
    if not tools:
        return []
    
    function_declarations = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            func_decl = {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
            }
            if "parameters" in func:
                func_decl["parameters"] = func["parameters"]
            function_declarations.append(func_decl)
    
    if function_declarations:
        return [{"functionDeclarations": function_declarations}]
    return []


def convert_openai_functions_to_gemini(functions: List[Dict]) -> List[Dict]:
    """
    Convert legacy OpenAI functions format to Gemini function declarations.
    
    OpenAI legacy format:
    [{"name": "...", "description": "...", "parameters": {...}}]
    """
    if not functions:
        return []
    
    function_declarations = []
    for func in functions:
        func_decl = {
            "name": func.get("name", ""),
            "description": func.get("description", ""),
        }
        if "parameters" in func:
            func_decl["parameters"] = func["parameters"]
        function_declarations.append(func_decl)
    
    if function_declarations:
        return [{"functionDeclarations": function_declarations}]
    return []


class GeminiClient:
    """
    Gemini Client that sends requests to the native endpoint.
    Supports function calling.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = GEMINI_ENDPOINT

    def generate_content(
        self, 
        model: str, 
        messages: List[Dict], 
        tools: List[Dict] = None,
        functions: List[Dict] = None,
        **kwargs
    ) -> Tuple[str, Optional[List[Dict]]]:
        """
        Generate content with optional function calling support.
        
        Returns:
            Tuple of (text_content, tool_calls)
            - text_content: The text response from the model
            - tool_calls: List of tool calls if the model wants to call functions, None otherwise
        """
        # 1. Convert OpenAI messages to Gemini contents structure
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if role == "system":
                # Gemini supports systemInstruction field separately
                if isinstance(content, str):
                    system_instruction = {"parts": [{"text": content}]}
                continue
            
            # Handle tool/function messages (results from tool calls)
            if role == "tool" or role == "function":
                # Convert tool result to Gemini functionResponse format
                tool_call_id = msg.get("tool_call_id") or msg.get("name", "")
                name = msg.get("name", tool_call_id)
                result_content = content or ""
                
                # Try to parse as JSON, otherwise use as string
                try:
                    response_data = json.loads(result_content) if isinstance(result_content, str) else result_content
                except json.JSONDecodeError:
                    response_data = {"result": result_content}
                
                contents.append({
                    "role": "user",  # Gemini expects function responses as user role
                    "parts": [{
                        "functionResponse": {
                            "name": name,
                            "response": response_data
                        }
                    }]
                })
                continue
            
            # Handle assistant messages with tool_calls
            if role == "assistant":
                parts = []
                
                # Add text content if present
                if content:
                    if isinstance(content, str):
                        parts.append({"text": content})
                    elif isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                parts.append({"text": item.get("text")})
                
                # Add function calls if present
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    if tc.get("type") == "function":
                        func = tc.get("function", {})
                        func_name = func.get("name", "")
                        func_args = func.get("arguments", "{}")
                        try:
                            args_dict = json.loads(func_args) if isinstance(func_args, str) else func_args
                        except json.JSONDecodeError:
                            args_dict = {}
                        parts.append({
                            "functionCall": {
                                "name": func_name,
                                "args": args_dict
                            }
                        })
                
                # Legacy function_call format
                function_call = msg.get("function_call")
                if function_call:
                    func_name = function_call.get("name", "")
                    func_args = function_call.get("arguments", "{}")
                    try:
                        args_dict = json.loads(func_args) if isinstance(func_args, str) else func_args
                    except json.JSONDecodeError:
                        args_dict = {}
                    parts.append({
                        "functionCall": {
                            "name": func_name,
                            "args": args_dict
                        }
                    })
                
                if parts:
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
        
        # 3. Add tools/functions if provided
        gemini_tools = []
        if tools:
            gemini_tools = convert_openai_tools_to_gemini(tools)
        elif functions:
            gemini_tools = convert_openai_functions_to_gemini(functions)
        
        if gemini_tools:
            payload["tools"] = gemini_tools

        # 4. Headers
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

        # 5. Execute Request
        print(f"[GeminiClient] Sending request with {len(contents)} messages, tools: {bool(gemini_tools)}")
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
        
        # 6. Parse response - extract text and function calls
        text_content = ""
        tool_calls = []
        
        for i, part in enumerate(content_parts):
            if "text" in part:
                text_content += part["text"]
            elif "functionCall" in part:
                func_call = part["functionCall"]
                func_name = func_call.get("name", "")
                func_args = func_call.get("args", {})
                
                # Convert to OpenAI tool_calls format
                tool_calls.append({
                    "id": f"call_{func_name}_{i}_{int(time.time())}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(func_args, ensure_ascii=False)
                    }
                })
        
        return text_content, tool_calls if tool_calls else None


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
        
        # Extract tools/functions for function calling
        tools = data.get("tools", [])
        functions = data.get("functions", [])  # Legacy format
        
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
        text_content, tool_calls = await asyncio.to_thread(
            client.generate_content,
            model=model,
            messages=messages,
            tools=tools,
            functions=functions,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Log for debugging
        print(f"[GeminiProxy] Response - text: {text_content[:200] if text_content else '(empty)'}, tool_calls: {len(tool_calls) if tool_calls else 0}")
        try:
            with open("response.log", "a", encoding="utf-8") as f:
                f.write(f"--- Request at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                f.write(f"Model: {model}\n")
                f.write(f"Tools: {json.dumps([t.get('function', {}).get('name') for t in tools] if tools else [], ensure_ascii=False)}\n")
                f.write(f"Messages count: {len(messages)}\n")
                f.write(f"Response Content: {text_content}\n")
                f.write(f"Tool Calls: {json.dumps(tool_calls, ensure_ascii=False) if tool_calls else 'None'}\n")
                f.write("-" * 50 + "\n\n")
        except Exception as log_err:
            print(f"[GeminiProxy] Failed to log response: {log_err}")
        
        # Construct standard OpenAI JSON response
        message = {
            "role": "assistant",
            "content": text_content if text_content else None
        }
        
        # Add tool_calls if present (function calling response)
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        else:
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
        import traceback
        print(f"[GeminiProxy] Error: {e}")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    print("Starting Gemini Proxy on port 51958...")
    uvicorn.run(app, host="0.0.0.0", port=51958)

