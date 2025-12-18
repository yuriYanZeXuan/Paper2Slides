"""
API Utilities for Paper2Slides
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional
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

def get_openai_client(
    api_key: Optional[str] = None, 
    base_url: Optional[str] = None,
    key_type: str = "text"
):
    """
    Get configured OpenAI client.
    Args:
        api_key: Optional API key. If not provided, will load from environment.
        base_url: Optional base URL. Defaults to local gemini_proxy.
        key_type: "text" (default) or "image" to select appropriate env vars if api_key not provided.
    """
    # Use provided api_key, or load from environment based on key_type
    final_api_key = api_key or load_env_api_key(key_type)
    
    # Use provided base_url, or default to local gemini_proxy
    final_base_url = base_url or "http://127.0.0.1:51958/v1"
    
    return OpenAI(api_key=final_api_key, base_url=final_base_url)
