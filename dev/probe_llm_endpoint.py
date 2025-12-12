"""Probe LLM gateway compatibility.

This script tries several common OpenAI/Azure-compatible endpoint shapes against
YOUR configured base URL and API key env vars, and prints status codes.

It does NOT print the API key.

Env vars used:
- Base URL: RAG_LLM_BASE_URL or OPENAI_BASE_URL or RUNWAY_API_BASE
- API key: RAG_LLM_API_KEY or OPENAI_API_KEY or GEMINI_TEXT_KEY or RUNWAY_API_KEY

Usage (on server):
  python dev/probe_llm_endpoint.py

Optional:
  PROBE_MODEL=gpt-4o
  PROBE_API_VERSION=2024-12-01-preview
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple

import requests


def _get_base_and_key() -> Tuple[str, str]:
    base = (
        os.getenv("RAG_LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("RUNWAY_API_BASE")
        or ""
    ).strip()
    key = (
        os.getenv("RAG_LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_TEXT_KEY")
        or os.getenv("RUNWAY_API_KEY")
        or ""
    ).strip()
    if not base:
        raise SystemExit("Missing base url env: set RAG_LLM_BASE_URL / OPENAI_BASE_URL / RUNWAY_API_BASE")
    if not key:
        raise SystemExit("Missing api key env: set RAG_LLM_API_KEY / OPENAI_API_KEY / GEMINI_TEXT_KEY / RUNWAY_API_KEY")
    return base.rstrip("/"), key


def _req(url: str, headers: Dict[str, str], params: Optional[Dict[str, str]], body: Dict[str, Any]) -> Tuple[Any, str]:
    try:
        r = requests.post(url, headers=headers, params=params, json=body, timeout=20)
        txt = (r.text or "")
        txt = txt.replace("\n", " ")
        return r.status_code, txt[:200]
    except Exception as e:
        return "EXC", str(e)[:200]


def main() -> None:
    base, key = _get_base_and_key()
    model = (os.getenv("PROBE_MODEL") or "gpt-4o").strip()
    api_version = (os.getenv("PROBE_API_VERSION") or "2024-12-01-preview").strip()

    print("Base URL:", base)
    print("Model:", model)
    print("API version:", api_version)
    print("---")

    body = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}

    candidates = []
    # OpenAI style (Bearer)
    candidates.append((base + "/v1/chat/completions", {"Authorization": f"Bearer {key}"}, None))
    candidates.append((base + "/chat/completions", {"Authorization": f"Bearer {key}"}, None))
    # OpenAI style (api-key)
    candidates.append((base + "/v1/chat/completions", {"api-key": key}, None))
    candidates.append((base + "/chat/completions", {"api-key": key}, None))
    # Add api-version param (common in gateways)
    candidates.append((base + "/v1/chat/completions", {"api-key": key}, {"api-version": api_version}))
    candidates.append((base + "/chat/completions", {"api-key": key}, {"api-version": api_version}))
    # Azure deployments style
    candidates.append((base + f"/openai/deployments/{model}/chat/completions", {"api-key": key}, {"api-version": api_version}))
    candidates.append((base + "/openai/deployments/gpt-4o/chat/completions", {"api-key": key}, {"api-version": api_version}))
    candidates.append((base + "/openai/deployments/gpt-4o-mini/chat/completions", {"api-key": key}, {"api-version": api_version}))

    for i, (url, headers, params) in enumerate(candidates, start=1):
        code, txt = _req(url, headers=headers, params=params, body=body)
        print(f"[{i}] {code} URL={url} params={params} headers={list(headers.keys())} :: {txt}")

    print("---")
    print("Env summary (no secrets):")
    print("RAG_LLM_BASE_URL=", os.getenv("RAG_LLM_BASE_URL"))
    print("OPENAI_BASE_URL=", os.getenv("OPENAI_BASE_URL"))
    print("RUNWAY_API_BASE=", os.getenv("RUNWAY_API_BASE"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise
