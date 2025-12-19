import json
import re
from typing import Any, Dict


def strip_markdown_fences(text: str) -> str:
    """Remove ``` / ```json fences if present."""
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if not lines:
        return ""
    # drop first line (``` or ```json)
    lines = lines[1:]
    # drop trailing ``` lines
    while lines and lines[-1].strip() == "```":
        lines.pop()
    return "\n".join(lines).strip()


def extract_json_fragment(text: str) -> str:
    """Best-effort: extract a JSON object/array substring from model output."""
    s = strip_markdown_fences(text)
    s = (s or "").strip()
    if not s:
        return ""
    if s[0] in "{[":
        return s
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        return m.group(0).strip()
    m = re.search(r"\[[\s\S]*\]", s)
    if m:
        return m.group(0).strip()
    return s


def parse_agent_final_json(raw: str) -> Dict[str, Any]:
    """Parse agent final output robustly; raises ValueError with context."""
    candidate = extract_json_fragment(raw)
    if not candidate or not candidate.strip():
        raise ValueError("empty/blank agent output (after stripping markdown fences)")
    try:
        obj = json.loads(candidate)
    except Exception as e:
        preview = (candidate[:400] + "...") if len(candidate) > 400 else candidate
        raise ValueError(f"invalid JSON from agent output; preview={preview!r}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"agent final output must be a JSON object, got {type(obj)}")
    return obj


def extract_json_from_response(content: str) -> Dict[str, Any]:
    """从 VLM 响应中提取 JSON 对象，处理 markdown 代码块和额外文本。
    
    Args:
        content: 模型响应的原始文本
        
    Returns:
        解析后的 dict，如果解析失败返回空字典
    """
    if not content or not content.strip():
        return {}
    
    # 复用现有的提取逻辑
    candidate = extract_json_fragment(content)
    if not candidate:
        return {}
    
    try:
        result = json.loads(candidate)
        if isinstance(result, dict):
            return result
        return {}
    except json.JSONDecodeError:
        return {}


