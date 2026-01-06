import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re
import difflib

from qwen_agent.tools.base import BaseTool, register_tool
from paper2slides.agents.tools.config_loader import get_text_match_config
from paper2slides.utils.agent_artifact_logging import save_json_log


BBox = Tuple[int, int, int, int]

_PLAN_SPANS_CACHE: dict[str, tuple[float, List[Dict[str, Any]]]] = {}


def _get_default_max_candidates() -> int:
    """获取默认最大候选数量。"""
    cfg = get_text_match_config()
    return int(cfg.get("max_candidates", 40))


def _load_plan_text_spans(plan_text_spans_path: str) -> List[Dict[str, Any]]:
    p = Path(plan_text_spans_path)
    assert p.exists(), f"plan_text_spans_path not found: {plan_text_spans_path}"
    assert p.is_file(), f"plan_text_spans_path is not a file: {plan_text_spans_path}"

    mtime = p.stat().st_mtime
    cache_key = str(p)
    if cache_key in _PLAN_SPANS_CACHE:
        cached_mtime, cached_data = _PLAN_SPANS_CACHE[cache_key]
        if cached_mtime == mtime:
            return cached_data

    data = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(data, list), f"plan_text_spans json must be a list, got: {type(data)}"
    # best-effort validation (do not be too strict)
    for item in data[:3]:
        assert isinstance(item, dict), "each span must be an object"

    _PLAN_SPANS_CACHE[cache_key] = (mtime, data)
    return data


def _load_ocr_text_map_from_ckpt(ckpt_path: str) -> dict[int, str]:
    """从 MinerU grounding checkpoint 里读取 raw_blocks，并返回 {id: content}."""
    with open(ckpt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_blocks = data.get("raw_blocks", []) or []
    out: dict[int, str] = {}
    for b in raw_blocks:
        try:
            bid = int(b.get("id"))
        except Exception:
            continue
        content = str(b.get("content") or "")
        if content.strip():
            out[bid] = content
    return out


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\u200b\u200c\u200d\uFEFF]")


def _norm_text(s: str) -> str:
    s = (s or "").strip()
    s = _PUNCT_RE.sub("", s)
    s = s.replace("\n", " ")
    s = _WS_RE.sub(" ", s)
    return s.lower()


def _similarity(a: str, b: str) -> float:
    """轻量相似度：difflib ratio（0~1）。"""
    a_n = _norm_text(a)
    b_n = _norm_text(b)
    if not a_n or not b_n:
        return 0.0
    return difflib.SequenceMatcher(None, a_n, b_n).ratio()


def match_plan_text_with_ocr(
    ocr_text: str,
    plan_text_spans: List[Dict[str, Any]],
    *,
    max_candidates: int | None = None,
) -> tuple[str | None, Dict[str, Any]]:
    """用 OCR 文本与 plan_text_spans 做字符串相似度匹配（不调用 VLM、不用 bbox crop）。"""
    if not ocr_text or not ocr_text.strip() or not plan_text_spans:
        return None, {"matched_index": None, "score": 0.0}

    if max_candidates is None:
        max_candidates = _get_default_max_candidates()
    candidates = plan_text_spans[: max(1, int(max_candidates))]

    best_i: int | None = None
    best_score = -1.0
    best_span: Dict[str, Any] | None = None
    for i, span in enumerate(candidates, start=1):
        t = str(span.get("text") or "")
        s = _similarity(ocr_text, t)
        if s > best_score:
            best_score = s
            best_i = i
            best_span = span

    matched_text = None
    if best_i is not None and best_span is not None:
        matched_text = str(best_span.get("text") or "").strip() or None

    meta = {
        "matched_index": best_i,
        "score": float(best_score),
        "ocr_text": str(ocr_text),
        "matched_span": best_span,
    }
    return matched_text, meta


@register_tool("poster_text_match")
class PosterTextMatch(BaseTool):
    """将 patch 里的文字匹配到 plan_text_spans（通过路径加载）。

    方案 B：
    - 调用方只传 plan_text_spans_path（避免把大候选塞进 tool 参数由 LLM 搬运）
    - tool 内部读取 spans，再调用 VLM 做匹配
    """

    description = "Match the text in a poster patch to plan text spans (loaded from a JSON file)."
    parameters = {
        "type": "object",
        "properties": {
            "plan_text_spans_path": {
                "type": "string",
                "description": "Path to a JSON file containing plan_text_spans list.",
            },
            "grounding_ckpt_path": {
                "type": "string",
                "description": "Path to the MinerU grounding checkpoint JSON file (returned by poster_text_grounding).",
            },
            "region_id": {
                "type": "integer",
                "description": "(Single) The 'id' of the text region from grounding results.",
            },
            "region_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "(Batch) List of region_id values. Use this to do ONE tool call for multiple regions.",
            },
            "bboxes": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
                "description": "Optional. Only used for echoing back to caller; matching does NOT use bbox crops.",
            },
            "log_root": {
                "type": "string",
                "description": "Optional log root path for logging.",
            },
        },
        "required": ["plan_text_spans_path", "grounding_ckpt_path"],
    }

    def call(self, params, **kwargs) -> str:
        params = self._verify_json_format_args(params)

        plan_text_spans_path = params["plan_text_spans_path"]
        ckpt_path = params["grounding_ckpt_path"]

        plan_text_spans = _load_plan_text_spans(plan_text_spans_path)
        ocr_map = _load_ocr_text_map_from_ckpt(ckpt_path)

        agent_name = str(params.get("agent_name") or "poster_refiner")
        log_root = params.get("log_root")

        # Batch mode (preferred)
        if params.get("region_ids") is not None:
            region_ids = params.get("region_ids") or []
            assert isinstance(region_ids, list) and len(region_ids) > 0, "region_ids must be a non-empty list"
            bboxes = params.get("bboxes") or []
            if bboxes:
                assert isinstance(bboxes, list) and len(bboxes) == len(region_ids), "bboxes length must match region_ids length"

            results: List[Dict[str, Any]] = []
            for i, rid in enumerate(region_ids):
                rid_i = int(rid)
                ocr_text = ocr_map.get(rid_i, "")
                matched_text, meta = match_plan_text_with_ocr(ocr_text, plan_text_spans)
                if log_root:
                    save_json_log(
                        agent_name=agent_name,
                        func_name="match_text_with_ocr",
                        payload={"region_id": rid_i, "meta": meta},
                        log_root=log_root,
                        suffix=f"rid_{rid_i}",
                    )
                results.append(
                    {
                        "region_id": rid_i,
                        "bbox": bboxes[i] if bboxes else None,
                        "matched_text": matched_text,
                        "meta": meta,
                    }
                )
            return json.dumps({"results": results}, ensure_ascii=False)

        # Single mode
        rid = params.get("region_id")
        assert rid is not None, "either region_ids or region_id must be provided"
        rid_i = int(rid)
        ocr_text = ocr_map.get(rid_i, "")
        matched_text, meta = match_plan_text_with_ocr(ocr_text, plan_text_spans)
        if log_root:
            save_json_log(
                agent_name=agent_name,
                func_name="match_text_with_ocr",
                payload={"region_id": rid_i, "meta": meta},
                log_root=log_root,
                suffix=f"rid_{rid_i}",
            )
        return json.dumps({"matched_text": matched_text, "meta": meta}, ensure_ascii=False)
