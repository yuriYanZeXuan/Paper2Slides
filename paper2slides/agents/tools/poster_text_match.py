import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re
import difflib

from qwen_agent.tools.base import BaseTool, register_tool


BBox = Tuple[int, int, int, int]

_PLAN_SPANS_CACHE: dict[str, tuple[float, List[Dict[str, Any]]]] = {}

_DEFAULT_MAX_CANDIDATES = 40


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


def _load_image_size_from_ckpt(ckpt_path: str) -> tuple[int, int] | None:
    """从 grounding checkpoint 读取 image_size (width/height)。"""
    with open(ckpt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    image_size = data.get("image_size") or {}
    w = int(image_size.get("width"))
    h = int(image_size.get("height"))
    if w > 0 and h > 0:
        return w, h


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _effective_text_len(line: str) -> float:
    """将一行文字折算成“em”长度，用于按宽度估算字体大小。

    - CJK 近似 1.0em/字
    - ASCII/数字/标点 近似 0.55em/字符
    - 空格 近似 0.33em
    """
    if not line:
        return 0.0
    total = 0.0
    for ch in line:
        if ch.isspace():
            total += 0.33
        elif _CJK_RE.match(ch):
            total += 1.0
        else:
            total += 0.55
    return total


def estimate_pptx_font_style_from_bbox(
    *,
    ocr_text: str,
    bbox: BBox,
    image_w: int,
    image_h: int,
    canvas_w_in: float = 48.0,
    canvas_h_in: float = 36.0,
    line_height_to_font_ratio: float = 1.2,
) -> Dict[str, Any]:
    """基于 OCR 文本 + bbox 估算 PPTX 字体大小（pt）与行高（pt）。

    这是启发式估计，目标是“看起来接近原图”，不追求像素级一致。
    """
    x0, y0, x1, y1 = [int(v) for v in bbox]
    bw_px = max(1, x1 - x0)
    bh_px = max(1, y1 - y0)

    # 行数估计：优先按换行；若无换行则认为 1 行
    raw_lines = [ln.strip() for ln in str(ocr_text or "").splitlines() if ln.strip()]
    num_lines = max(1, len(raw_lines))
    lines = raw_lines if raw_lines else [str(ocr_text or "").strip()]

    # px -> inch
    w_in = (bw_px / max(1, int(image_w))) * float(canvas_w_in)
    h_in = (bh_px / max(1, int(image_h))) * float(canvas_h_in)

    # line-height (pt) via bbox height
    line_height_in = h_in / float(num_lines)
    line_height_pt = max(1.0, line_height_in * 72.0)

    # font size via height (pt)
    denom = float(line_height_to_font_ratio) if line_height_to_font_ratio > 0 else 1.2
    font_pt_by_h = line_height_pt / denom

    # font size via width (pt)
    eff_lens = [_effective_text_len(ln) for ln in lines if ln]
    max_eff_len = max(eff_lens) if eff_lens else 1.0
    font_pt_by_w = (w_in * 72.0) / max(1.0, max_eff_len)

    # conservative: choose smaller, clamp
    font_pt = max(6.0, min(200.0, min(font_pt_by_h, font_pt_by_w)))

    return {
        "estimated_num_lines": int(num_lines),
        "estimated_line_height_pt": float(round(line_height_pt, 2)),
        "estimated_font_size_pt": float(round(font_pt, 2)),
        "canvas_w_in": float(canvas_w_in),
        "canvas_h_in": float(canvas_h_in),
        "line_height_to_font_ratio": float(denom),
        "method": "bbox_height_and_width_heuristic",
    }


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
        max_candidates = _DEFAULT_MAX_CANDIDATES
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
    """仅支持 Batch 模式：一次性匹配多个 region_id 的 OCR 文本到 plan_text_spans。"""

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
            "region_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "(Batch) List of region_id values. Use this to do ONE tool call for multiple regions.",
            },
            "bboxes": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
                "description": "Optional. Pixel bboxes aligned with region_ids. Used only for style_hint estimation.",
            },
            "max_candidates": {
                "type": "integer",
                "description": "Optional. Only consider the first N plan_text_spans as candidates (default 40).",
            },
        },
        "required": ["plan_text_spans_path", "grounding_ckpt_path", "region_ids"],
    }

    def call(self, params, **kwargs) -> str:
        params = self._verify_json_format_args(params)

        plan_text_spans_path = params["plan_text_spans_path"]
        ckpt_path = params["grounding_ckpt_path"]

        plan_text_spans = _load_plan_text_spans(plan_text_spans_path)
        ocr_map = _load_ocr_text_map_from_ckpt(ckpt_path)
        img_size = _load_image_size_from_ckpt(ckpt_path)

        region_ids = params.get("region_ids") or []
        assert isinstance(region_ids, list) and len(region_ids) > 0, "region_ids must be a non-empty list"

        bboxes = params.get("bboxes") or []
        if bboxes:
            assert isinstance(bboxes, list) and len(bboxes) == len(region_ids), "bboxes length must match region_ids length"

        max_candidates = params.get("max_candidates", None)
        try:
            max_candidates_i = int(max_candidates) if max_candidates is not None else None
        except Exception:
            max_candidates_i = None

        results: List[Dict[str, Any]] = []
        for i, rid in enumerate(region_ids):
            rid_i = int(rid)
            ocr_text = ocr_map.get(rid_i, "")
            matched_text, meta = match_plan_text_with_ocr(
                ocr_text=ocr_text,
                plan_text_spans=plan_text_spans,
                max_candidates=max_candidates_i,
            )

            style_hint = None
            if bboxes and img_size is not None:
                bbox_i = tuple(map(int, bboxes[i]))
                iw, ih = img_size
                style_hint = estimate_pptx_font_style_from_bbox(
                    ocr_text=ocr_text,
                    bbox=bbox_i,  # type: ignore[arg-type]
                    image_w=iw,
                    image_h=ih,
                )

            results.append(
                {
                    "region_id": rid_i,
                    "bbox": bboxes[i] if bboxes else None,
                    "matched_text": matched_text,
                    "meta": meta,
                    "style_hint": style_hint,
                }
            )

        return json.dumps({"results": results}, ensure_ascii=False)
