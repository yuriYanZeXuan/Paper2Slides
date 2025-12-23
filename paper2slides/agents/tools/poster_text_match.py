import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

from qwen_agent.tools.base import BaseTool, register_tool
from paper2slides.agents.tools.config_loader import get_text_match_config
from paper2slides.utils.api_utils import get_openai_client
from paper2slides.utils.agent_artifact_logging import save_json_log
from paper2slides.utils.agent_output_parsing import extract_json_from_response


BBox = Tuple[int, int, int, int]

_PLAN_SPANS_CACHE: dict[str, tuple[float, List[Dict[str, Any]]]] = {}


def _get_default_model() -> str:
    """获取默认匹配模型。"""
    cfg = get_text_match_config()
    model = cfg.get("model", "").strip()
    if model:
        return model
    return os.getenv("POSTER_TEXT_MATCH_MODEL", "gemini-3-pro")


def _get_default_max_candidates() -> int:
    """获取默认最大候选数量。"""
    cfg = get_text_match_config()
    return int(cfg.get("max_candidates", 40))


def _encode_image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

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


def _load_ocr_hint_from_ckpt(ckpt_path: str, target_bbox: BBox | None = None, target_id: int | None = None) -> str | None:
    """从 grounding checkpoint 中查找 OCR 内容作为 hint。
    
    优先使用 target_id 查找；如果未提供，则尝试使用 bbox 匹配（IoU 或距离）。
    """
    with open(ckpt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    raw_blocks = data.get("raw_blocks", [])
    if not raw_blocks:
        return None
        
    # 方式 1: 通过 ID 查找
    if target_id is not None:
        for block in raw_blocks:
            # 兼容旧格式（没有 id 字段）和新格式
            if block.get("id") == target_id:
                return block.get("content", "")
        return None
            

def match_plan_text_for_patch(
    patch: Image.Image,
    bbox: BBox,
    plan_text_spans: List[Dict[str, Any]],
    *,
    max_candidates: int | None = None,
    model: str | None = None,
    hint_text: str | None = None,
    agent_name: str = "poster_refiner",
    log_root: str | None = None,
) -> tuple[str | None, Dict[str, Any] | None]:
    """用 VLM 将 patch 中的文字匹配到 plan_text_spans。

    返回：
    - matched_text: 可能为 None
    - meta: 匹配元信息（用于日志）

    说明：该函数放在 tools 目录下，便于与 poster_refiner 解耦维护。
    """

    if not plan_text_spans:
        return None, None

    # 使用配置中的默认值
    if max_candidates is None:
        max_candidates = _get_default_max_candidates()
    if model is None:
        model = _get_default_model()
    
    model = model.strip()
    assert model, "model must be non-empty"

    client = get_openai_client(key_type="text")
    w, h = patch.size

    candidates = plan_text_spans[: max(1, int(max_candidates))]
    candidates_str_lines: List[str] = []
    for idx, span in enumerate(candidates, start=1):
        text = (span.get("text") or "").replace("\n", " ").strip()
        if len(text) > 200:
            text = text[:200] + "..."
        candidates_str_lines.append(f"{idx}. {text}")
    candidates_str = "\n".join(candidates_str_lines)

    b64 = _encode_image_to_base64(patch)

    system_prompt = (
        "You are an expert at reading small text on academic posters and matching it to candidate text spans. "
        "Your task is to find which candidate text best corresponds to the text appearing inside the given image patch. "
        "You must respond with pure JSON only, no markdown, no code blocks, no extra text."
    )
    
    ocr_hint_str = ""
    # 优先使用 hint_text (通常来自 ckpt)
    if hint_text and len(hint_text.strip()) > 0:
        ocr_hint_str = f"Hint: An upstream OCR model read the text in this region as: \"{hint_text}\". Use this as a reference if the image is blurry.\n"

    user_text = (
        f"Here is a small image patch from a poster. {ocr_hint_str}"
        "First, read the text inside the patch.\n"
        "Then, from the candidate list below, choose the SINGLE candidate that best matches "
        "the text in this patch (based on semantic content, not style).\n\n"
        "Respond with ONLY the raw JSON object, do NOT wrap it in ```json``` or any markdown.\n"
        "Example response: {\"matched_index\": 1, \"matched_text\": \"example text\"}\n\n"
        "JSON format:\n"
        "{\n"
        '  \"matched_index\": <integer index in [1..N]> or null if no good match,\n'
        '  \"matched_text\": \"the chosen candidate text or empty string\"\n'
        "}\n\n"
        "Candidate text spans:\n"
        f"{candidates_str}\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )
    content = response.choices[0].message.content
    
    # 从响应中提取 JSON（处理模型可能输出额外文本的情况）
    data = extract_json_from_response(content)

    raw_idx = data.get("matched_index")
    raw_text = data.get("matched_text")

    matched_index: int | None = None
    matched_text: str | None = None

    if isinstance(raw_idx, int) and 1 <= raw_idx <= len(candidates):
        matched_index = raw_idx
        matched_span = candidates[matched_index - 1]
        matched_text = str(raw_text or matched_span.get("text") or "")
        meta: Dict[str, Any] = {
            "matched_index": matched_index,
            "matched_span": matched_span,
            "bbox": [int(v) for v in bbox],
            "patch_size": [int(w), int(h)],
        }
    else:
        meta = {
            "matched_index": None,
            "bbox": [int(v) for v in bbox],
            "patch_size": [int(w), int(h)],
        }

    if log_root:
        save_json_log(
            agent_name=agent_name,
            func_name="match_text_for_patch",
            payload=meta,
            log_root=log_root,
        )

    if matched_index is None or not matched_text:
        return None, meta

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
            "image_path": {
                "type": "string",
                "description": "Path to the full poster image file. The tool will crop bbox as the patch.",
            },
            "bbox": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Bounding box [x0, y0, x1, y1] in the original image coordinate system.",
            },
            "plan_text_spans_path": {
                "type": "string",
                "description": "Path to a JSON file containing plan_text_spans list.",
            },
            "grounding_ckpt_path": {
                "type": "string",
                "description": "Path to the grounding checkpoint JSON file (returned by poster_text_grounding).",
            },
            "region_id": {
                "type": "integer",
                "description": "The 'id' of the text region from grounding results. Used to lookup OCR content in checkpoint.",
            },
            "log_root": {
                "type": "string",
                "description": "Optional log root path for logging.",
            },
        },
        "required": ["image_path", "bbox", "plan_text_spans_path"],
    }

    def call(self, params, **kwargs) -> str:
        params = self._verify_json_format_args(params)

        image_path = params["image_path"]
        bbox = params["bbox"]
        plan_text_spans_path = params["plan_text_spans_path"]

        assert isinstance(bbox, (list, tuple)) and len(bbox) == 4, f"invalid bbox: {bbox}"
        bbox_t: BBox = tuple(map(int, bbox))  # type: ignore[assignment]

        full_img = Image.open(image_path).convert("RGB")
        patch = full_img.crop(bbox_t)
        plan_text_spans = _load_plan_text_spans(plan_text_spans_path)
        
        # 优先从 checkpoint 获取 hint
        hint_text = None
        ckpt_path = params.get("grounding_ckpt_path")
        region_id = params.get("region_id")
        
        if ckpt_path:
            # 尝试从 checkpoint 加载
            ckpt_hint = _load_ocr_hint_from_ckpt(ckpt_path, target_id=region_id)
            if ckpt_hint and ckpt_hint.strip():
                hint_text = ckpt_hint
                
        agent_name = str(params.get("agent_name") or "poster_refiner")
        log_root = params.get("log_root")

        matched_text, meta = match_plan_text_for_patch(
            patch=patch,
            bbox=bbox_t,
            plan_text_spans=plan_text_spans,
            hint_text=hint_text,
            agent_name=agent_name,
            log_root=log_root,
        )

        return json.dumps(
            {"matched_text": matched_text, "meta": meta},
            ensure_ascii=False,
        )
