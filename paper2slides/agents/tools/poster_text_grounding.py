import base64
import io
import json
import os
from typing import List, Tuple, Union

from PIL import Image

from qwen_agent.tools.base import BaseTool, register_tool
from paper2slides.utils.agent_logging import log_agent_info, log_agent_warning
from paper2slides.utils.agent_artifact_logging import save_bbox_visualization, save_json_log
from paper2slides.utils.api_utils import get_openai_client


BBox = Tuple[int, int, int, int]

DEFAULT_GROUNDING_VLM_MODEL = os.getenv("POSTER_TEXT_GROUNDING_MODEL", "gpt-4o")


def _encode_image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _maybe_denormalize_bbox(
    box: List[float], w: int, h: int
) -> Tuple[int, int, int, int]:
    """将可能的归一化坐标 (0-1) 转换为像素坐标。

    判断逻辑：如果所有坐标值都在 [0, 1] 范围内，则认为是归一化坐标。
    """
    x0, y0, x1, y1 = box
    # 判断是否是归一化坐标：所有值都在 [0, 1] 范围内
    is_normalized = all(0 <= v <= 1 for v in [x0, y0, x1, y1])
    
    if is_normalized:
        # 归一化坐标 -> 像素坐标
        x0 = int(x0 * w)
        y0 = int(y0 * h)
        x1 = int(x1 * w)
        y1 = int(y1 * h)
    else:
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    
    return x0, y0, x1, y1


def ground_poster_text_regions_with_vlm(image: Image.Image) -> List[BBox]:
    """使用多模态模型（默认 gpt-4o）在海报中定位需要增强的小文字区域。

    返回值为像素级别的 bbox 列表，便于与后续 FlowEdit / Refiner 直接对接。
    如需替换为本地开源 VLM 或其它接口，只需保持该函数签名不变。
    """
    w, h = image.size

    client = get_openai_client(key_type="text")

    b64 = _encode_image_to_base64(image)

    system_prompt = (
        "You are an expert at analyzing poster layouts and detecting small or unclear text regions.\n"
        "You must return bounding boxes for regions where text is likely small, low-contrast, or hard to read.\n"
        "You should provide your reasoning process and the text content within each region."
    )
    user_instructions = (
        f"Analyze this poster image with dimensions: WIDTH = {w} pixels, HEIGHT = {h} pixels.\n\n"
        "Task: Identify up to 5 rectangular regions containing small or unclear text that could benefit from enhancement.\n\n"
        "COORDINATE SYSTEM:\n"
        f"- Image origin (0, 0) is at TOP-LEFT corner.\n"
        f"- X-axis: 0 (left) to {w} (right). Y-axis: 0 (top) to {h} (bottom).\n"
        f"- All coordinates MUST be INTEGER PIXEL VALUES (NOT normalized 0-1 or per-thousand 0-1000).\n"
        f"- Valid ranges: 0 <= x0 < x1 <= {w}, 0 <= y0 < y1 <= {h}.\n\n"
        "EXAMPLES for this image:\n"
        f"- Top-left quarter: [0, 0, {w//2}, {h//2}]\n"
        f"- Bottom-right quarter: [{w//2}, {h//2}, {w}, {h}]\n"
        f"- A text region at upper-right: [{int(w*0.7)}, {int(h*0.1)}, {int(w*0.95)}, {int(h*0.25)}]\n\n"
        "Return ONLY a JSON object with the following structure:\n"
        "{\n"
        '  "thinking": "Your step-by-step reasoning: how you analyzed the poster, what regions you noticed, why they need enhancement...",\n'
        f'  "image_size": [{w}, {h}],\n'
        '  "regions": [\n'
        "    {\n"
        '      "bbox": [x0, y0, x1, y1],\n'
        '      "text_content": "The actual text you can read in this region (transcribe as accurately as possible)",\n'
        '      "reason": "Why this region needs enhancement (e.g., small font, low contrast, blurry)"\n'
        "    },\n"
        "    ... up to 5 regions ...\n"
        "  ]\n"
        "}\n\n"
        "IMPORTANT:\n"
        "- Coordinates MUST be integer pixel values matching the image dimensions above.\n"
        "- Each bbox should be at least 50x50 pixels.\n"
        "- Transcribe the text_content as accurately as you can see it."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_instructions},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}"
                    },
                },
            ],
        },
    ]

    client_response = client.chat.completions.create(
        model=DEFAULT_GROUNDING_VLM_MODEL,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    content = client_response.choices[0].message.content
    data = json.loads(content)
    
    # 解析新格式的响应
    thinking = data.get("thinking", "")
    vlm_reported_size = data.get("image_size", None)
    raw_regions = data.get("regions", [])
    
    # 兼容旧格式（如果 VLM 返回的是 bboxes 而不是 regions）
    if not raw_regions and "bboxes" in data:
        raw_bboxes = data.get("bboxes", [])
        raw_regions = [{"bbox": box, "text_content": "", "reason": ""} for box in raw_bboxes]

    # 记录 VLM 的原始响应到日志文件，便于调试
    save_json_log(
        agent_name="poster_text_grounding",
        func_name="vlm_grounding_raw",
        payload={
            "actual_image_size": {"width": w, "height": h},
            "vlm_reported_size": vlm_reported_size,
            "model": DEFAULT_GROUNDING_VLM_MODEL,
            "thinking": thinking,
            "raw_response": content,
            "raw_regions": raw_regions,
        },
    )
    log_agent_info(
        "poster_text_grounding",
        f"vlm thinking: {thinking[:200]}..." if len(thinking) > 200 else f"vlm thinking: {thinking}"
    )
    log_agent_info(
        "poster_text_grounding",
        f"vlm raw response: actual_size=({w}, {h}), vlm_size={vlm_reported_size}, regions_count={len(raw_regions)}"
    )

    bboxes: List[BBox] = []
    region_details: List[dict] = []  # 保存完整的 region 信息用于日志
    
    for idx, region in enumerate(raw_regions):
        if not isinstance(region, dict):
            log_agent_warning("poster_text_grounding", f"invalid region format: {region}")
            continue
        
        box = region.get("bbox", [])
        text_content = region.get("text_content", "")
        reason = region.get("reason", "")
        
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            log_agent_warning("poster_text_grounding", f"invalid bbox format in region {idx}: {box}")
            continue
        
        # 处理可能的归一化坐标
        raw_box = list(map(float, box))
        x0, y0, x1, y1 = _maybe_denormalize_bbox(raw_box, w, h)
        
        # 简单合法性检查与裁剪
        x0 = max(0, min(x0, w))
        x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h))
        y1 = max(0, min(y1, h))
        
        if x1 <= x0 or y1 <= y0:
            log_agent_warning("poster_text_grounding", f"invalid bbox after denorm: ({x0}, {y0}, {x1}, {y1})")
            continue
        
        bboxes.append((x0, y0, x1, y1))
        region_details.append({
            "idx": idx,
            "raw_bbox": raw_box,
            "final_bbox": [x0, y0, x1, y1],
            "text_content": text_content,
            "reason": reason,
        })
        
        log_agent_info(
            "poster_text_grounding",
            f"region {idx}: bbox={[x0, y0, x1, y1]}, text='{text_content[:50]}...'" if len(text_content) > 50 
            else f"region {idx}: bbox={[x0, y0, x1, y1]}, text='{text_content}'"
        )
    
    # 保存详细的 region 信息到日志
    save_json_log(
        agent_name="poster_text_grounding",
        func_name="vlm_grounding_regions",
        payload={
            "image_size": {"width": w, "height": h},
            "thinking": thinking,
            "valid_regions_count": len(bboxes),
            "region_details": region_details,
        },
    )
    
    if len(bboxes) == 0:
        log_agent_warning("poster_text_grounding", "No valid bboxes found from VLM response")
        raise ValueError(f"No valid bboxes found. raw_regions={raw_regions}, image_size=({w}, {h})")

    log_agent_info("poster_text_grounding", f"vlm grounded {len(bboxes)} regions (after denorm): {bboxes}")
    return bboxes


@register_tool('poster_text_grounding')
class PosterTextGrounding(BaseTool):
    """在海报中定位需要增强的小文字区域。

    当前实现：使用 OpenAI 兼容多模态模型（默认 gpt-4o）做视觉 grounding；
    如需使用本地开源模型或其它服务，可替换 ground_poster_text_regions_with_vlm 的内部实现。
    """

    description = 'Roughly locate small / unclear text regions in a poster image.'
    parameters = {
        'type': 'object',
        'properties': {
            'image_path': {
                'type': 'string',
                'description': 'Path to the poster image file.'
            }
        },
        'required': ['image_path'],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        image_path = params['image_path']
        img = Image.open(image_path).convert("RGB")
        bboxes = ground_poster_text_regions_with_vlm(img)

        # 保存 bbox 可视化结果
        vis_path = save_bbox_visualization(
            agent_name="poster_text_grounding",
            func_name="vlm_grounding",
            image=img,
            bboxes=bboxes,
        )
        log_agent_info("poster_text_grounding", f"bbox visualization saved to {vis_path}")
      
        return json.dumps({'bboxes': bboxes}, ensure_ascii=False)
