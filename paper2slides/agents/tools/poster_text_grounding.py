import base64
import io
import json
import os
from typing import List, Tuple, Union

from PIL import Image

from qwen_agent.tools.base import BaseTool, register_tool
from paper2slides.utils.agent_logging import log_agent_info, log_agent_warning
from paper2slides.utils.api_utils import get_openai_client


BBox = Tuple[int, int, int, int]

DEFAULT_GROUNDING_VLM_MODEL = os.getenv("POSTER_TEXT_GROUNDING_MODEL", "gpt-4o")


def _encode_image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


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
        "You must return bounding boxes for regions where text is likely small, low-contrast, or hard to read."
    )
    user_instructions = (
        "Given this poster image (width: {w}px, height: {h}px), "
        "identify up to 5 regions where text is small or unclear and could benefit from enhancement.\n\n"
        "Return ONLY a JSON object of the form:\n"
        "{\n"
        '  "bboxes": [\n'
        "    [x0, y0, x1, y1],\n"
        "    ... up to 5 items ...\n"
        "  ]\n"
        "}\n\n"
        "Coordinates must be integer pixel values in the range:\n"
        f"0 <= x0 < x1 <= {w}, 0 <= y0 < y1 <= {h}.\n"
    ).format(w=w, h=h)

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
    raw_bboxes = data.get("bboxes", [])

    bboxes: List[BBox] = []
    for box in raw_bboxes:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x0, y0, x1, y1 = map(int, box)
        # 简单合法性检查与裁剪
        x0 = max(0, min(x0, w))
        x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h))
        y1 = max(0, min(y1, h))
        if x1 <= x0 or y1 <= y0:
            continue
        bboxes.append((x0, y0, x1, y1))
    assert len(bboxes) > 0, "No bboxes found"

    log_agent_info("poster_text_grounding", f"vlm grounded {len(bboxes)} regions: {bboxes}")
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
        return json.dumps({'bboxes': bboxes}, ensure_ascii=False)
