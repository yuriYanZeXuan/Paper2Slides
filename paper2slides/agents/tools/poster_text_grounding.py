import json
from typing import List, Tuple, Union

from PIL import Image

from qwen_agent.tools.base import BaseTool, register_tool
from paper2slides.utils.agent_logging import log_agent_info


BBox = Tuple[int, int, int, int]


@register_tool('poster_text_grounding')
class PosterTextGrounding(BaseTool):
    """最小实现：在海报中粗略定位需要增强的小文字区域。

    当前实现：返回图像中心附近的一个矩形 bbox，主要用于演示工具调用/调试流程。
    后续可以替换为 grounding DINO + OCR，或 VLM grounded bbox。"""

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

        img = Image.open(image_path)
        w, h = img.size
        cx, cy = w // 2, h // 2
        bw, bh = w // 3, h // 6
        x0 = max(cx - bw // 2, 0)
        y0 = max(cy - bh // 2, 0)
        x1 = min(cx + bw // 2, w)
        y1 = min(cy + bh // 2, h)
        bboxes: List[BBox] = [(x0, y0, x1, y1)]

        log_agent_info("poster_text_grounding", f"image={image_path}, bboxes={bboxes}")
        return json.dumps({'bboxes': bboxes}, ensure_ascii=False)
