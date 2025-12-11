import json
from typing import Union

from PIL import Image

from qwen_agent.tools.base import BaseTool, register_tool
from paper2slides.utils.agent_logging import log_agent_info, log_agent_warning


@register_tool('poster_text_score')
class PosterTextScore(BaseTool):
    """最小实现：对海报文字清晰度进行打分（占位实现）。

    真实场景下，这里应该调用 Qwen-VL / 其它 VLM，根据图像和提示语返回一个 0-10 的分数。
    当前仅返回固定分数，目的是提供一个符合 Qwen-Agent 工具范式的壳，方便集成和调试。
    """

    description = 'Score the clarity / legibility of texts in a poster image.'
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

        # 这里仅验证文件是否存在 / 能否打开，实际不使用图像内容。
        try:
            _ = Image.open(image_path)
        except Exception:
            score = 0.0
            log_agent_warning("poster_text_score", f"failed to open image: {image_path}, return score={score}")
        else:
            # 占位：恒定 5.0，后续替换为真实 VLM 调用
            score = 5.0
            log_agent_info("poster_text_score", f"dummy score={score} for image={image_path}")

        return json.dumps({'score': float(score)}, ensure_ascii=False)
