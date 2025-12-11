import base64
import io
import json
import os
from typing import Union

from PIL import Image

from qwen_agent.tools.base import BaseTool, register_tool
from paper2slides.utils.agent_logging import log_agent_info, log_agent_warning
from paper2slides.utils.api_utils import get_openai_client


DEFAULT_TEXT_VLM_MODEL = os.getenv("POSTER_TEXT_VLM_MODEL", "gpt-4o")


def _encode_image_to_base64(image: Image.Image) -> str:
    """将 PIL Image 编码为 base64 data URL 所需的字符串。"""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def score_poster_text_clarity_with_vlm(image: Image.Image) -> float:
    """使用 OpenAI 兼容的多模态模型（默认 gpt-4o）对文字清晰度打分。

    设计为一个可复用的函数，后续可以替换为本地开源 VLM，只需提供同样接口。
    """
    try:
        client = get_openai_client(key_type="text")
    except Exception as e:
        log_agent_warning("poster_text_score", f"failed to init VLM client: {e}, fallback to dummy score=5.0")
        return 5.0

    b64 = _encode_image_to_base64(image)

    system_prompt = (
        "You are an expert at evaluating the visual clarity and legibility of text in poster images. "
        "Score ONLY the readability of all text in the image."
    )
    user_instructions = (
        "Look at this poster image and evaluate how clear and readable the text is overall.\n\n"
        "- Score range: 0.0 (completely unreadable) to 10.0 (perfectly sharp and legible).\n"
        "- Consider font sharpness, contrast, size, and whether small text can be read.\n"
        "- Return ONLY a JSON object with a single field 'score', for example: {\"score\": 7.5}.\n"
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

    try:
        response = client.chat.completions.create(
            model=DEFAULT_TEXT_VLM_MODEL,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        score = float(data.get("score", 5.0))
        # clamp to [0, 10]
        score = max(0.0, min(10.0, score))
        log_agent_info("poster_text_score", f"vlm score={score:.2f}")
        return score
    except Exception as e:
        log_agent_warning("poster_text_score", f"VLM scoring failed: {e}, fallback to dummy score=5.0")
        return 5.0


@register_tool('poster_text_score')
class PosterTextScore(BaseTool):
    """对海报文字清晰度进行打分的工具。

    当前实现：使用 OpenAI 兼容的多模态模型（默认 gpt-4o）进行视觉打分；
    后续如需切换到本地开源模型或其他 API，只需替换 score_poster_text_clarity_with_vlm 的内部实现。
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

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            score = 0.0
            log_agent_warning("poster_text_score", f"failed to open image: {image_path}, return score={score}")
            return json.dumps({'score': float(score)}, ensure_ascii=False)

        score = score_poster_text_clarity_with_vlm(img)
        return json.dumps({'score': float(score)}, ensure_ascii=False)
