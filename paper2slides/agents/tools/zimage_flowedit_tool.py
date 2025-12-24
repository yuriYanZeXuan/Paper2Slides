import json
import os
from typing import Union

from PIL import Image
import torch
from diffusers import ZImagePipeline

from qwen_agent.tools.base import BaseTool, register_tool

from paper2slides.agents.tools.config_loader import get_flowedit_config
from paper2slides.agents.tools.zimage_flowedit_core import FlowEditZImage
from paper2slides.utils.agent_logging import log_agent_info, log_agent_success


_PIPE_CACHE: dict[str, ZImagePipeline] = {}


def _get_zimage_pipe(model_name: str, device: str) -> ZImagePipeline:
    key = f'{model_name}@{device}'
    if key in _PIPE_CACHE:
        return _PIPE_CACHE[key]

    pipe = ZImagePipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if 'cuda' in device else torch.float32,
        low_cpu_mem_usage=False,
    )
    pipe.to(device)
    _PIPE_CACHE[key] = pipe
    return pipe


@register_tool('zimage_flowedit')
class ZImageFlowEdit(BaseTool):
    """使用 Z-Image FlowEdit 对整张图像做一次 inversion-free 文本编辑。

    最小实现：
    - 输入：源图路径 + 源/目标 prompt + 输出路径
    - 过程：加载/复用 Z-Image pipeline，调用 FlowEditZImage
    - 输出：写入一张新图，并返回输出路径
    """

    description = 'Perform inversion-free FlowEdit editing on a whole image using Z-Image.'
    parameters = {
        'type': 'object',
        'properties': {
            'src_image_path': {
                'type': 'string',
                'description': 'Path to the source image file.'
            },
            'src_prompt': {
                'type': 'string',
                'description': 'Text description of the current image content.'
            },
            'tar_prompt': {
                'type': 'string',
                'description': 'Target text description for editing.'
            },
            'output_path': {
                'type': 'string',
                'description': 'Where to save the edited image.'
            },
            'in_context': {
                'type': 'boolean',
                'description': 'Enable In-Context-Aware mode: concatenate reference image during denoising to preserve similarity with original image. Default is False.',
                'default': False,
            },
        },
        'required': ['src_image_path', 'src_prompt', 'tar_prompt', 'output_path'],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)

        src_image_path: str = params['src_image_path']
        src_prompt: str = params['src_prompt']
        tar_prompt: str = params['tar_prompt']
        output_path: str = params['output_path']
        in_context: bool = params.get('in_context', False)
        
        # 从配置文件读取固定参数
        cfg = get_flowedit_config()
        num_inference_steps = int(cfg.get("num_inference_steps", 20))
        src_guidance_scale = float(cfg.get("src_guidance_scale", 1.5))
        tar_guidance_scale = float(cfg.get("tar_guidance_scale", 5.5))
        # 整图编辑的 n_max/n_min 使用不同默认值
        n_max = int(cfg.get("n_max", 20))
        n_min = int(cfg.get("n_min", 10))
        seed = int(cfg.get("seed", 42))
        model_name = cfg.get("model_name") or 'Tongyi-MAI/Z-Image-Turbo'
        device = cfg.get("device") or ('cuda' if torch.cuda.is_available() else 'cpu')

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        log_agent_info(
            "zimage_flowedit_tool",
            f"start | src={src_image_path} -> out={output_path}, model={model_name}, device={device}, in_context={in_context}",
        )

        image = Image.open(src_image_path).convert('RGB')
        pipe = _get_zimage_pipe(model_name, device)

        edited = FlowEditZImage(
            pipe=pipe,
            x_src_image=image,
            src_prompt=src_prompt,
            tar_prompt=tar_prompt,
            num_inference_steps=num_inference_steps,
            src_guidance_scale=src_guidance_scale,
            tar_guidance_scale=tar_guidance_scale,
            n_max=n_max,
            n_min=n_min,
            seed=seed,
            in_context=in_context,
        )
        edited.save(output_path)

        log_agent_success("zimage_flowedit_tool", f"saved edited image to {output_path}")
        return json.dumps({'output_path': output_path}, ensure_ascii=False)
