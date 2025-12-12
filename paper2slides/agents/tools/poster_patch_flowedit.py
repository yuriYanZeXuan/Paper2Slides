import json
import os
from typing import Tuple, Union

from PIL import Image
import torch
from diffusers import ZImagePipeline

from qwen_agent.tools.base import BaseTool, register_tool

from paper2slides.agents.tools.zimage_flowedit_core import FlowEditZImage
from paper2slides.utils.agent_logging import log_agent_info, log_agent_success


BBox = Tuple[int, int, int, int]

_PIPE_CACHE: dict[str, ZImagePipeline] = {}


def _get_zimage_pipe(model_name: str, device: str) -> ZImagePipeline:
    key = f"{model_name}@{device}"
    if key in _PIPE_CACHE:
        return _PIPE_CACHE[key]

    pipe = ZImagePipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
        low_cpu_mem_usage=False,
    )
    pipe.to(device)
    _PIPE_CACHE[key] = pipe
    return pipe


@register_tool("poster_patch_flowedit")
class PosterPatchFlowEdit(BaseTool):
    """对整图的一个 bbox 做局部放大+FlowEdit，并回填到整图。

    该工具是闭环的关键：agent 只需提供 image_path、bbox、prompts、output_path，
    即可获得一张更新后的整图，不需要在 LLM 侧理解拼接逻辑。
    """

    description = "Perform FlowEdit on a bbox patch and paste it back to the full poster image."
    parameters = {
        "type": "object",
        "properties": {
            "image_path": {"type": "string", "description": "Path to current full poster image."},
            "bbox": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "BBox [x0,y0,x1,y1] in pixels of the full image.",
            },
            "src_prompt": {"type": "string", "description": "Source prompt describing current image/region."},
            "tar_prompt": {"type": "string", "description": "Target prompt describing desired edit in this region."},
            "output_image_path": {"type": "string", "description": "Where to save the updated full image."},
            "upscale_factor": {"type": "integer", "default": 2, "description": "Upscale factor for the bbox patch."},
            "num_inference_steps": {"type": "integer", "default": 20},
            "src_guidance_scale": {"type": "number", "default": 1.5},
            "tar_guidance_scale": {"type": "number", "default": 5.5},
            "n_max": {"type": "integer", "default": 18},
            "n_min": {"type": "integer", "default": 0},
            "seed": {"type": "integer", "default": 42},
            "model_name": {
                "type": "string",
                "default": "Tongyi-MAI/Z-Image-Turbo",
                "description": "Z-Image model name to use.",
            },
            "device": {
                "type": "string",
                "default": "",
                "description": "Device to run on, e.g. cuda/cpu. Default auto.",
            },
        },
        "required": ["image_path", "bbox", "src_prompt", "tar_prompt", "output_image_path"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)

        image_path: str = params["image_path"]
        bbox_raw = params["bbox"]
        src_prompt: str = params["src_prompt"]
        tar_prompt: str = params["tar_prompt"]
        output_image_path: str = params["output_image_path"]

        assert isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4, f"invalid bbox: {bbox_raw}"
        x0, y0, x1, y1 = map(int, bbox_raw)
        bbox: BBox = (x0, y0, x1, y1)

        upscale_factor = int(params.get("upscale_factor", 2))
        num_inference_steps = int(params.get("num_inference_steps", 20))
        src_guidance_scale = float(params.get("src_guidance_scale", 1.5))
        tar_guidance_scale = float(params.get("tar_guidance_scale", 5.5))
        n_max = int(params.get("n_max", 18))
        n_min = int(params.get("n_min", 0))
        seed = int(params.get("seed", 42))
        model_name = str(params.get("model_name") or "Tongyi-MAI/Z-Image-Turbo")
        device = str(params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

        os.makedirs(os.path.dirname(output_image_path) or ".", exist_ok=True)

        log_agent_info(
            "poster_patch_flowedit",
            f"start | img={image_path} bbox={bbox} -> out={output_image_path}, model={model_name}, device={device}",
        )

        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, f"bbox out of bounds: {bbox}, image_size=({w},{h})"

        crop = image.crop((x0, y0, x1, y1))
        if upscale_factor > 1:
            crop = crop.resize(((x1 - x0) * upscale_factor, (y1 - y0) * upscale_factor), Image.LANCZOS)

        pipe = _get_zimage_pipe(model_name=model_name, device=device)

        edited = FlowEditZImage(
            pipe=pipe,
            x_src_image=crop,
            src_prompt=src_prompt,
            tar_prompt=tar_prompt,
            num_inference_steps=num_inference_steps,
            src_guidance_scale=src_guidance_scale,
            tar_guidance_scale=tar_guidance_scale,
            n_max=n_max,
            n_min=n_min,
            seed=seed,
        )

        # resize back to original bbox size
        edited = edited.resize((x1 - x0, y1 - y0), Image.LANCZOS)

        out = image.copy()
        out.paste(edited, (x0, y0, x1, y1))
        out.save(output_image_path)

        log_agent_success("poster_patch_flowedit", f"saved updated image to {output_image_path}")
        return json.dumps({"output_image_path": output_image_path}, ensure_ascii=False)

