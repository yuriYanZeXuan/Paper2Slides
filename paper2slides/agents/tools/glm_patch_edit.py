import json
import os
from typing import Tuple, Union

from PIL import Image
import torch
from diffusers.pipelines.glm_image import GlmImagePipeline

from qwen_agent.tools.base import BaseTool, register_tool

from paper2slides.agents.tools.config_loader import get_glm_image_config
from paper2slides.utils.agent_artifact_logging import (
    save_before_after_image,
    save_bbox_visualization,
    save_json_log,
)
from paper2slides.utils.agent_logging import (
    log_agent_info,
    log_agent_success,
    log_agent_error,
)


BBox = Tuple[int, int, int, int]

_PIPE_CACHE: dict[str, GlmImagePipeline] = {}


def _get_glm_pipe(model_name: str, device: str) -> GlmImagePipeline:
    key = f"{model_name}@{device}"
    if key in _PIPE_CACHE:
        return _PIPE_CACHE[key]

    torch_dtype = torch.bfloat16 if "cuda" in device else torch.float32
    pipe = GlmImagePipeline.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device)
    _PIPE_CACHE[key] = pipe
    return pipe


def _align_to_multiple(value: int, multiple: int = 32) -> int:
    if multiple <= 1:
        return int(value)
    return max(multiple, int(round(value / multiple)) * multiple)


@register_tool("glm_patch_edit")
class GLMPatchEdit(BaseTool):
    """对整图的一个 bbox 做局部放大 + GLM-Image 编辑，并回填到整图。

    用于小字区域放大后精修，兼顾清晰度与整体风格一致。
    """

    description = "Edit a bbox patch with GLM-Image and paste it back to the full poster image."
    parameters = {
        "type": "object",
        "properties": {
            "image_path": {"type": "string", "description": "Path to current full poster image."},
            "bbox": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "BBox [x0,y0,x1,y1] in pixels of the full image.",
            },
            "prompt": {"type": "string", "description": "Prompt describing the desired edit in this region."},
            "output_image_path": {"type": "string", "description": "Where to save the updated full image."},
        },
        "required": ["image_path", "bbox", "prompt", "output_image_path"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self._verify_json_format_args(params)
        except Exception as e:
            log_agent_error("glm_patch_edit", f"invalid params: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

        image_path: str = params["image_path"]
        bbox_raw = params["bbox"]
        prompt: str = params["prompt"]
        output_image_path: str = params["output_image_path"]

        assert isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4, f"invalid bbox: {bbox_raw}"
        x0, y0, x1, y1 = map(int, bbox_raw)
        bbox: BBox = (x0, y0, x1, y1)

        cfg = get_glm_image_config()
        max_resolution = int(cfg.get("max_resolution", 1024))
        num_inference_steps = int(cfg.get("num_inference_steps", 50))
        guidance_scale = float(cfg.get("guidance_scale", 1.5))
        seed = int(cfg.get("seed", 42))
        model_name = cfg.get("model_name") or "THUDM/GLM-Image-1.0"
        device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(os.path.dirname(output_image_path) or ".", exist_ok=True)

        log_agent_info(
            "glm_patch_edit",
            f"start | img={image_path} bbox={bbox} -> out={output_image_path}, model={model_name}, device={device}, max_res={max_resolution}",
        )

        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, f"bbox out of bounds: {bbox}, image_size=({w},{h})"

        crop = image.crop((x0, y0, x1, y1))
        crop_w, crop_h = crop.size

        if max(crop_w, crop_h) < max_resolution:
            scale = max_resolution / max(crop_w, crop_h)
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            crop = crop.resize((new_w, new_h), Image.LANCZOS)

        aligned_w = _align_to_multiple(crop.size[0], 32)
        aligned_h = _align_to_multiple(crop.size[1], 32)
        if (aligned_w, aligned_h) != crop.size:
            crop = crop.resize((aligned_w, aligned_h), Image.LANCZOS)

        pipe = _get_glm_pipe(model_name=os.getenv("LOCAL_GLM_IMAGE_MODEL", model_name), device=device)

        generator = torch.Generator(device=device).manual_seed(seed)
        edited = pipe(
            prompt=prompt,
            image=[crop],
            height=aligned_h,
            width=aligned_w,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        edited = edited.resize((x1 - x0, y1 - y0), Image.LANCZOS)

        out = image.copy()
        out.paste(edited, (x0, y0, x1, y1))
        out.save(output_image_path)

        save_bbox_visualization(
            agent_name="glm_patch_edit",
            func_name="bbox_region",
            image=image,
            bboxes=[bbox],
            suffix=f"x{x0}_y{y0}",
        )
        save_before_after_image(
            agent_name="glm_patch_edit",
            func_name="patch_edit",
            before_img=image.crop((x0, y0, x1, y1)),
            after_img=edited,
            suffix=f"x{x0}_y{y0}",
        )
        save_before_after_image(
            agent_name="glm_patch_edit",
            func_name="full_image",
            before_img=image,
            after_img=out,
            suffix=f"x{x0}_y{y0}",
        )
        save_json_log(
            agent_name="glm_patch_edit",
            func_name="full_image",
            payload={
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "prompt": prompt,
                "crop_size": (crop_w, crop_h),
                "edited_size": edited.size,
                "output_image_path": output_image_path,
            },
            suffix=f"x{x0}_y{y0}",
        )
        log_agent_success("glm_patch_edit", f"saved updated image to {output_image_path}")
        return json.dumps({"output_image_path": output_image_path}, ensure_ascii=False)
