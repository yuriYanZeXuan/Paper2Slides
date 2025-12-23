import json
import os
from typing import Tuple, Union

from PIL import Image
from Paper2Slides.paper2slides.utils import save_json_log
import torch
from diffusers import ZImagePipeline

from qwen_agent.tools.base import BaseTool, register_tool

from paper2slides.agents.tools.config_loader import get_flowedit_config
from paper2slides.agents.tools.zimage_flowedit_core import FlowEditZImage
from paper2slides.utils.agent_artifact_logging import save_before_after_image, save_bbox_visualization
from paper2slides.utils.agent_logging import log_agent_info, log_agent_success, log_agent_error


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
        },
        "required": ["image_path", "bbox", "src_prompt", "tar_prompt", "output_image_path"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self._verify_json_format_args(params)
        except Exception as e:
            log_agent_error("poster_patch_flowedit", f"invalid params: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

        image_path: str = params["image_path"]
        bbox_raw = params["bbox"]
        src_prompt: str = params["src_prompt"]
        tar_prompt: str = params["tar_prompt"]
        output_image_path: str = params["output_image_path"]

        assert isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4, f"invalid bbox: {bbox_raw}"
        x0, y0, x1, y1 = map(int, bbox_raw)
        bbox: BBox = (x0, y0, x1, y1)

        # 从配置文件读取固定参数
        cfg = get_flowedit_config()
        max_resolution = int(cfg.get("max_resolution", 1024))
        num_inference_steps = int(cfg.get("num_inference_steps", 20))
        src_guidance_scale = float(cfg.get("src_guidance_scale", 1.5))
        tar_guidance_scale = float(cfg.get("tar_guidance_scale", 5.5))
        n_max = int(cfg.get("n_max", 18))
        n_min = int(cfg.get("n_min", 0))
        seed = int(cfg.get("seed", 42))
        model_name = cfg.get("model_name") or "Tongyi-MAI/Z-Image-Turbo"
        device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(os.path.dirname(output_image_path) or ".", exist_ok=True)

        log_agent_info(
            "poster_patch_flowedit",
            f"start | img={image_path} bbox={bbox} -> out={output_image_path}, model={model_name}, device={device}, max_res={max_resolution}",
        )

        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, f"bbox out of bounds: {bbox}, image_size=({w},{h})"

        crop = image.crop((x0, y0, x1, y1))
        crop_w, crop_h = crop.size
        
        # 按最长边缩放到 max_resolution，保持长宽比
        if max(crop_w, crop_h) < max_resolution:
            scale = max_resolution / max(crop_w, crop_h)
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            crop = crop.resize((new_w, new_h), Image.LANCZOS)

        pipe = _get_zimage_pipe(model_name=os.getenv("LOCAL_IMAGE_MODEL", model_name), device=device)

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

        # ============ 保存可视化日志 ============
        # 1. 保存 bbox 可视化：在原图上标注编辑区域
        save_bbox_visualization(
            agent_name="poster_patch_flowedit",
            func_name="bbox_region",
            image=image,
            bboxes=[bbox],
            suffix=f"x{x0}_y{y0}",
        )

        # 2. 保存 patch 级别的 before/after 对比（crop vs edited patch）
        save_before_after_image(
            agent_name="poster_patch_flowedit",
            func_name="patch_edit",
            before_img=image.crop((x0, y0, x1, y1)),
            after_img=edited,
            suffix=f"x{x0}_y{y0}",
        )

        # 3. 保存整图级别的 before/after 对比
        save_before_after_image(
            agent_name="poster_patch_flowedit",
            func_name="full_image",
            before_img=image,
            after_img=out,
            suffix=f"x{x0}_y{y0}",
        )
        save_json_log(
            agent_name="poster_patch_flowedit",
            func_name="full_image",
            payload={
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "src_prompt": src_prompt,
                "tar_prompt": tar_prompt,
                "crop_size": (crop_w, crop_h),
                "edited_size": edited.size,
                "output_image_path": output_image_path,
            },
            suffix=f"{src_prompt[:10]}_{tar_prompt[:10]}",
        )
        log_agent_success("poster_patch_flowedit", f"saved updated image to {output_image_path}")
        return json.dumps({"output_image_path": output_image_path}, ensure_ascii=False)
