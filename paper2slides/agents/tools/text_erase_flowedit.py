"""
Text Erase FlowEdit Tool

使用 Z-Image FlowEdit 擦除图像中的模糊文字区域，生成干净的背景。
这是新 refine 流程的关键工具：先擦除模糊文字，再用 PPTX 渲染清晰文字叠加。
"""

import json
import os
from typing import Tuple, Union, List

from PIL import Image
import torch
from diffusers import ZImagePipeline

from qwen_agent.tools.base import BaseTool, register_tool

from paper2slides.agents.tools.config_loader import get_flowedit_config
from paper2slides.agents.tools.zimage_flowedit_core import FlowEditZImage
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

_PIPE_CACHE: dict[str, ZImagePipeline] = {}


def _get_zimage_pipe(model_name: str, device: str) -> ZImagePipeline:
    """获取或缓存 Z-Image pipeline"""
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


def erase_text_region(
    image: Image.Image,
    bbox: BBox,
    background_description: str = "clean background with seamless texture",
    model_name: str = None,
    device: str = None,
    max_resolution: int = 1024,
) -> Image.Image:
    """擦除单个文字区域，返回更新后的完整图像。
    
    Args:
        image: 完整图像
        bbox: 要擦除的文字区域 [x0, y0, x1, y1]
        background_description: 背景描述（用于生成干净背景）
        model_name: Z-Image 模型名称
        device: 设备
        max_resolution: 最大分辨率
    
    Returns:
        更新后的完整图像
    """
    cfg = get_flowedit_config()
    model_name = model_name or cfg.get("model_name") or "Tongyi-MAI/Z-Image-Turbo"
    device = device or cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    num_inference_steps = int(cfg.get("num_inference_steps", 20))
    src_guidance_scale = float(cfg.get("src_guidance_scale", 1.5))
    tar_guidance_scale = float(cfg.get("tar_guidance_scale", 5.5))
    n_max = int(cfg.get("n_max", 18))
    n_min = int(cfg.get("n_min", 0))
    seed = int(cfg.get("seed", 42))
    
    x0, y0, x1, y1 = bbox
    w, h = image.size
    
    # 验证 bbox
    assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, f"bbox out of bounds: {bbox}, image_size=({w},{h})"
    
    # 裁剪区域
    crop = image.crop((x0, y0, x1, y1))
    crop_w, crop_h = crop.size
    
    # 按最长边缩放到 max_resolution，保持长宽比
    if max(crop_w, crop_h) < max_resolution:
        scale = max_resolution / max(crop_w, crop_h)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        crop = crop.resize((new_w, new_h), Image.LANCZOS)
    
    # 构造擦除 prompt
    # src_prompt 描述当前有文字的状态
    src_prompt = "text content with characters and words on the image"
    # tar_prompt 描述期望的干净背景状态
    tar_prompt = f"clean empty background without any text, {background_description}"
    
    pipe = _get_zimage_pipe(model_name=os.getenv("LOCAL_IMAGE_MODEL", model_name), device=device)
    
    # 使用 FlowEdit 擦除文字
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
        in_context=True,  # 启用 in-context 保持背景一致性
    )
    
    # 缩放回原始尺寸
    edited = edited.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    
    # 粘贴回原图
    out = image.copy()
    out.paste(edited, (x0, y0, x1, y1))
    
    return out


def erase_multiple_text_regions(
    image: Image.Image,
    bboxes: List[BBox],
    background_description: str = "clean background with seamless texture",
    model_name: str = None,
    device: str = None,
    max_resolution: int = 1024,
) -> Image.Image:
    """擦除多个文字区域，返回更新后的完整图像。
    
    按区域面积从大到小排序处理，以获得更好的擦除效果。
    """
    # 按面积从大到小排序
    sorted_bboxes = sorted(
        bboxes,
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
        reverse=True
    )
    
    current_image = image
    for i, bbox in enumerate(sorted_bboxes):
        log_agent_info("text_erase", f"erasing region {i+1}/{len(bboxes)}: bbox={bbox}")
        current_image = erase_text_region(
            current_image,
            bbox,
            background_description=background_description,
            model_name=model_name,
            device=device,
            max_resolution=max_resolution,
        )
    
    return current_image


@register_tool("text_erase_flowedit")
class TextEraseFlowEdit(BaseTool):
    """使用 Z-Image FlowEdit 擦除图像中的文字区域，生成干净背景。
    
    这是新 refine 流程的关键工具：先擦除模糊文字，再用 PPTX 渲染清晰文字叠加。
    支持单个或多个文字区域的擦除。
    """

    description = (
        "Erase text regions from an image using Z-Image FlowEdit to create a clean background. "
        "Supports erasing single or multiple text regions. "
        "The result is a clean background image without the original text."
    )
    
    parameters = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the source image file.",
            },
            "bboxes": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "description": "List of bounding boxes [[x0, y0, x1, y1], ...] for text regions to erase.",
            },
            "output_path": {
                "type": "string",
                "description": "Where to save the erased background image.",
            },
            "background_description": {
                "type": "string",
                "description": "Description of the desired background (e.g., 'clean academic poster background').",
                "default": "clean background with seamless texture",
            },
        },
        "required": ["image_path", "bboxes", "output_path"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self._verify_json_format_args(params)
        except Exception as e:
            log_agent_error("text_erase_flowedit", f"invalid params: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

        image_path: str = params["image_path"]
        bboxes_raw = params["bboxes"]
        output_path: str = params["output_path"]
        background_description: str = params.get(
            "background_description", "clean background with seamless texture"
        )

        # 验证 bboxes
        bboxes: List[BBox] = []
        for bbox_raw in bboxes_raw:
            assert isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4, f"invalid bbox: {bbox_raw}"
            bboxes.append(tuple(map(int, bbox_raw)))

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        log_agent_info(
            "text_erase_flowedit",
            f"start | image={image_path}, num_bboxes={len(bboxes)} -> output={output_path}",
        )

        image = Image.open(image_path).convert("RGB")
        
        # 保存 bbox 可视化
        save_bbox_visualization(
            agent_name="text_erase_flowedit",
            func_name="erase_regions",
            image=image,
            bboxes=bboxes,
        )

        # 擦除所有文字区域
        erased_image = erase_multiple_text_regions(
            image=image,
            bboxes=bboxes,
            background_description=background_description,
        )

        erased_image.save(output_path)

        # 保存 before/after 对比
        save_before_after_image(
            agent_name="text_erase_flowedit",
            func_name="erase_result",
            before_img=image,
            after_img=erased_image,
        )
        
        # 保存日志
        save_json_log(
            agent_name="text_erase_flowedit",
            func_name="erase_result",
            payload={
                "image_path": image_path,
                "output_path": output_path,
                "bboxes": bboxes,
                "num_regions": len(bboxes),
                "background_description": background_description,
            },
        )

        log_agent_success("text_erase_flowedit", f"saved erased background to {output_path}")
        
        return json.dumps({
            "output_path": output_path,
            "num_regions_erased": len(bboxes),
        }, ensure_ascii=False)

