"""
Eraser Qwen Editor Tool

用途：用 Qwen-Image-Edit 对整幅图像做一次编辑来擦除文字/figure 内容，
然后可选地将 bbox 外的区域恢复为原图（等价于只应用 bbox 区域的编辑结果）。

接口与日志风格参考 text_erase_flowedit.py：
- 输入：image_path + bboxes + output_path
- 输出：保存擦除后的背景图，并记录 bbox 可视化、before/after、json log
"""

import json
import os
from typing import Any, List, Tuple, Union

import torch
from PIL import Image
import gc

from qwen_agent.tools.base import BaseTool, register_tool

from paper2slides.agents.tools.config_loader import get_config
from paper2slides.utils.agent_artifact_logging import (
    save_before_after_image,
    save_bbox_visualization,
    save_json_log,
)
from paper2slides.utils.agent_logging import (
    log_agent_error,
    log_agent_info,
    log_agent_success,
    log_agent_warning,
)

BBox = Tuple[int, int, int, int]

_PIPE_CACHE: dict[str, object] = {}


def _cfg() -> dict[str, Any]:
    return (get_config() or {}).get("qwen_edit", {}) or {}


def _validate_bbox(image: Image.Image, bbox: BBox) -> None:
    x0, y0, x1, y1 = bbox
    w, h = image.size
    assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, f"bbox out of bounds: {bbox}, image_size=({w},{h})"


def _expand_bbox(image: Image.Image, bbox: BBox, pad_ratio: float) -> BBox:
    if pad_ratio <= 0:
        return bbox
    x0, y0, x1, y1 = bbox
    w, h = image.size
    bw = x1 - x0
    bh = y1 - y0
    pad_x = int(round(bw * pad_ratio))
    pad_y = int(round(bh * pad_ratio))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(w, x1 + pad_x),
        min(h, y1 + pad_y),
    )


def _resize_longest_side(img: Image.Image, max_resolution: int) -> Image.Image:
    w, h = img.size
    if max_resolution <= 0 or max(w, h) == max_resolution:
        return img
    scale = max_resolution / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if (new_w, new_h) == (w, h):
        return img
    return img.resize((new_w, new_h), Image.LANCZOS)


def _default_prompt(cfg: dict[str, Any]) -> str:
    mode = str(cfg.get("mode", "text_and_figure") or "").strip().lower()
    bg = str(cfg.get("background_description", "clean background with seamless texture") or "").strip()
    if mode == "text":
        return f"Remove all text. Fill with {bg}. No text."
    if mode == "figure":
        return f"Remove the figure/chart/diagram content. Fill with {bg}. No text."
    return (
        "Remove all text and any figure/chart/diagram content (including axes, legend, labels, markers). "
        f"Fill with {bg}. No text."
    )


def _get_pipe(model_name: str, device: str):
    key = f"{model_name}@{device}"
    if key in _PIPE_CACHE:
        return _PIPE_CACHE[key]
    from diffusers import QwenImageEditPlusPipeline  # type: ignore
    dtype = torch.bfloat16 if "cuda" in device else torch.float32
    pipe = QwenImageEditPlusPipeline.from_pretrained(model_name, torch_dtype=dtype)
    pipe.to(device)
    _PIPE_CACHE[key] = pipe
    return pipe


def _unload_pipe(model_name: str, device: str) -> None:
    """best-effort 卸载 pipeline 并释放 CUDA 显存。"""
    key = f"{model_name}@{device}"
    pipe = _PIPE_CACHE.pop(key, None)
    if pipe is None:
        return
    try:
        pipe.to("cpu")
    except Exception:
        pass
    try:
        del pipe
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


@torch.inference_mode()
def erase_with_qwen(image: Image.Image, bboxes: List[BBox]) -> tuple[Image.Image, dict[str, Any]]:
    cfg = _cfg()
    model_name = os.getenv("LOCAL_QWEN_EDIT_MODEL") or str(cfg.get("model_name") or "").strip() or "Qwen/Qwen-Image-Edit-2511"
    device = str(cfg.get("device") or "").strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    max_resolution = int(cfg.get("max_resolution", 1024))
    num_inference_steps = int(cfg.get("num_inference_steps", 40))
    guidance_scale = float(cfg.get("guidance_scale", 1.0))
    true_cfg_scale = float(cfg.get("true_cfg_scale", 4.0))
    seed = int(cfg.get("seed", 0))
    restore_outside = bool(cfg.get("restore_outside_bboxes", True))
    pad_ratio = float(cfg.get("pad_ratio", 0.0))
    custom_prompt = str(cfg.get("prompt", "") or "").strip()
    used_prompt = custom_prompt or _default_prompt(cfg)

    if "cuda" in device and not torch.cuda.is_available():
        log_agent_warning("eraser_qwen_editor", f"cuda not available, fallback to cpu (requested device={device})")
        device = "cpu"

    pipe = _get_pipe(model_name=model_name, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)

    try:
        resized = _resize_longest_side(image, max_resolution=max_resolution)
        out = pipe(
            image=[resized],
            prompt=used_prompt,
            generator=gen,
            true_cfg_scale=true_cfg_scale,
            negative_prompt=" ",
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
        )
        edited_full = out.images[0].resize(image.size, Image.LANCZOS)
    finally:
        # 用完立即卸载，避免与其它模型（Z-Image）显存冲突
        _unload_pipe(model_name=model_name, device=device)

    if restore_outside and bboxes:
        final_img = image.copy()
        for bbox in bboxes:
            _validate_bbox(image, bbox)
            x0, y0, x1, y1 = _expand_bbox(image, bbox, pad_ratio=pad_ratio)
            final_img.paste(edited_full.crop((x0, y0, x1, y1)), (x0, y0, x1, y1))
    else:
        final_img = edited_full

    meta = {
        "model_name": model_name,
        "device": device,
        "max_resolution": max_resolution,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "true_cfg_scale": true_cfg_scale,
        "seed": seed,
        "restore_outside_bboxes": restore_outside,
        "pad_ratio": pad_ratio,
        "prompt": custom_prompt,
        "used_prompt": used_prompt,
    }
    return final_img, meta


@register_tool("eraser_qwen_editor")
class EraserQwenEditor(BaseTool):
    description = (
        "Erase text/figure content from an image using Qwen-Image-Edit. "
        "Edits the full image once and optionally restores areas outside bboxes from the original image."
    )

    parameters = {
        "type": "object",
        "properties": {
            "image_path": {"type": "string", "description": "Path to the source image file."},
            "bboxes": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
                "description": "List of bounding boxes [[x0, y0, x1, y1], ...].",
            },
            "output_path": {"type": "string", "description": "Where to save the erased background image."},
        },
        "required": ["image_path", "bboxes", "output_path"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self._verify_json_format_args(params)
        except Exception as e:
            log_agent_error("eraser_qwen_editor", f"invalid params: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

        image_path: str = params["image_path"]
        bboxes_raw = params["bboxes"]
        output_path: str = params["output_path"]

        bboxes: List[BBox] = []
        for bbox_raw in bboxes_raw:
            assert isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4, f"invalid bbox: {bbox_raw}"
            bboxes.append(tuple(map(int, bbox_raw)))

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        log_agent_info("eraser_qwen_editor", f"start | image={image_path}, num_bboxes={len(bboxes)} -> output={output_path}")

        image = Image.open(image_path).convert("RGB")

        save_bbox_visualization(
            agent_name="eraser_qwen_editor",
            func_name="erase_regions",
            image=image,
            bboxes=bboxes,
        )

        erased, meta = erase_with_qwen(image=image, bboxes=bboxes)
        erased.save(output_path)

        save_before_after_image(
            agent_name="eraser_qwen_editor",
            func_name="erase_result",
            before_img=image,
            after_img=erased,
        )

        save_json_log(
            agent_name="eraser_qwen_editor",
            func_name="erase_result",
            payload={
                "image_path": image_path,
                "output_path": output_path,
                "bboxes": bboxes,
                "num_regions": len(bboxes),
                "qwen_edit": meta,
            },
        )

        log_agent_success("eraser_qwen_editor", f"saved erased background to {output_path}")
        return json.dumps({"output_path": output_path, "num_regions_erased": len(bboxes)}, ensure_ascii=False)


