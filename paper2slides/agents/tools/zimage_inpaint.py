"""
Z-Image Inpainting Tool

使用 Z-Image 进行 inpainting：
- mask 区域：从纯噪声开始去噪（完全重新生成）
- 非 mask 区域：保持原始图像内容

这与 FlowEdit 不同：FlowEdit 是基于 prompt 差异的编辑，
而 inpainting 是在指定区域完全重新生成内容。
"""

import json
import os
from typing import Tuple, Union, List, Optional

import numpy as np
from PIL import Image
import torch
from diffusers import ZImagePipeline
from diffusers.pipelines.z_image.pipeline_z_image import retrieve_timesteps, calculate_shift
from tqdm import tqdm

from qwen_agent.tools.base import BaseTool, register_tool

from paper2slides.agents.tools.config_loader import get_flowedit_config
from paper2slides.utils.agent_artifact_logging import (
    save_before_after_image,
    save_bbox_visualization,
    save_json_log,
)
from paper2slides.utils.agent_logging import (
    log_agent_info,
    log_agent_success,
    log_agent_error,
    log_agent_start,
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


def create_mask_from_bboxes(
    image_size: Tuple[int, int],
    bboxes: List[BBox],
    latent_scale: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """从 bboxes 创建 mask
    
    Args:
        image_size: (width, height) 图像尺寸
        bboxes: 要 inpaint 的区域列表
        latent_scale: latent 空间的缩放比例（通常为 8）
    
    Returns:
        image_mask: [1, 1, H, W] 图像空间的 mask
        latent_mask: [1, 1, H//8, W//8] latent 空间的 mask
    """
    w, h = image_size
    
    # 创建图像空间 mask
    mask = np.zeros((h, w), dtype=np.float32)
    for x0, y0, x1, y1 in bboxes:
        mask[y0:y1, x0:x1] = 1.0
    
    image_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # 创建 latent 空间 mask（使用 max pooling 确保覆盖）
    latent_h = h // latent_scale
    latent_w = w // latent_scale
    
    # 使用 max pooling 降采样到 latent 空间
    latent_mask = torch.nn.functional.max_pool2d(
        image_mask, 
        kernel_size=latent_scale, 
        stride=latent_scale
    )
    
    return image_mask, latent_mask


def calc_v_zimage(pipe, latents, prompt_embeds_list, negative_prompt_embeds_list, guidance_scale, t):
    """Z-Image 的单步 velocity 计算"""
    timestep = t.expand(latents.shape[0])
    timestep = (1000 - timestep) / 1000

    apply_cfg = pipe.do_classifier_free_guidance and guidance_scale > 0

    if apply_cfg:
        latents_typed = latents.to(pipe.transformer.dtype)
        latent_model_input = latents_typed.repeat(2, 1, 1, 1)
        prompt_embeds_model_input = prompt_embeds_list + negative_prompt_embeds_list
        timestep_model_input = timestep.repeat(2)
    else:
        latent_model_input = latents.to(pipe.transformer.dtype)
        prompt_embeds_model_input = prompt_embeds_list
        timestep_model_input = timestep

    # Z-Image: 增加 frame 维度
    latent_model_input = latent_model_input.unsqueeze(2)
    latent_model_input_list = list(latent_model_input.unbind(dim=0))

    model_out_list = pipe.transformer(
        latent_model_input_list, timestep_model_input, prompt_embeds_model_input, return_dict=False
    )[0]

    actual_batch_size = latents.shape[0]

    if apply_cfg:
        pos_out = model_out_list[:actual_batch_size]
        neg_out = model_out_list[actual_batch_size:]

        noise_pred = []
        for j in range(actual_batch_size):
            pos = pos_out[j].float()
            neg = neg_out[j].float()
            pred = pos + guidance_scale * (pos - neg)
            noise_pred.append(pred)

        noise_pred = torch.stack(noise_pred, dim=0)
    else:
        noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

    noise_pred = noise_pred.squeeze(2)
    noise_pred = -noise_pred

    return noise_pred


@torch.no_grad()
def zimage_inpaint(
    pipe: ZImagePipeline,
    image: Image.Image,
    bboxes: List[BBox],
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    strength: float = 1.0,
    seed: int = 42,
) -> Image.Image:
    """Z-Image Inpainting 核心逻辑
    
    Args:
        pipe: ZImagePipeline
        image: 原始图像
        bboxes: 要 inpaint 的区域列表
        prompt: 生成内容的描述
        negative_prompt: 负面提示
        num_inference_steps: 去噪步数
        guidance_scale: CFG 强度
        strength: inpaint 强度 (0-1)，1.0 表示完全从噪声开始
        seed: 随机种子
    
    Returns:
        inpainted 图像
    """
    agent = "zimage_inpaint"
    log_agent_start(agent)
    
    device = pipe._execution_device
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # 1. Encode prompt
    log_agent_info(agent, f"encoding prompt | guidance_scale={guidance_scale}")
    pipe._guidance_scale = guidance_scale
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        device=device,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
    )
    
    # 2. Encode source image
    image_tensor = pipe.image_processor.preprocess(image)
    image_tensor = image_tensor.to(device=device, dtype=pipe.vae.dtype)
    
    latents = pipe.vae.encode(image_tensor).latent_dist.mode()
    shift = getattr(pipe.vae.config, "shift_factor", 0.0)
    scale = getattr(pipe.vae.config, "scaling_factor", 1.0)
    x0 = (latents - shift) * scale
    x0 = x0.to(dtype=torch.float32)
    
    # 3. Create mask
    log_agent_info(agent, f"creating mask for {len(bboxes)} regions")
    _, latent_mask = create_mask_from_bboxes(image.size, bboxes)
    latent_mask = latent_mask.to(device=device, dtype=x0.dtype)
    
    # 4. Prepare timesteps
    image_seq_len = (x0.shape[2] // 2) * (x0.shape[3] // 2)
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    pipe.scheduler.sigma_min = 0.0
    scheduler_kwargs = {"mu": mu}
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        device,
        **scheduler_kwargs,
    )
    
    # 5. 根据 strength 调整起始步骤
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = timesteps[t_start:]
    
    # 6. 初始化 latent
    # mask 区域从噪声开始，非 mask 区域从原图开始
    noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
    
    if t_start > 0:
        # 如果 strength < 1，需要对原图加噪
        t_0 = timesteps[0] / 1000.0
        noisy_x0 = (1 - t_0) * x0 + t_0 * noise
    else:
        # strength = 1，从纯噪声开始
        noisy_x0 = noise
    
    # 组合：mask 区域用噪声，非 mask 区域用原图
    zt = latent_mask * noisy_x0 + (1 - latent_mask) * x0
    
    # 7. Denoising loop
    log_agent_info(agent, f"denoising | steps={len(timesteps)}, strength={strength}")
    
    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Inpainting"):
        t_i = t / 1000.0
        if i + 1 < len(timesteps):
            t_im1 = timesteps[i + 1] / 1000.0
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)
        
        # 计算 velocity
        vt = calc_v_zimage(
            pipe, zt, prompt_embeds, negative_prompt_embeds, guidance_scale, t
        )
        
        # 更新 latent
        zt_next = zt + (t_im1 - t_i) * vt
        
        # 非 mask 区域保持原图
        # 在每个时间步，非 mask 区域应该是 (1-t_{i+1}) * x0 + t_{i+1} * noise
        if t_im1 > 0:
            noisy_x0_next = (1 - t_im1) * x0 + t_im1 * noise
        else:
            noisy_x0_next = x0
        
        zt = latent_mask * zt_next + (1 - latent_mask) * noisy_x0_next
    
    # 8. Decode
    latents_out = zt.to(pipe.vae.dtype)
    latents_out = (latents_out / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    
    image_out = pipe.vae.decode(latents_out, return_dict=False)[0]
    image_out = pipe.image_processor.postprocess(image_out, output_type="pil")[0]
    
    log_agent_success(agent, f"inpainting finished, {len(bboxes)} regions processed")
    return image_out


def inpaint_text_regions(
    image: Image.Image,
    bboxes: List[BBox],
    prompt: str = "clean empty background, seamless texture, no text",
    negative_prompt: str = "text, letters, words, characters, watermark",
    model_name: str = None,
    device: str = None,
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    strength: float = 1.0,
    seed: int = 42,
) -> Image.Image:
    """使用 Z-Image inpainting 擦除多个文字区域
    
    Args:
        image: 原始图像
        bboxes: 要擦除的文字区域列表
        prompt: 生成背景的描述
        negative_prompt: 负面提示（避免生成文字）
        model_name: Z-Image 模型名称
        device: 设备
        num_inference_steps: 去噪步数
        guidance_scale: CFG 强度
        strength: inpaint 强度
        seed: 随机种子
    
    Returns:
        擦除文字后的图像
    """
    cfg = get_flowedit_config()
    model_name = model_name or cfg.get("model_name") or "Tongyi-MAI/Z-Image-Turbo"
    device = device or cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用环境变量覆盖模型路径
    model_name = os.getenv("LOCAL_IMAGE_MODEL", model_name)
    
    log_agent_info("zimage_inpaint", f"loading model: {model_name}")
    pipe = _get_zimage_pipe(model_name, device)
    
    result = zimage_inpaint(
        pipe=pipe,
        image=image,
        bboxes=bboxes,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        seed=seed,
    )
    
    return result


@register_tool("zimage_inpaint")
class ZImageInpaintTool(BaseTool):
    """使用 Z-Image Inpainting 擦除图像中的指定区域
    
    与 FlowEdit 不同，inpainting 在 mask 区域从纯噪声开始生成，
    非 mask 区域保持原图不变，实现更干净的擦除效果。
    """
    
    description = (
        "Inpaint specified regions in an image using Z-Image. "
        "The masked regions will be regenerated from noise while "
        "the rest of the image remains unchanged. "
        "Ideal for erasing text or unwanted content."
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
                "description": "List of bounding boxes [[x0, y0, x1, y1], ...] for regions to inpaint.",
            },
            "output_path": {
                "type": "string",
                "description": "Where to save the inpainted image.",
            },
            "prompt": {
                "type": "string",
                "description": "Description of what to generate in the masked regions.",
                "default": "clean empty background, seamless texture, no text",
            },
            "negative_prompt": {
                "type": "string",
                "description": "What to avoid generating.",
                "default": "text, letters, words, characters, watermark",
            },
            "strength": {
                "type": "number",
                "description": "Inpainting strength (0-1). 1.0 means fully regenerate from noise.",
                "default": 1.0,
            },
            "num_inference_steps": {
                "type": "integer",
                "description": "Number of denoising steps.",
                "default": 30,
            },
            "guidance_scale": {
                "type": "number",
                "description": "Classifier-free guidance scale.",
                "default": 5.0,
            },
        },
        "required": ["image_path", "bboxes", "output_path"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)

        image_path: str = params["image_path"]
        bboxes_raw = params["bboxes"]
        output_path: str = params["output_path"]
        prompt: str = params.get("prompt", "clean empty background, seamless texture, no text")
        negative_prompt: str = params.get("negative_prompt", "text, letters, words, characters, watermark")
        strength: float = float(params.get("strength", 1.0))
        num_inference_steps: int = int(params.get("num_inference_steps", 30))
        guidance_scale: float = float(params.get("guidance_scale", 5.0))

        # 验证 bboxes
        bboxes: List[BBox] = []
        for bbox_raw in bboxes_raw:
            bboxes.append(tuple(map(int, bbox_raw)))

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        log_agent_info(
            "zimage_inpaint",
            f"start | image={image_path}, num_bboxes={len(bboxes)}, strength={strength}",
        )

        image = Image.open(image_path).convert("RGB")
        
        # 保存 bbox 可视化
        save_bbox_visualization(
            agent_name="zimage_inpaint",
            func_name="inpaint_regions",
            image=image,
            bboxes=bboxes,
        )

        # Inpainting
        result = inpaint_text_regions(
            image=image,
            bboxes=bboxes,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
        )

        result.save(output_path)

        # 保存 before/after 对比
        save_before_after_image(
            agent_name="zimage_inpaint",
            func_name="inpaint_result",
            before_img=image,
            after_img=result,
        )
        
        # 保存日志
        save_json_log(
            agent_name="zimage_inpaint",
            func_name="inpaint_result",
            payload={
                "image_path": image_path,
                "output_path": output_path,
                "bboxes": bboxes,
                "num_regions": len(bboxes),
                "prompt": prompt,
                "strength": strength,
            },
        )

        log_agent_success("zimage_inpaint", f"saved inpainted image to {output_path}")
        
        return json.dumps({
            "output_path": output_path,
            "num_regions_inpainted": len(bboxes),
        }, ensure_ascii=False)

