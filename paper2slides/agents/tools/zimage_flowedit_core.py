import torch
from diffusers import ZImagePipeline
from diffusers.pipelines.z_image.pipeline_z_image import retrieve_timesteps, calculate_shift
from tqdm import tqdm
from PIL import Image

from paper2slides.utils.agent_logging import (
    log_agent_start,
    log_agent_info,
    log_agent_success,
)


def calc_v_zimage(pipe, latents, prompt_embeds_list, negative_prompt_embeds_list, guidance_scale, t,
                  cfg_normalization: float | bool = False, cfg_truncation: float = 1.0):
    """Z-Image 的单步 velocity 计算，抽取自原 pipeline 的 CFG 逻辑。

    - `prompt_embeds_list` / `negative_prompt_embeds_list` 都是 List[Tensor]
    - 与 `ZImagePipeline.__call__` 中的实现保持一致：通过 list 相加做 CFG
    """
    # broadcast timestep
    timestep = t.expand(latents.shape[0])
    timestep = (1000 - timestep) / 1000
    t_norm = timestep[0].item()

    # cfg truncation
    current_guidance_scale = guidance_scale
    if pipe.do_classifier_free_guidance and cfg_truncation is not None and float(cfg_truncation) <= 1:
        if t_norm > cfg_truncation:
            current_guidance_scale = 0.0

    apply_cfg = pipe.do_classifier_free_guidance and current_guidance_scale > 0

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

            pred = pos + current_guidance_scale * (pos - neg)

            if cfg_normalization and float(cfg_normalization) > 0.0:
                ori_pos_norm = torch.linalg.vector_norm(pos)
                new_pos_norm = torch.linalg.vector_norm(pred)
                max_new_norm = ori_pos_norm * float(cfg_normalization)
                if new_pos_norm > max_new_norm:
                    pred = pred * (max_new_norm / new_pos_norm)

            noise_pred.append(pred)

        noise_pred = torch.stack(noise_pred, dim=0)
    else:
        noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

    noise_pred = noise_pred.squeeze(2)
    noise_pred = -noise_pred

    return noise_pred


def _calc_v_zimage_incontext(
    pipe, latents_concat, prompt_embeds_list, negative_prompt_embeds_list, 
    guidance_scale, t, ref_width, cfg_normalization: float | bool = False, cfg_truncation: float = 1.0
):
    """In-context aware velocity 计算：处理拼接后的 latent，只返回编辑部分的 velocity。
    
    latents_concat: 拼接后的 latent [edit_latent | ref_latent]
    ref_width: 参考图像在 latent 空间的宽度
    """
    # 使用完整拼接 latent 计算 velocity
    full_velocity = calc_v_zimage(
        pipe, latents_concat, prompt_embeds_list, negative_prompt_embeds_list,
        guidance_scale, t, cfg_normalization, cfg_truncation
    )
    # 只返回编辑部分的 velocity（左侧）
    edit_width = latents_concat.shape[3] - ref_width
    return full_velocity[:, :, :, :edit_width]


@torch.no_grad()
def FlowEditZImage(
    pipe: ZImagePipeline,
    x_src_image: Image.Image,
    src_prompt: str,
    tar_prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 5.0,
    tar_guidance_scale: float = 5.0,
    n_min: int = 0,
    n_max: int = 20,
    seed: int = 42,
    in_context: bool = False,
) -> Image.Image:
    """Z-Image 的 FlowEdit inversion-free 编辑核心逻辑。

    这里仅保留必要参数，方便作为独立工具被调用。
    
    Args:
        in_context: 是否启用 In-Context-Aware 增强。
            启用后，会将原始参考图像拼接在噪声去噪过程的右侧作为清晰参考，
            使用 mask 确保参考图像部分在每个时间步恢复原状，不受模型输出影响。
            这有助于保持编辑结果与原图的一致性。
    """
    agent = "zimage_flowedit_core"
    log_agent_start(agent)

    device = pipe._execution_device

    # 1. encode prompts
    log_agent_info(agent, f"encode prompts | src_g={src_guidance_scale}, tar_g={tar_guidance_scale}, in_context={in_context}")
    pipe._guidance_scale = src_guidance_scale
    src_prompt_embeds, src_negative_prompt_embeds = pipe.encode_prompt(
        prompt=src_prompt,
        negative_prompt=negative_prompt,
        device=device,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
    )

    pipe._guidance_scale = tar_guidance_scale
    tar_prompt_embeds, tar_negative_prompt_embeds = pipe.encode_prompt(
        prompt=tar_prompt,
        negative_prompt=negative_prompt,
        device=device,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
    )

    # 2. encode source image -> latent x_src
    image = pipe.image_processor.preprocess(x_src_image)
    image = image.to(device=device, dtype=pipe.vae.dtype)

    latents = pipe.vae.encode(image).latent_dist.mode()
    shift = getattr(pipe.vae.config, "shift_factor", 0.0)
    scale = getattr(pipe.vae.config, "scaling_factor", 1.0)
    x_src = (latents - shift) * scale
    x_src = x_src.to(dtype=torch.float32)
    
    # In-Context: 保存原始参考 latent（清晰的，用于拼接）
    ref_latent_width = x_src.shape[3]  # latent 空间的宽度
    x_ref = x_src.clone()  # 参考图像的 latent，始终保持清晰

    # 3. timesteps
    log_agent_info(agent, f"prepare timesteps | num_steps={num_inference_steps}, n_min={n_min}, n_max={n_max}")
    
    # 计算 image_seq_len 时需要考虑 in_context 的拼接
    if in_context:
        # 拼接后的总宽度是原来的 2 倍
        concat_width = x_src.shape[3] * 2
        image_seq_len = (x_src.shape[2] // 2) * (concat_width // 2)
    else:
        image_seq_len = (x_src.shape[2] // 2) * (x_src.shape[3] // 2)
    
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

    zt_edit = x_src.clone()

    # 4. FlowEdit 迭代
    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
        if num_inference_steps - i > n_max:
            continue

        t_i = t / 1000.0
        if i + 1 < len(timesteps):
            t_im1 = timesteps[i + 1] / 1000.0
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)

        if num_inference_steps - i > n_min:
            # 编辑阶段
            V_delta_avg = torch.zeros_like(x_src)

            for _ in range(n_avg):
                fwd_noise = torch.randn_like(x_src).to(device)
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src

                if in_context:
                    # In-Context: 将参考图像拼接在右侧
                    # 参考图像始终保持清晰（x_ref），不加噪声
                    zt_src_concat = torch.cat([zt_src, x_ref], dim=3)  # [B, C, H, W*2]
                    zt_tar_concat = torch.cat([zt_tar, x_ref], dim=3)
                    
                    vt_src = _calc_v_zimage_incontext(
                        pipe,
                        zt_src_concat,
                        src_prompt_embeds,
                        src_negative_prompt_embeds,
                        src_guidance_scale,
                        t,
                        ref_latent_width,
                    )
                    vt_tar = _calc_v_zimage_incontext(
                        pipe,
                        zt_tar_concat,
                        tar_prompt_embeds,
                        tar_negative_prompt_embeds,
                        tar_guidance_scale,
                        t,
                        ref_latent_width,
                    )
                else:
                    vt_src = calc_v_zimage(
                        pipe,
                        zt_src,
                        src_prompt_embeds,
                        src_negative_prompt_embeds,
                        src_guidance_scale,
                        t,
                    )
                    vt_tar = calc_v_zimage(
                        pipe,
                        zt_tar,
                        tar_prompt_embeds,
                        tar_negative_prompt_embeds,
                        tar_guidance_scale,
                        t,
                    )

                V_delta_avg += (vt_tar - vt_src) / max(n_avg, 1)

            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
        else:
            # 收尾生成阶段（可选）
            if i == num_inference_steps - n_min:
                fwd_noise = torch.randn_like(x_src).to(device)
                xt_src = (1 - t_i) * x_src + t_i * fwd_noise
                xt_tar = zt_edit + xt_src - x_src

            if in_context:
                # In-Context: 收尾阶段也拼接参考图像
                xt_tar_concat = torch.cat([xt_tar, x_ref], dim=3)
                vt_tar = _calc_v_zimage_incontext(
                    pipe,
                    xt_tar_concat,
                    tar_prompt_embeds,
                    tar_negative_prompt_embeds,
                    tar_guidance_scale,
                    t,
                    ref_latent_width,
                )
            else:
                vt_tar = calc_v_zimage(
                    pipe,
                    xt_tar,
                    tar_prompt_embeds,
                    tar_negative_prompt_embeds,
                    tar_guidance_scale,
                    t,
                )
            prev_sample = xt_tar + (t_im1 - t_i) * vt_tar
            xt_tar = prev_sample

    latents_out = zt_edit if n_min == 0 else xt_tar
    latents_out = latents_out.to(pipe.vae.dtype)
    latents_out = (latents_out / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

    image_out = pipe.vae.decode(latents_out, return_dict=False)[0]
    image_out = pipe.image_processor.postprocess(image_out, output_type="pil")[0]

    log_agent_success(agent, f"FlowEditZImage finished (in_context={in_context})")
    return image_out
