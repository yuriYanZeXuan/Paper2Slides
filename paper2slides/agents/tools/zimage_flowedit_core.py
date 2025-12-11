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
) -> Image.Image:
    """Z-Image 的 FlowEdit inversion-free 编辑核心逻辑。

    这里仅保留必要参数，方便作为独立工具被调用。
    """
    agent = "zimage_flowedit_core"
    log_agent_start(agent)

    device = pipe._execution_device

    # 1. encode prompts
    log_agent_info(agent, f"encode prompts | src_g={src_guidance_scale}, tar_g={tar_guidance_scale}")
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

    # 3. timesteps
    log_agent_info(agent, f"prepare timesteps | num_steps={num_inference_steps}, n_min={n_min}, n_max={n_max}")
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

    log_agent_success(agent, "FlowEditZImage finished")
    return image_out
