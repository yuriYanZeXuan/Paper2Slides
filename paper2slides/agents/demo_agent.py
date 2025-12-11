"""Demo script for PosterRefinerAgent.

用法示例（在项目根目录运行）：

    python -m paper2slides.agents.demo_agent

该 demo 会：
1. 加载 Z-Image pipeline
2. 用一个简单 prompt 生成初始海报
3. 通过 PosterRefinerAgent 对小文字区域进行一次 FlowEdit 增强
4. 将中间结果保存在 `dev/outputs/` 目录下
"""

import os

from PIL import Image
import torch

from diffusers import ZImagePipeline

from paper2slides.utils.logging import get_logger
from paper2slides.agents.poster_refiner import PosterRefinerAgent


logger = get_logger(__name__)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载 Z-Image pipeline
    logger.info("Loading Z-Image pipeline for demo ...")
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
        low_cpu_mem_usage=False,
    )
    pipe.to(device)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dev", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # 2. 生成一张初始海报
    src_prompt = "An academic poster layout with multiple text boxes and a title, small fonts in the lower area."
    logger.info("Generating base poster image ...")
    base_image = pipe(
        prompt=src_prompt,
        height=1024,
        width=1024,
        num_inference_steps=20,
        guidance_scale=0.0,
        generator=torch.Generator(device).manual_seed(42),
    ).images[0]
    base_path = os.path.join(out_dir, "demo_base_poster.png")
    base_image.save(base_path)
    logger.info(f"Base poster saved to {base_path}")

    # 3. 构造 PosterRefinerAgent，并进行小文字区域增强
    agent = PosterRefinerAgent(device=device)

    refined = agent.run(
        image=base_image,
        src_prompt=src_prompt,
        tar_prompt_for_text="The small text in the lower area is very sharp and high-resolution, easy to read.",
        clarity_threshold=10.0,  # 强制执行一次增强，方便可视化
    )

    # 4. 保存增强后的结果
    refined_path = os.path.join(out_dir, "demo_refined_poster.png")
    refined.save(refined_path)
    logger.info(f"Refined poster saved to {refined_path}")


if __name__ == "__main__":
    main()
