import os
import torch
from diffusers import ZImagePipeline
from PIL import Image

from paper2slides.agents.tools.zimage_flowedit_core import FlowEditZImage


if __name__ == "__main__":
    # Load Pipeline（本地 Z-Image 权重，路径与 run_poster.sh 保持一致）
    model_path = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image"
    try:
        pipe = ZImagePipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        pipe.to("cuda")
        # pipe.enable_model_cpu_offload() # Optional
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        exit(1)

    # Load Source Image (Create dummy if not exists)
    import os
    os.makedirs("Paper2Slides/dev/outputs", exist_ok=True)
    
    # Just create a simple image for testing if none exists
    # Or generate one
    print("Generating source image...")
    src_prompt = "A photo of a cat sitting on a bench."
    src_image = pipe(
        prompt=src_prompt,
        height=1024,
        width=1024,
        num_inference_steps=20, # Fast gen
        guidance_scale=0.0, # Turbo default
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    src_image.save("Paper2Slides/dev/outputs/src_image.png")
    
    # Edit
    print("Running FlowEdit...")
    tar_prompt = "A photo of a dog sitting on a bench."
    
    res_image = FlowEditZImage(
        pipe=pipe,
        x_src_image=src_image,
        src_prompt=src_prompt,
        tar_prompt=tar_prompt,
        num_inference_steps=20, # Match gen steps or similar
        src_guidance_scale=1.5, # Guidance usually needed for edit? Z-Image Turbo uses 0 for gen, but maybe 1.5 for edit guidance?
        tar_guidance_scale=5.5,
        n_max=18, # Stop editing at last few steps
        n_min=0,
        seed=42
    )
    
    res_image.save("Paper2Slides/dev/outputs/flowedit_result.png")
    print("Saved result to Paper2Slides/dev/outputs/flowedit_result.png")
