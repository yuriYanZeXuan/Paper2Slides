from typing import List, Tuple

from PIL import Image, ImageDraw
import torch

from diffusers import ZImagePipeline

from paper2slides.utils.logging import get_logger
from paper2slides.agents.tools.zimage_flowedit_core import FlowEditZImage
from paper2slides.utils.agent_logging import (
    log_agent_start,
    log_agent_info,
    log_agent_success,
)
from paper2slides.utils.agent_artifact_logging import (
    save_before_after_image,
    save_json_log,
    get_default_log_root,
)


logger = get_logger(__name__)


BBox = Tuple[int, int, int, int]


_AGENT_NAME = "poster_refiner"
_LOG_ROOT = get_default_log_root(_AGENT_NAME)

class PosterRefinerAgent:
    """Agent that refines poster small-text regions using Z-Image FlowEdit.

    Pipeline (high level):
    1. 接收一张初始海报图（例如由 Z-Image 生成）
    2. 使用 VLM (占位) 评估整体文字清晰度
    3. 使用 grounding (占位) 找到小字/模糊文字区域的 bbox
    4. 对每个 bbox 进行局部放大 + FlowEditZImage 重绘
    5. 将重绘后的 patch 拼回原图
    """

    def __init__(
        self,
        zimage_model_name: str = "Tongyi-MAI/Z-Image-Turbo",
        device: str = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        log_agent_start("poster_refiner_agent")
        logger.info(f"Loading Z-Image pipeline {zimage_model_name} on {self.device} ...")
        self.pipe: ZImagePipeline = ZImagePipeline.from_pretrained(
            zimage_model_name,
            torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
            low_cpu_mem_usage=False,
        )
        self.pipe.to(self.device)

    # ============ 占位 / 简化实现部分（后续可替换为真实 VLM & grounding） ============
    def assess_clarity(self, image: Image.Image) -> float:
        """使用 VLM 评估文字清晰度.

        当前实现是占位: 简单返回 1.0~10.0 的常数, 方便集成和调试.
        后续可替换为真实的 Qwen-VL 调用。
        """

        # TODO: 接入 Qwen-VL / 其他 VLM, 根据 prompt + 图像返回评分
        score = 5.0

        # 将打分结果记录为 json，方便后续调试多轮闭环
        save_json_log(
            agent_name=_AGENT_NAME,
            func_name="assess_clarity",
            payload={
                "score": float(score),
                "note": "dummy implementation; replace with real VLM score.",
            },
            log_root=_LOG_ROOT,
        )
        return score

    def ground_small_text(self, image: Image.Image) -> List[BBox]:
        """定位需要润色的小文字区域.

        当前实现是占位: 简单返回全图中心附近的一个 bbox.
        后续可替换为 grounding DINO + OCR / VLM grounded bbox.
        """

        w, h = image.size
        cx, cy = w // 2, h // 2
        bw, bh = w // 3, h // 6
        x0 = max(cx - bw // 2, 0)
        y0 = max(cy - bh // 2, 0)
        x1 = min(cx + bw // 2, w)
        y1 = min(cy + bh // 2, h)
        bboxes = [(x0, y0, x1, y1)]

        # 可视化：在复制的图像上绘制 bbox，作为 "after" 图用于日志记录
        vis_img = image.copy()
        draw = ImageDraw.Draw(vis_img)
        for (bx0, by0, bx1, by1) in bboxes:
            draw.rectangle((bx0, by0, bx1, by1), outline="red", width=3)

        save_before_after_image(
            agent_name=_AGENT_NAME,
            func_name="ground_small_text",
            before_img=image,
            after_img=vis_img,
            log_root=_LOG_ROOT,
        )

        # 将 grounding 结果记录为 json
        save_json_log(
            agent_name=_AGENT_NAME,
            func_name="ground_small_text",
            payload={
                "image_size": [int(w), int(h)],
                "bboxes": [list(map(int, b)) for b in bboxes],
            },
            log_root=_LOG_ROOT,
        )
        return bboxes

    # ============ FlowEdit 增强 ============
    def refine_patch(
        self,
        image: Image.Image,
        bbox: BBox,
        src_prompt: str,
        tar_prompt: str,
        upscale_factor: int = 2,
    ) -> Image.Image:
        """对给定 bbox 区域进行放大 + FlowEditZImage 润色.

        - src_prompt: 原图描述（可较粗略）
        - tar_prompt: 希望在 patch 内达到的效果（例如："文字更加清晰锐利"）
        """

        x0, y0, x1, y1 = bbox
        crop = image.crop((x0, y0, x1, y1))

        # 放大，以便模型更有空间优化字体细节
        new_w = (x1 - x0) * upscale_factor
        new_h = (y1 - y0) * upscale_factor
        crop_resized = crop.resize((new_w, new_h), Image.LANCZOS)

        # 调用 FlowEditZImage 进行 inversion-free 文本/局部编辑
        edited = FlowEditZImage(
            pipe=self.pipe,
            x_src_image=crop_resized,
            src_prompt=src_prompt,
            tar_prompt=tar_prompt,
            num_inference_steps=20,
            src_guidance_scale=1.5,
            tar_guidance_scale=5.5,
            n_max=18,
            n_min=0,
            seed=42,
        )

        # 缩放回原始 bbox 大小
        edited_resized = edited.resize((x1 - x0, y1 - y0), Image.LANCZOS)

        # 日志：记录一次 patch 编辑的前后图片 + 相关参数
        save_before_after_image(
            agent_name=_AGENT_NAME,
            func_name="refine_patch",
            before_img=crop,
            after_img=edited_resized,
            log_root=_LOG_ROOT,
        )
        save_json_log(
            agent_name=_AGENT_NAME,
            func_name="refine_patch",
            payload={
                "bbox": [int(x0), int(y0), int(x1), int(y1)],
                "upscale_factor": int(upscale_factor),
                "src_prompt": src_prompt,
                "tar_prompt": tar_prompt,
            },
            log_root=_LOG_ROOT,
        )

        return edited_resized

    def stitch_patches(
        self,
        base_image: Image.Image,
        patches: List[Image.Image],
        bboxes: List[BBox],
    ) -> Image.Image:
        """将编辑后的 patch 拼回原图."""

        assert len(patches) == len(bboxes)
        out = base_image.copy()
        for patch, (x0, y0, x1, y1) in zip(patches, bboxes):
            out.paste(patch, (x0, y0, x1, y1))

        # 日志：整张图拼接前后对比
        save_before_after_image(
            agent_name=_AGENT_NAME,
            func_name="stitch_patches",
            before_img=base_image,
            after_img=out,
            log_root=_LOG_ROOT,
        )
        save_json_log(
            agent_name=_AGENT_NAME,
            func_name="stitch_patches",
            payload={
                "num_patches": len(patches),
                "bboxes": [list(map(int, b)) for b in bboxes],
            },
            log_root=_LOG_ROOT,
        )
        return out

    def run(
        self,
        image: Image.Image,
        src_prompt: str,
        tar_prompt_for_text: str = "The text in the image is sharp, high-contrast, and highly legible.",
        clarity_threshold: float = 7.0,
    ) -> Image.Image:
        """完整执行一次海报文字清晰度增强流程.

        - 如果 clarity 已经高于阈值，则直接返回原图
        - 否则：ground -> patch refine -> stitch
        """

        score = self.assess_clarity(image)
        log_agent_info("poster_refiner_agent", f"clarity score={score:.2f}")
        if score >= clarity_threshold:
            log_agent_info("poster_refiner_agent", "clarity good enough; skip refinement")
            return image

        bboxes = self.ground_small_text(image)
        log_agent_info("poster_refiner_agent", f"grounded {len(bboxes)} regions: {bboxes}")

        patches: List[Image.Image] = []
        for bbox in bboxes:
            patch = self.refine_patch(
                image=image,
                bbox=bbox,
                src_prompt=src_prompt,
                tar_prompt=tar_prompt_for_text,
            )
            patches.append(patch)

        refined = self.stitch_patches(image, patches, bboxes)
        log_agent_success("poster_refiner_agent", "refinement finished")
        return refined
