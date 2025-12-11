import base64
import io
import os
import json
from typing import List, Tuple, Dict, Any

from PIL import Image, ImageDraw
import torch

from diffusers import ZImagePipeline

from paper2slides.utils.logging import get_logger
from paper2slides.agents.tools.zimage_flowedit_core import FlowEditZImage
from paper2slides.utils.agent_logging import (
    log_agent_start,
    log_agent_info,
    log_agent_success,
    log_agent_warning,
)
from paper2slides.utils.agent_artifact_logging import (
    save_before_after_image,
    save_json_log,
    get_default_log_root,
)
from paper2slides.utils.api_utils import get_openai_client


logger = get_logger(__name__)


BBox = Tuple[int, int, int, int]


_AGENT_NAME = "poster_refiner"
_LOG_ROOT = get_default_log_root(_AGENT_NAME)
_TEXT_MATCH_MODEL = os.getenv("POSTER_TEXT_MATCH_MODEL", "gpt-4o")

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
        style_name: str = "academic",
        plan_text_spans: List[Dict[str, Any]] | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.style_name = style_name or "academic"
        # 每个元素形如 {"id": ..., "section_id": ..., "section_title": ..., "text": ...}
        self.plan_text_spans: List[Dict[str, Any]] = list(plan_text_spans or [])
        self._vlm_client = None

        log_agent_start("poster_refiner_agent")
        logger.info(f"Loading Z-Image pipeline {zimage_model_name} on {self.device} ...")
        self.pipe: ZImagePipeline = ZImagePipeline.from_pretrained(
            zimage_model_name,
            torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
            low_cpu_mem_usage=False,
        )
        self.pipe.to(self.device)

    # ============ Prompt 构造与文字匹配辅助函数 ============
    def _build_src_prompt(self) -> str:
        """基于 style_name 构造全局 src_prompt，用于保持整体画风不变。"""
        style = (self.style_name or "academic").strip()
        return (
            f"An {style} academic research poster. "
            "Preserve the current layout, colors, fonts and overall visual style of the original image."
        )

    def _build_tar_prompt(self, matched_text: str) -> str:
        """基于匹配到的文字内容构造局部 tar_prompt。"""
        style = (self.style_name or "academic").strip()
        snippet = (matched_text or "").strip()
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        if not snippet:
            # 兜底：无具体文字时退回到通用描述
            return (
                "The text in this region is sharp, high-contrast, and highly legible, "
                "without changing the overall poster layout or colors."
            )
        return (
            f"Same {style} academic poster style and layout as the original image, "
            f"but ensure that the text '{snippet}' in this region is sharp, high-contrast, and highly legible, "
            "without changing the overall composition, fonts, or colors outside this region."
        )

    def _get_vlm_client(self):
        if self._vlm_client is None:
            self._vlm_client = get_openai_client(key_type="text")
        return self._vlm_client

    def _encode_patch_to_base64(self, patch: Image.Image) -> str:
        buf = io.BytesIO()
        patch.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _match_text_for_patch(
        self,
        patch: Image.Image,
        bbox: BBox,
        max_candidates: int = 40,
    ) -> Tuple[str | None, Dict[str, Any] | None]:
        """使用 GPT-4o 在 plan_text_spans 中为该 patch 匹配最相关的文字内容."""
        if not self.plan_text_spans:
            return None, None

        client = self._get_vlm_client()
        w, h = patch.size

        # 截断候选，以避免 prompt 过长
        candidates = self.plan_text_spans[:max_candidates]
        candidates_str_lines = []
        for idx, span in enumerate(candidates, start=1):
            text = (span.get("text") or "").replace("\n", " ").strip()
            if len(text) > 200:
                text = text[:200] + "..."
            candidates_str_lines.append(f"{idx}. {text}")
        candidates_str = "\n".join(candidates_str_lines)

        b64 = self._encode_patch_to_base64(patch)

        system_prompt = (
            "You are an expert at reading small text on academic posters and matching it to candidate text spans. "
            "Your task is to find which candidate text best corresponds to the text appearing inside the given image patch."
        )
        user_text = (
            "Here is a small image patch from a poster. First, read the text inside the patch.\n"
            "Then, from the candidate list below, choose the SINGLE candidate that best matches "
            "the text in this patch (based on semantic content, not style).\n\n"
            "Return ONLY a JSON object of the form:\n"
            "{\n"
            '  \"matched_index\": <integer index in [1..N]> or null if no good match,\n'
            '  \"matched_text\": \"the chosen candidate text or empty string\"\n'
            "}\n\n"
            "Candidate text spans:\n"
            f"{candidates_str}\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
        ]

        matched_index: int | None = None
        matched_text: str | None = None
        try:
            response = client.chat.completions.create(
                model=_TEXT_MATCH_MODEL,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            raw_idx = data.get("matched_index")
            raw_text = data.get("matched_text")
            if isinstance(raw_idx, int) and 1 <= raw_idx <= len(candidates):
                matched_index = raw_idx
                matched_span = candidates[matched_index - 1]
                matched_text = str(raw_text or matched_span.get("text") or "")
                meta = {
                    "matched_index": matched_index,
                    "matched_span": matched_span,
                    "bbox": [int(v) for v in bbox],
                    "patch_size": [int(w), int(h)],
                }
            else:
                matched_index = None
                meta = {
                    "matched_index": None,
                    "bbox": [int(v) for v in bbox],
                    "patch_size": [int(w), int(h)],
                }
        except Exception as e:
            log_agent_warning(
                _AGENT_NAME,
                f"text matching VLM call failed for bbox={bbox}: {e}",
            )
            matched_index = None
            matched_text = None
            meta = {
                "matched_index": None,
                "error": str(e),
                "bbox": [int(v) for v in bbox],
                "patch_size": [int(w), int(h)],
            }

        # 记录匹配日志
        save_json_log(
            agent_name=_AGENT_NAME,
            func_name="match_text_for_patch",
            payload=meta,
            log_root=_LOG_ROOT,
        )

        if matched_index is None or not matched_text:
            return None, meta

        return matched_text, meta

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
        matched_meta: Dict[str, Any] | None = None,
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
                "matched_meta": matched_meta or {},
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
        src_prompt: str | None = None,
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

        # 若未显式提供 src_prompt，则基于 style_name 构造
        effective_src_prompt = src_prompt or self._build_src_prompt()
        default_tar_prompt = tar_prompt_for_text

        patches: List[Image.Image] = []
        for idx, bbox in enumerate(bboxes):
            # 每个 bbox 尝试与计划文本做一次匹配，获得更精确的文字内容
            matched_text: str | None = None
            matched_meta: Dict[str, Any] | None = None
            if self.plan_text_spans:
                patch_for_match = image.crop(bbox)
                matched_text, matched_meta = self._match_text_for_patch(
                    patch_for_match,
                    bbox=bbox,
                )

            if matched_text:
                effective_tar_prompt = self._build_tar_prompt(matched_text)
            else:
                # 匹配失败时退回到通用 tar_prompt
                effective_tar_prompt = default_tar_prompt

            patch = self.refine_patch(
                image=image,
                bbox=bbox,
                src_prompt=effective_src_prompt,
                tar_prompt=effective_tar_prompt,
                matched_meta=matched_meta,
            )
            patches.append(patch)

        refined = self.stitch_patches(image, patches, bboxes)
        log_agent_success("poster_refiner_agent", "refinement finished")
        return refined
