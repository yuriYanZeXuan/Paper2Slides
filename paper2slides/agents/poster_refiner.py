import os
import json
from typing import List, Tuple, Dict, Any

from PIL import Image, ImageDraw
import torch

from diffusers import ZImagePipeline

from paper2slides.utils.logging import get_logger
from paper2slides.agents.tools.zimage_flowedit_core import FlowEditZImage
# Ensure tools are imported so @register_tool side-effects run (tool registry is populated).
from paper2slides.agents.tools import poster_text_score as _poster_text_score_tool  # noqa: F401
from paper2slides.agents.tools import poster_text_grounding as _poster_text_grounding_tool  # noqa: F401
from paper2slides.agents.tools import zimage_flowedit_tool as _zimage_flowedit_tool  # noqa: F401
from paper2slides.agents.tools.poster_text_match import PosterTextMatch
from qwen_agent.agents import Assistant
from paper2slides.utils.agent_logging import *
from paper2slides.utils.agent_artifact_logging import (
    save_before_after_image,
    save_json_log,
    get_default_log_root,
)
logger = get_logger(__name__)


BBox = Tuple[int, int, int, int]


_AGENT_NAME = "poster_refiner"
_LOG_ROOT = get_default_log_root(_AGENT_NAME)
_TOOL_AGENT_MODEL = os.getenv("POSTER_TOOL_AGENT_MODEL", "gpt-4o")

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
        plan_text_spans_path: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.style_name = style_name or "academic"
        # 每个元素形如 {"id": ..., "section_id": ..., "section_title": ..., "text": ...}
        self.plan_text_spans: List[Dict[str, Any]] = list(plan_text_spans or [])
        self.plan_text_spans_path: str | None = plan_text_spans_path
        # Qwen-Agent 工具调度 Agent：用于自主决定是否需要继续 grounding/refine
        # 注意：这里使用 OpenAI 兼容的配置（api_key/base_url/model），以适配项目现有网关。
        llm_cfg = {
            "model": _TOOL_AGENT_MODEL,
            "api_key": os.getenv("RAG_LLM_API_KEY") or os.getenv("GEMINI_TEXT_KEY") or os.getenv("RUNWAY_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
            "base_url": os.getenv("RAG_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("RUNWAY_API_BASE") or "",
        }
        assert llm_cfg["api_key"], "No API key found for tool agent (RAG_LLM_API_KEY/GEMINI_TEXT_KEY/RUNWAY_API_KEY/OPENAI_API_KEY)"
        self._tool_assistant = Assistant(
            llm=llm_cfg,
            function_list=["poster_text_score", "poster_text_grounding", "zimage_flowedit"],
            system_message=(
                "You are a tool-using agent for refining poster text clarity. "
                "For clarity decision, ONLY use poster_text_score and poster_text_grounding. "
                "Do NOT call zimage_flowedit unless explicitly asked. "
                "Output ONLY JSON."
            ),
        )

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

    # ============ VLM tools (via qwen_agent tool dispatch) ============
    def _save_tmp_image_for_tool(self, image: Image.Image, tag: str) -> str:
        """Save image into agent log root for tool consumption (tools take image_path)."""
        assert isinstance(tag, str) and tag.strip()
        tmp_dir = os.path.join(_LOG_ROOT, "tool_inputs")
        os.makedirs(tmp_dir, exist_ok=True)
        path = os.path.join(tmp_dir, f"{tag}.png")
        image.save(path)
        return path

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

        # 让 Qwen-Agent 自主调度工具：先打分，再按阈值决定是否 grounding
        image_path = self._save_tmp_image_for_tool(image, tag="poster_tool_agent")
        user_prompt = (
            "Given a poster image at image_path, decide whether refinement is needed.\n"
            "Rules:\n"
            f"- First call tool poster_text_score(image_path) to get score in [0,10].\n"
            f"- If score >= {float(clarity_threshold)}, reply with a JSON object: "
            "{\"score\": <float>, \"should_refine\": false, \"bboxes\": []}\n"
            f"- If score < {float(clarity_threshold)}, call tool poster_text_grounding(image_path) "
            "to get bboxes, then reply with JSON: "
            "{\"score\": <float>, \"should_refine\": true, \"bboxes\": [[x0,y0,x1,y1], ...]}\n"
            "Return ONLY JSON."
        )
        messages = [{"role": "user", "content": f"image_path={image_path}\n\n{user_prompt}"}]

        final_content: str | None = None
        for chunk in self._tool_assistant.run(messages):
            for msg in chunk:
                if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                    final_content = msg["content"]
        assert final_content, "Qwen-Agent did not produce assistant content"
        result = json.loads(final_content)
        assert isinstance(result, dict), f"agent final output must be json object, got: {type(result)}"

        score = float(result["score"])
        assert 0.0 <= score <= 10.0, f"clarity score out of range: {score}"
        should_refine = bool(result.get("should_refine", False))
        bboxes = list(result.get("bboxes") or [])

        # 记录打分（保持与之前日志接口一致）
        save_json_log(
            agent_name=_AGENT_NAME,
            func_name="assess_clarity",
            payload={"score": float(score)},
            log_root=_LOG_ROOT,
        )

        log_agent_info("poster_refiner_agent", f"clarity score={score:.2f}")
        if not should_refine:
            assert len(bboxes) == 0, "should_refine=false but bboxes provided"
            log_agent_info("poster_refiner_agent", "clarity good enough; skip refinement")
            return image

        w, h = image.size
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
        save_json_log(
            agent_name=_AGENT_NAME,
            func_name="ground_small_text",
            payload={"image_size": [int(w), int(h)], "bboxes": [list(map(int, b)) for b in bboxes]},
            log_root=_LOG_ROOT,
        )

        log_agent_info("poster_refiner_agent", f"grounded {len(bboxes)} regions: {bboxes}")

        # 若未显式提供 src_prompt，则基于 style_name 构造
        effective_src_prompt = src_prompt or self._build_src_prompt()
        default_tar_prompt = tar_prompt_for_text

        patches: List[Image.Image] = []
        for idx, bbox in enumerate(bboxes):
            # 每个 bbox 尝试与计划文本做一次匹配，获得更精确的文字内容
            matched_text: str | None = None
            matched_meta: Dict[str, Any] | None = None
            if self.plan_text_spans_path:
                patch_for_match = image.crop(bbox)
                patch_path = self._save_tmp_image_for_tool(patch_for_match, tag=f"patch_match_{idx}")
                tool = PosterTextMatch()
                out = tool.call(
                    {
                        "patch_image_path": patch_path,
                        "bbox": [int(v) for v in bbox],
                        "plan_text_spans_path": self.plan_text_spans_path,
                        "max_candidates": 40,
                        "agent_name": _AGENT_NAME,
                        "log_root": str(_LOG_ROOT),
                    }
                )
                data = json.loads(out)
                matched_text = data.get("matched_text")
                matched_meta = data.get("meta")

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
