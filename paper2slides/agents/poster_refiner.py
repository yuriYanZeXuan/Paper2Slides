import os
import json
from typing import List, Tuple, Dict, Any

from PIL import Image

from paper2slides.utils.logging import get_logger
from paper2slides.utils.agent_output_parsing import parse_agent_final_json
# Ensure tools are imported so @register_tool side-effects run (tool registry is populated).
from paper2slides.agents.tools import poster_text_score as _poster_text_score_tool  # noqa: F401
# 使用 MinerU 版本的 text grounding（替换原 VLM 版本）
# from paper2slides.agents.tools import poster_text_grounding as _poster_text_grounding_tool  # noqa: F401
from paper2slides.agents.tools import poster_minerU_grounding as _poster_text_grounding_tool  # noqa: F401
from paper2slides.agents.tools import zimage_flowedit_tool as _zimage_flowedit_tool  # noqa: F401
from paper2slides.agents.tools import poster_text_match as _poster_text_match_tool  # noqa: F401
from paper2slides.agents.tools import poster_patch_flowedit as _poster_patch_flowedit_tool  # noqa: F401
from qwen_agent.agents import Assistant
from paper2slides.utils.agent_logging import *
from paper2slides.utils.agent_artifact_logging import (
    save_json_log,
    get_default_log_root,
)
from paper2slides.utils.api_utils import DEFAULT_CHAT_COMPLETIONS_URL, load_env_api_key
logger = get_logger(__name__)


BBox = Tuple[int, int, int, int]


_AGENT_NAME = "poster_refiner"
_TOOL_AGENT_MODEL = "gpt-4o"
_MAX_ROUNDS_DEFAULT = 3
_BBOX_LIMIT_DEFAULT = 5



class PosterRefinerAgent:
    """PosterRefinerAgent (agent-driven).

    目标：将“评估→定位→匹配→局部重绘回填→复评”的控制权交给 Qwen-Agent 的 tool calling，
    在一次 `Assistant.run()` 内多轮迭代（默认最多 3 轮）。
    """

    def __init__(
        self,
        zimage_model_name: str = "Tongyi-MAI/Z-Image-Turbo",
        device: str = None,
        style_name: str = "academic",
        plan_text_spans: List[Dict[str, Any]] | None = None,
        plan_text_spans_path: str | None = None,
    ) -> None:
        self.device = device or "cuda"
        self.zimage_model_name = zimage_model_name
        self.style_name = style_name or "academic"
        # 每个元素形如 {"id": ..., "section_id": ..., "section_title": ..., "text": ...}
        self.plan_text_spans: List[Dict[str, Any]] = list(plan_text_spans or [])
        self.plan_text_spans_path: str | None = plan_text_spans_path
        # Qwen-Agent 工具调度 Agent：用于自主决定是否需要继续 grounding/refine
        # 注意：这里使用 OpenAI 兼容的配置（api_key/base_url/model），以适配项目现有网关。
        raw_key = load_env_api_key("text")
        assert raw_key, "No API key found for tool agent (RAG_LLM_API_KEY/GEMINI_TEXT_KEY/RUNWAY_API_KEY/OPENAI_API_KEY)"

        # base_url 写死（不从环境变量读取），避免 /openai vs /openai/v1 导致 404
        # qwen_agent 对 OpenAI 兼容配置一般使用 model_type=openai + base_url
        self._llm_cfg = {
            "model_type": "azure",
            "model": _TOOL_AGENT_MODEL,
            "api_key": raw_key,
            "base_url": DEFAULT_CHAT_COMPLETIONS_URL,
        }
        self._function_list = [
            "poster_text_score",
            "poster_text_grounding",
            "poster_text_match",
            "poster_patch_flowedit",
            # optional fallback (whole-image edit)
            "zimage_flowedit",
        ]
        self._system_message = (
            "You are a helpful assistant that improves text clarity in academic poster images.\n"
            "You have access to tools for: scoring text clarity, locating unclear text regions, matching patch text to plan spans, "
            "and applying FlowEdit to enhance specific regions.\n\n"
            "Important behavior:\n"
            "- Execute tool calls in sequence without pausing to explain between calls.\n"
            "- Continue calling tools until the task is complete.\n"
            "- Only output your final JSON response after all tool calls are done.\n\n"
            "Workflow:\n"
            "1. Keep track of the current working image_path after each edit.\n"
            "2. Use poster_text_score to assess clarity. If score >= clarity_threshold, output final JSON.\n"
            "3. If score < clarity_threshold, immediately call poster_text_grounding to get bboxes (limit to bbox_limit).\n"
            "4. For each bbox, if plan_text_spans_path is provided, call poster_text_match(image_path,bbox,plan_text_spans_path)\n"
            "   to get matched_text; use it to craft a concise tar_prompt (keep matched_text <= 200 chars).\n"
            "   If matched_text is empty or plan_text_spans_path is not available, use a generic tar_prompt:\n"
            "   \"The text in this region is sharp, high-contrast, and highly legible, without changing layout/colors.\"\n"
            "5. Apply edits with poster_patch_flowedit, saving outputs under work_dir.\n"
            "6. Use zimage_flowedit as a fallback option (whole-image edit), and save outputs under work_dir.\n"
            "7. You may iterate up to max_rounds.\n\n"
            "Final output format (JSON only):\n"
            "{\"final_image_path\": \"...\", \"final_score\": float, \"rounds\": int, \"history\": [...], \"thoughts\": [...]}\n"
        )

        log_agent_start("poster_refiner_agent")
        logger.info("PosterRefinerAgent initialized (agent-driven).")

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
        assert snippet and len(snippet) < 200, "matched_text is empty or too long (>200 chars)"
        return (
            f"Same {style} academic poster style and layout as the original image, "
            f"but ensure that the text '{snippet}' in this region is sharp, high-contrast, and highly legible, "
            "without changing the overall composition, fonts, or colors outside this region."
        )

    # ============ VLM tools (via qwen_agent tool dispatch) ============
    def _save_tmp_image_for_tool(self, image: Image.Image, tag: str) -> str:
        """Save image into agent log root for tool consumption (tools take image_path)."""
        assert isinstance(tag, str) and tag.strip()
        log_root = get_default_log_root(_AGENT_NAME)
        tmp_dir = os.path.join(log_root, "tool_inputs")
        os.makedirs(tmp_dir, exist_ok=True)
        path = os.path.join(tmp_dir, f"{tag}.png")
        image.save(path)
        return path

    def run(
        self,
        image: Image.Image,
        clarity_threshold: float = 7.0,
        max_rounds: int = _MAX_ROUNDS_DEFAULT,
        bbox_limit: int = _BBOX_LIMIT_DEFAULT,
    ) -> Image.Image:
        """由 Qwen-Agent 在一次 run 内多轮调用工具完成闭环，返回最终图像。"""

        assert max_rounds > 0
        assert bbox_limit > 0

        # 硬上限来自 qwen_agent.settings.MAX_LLM_CALL_PER_RUN（默认 20，可能不足以跑完多轮闭环）。
        # 这里按 max_rounds/bbox_limit 估算需要的 LLM call budget，并同时设置 env + settings 变量：
        # - env：供依赖 os.getenv 的场景
        # - settings：供已导入 settings 的运行时读取
        tool_calls_per_round = 3 + 2 * int(bbox_limit)  # score + grounding + bbox*(match+patch_flowedit) + score_after
        est_tool_calls = int(max_rounds) * tool_calls_per_round + 5
        llm_budget = max(20, 2 * est_tool_calls)  # 2x safety factor
        os.environ["QWEN_AGENT_MAX_LLM_CALL_PER_RUN"] = str(llm_budget)
        from qwen_agent import settings as qwen_settings
        qwen_settings.MAX_LLM_CALL_PER_RUN = int(llm_budget)

        tool_assistant = Assistant(
            llm=self._llm_cfg,
            function_list=self._function_list,
            system_message=self._system_message,
        )

        # 动态获取当前 session 的日志目录
        log_root = get_default_log_root(_AGENT_NAME)
        work_dir = os.path.join(log_root, "autonomous")
        os.makedirs(work_dir, exist_ok=True)

        init_image_path = os.path.join(work_dir, "round0_init.png")
        image.save(init_image_path)

        src_prompt = self._build_src_prompt()

        save_json_log(
            agent_name=_AGENT_NAME,
            func_name="run_context",
            payload={
                "init_image_path": init_image_path,
                "plan_text_spans_path": self.plan_text_spans_path,
                "clarity_threshold": float(clarity_threshold),
                "max_rounds": int(max_rounds),
                "bbox_limit": int(bbox_limit),
                "src_prompt": src_prompt,
                "style_name": self.style_name,
                "zimage_model_name": self.zimage_model_name,
                "device": self.device,
                "work_dir": work_dir,
                # 不记录 api_key，避免泄露；仅记录与 404 相关的 url/配置
                "llm_cfg_public": {
                    "model_type": self._llm_cfg.get("model_type"),
                    "model": self._llm_cfg.get("model"),
                    "model_server": self._llm_cfg.get("model_server"),
                },
            },
            log_root=log_root,
        )

        user_prompt = (
            "Improve the text clarity of a poster image by calling the available tools.\n\n"
            "Context:\n"
            "- init_image_path: the starting image\n"
            "- plan_text_spans_path: optional JSON file for matching\n"
            "- work_dir: directory for saving output images\n"
            "- src_prompt: global style description\n"
            "- clarity_threshold, max_rounds, bbox_limit\n\n"
            "Execute these steps by calling tools directly:\n"
            "1) Set current_image_path = init_image_path.\n"
            "2) For round=1..max_rounds:\n"
            "   - Call poster_text_score(image_path=current_image_path) to get score.\n"
            "   - If score >= clarity_threshold: output final JSON and stop.\n"
            "   - Otherwise, immediately call poster_text_grounding(image_path=current_image_path) to get bboxes.\n"
            "   - Process up to bbox_limit bboxes.\n"
            "   - For each bbox i:\n"
            "       * If plan_text_spans_path is provided: call poster_text_match to get matched_text.\n"
            "         If matched_text is empty, use a generic tar_prompt.\n"
            "         Use matched_text to create tar_prompt (keep matched_text <= 200 chars; keep tar_prompt concise).\n"
            "       * Call poster_patch_flowedit with bbox, src_prompt, tar_prompt,\n"
            "         output_image_path=f\"{work_dir}/r{round}_b{i}.png\", model_name, device, upscale_factor=2.\n"
            "       * Update current_image_path to the returned output_image_path.\n"
            "3) After finishing, call poster_text_score once more for final_score, then output final JSON.\n\n"
            "Begin by calling poster_text_score now. Do not output text between tool calls.\n\n"
            "Final JSON format:\n"
            "{\"final_image_path\": \"path\", \"final_score\": 8.5, \"rounds\": 2, \"history\": [...], \"thoughts\": [...]}\n"
        )

        context = {
            "init_image_path": init_image_path,
            "plan_text_spans_path": self.plan_text_spans_path,
            "work_dir": work_dir,
            "src_prompt": src_prompt,
            "clarity_threshold": float(clarity_threshold),
            "max_rounds": int(max_rounds),
            "bbox_limit": int(bbox_limit),
            "zimage_model_name": self.zimage_model_name,
            "device": self.device,
        }
        messages = [{"role": "user", "content": f"{user_prompt}\n\nContext(JSON): {json.dumps(context, ensure_ascii=False)}"}]

        final_content: str | None = None
        all_assistant_contents: list[str] = []
        for chunk in tool_assistant.run(messages):
            for msg in chunk:
                if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                    c = msg["content"]
                    # Some gateways may return whitespace-only content; ignore those.
                    if isinstance(c, str) and c.strip():
                        all_assistant_contents.append(c)
                        # Only accept content that looks like JSON as final output
                        stripped = c.strip()
                        if stripped.startswith("{") or stripped.startswith("```"):
                            final_content = c

        if not final_content:
            # INSERT_YOUR_CODE
            # 保存 all_assistant_contents 到 agent_logs 目录下，便于调试与追溯
            with open("Agent_debug.log", "w", encoding="utf-8") as f:
                for i, content in enumerate(all_assistant_contents):
                    f.write(f"--- chunk {i+1} ---\n")
                    f.write(content)
                    f.write("\n\n")
            raise RuntimeError("Qwen-Agent did not produce non-empty assistant content")

        # Parse robustly; if invalid, provide more context in error
        result = parse_agent_final_json(final_content)
        
        # 保存解析后的 agent 输出
        save_json_log(
            agent_name=_AGENT_NAME,
            func_name="agent_final",
            payload=result,
            log_root=log_root,
        )

        assert "final_image_path" in result, "missing final_image_path in agent output"
        assert "final_score" in result, "missing final_score in agent output"
        assert "rounds" in result, "missing rounds in agent output"
        assert "history" in result, "missing history in agent output"

        final_image_path = str(result["final_image_path"])

        out = Image.open(final_image_path).convert("RGB")
        log_agent_success("poster_refiner_agent", f"refinement finished: {final_image_path}")
        return out
