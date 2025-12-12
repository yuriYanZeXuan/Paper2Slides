import os
import json
from typing import List, Tuple, Dict, Any

from PIL import Image

from paper2slides.utils.logging import get_logger
# Ensure tools are imported so @register_tool side-effects run (tool registry is populated).
from paper2slides.agents.tools import poster_text_score as _poster_text_score_tool  # noqa: F401
from paper2slides.agents.tools import poster_text_grounding as _poster_text_grounding_tool  # noqa: F401
from paper2slides.agents.tools import zimage_flowedit_tool as _zimage_flowedit_tool  # noqa: F401
from paper2slides.agents.tools import poster_text_match as _poster_text_match_tool  # noqa: F401
from paper2slides.agents.tools import poster_patch_flowedit as _poster_patch_flowedit_tool  # noqa: F401
from qwen_agent.agents import Assistant
from paper2slides.utils.agent_logging import *
from paper2slides.utils.agent_artifact_logging import (
    save_json_log,
    get_default_log_root,
)
logger = get_logger(__name__)


BBox = Tuple[int, int, int, int]


_AGENT_NAME = "poster_refiner"
_LOG_ROOT = get_default_log_root(_AGENT_NAME)
_TOOL_AGENT_MODEL = os.getenv("POSTER_TOOL_AGENT_MODEL", "gpt-4o")
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
        self._llm_cfg = {
            "model": _TOOL_AGENT_MODEL,
            "api_key": os.getenv("RAG_LLM_API_KEY") or os.getenv("GEMINI_TEXT_KEY") or os.getenv("RUNWAY_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
            # Qwen-Agent 里 OpenAI 兼容服务端字段名为 model_server（见 qwen_agent.llm.get_chat_model）
            "model_server": os.getenv("RAG_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("RUNWAY_API_BASE") or "",
        }
        assert self._llm_cfg["api_key"], "No API key found for tool agent (RAG_LLM_API_KEY/GEMINI_TEXT_KEY/RUNWAY_API_KEY/OPENAI_API_KEY)"
        assert self._llm_cfg["model_server"], "No model_server found for tool agent (RAG_LLM_BASE_URL/OPENAI_BASE_URL/RUNWAY_API_BASE)"
        self._function_list = [
            "poster_text_score",
            "poster_text_grounding",
            "poster_text_match",
            "poster_patch_flowedit",
            # optional fallback (whole-image edit)
            "zimage_flowedit",
        ]
        self._system_message = (
            "You are an autonomous agent for refining text clarity in academic poster images.\n"
            "You can call tools to: score text clarity, locate unclear text regions, match patch text to plan spans, "
            "and apply FlowEdit to a bbox patch and paste it back.\n\n"
            "Rules:\n"
            "- You MUST keep track of the current working image_path after each edit.\n"
            "- Use poster_text_score to assess clarity. If score >= clarity_threshold, stop.\n"
            "- If score < clarity_threshold, call poster_text_grounding to get bboxes (limit to bbox_limit).\n"
            "- For each bbox, if plan_text_spans_path is provided, call poster_text_match(image_path,bbox,plan_text_spans_path)\n"
            "  to get matched_text; use it to craft a concise tar_prompt (keep matched_text <= 200 chars).\n"
            "  If matched_text is null/empty or plan_text_spans_path is null, use a generic tar_prompt:\n"
            "  \"The text in this region is sharp, high-contrast, and highly legible, without changing layout/colors.\"\n"
            "- Apply edits with poster_patch_flowedit, saving outputs ONLY under work_dir.\n"
            "- Use zimage_flowedit only as a LAST RESORT (whole-image edit), and still save outputs under work_dir.\n"
            "- Iterate up to max_rounds.\n\n"
            "Final output MUST be ONLY a JSON object with keys:\n"
            "final_image_path (string), final_score (float), rounds (int), history (list).\n"
            "history items should include at least: round, score_before, bboxes, edits, score_after.\n"
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
        tmp_dir = os.path.join(_LOG_ROOT, "tool_inputs")
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
        try:
            from qwen_agent import settings as qwen_settings

            qwen_settings.MAX_LLM_CALL_PER_RUN = int(llm_budget)
        except Exception:
            # 若运行环境不允许/版本差异，至少 env 已设置
            pass

        tool_assistant = Assistant(
            llm=self._llm_cfg,
            function_list=self._function_list,
            system_message=self._system_message,
        )

        work_dir = os.path.join(_LOG_ROOT, "autonomous")
        os.makedirs(work_dir, exist_ok=True)

        init_image_path = os.path.join(work_dir, "round0_init.png")
        image.save(init_image_path)

        src_prompt = self._build_src_prompt()

        # 记录本次 run 的上下文，便于 server 端排查
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
            },
            log_root=_LOG_ROOT,
        )

        user_prompt = (
            "You will improve the text clarity of a poster image.\n"
            "You are given:\n"
            "- init_image_path: the starting image\n"
            "- plan_text_spans_path: optional JSON file for matching\n"
            "- work_dir: where you MUST write any output images\n"
            "- src_prompt: global style description\n"
            "- clarity_threshold, max_rounds, bbox_limit\n\n"
            "Process:\n"
            "1) Set current_image_path = init_image_path.\n"
            "2) For round=1..max_rounds:\n"
            "   - Call poster_text_score(image_path=current_image_path) -> score.\n"
            "   - If score >= clarity_threshold: stop.\n"
            "   - Call poster_text_grounding(image_path=current_image_path) -> bboxes.\n"
            "   - Take up to bbox_limit bboxes.\n"
            "   - For each bbox i:\n"
            "       * If plan_text_spans_path is provided: call poster_text_match(image_path=current_image_path, bbox=bbox, plan_text_spans_path=...).\n"
            "         If matched_text is empty/null, use a generic tar_prompt.\n"
            "         Use matched_text to create tar_prompt (matched_text MUST be <= 200 chars; keep tar_prompt concise).\n"
            "       * Call poster_patch_flowedit(image_path=current_image_path, bbox=bbox, src_prompt=src_prompt, tar_prompt=tar_prompt,\n"
            "         output_image_path=f\"{work_dir}/r{round}_b{i}.png\", model_name=zimage_model_name, device=device, upscale_factor=2).\n"
            "       * Set current_image_path to the returned output_image_path.\n"
            "3) After loop, call poster_text_score once more for final_score.\n\n"
            "Return ONLY JSON with:\n"
            "{\n"
            "  \"final_image_path\": \"...\",\n"
            "  \"final_score\": <float>,\n"
            "  \"rounds\": <int>,\n"
            "  \"history\": [ ... ]\n"
            "}\n"
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
        for chunk in tool_assistant.run(messages):
            for msg in chunk:
                if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                    final_content = msg["content"]

        assert final_content, "Qwen-Agent did not produce assistant content"
        result = json.loads(final_content)
        assert isinstance(result, dict), f"agent final output must be json object, got: {type(result)}"
        assert "final_image_path" in result, "missing final_image_path in agent output"
        assert "final_score" in result, "missing final_score in agent output"
        assert "rounds" in result, "missing rounds in agent output"
        assert "history" in result, "missing history in agent output"

        # 保存 agent 的最终输出，便于服务端排查
        save_json_log(
            agent_name=_AGENT_NAME,
            func_name="agent_final",
            payload=result,
            log_root=_LOG_ROOT,
        )
        final_image_path = str(result["final_image_path"])

        out = Image.open(final_image_path).convert("RGB")
        log_agent_success("poster_refiner_agent", f"refinement finished: {final_image_path}")
        return out
