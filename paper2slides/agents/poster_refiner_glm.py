import os
import json
from typing import List, Dict, Any

from PIL import Image

from paper2slides.utils.logging import get_logger
from paper2slides.utils.agent_output_parsing import parse_agent_final_json
# Ensure tools are imported so @register_tool side-effects run (tool registry is populated).
from paper2slides.agents.tools import poster_text_score as _poster_text_score_tool  # noqa: F401
from paper2slides.agents.tools import image_content_grounding as _poster_text_grounding_tool  # noqa: F401
from paper2slides.agents.tools import poster_text_match as _poster_text_match_tool  # noqa: F401
from paper2slides.agents.tools import glm_patch_edit as _glm_patch_edit_tool  # noqa: F401
from paper2slides.utils.agent_logging import log_agent_start, log_agent_success
from paper2slides.utils.agent_artifact_logging import (
    save_json_log,
    get_default_log_root,
)
from paper2slides.utils.api_utils import load_env_api_key

from qwen_agent.agents import Assistant
from qwen_agent import settings as qwen_settings

logger = get_logger(__name__)


_AGENT_NAME = "poster_refiner_glm"
_TOOL_AGENT_MODEL = "gemini-3-pro"
_MAX_ROUNDS_DEFAULT = 3
_BBOX_LIMIT_DEFAULT = 5


class PosterRefinerGLMAgent:
    """PosterRefinerGLMAgent (agent-driven).

    目标：使用 GLM-Image 对文字模糊区域进行局部放大编辑，
    并结合 VLM 评分 + MinerU 定位进行多轮迭代。
    """

    def __init__(
        self,
        glm_model_name: str = "THUDM/GLM-Image-1.0",
        device: str = None,
        style_name: str = "academic",
        plan_text_spans: List[Dict[str, Any]] | None = None,
        plan_text_spans_path: str | None = None,
    ) -> None:
        self.device = device or "cuda"
        self.glm_model_name = glm_model_name
        self.style_name = style_name or "academic"
        self.plan_text_spans: List[Dict[str, Any]] = list(plan_text_spans or [])
        self.plan_text_spans_path: str | None = plan_text_spans_path

        raw_key = load_env_api_key("text")
        assert raw_key, "No API key found for tool agent"

        self._llm_cfg = {
            "model_type": "oai",
            "model": _TOOL_AGENT_MODEL,
            "api_key": raw_key,
            "base_url": "http://127.0.0.1:51958/v1",
        }
        self._function_list = [
            "poster_text_score",
            "poster_text_grounding",
            "poster_text_match",
            "glm_patch_edit",
        ]
        self._system_message = (
            "You are a helpful assistant refining academic poster images using GLM-Image patch editing.\n"
            "Goal: improve text clarity while keeping layout and style unchanged.\n\n"
            "Tools available:\n"
            "- poster_text_score: Score text clarity (0-10).\n"
            "- poster_text_grounding: Locate text regions using MinerU OCR.\n"
            "- poster_text_match: Match OCR text to plan spans (batch call).\n"
            "- glm_patch_edit: Enlarge a patch, edit with GLM-Image, and paste it back.\n\n"
            "Operational Guide (MANDATORY):\n"
            "1) Call poster_text_score on the current image. If score >= clarity_threshold, stop.\n"
            "2) Call poster_text_grounding to get regions + grounding_ckpt_path.\n"
            "3) Call poster_text_match EXACTLY ONCE in batch mode with region_ids + bboxes + grounding_ckpt_path + plan_text_spans_path.\n"
            "4) Select up to bbox_limit regions to refine. For each region, call glm_patch_edit sequentially:\n"
            "   - Use the latest image_path from the previous edit.\n"
            "   - prompt should include matched_text (if any) and instructions like:\n"
            "     'Replace the text in this region with: <matched_text>. Make it crisp, high-contrast, and legible. Keep layout and style.'\n"
            "5) After edits, call poster_text_score again. Iterate up to max_rounds.\n"
            "6) Finish by outputting JSON with: final_image_path, final_score, rounds, history.\n"
        )

        log_agent_start("poster_refiner_glm_agent")
        logger.info("PosterRefinerGLMAgent initialized (agent-driven).")

    def _build_src_prompt(self) -> str:
        style = (self.style_name or "academic").strip()
        return (
            f"An {style} academic research poster. "
            "Preserve the current layout, colors, fonts and overall visual style of the original image."
        )

    def run(
        self,
        image: Image.Image,
        clarity_threshold: float = 7.0,
        max_rounds: int = _MAX_ROUNDS_DEFAULT,
        bbox_limit: int = _BBOX_LIMIT_DEFAULT,
    ) -> Image.Image:
        assert max_rounds > 0
        assert bbox_limit > 0

        tool_calls_per_round = 4 + int(bbox_limit)  # score + grounding + match + edits + score
        est_tool_calls = int(max_rounds) * tool_calls_per_round + 5
        llm_budget = max(20, 2 * est_tool_calls)
        os.environ["QWEN_AGENT_MAX_LLM_CALL_PER_RUN"] = str(llm_budget)
        qwen_settings.MAX_LLM_CALL_PER_RUN = int(llm_budget)

        tool_assistant = Assistant(
            llm=self._llm_cfg,
            function_list=self._function_list,
            system_message=self._system_message,
        )

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
                "glm_model_name": self.glm_model_name,
                "device": self.device,
                "work_dir": work_dir,
                "llm_cfg_public": {
                    "model_type": self._llm_cfg.get("model_type"),
                    "model": self._llm_cfg.get("model"),
                    "model_server": self._llm_cfg.get("model_server"),
                },
            },
            log_root=log_root,
        )

        user_prompt = (
            "Please refine the text clarity of the poster image with GLM-Image patch editing.\n\n"
            "Context:\n"
            "- init_image_path: starting image\n"
            "- plan_text_spans_path: optional JSON file for matching\n"
            "- work_dir: directory for saving output images\n"
            "- src_prompt: global style description\n"
            "- clarity_threshold, max_rounds, bbox_limit\n\n"
            "Task instructions:\n"
            "1) Score -> Ground regions -> Match text (ONE batch call) -> Patch edit -> Re-score.\n"
            "2) You MUST call poster_text_match only once (batch mode).\n"
            "3) You MUST use glm_patch_edit for each region you decide to refine.\n"
            "4) Save intermediate images under work_dir and keep the latest image_path.\n\n"
            "Once finished, output a JSON object with keys: final_image_path, final_score, rounds, history."
        )

        context = {
            "init_image_path": init_image_path,
            "plan_text_spans_path": self.plan_text_spans_path,
            "work_dir": work_dir,
            "src_prompt": src_prompt,
            "clarity_threshold": float(clarity_threshold),
            "max_rounds": int(max_rounds),
            "bbox_limit": int(bbox_limit),
            "glm_model_name": self.glm_model_name,
            "device": self.device,
        }
        messages = [{"role": "user", "content": f"{user_prompt}\n\nContext(JSON): {json.dumps(context, ensure_ascii=False)}"}]

        final_content: str | None = None
        all_assistant_contents: list[str] = []
        for chunk in tool_assistant.run(messages):
            for msg in chunk:
                if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                    c = msg["content"]
                    if isinstance(c, str) and c.strip():
                        all_assistant_contents.append(c)
                        stripped = c.strip()
                        if stripped.startswith("{") or stripped.startswith("```"):
                            final_content = c

        if not final_content:
            debug_path = os.path.join(work_dir, "agent_debug.log")
            with open(debug_path, "w", encoding="utf-8") as f:
                for i, content in enumerate(all_assistant_contents):
                    f.write(f"--- chunk {i+1} ---\n")
                    f.write(content)
                    f.write("\n\n")
            raise RuntimeError("Qwen-Agent did not produce non-empty assistant content")

        result = parse_agent_final_json(final_content)

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
        log_agent_success("poster_refiner_glm_agent", f"refinement finished: {final_image_path}")
        return out
