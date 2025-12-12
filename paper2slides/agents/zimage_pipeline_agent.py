"""Z-Image agent pipeline entry.

1. 复用 Paper2Slides 的主 RAG / summary / plan / generate 流水线
2. 固定使用 Z-Image 本地后端生成 poster 或 slides
3. 对生成的图片（poster 或每一张 slide）调用 PosterRefinerAgent 做一次小字强化

用法示例（项目根目录）：

    python -m paper2slides.agents.zimage_pipeline_agent \
        --input path/to/paper.pdf \
        --output poster \
        --style academic

"""

import argparse
import asyncio
import logging
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image

from paper2slides.utils import setup_logging, load_json
from paper2slides.utils.path_utils import normalize_input_path, get_project_name, parse_style
from paper2slides.core.paths import get_base_dir, get_config_dir, get_plan_checkpoint
from paper2slides.core.pipeline import run_pipeline, STAGES
from paper2slides.utils.agent_logging import (
    log_agent_start,
    log_agent_info,
    log_agent_success,
    log_agent_warning,
)
from paper2slides.agents.poster_refiner import PosterRefinerAgent


logger = logging.getLogger(__name__)


def _split_content_to_spans(text: str) -> List[str]:
    """将 section 的长 content 按句子/段落拆分为较短的文字片段."""
    if not text:
        return []
    # 先按换行粗略切分，再按常见句末标点进一步分句
    spans: List[str] = []
    for block in re.split(r"[\r\n]+", text):
        block = block.strip()
        if not block:
            continue
        parts = re.split(r"[。！？!?；;]+", block)
        for p in parts:
            p = p.strip()
            if p:
                spans.append(p)
    return spans


def _build_plan_text_spans(plan_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 checkpoint_plan.json 中构造 poster 用的文字片段列表."""
    result: List[Dict[str, Any]] = []
    plan = plan_data.get("plan") or {}
    sections = plan.get("sections") or []

    for sec in sections:
        sec_id = sec.get("id", "")
        title = sec.get("title", "")
        content = sec.get("content", "") or ""
        spans = _split_content_to_spans(content)
        for idx, span in enumerate(spans, start=1):
            result.append(
                {
                    "id": f"{sec_id or 'section'}_{idx}",
                    "section_id": sec_id,
                    "section_title": title,
                    "text": span,
                }
            )
    return result


def _find_latest_output_dir(config_dir: Path) -> Path:
    """在 config_dir 下找到最新的时间戳输出目录（与 get_output_dir 对应）。"""
    if not config_dir.exists():
        raise FileNotFoundError(f"Config dir not found: {config_dir}")

    subdirs: List[Path] = [d for d in config_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No output subdirs under: {config_dir}")

    # 目录名形如 20250101_120000，按名字排序等价于时间排序
    subdirs.sort(key=lambda p: p.name)
    return subdirs[-1]


def _collect_images(output_dir: Path, output_type: str) -> List[Path]:
    """收集需要进行 agent 强化的图片路径。"""
    if output_type == "poster":
        # 生成阶段约定 poster 的 section_id 为 "poster"
        candidates = list(output_dir.glob("poster.*"))
        return candidates
    else:
        # slides: 所有非 PDF 图片都进行一次增强
        imgs = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            imgs.extend(output_dir.glob(ext))
        # 排除 slides.pdf
        imgs = [p for p in imgs if p.name != "slides.pdf"]
        return imgs


def run_zimage_agent_pipeline(args: argparse.Namespace) -> None:
    agent = "zimage_pipeline_agent"
    log_agent_start(agent)

    # 1. 解析/归一化输入
    input_path = normalize_input_path(args.input)
    project_name = get_project_name(args.input)

    style_type, custom_style = parse_style(args.style)

    config = {
        "input_path": input_path,
        "content_type": args.content,
        "output_type": args.output,
        "image_backend": "zimage",  # 固定使用本地 Z-Image
        "local_image_model": args.local_image_model,
        "style": style_type,
        "custom_style": custom_style,
        "slides_length": args.length,
        "poster_density": args.density,
        "fast_mode": args.fast,
    }

    base_dir = get_base_dir(args.output_dir, project_name, args.content)
    config_dir = get_config_dir(base_dir, config)

    log_agent_info(agent, f"project={project_name}, base={base_dir}, config={config_dir}")

    # 2. 运行主流水线（从自动检测到的起始 stage 开始）
    # detect_start_stage 定义在 paper2slides.core 包的 __init__ 中，而不是 pipeline.py
    from paper2slides.core import detect_start_stage

    from_stage = detect_start_stage(base_dir, config_dir, config)
    if from_stage != "rag":
        logger.info(f"Reusing existing checkpoints, starting from: {from_stage}")

    asyncio.run(run_pipeline(base_dir, config_dir, config, from_stage))

    # 2.5 加载本次生成使用的内容规划（checkpoint_plan.json），准备用于文字精准对齐
    plan_text_spans: List[Dict[str, Any]] = []
    plan_ckpt = get_plan_checkpoint(config_dir)
    if plan_ckpt.exists():
        plan_data = load_json(plan_ckpt)
        plan_text_spans = _build_plan_text_spans(plan_data)
        log_agent_info(agent, f"loaded {len(plan_text_spans)} text spans from plan checkpoint")
    else:
        log_agent_warning(agent, f"plan checkpoint not found: {plan_ckpt}")
    
    # 2.6 将 plan_text_spans 落盘，供 poster_text_match tool 使用（方案 B：通过路径加载，避免把大候选塞进 tool 参数）
    plan_text_spans_path: str | None = None
    if plan_text_spans:
        plan_text_spans_file = config_dir / "plan_text_spans.json"
        with plan_text_spans_file.open("w", encoding="utf-8") as f:
            json.dump(plan_text_spans, f, ensure_ascii=False, indent=2)
        plan_text_spans_path = str(plan_text_spans_file)
        log_agent_info(agent, f"saved plan_text_spans to {plan_text_spans_file}")

    # 3. 找到本次生成的输出图片目录
    output_dir = _find_latest_output_dir(config_dir)
    log_agent_info(agent, f"raw outputs dir={output_dir}")

    # 4. 收集需要强化的图片
    image_paths = _collect_images(output_dir, args.output)
    if not image_paths:
        log_agent_warning(agent, f"no images found under {output_dir}, skip refinement")
        return

    # 5. 构造 PosterRefinerAgent，并依次对图片做强化
    # Z-Image 权重路径：优先使用 CLI/local env，其次退回到与 run_poster.sh 一致的默认路径
    zimage_model = (
        args.local_image_model
        or "Tongyi-MAI/Z-Image-Turbo"
    )
    refiner = PosterRefinerAgent(
        zimage_model_name=zimage_model,
        device=args.device,
        style_name=style_type,
        plan_text_spans=[],  # 方案 B：不直接传大列表给 agent，交由 tool 通过路径加载
        plan_text_spans_path=plan_text_spans_path,
    )

    for img_path in image_paths:
        log_agent_info(agent, f"refine image={img_path.name}")
        img = Image.open(img_path).convert("RGB")

        # 让 PosterRefinerAgent 内部基于 style 和 plan_text_spans 构造 src/tar prompt，
        # 这里仅设置清晰度阈值，保持调用接口简洁。
        refined = refiner.run(
            image=img,
            clarity_threshold=7.0,
            max_rounds=3,
            bbox_limit=5,
        )

        refined.save(img_path)
        log_agent_success(agent, f"refined image saved: {img_path.name}")

    log_agent_success(agent, "all images refined")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper2Slides Z-Image Agent Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", "-i", help="Input file or directory path (relative or absolute)")
    parser.add_argument("--content", choices=["paper", "general"], default="paper", help="Content type")
    parser.add_argument("--output", choices=["poster", "slides"], default="poster", help="Output type")
    parser.add_argument("--style", default="academic", help="Style: academic, doraemon, or custom description")
    parser.add_argument("--length", choices=["short", "medium", "long"], default="short",
                        help="Slides length when output=slides")
    parser.add_argument("--density", choices=["sparse", "medium", "dense"], default="medium",
                        help="Poster density when output=poster")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent.parent.parent / "outputs"),
                        help="Output root directory")
    parser.add_argument("--fast", action="store_true", help="Fast mode: parse only, no RAG indexing")
    parser.add_argument("--local-image-model", default="Tongyi-MAI/Z-Image-Turbo",
                        help="Local Z-Image model path or repo id")
    parser.add_argument("--device", default="cuda",
                        help="Device for local Z-Image (cuda/cpu)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # 常规 logging（用于 pipeline 自身）
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    if not args.input:
        parser.print_help()
        return

    run_zimage_agent_pipeline(args)


if __name__ == "__main__":
    main()
