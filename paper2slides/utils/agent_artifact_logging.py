import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from paper2slides.utils.logging import get_logger


logger = get_logger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def get_default_log_root(agent_name: str) -> Path:
    """返回给定 agent 的默认日志根目录：cwd/agent_logs/<agent_name>."""

    return Path(os.getcwd()) / "agent_logs" / agent_name


def save_json_log(
    agent_name: str,
    func_name: str,
    payload: Dict[str, Any],
    suffix: Optional[str] = None,
    log_root: Optional[Path] = None,
) -> None:
    """将数值 / LLM 输入输出等信息保存为 json 日志.

    日志路径结构：
        <log_root_or_default>/<agent_name>/<func_name>/<func_name>[_suffix]_timestamp.json
    """

    root = log_root or get_default_log_root(agent_name)
    subdir = root / func_name
    _ensure_dir(subdir)

    ts = _timestamp()
    suffix_str = f"_{suffix}" if suffix else ""
    path = subdir / f"{func_name}{suffix_str}_{ts}.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON log to {path}")

def save_before_after_image(
    agent_name: str,
    func_name: str,
    before_img: Image.Image,
    after_img: Image.Image,
    suffix: Optional[str] = None,
    log_root: Optional[Path] = None,
) -> None:
    """将处理前后的图片横向拼接后保存为 PNG.

    日志路径结构：
        <log_root_or_default>/<agent_name>/<func_name>/<func_name>[_suffix]_timestamp.png
    """

    root = log_root or get_default_log_root(agent_name)
    subdir = root / func_name
    _ensure_dir(subdir)

    ts = _timestamp()
    suffix_str = f"_{suffix}" if suffix else ""
    path = subdir / f"{func_name}{suffix_str}_{ts}.png"

    # 统一高度，按比例缩放
    w1, h1 = before_img.size
    w2, h2 = after_img.size
    target_h = max(h1, h2)

    def _resize_to_h(img: Image.Image, h: int) -> Image.Image:
        w, _ = img.size
        if h == img.size[1]:
            return img
        new_w = int(w * h / img.size[1])
        return img.resize((new_w, h), Image.LANCZOS)

    b = _resize_to_h(before_img, target_h)
    a = _resize_to_h(after_img, target_h)

    bw, _ = b.size
    aw, _ = a.size
    canvas = Image.new("RGB", (bw + aw, target_h))
    canvas.paste(b, (0, 0))
    canvas.paste(a, (bw, 0))

    canvas.save(path)
    print(f"Saved image log to {path}")


# 定义一组高对比度颜色用于绘制多个bbox
_BBOX_COLORS = [
    (255, 0, 0),      # 红
    (0, 255, 0),      # 绿
    (0, 0, 255),      # 蓝
    (255, 255, 0),    # 黄
    (255, 0, 255),    # 品红
    (0, 255, 255),    # 青
    (255, 128, 0),    # 橙
    (128, 0, 255),    # 紫
]


def save_bbox_visualization(
    agent_name: str,
    func_name: str,
    image: Image.Image,
    bboxes: List[Tuple[int, int, int, int]],
    suffix: Optional[str] = None,
    log_root: Optional[Path] = None,
    line_width: int = 3,
) -> Path:
    """将带有 bbox 框的可视化图片保存为 PNG.

    每个 bbox 使用不同颜色绘制，并在左上角标注序号。

    Args:
        agent_name: agent 名称
        func_name: 函数名称
        image: 原始图片
        bboxes: bbox 列表，每个 bbox 为 (x0, y0, x1, y1)
        suffix: 可选的文件名后缀
        log_root: 日志根目录，默认为 agent_logs/<agent_name>
        line_width: 边框线宽

    Returns:
        保存的文件路径
    """
    root = log_root or get_default_log_root(agent_name)
    subdir = root / func_name
    _ensure_dir(subdir)

    ts = _timestamp()
    suffix_str = f"_{suffix}" if suffix else ""
    path = subdir / f"{func_name}{suffix_str}_{ts}.png"

    # 复制图片以避免修改原图
    vis_img = image.copy()
    draw = ImageDraw.Draw(vis_img)

    # 尝试加载字体，失败则使用默认字体
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except (OSError, IOError):
            font = ImageFont.load_default()

    for i, bbox in enumerate(bboxes):
        x0, y0, x1, y1 = bbox
        color = _BBOX_COLORS[i % len(_BBOX_COLORS)]

        # 绘制矩形边框
        for offset in range(line_width):
            draw.rectangle(
                [x0 + offset, y0 + offset, x1 - offset, y1 - offset],
                outline=color,
            )

        # 在左上角绘制序号标签
        label = str(i + 1)
        # 绘制背景以提高可读性
        text_bbox = draw.textbbox((x0, y0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        padding = 2
        draw.rectangle(
            [x0, y0, x0 + text_w + 2 * padding, y0 + text_h + 2 * padding],
            fill=color,
        )
        draw.text((x0 + padding, y0 + padding), label, fill=(255, 255, 255), font=font)

    vis_img.save(path)
    logger.info(f"Saved bbox visualization to {path}")
    print(f"Saved bbox visualization to {path}")
    return path