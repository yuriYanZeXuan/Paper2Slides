import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

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
