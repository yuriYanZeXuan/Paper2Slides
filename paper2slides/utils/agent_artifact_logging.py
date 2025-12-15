"""Agent artifact logging utilities.

重构后的日志管理：
- 每次运行（session）创建一个独立的序号目录：agent_logs/run_001/, run_002/, ...
- 所有 agent 的日志和输出统一保存在当前 session 目录下
- 目录结构：agent_logs/run_XXX/<agent_name>/<func_name>/...
"""

import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from paper2slides.utils.logging import get_logger


logger = get_logger(__name__)


# ============ 全局 Session 管理 ============

_session_lock = threading.Lock()
_current_session_dir: Optional[Path] = None


def _get_agent_logs_root() -> Path:
    """返回 agent_logs 根目录，统一放在 outputs 目录下。"""
    return Path(os.getcwd()) / "outputs" / "agent_logs"


def _find_next_run_number(root: Path) -> int:
    """找到下一个可用的 run 序号。"""
    if not root.exists():
        return 1
    
    max_num = 0
    pattern = re.compile(r"^run_(\d+)$")
    
    for item in root.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)
    
    return max_num + 1


def init_session(session_name: Optional[str] = None) -> Path:
    """初始化一个新的日志 session，返回 session 目录路径。
    
    Args:
        session_name: 可选的 session 名称。如果不提供，则自动使用 run_XXX 格式。
    
    Returns:
        session 目录的 Path
    """
    global _current_session_dir
    
    with _session_lock:
        root = _get_agent_logs_root()
        root.mkdir(parents=True, exist_ok=True)
        
        if session_name:
            session_dir = root / session_name
        else:
            next_num = _find_next_run_number(root)
            session_dir = root / f"run_{next_num:03d}"
        
        session_dir.mkdir(parents=True, exist_ok=True)
        _current_session_dir = session_dir
        
        # 保存 session 元信息
        meta_path = session_dir / "session_meta.json"
        meta = {
            "session_name": session_dir.name,
            "created_at": datetime.now().isoformat(),
            "cwd": os.getcwd(),
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Initialized logging session: {session_dir}")
        print(f"[agent_logging] Session initialized: {session_dir}")
        
        return session_dir


def get_current_session_dir() -> Path:
    """获取当前 session 目录。如果尚未初始化，则自动初始化。"""
    global _current_session_dir
    
    with _session_lock:
        if _current_session_dir is None:
            # 自动初始化一个新 session
            return init_session()
        return _current_session_dir


def has_active_session() -> bool:
    """检查是否有活跃的 session（已初始化但未显式关闭）。"""
    with _session_lock:
        return _current_session_dir is not None


def get_session_output_dir() -> Path:
    """获取当前 session 的 outputs 目录：<session_dir>/outputs.
    
    用于保存生成的图片等输出文件，与 agent 日志统一在同一 session 下。
    """
    session_dir = get_current_session_dir()
    output_dir = session_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def set_session_dir(session_dir: Path) -> None:
    """手动设置当前 session 目录（用于恢复之前的 session）。"""
    global _current_session_dir
    
    with _session_lock:
        if not session_dir.exists():
            session_dir.mkdir(parents=True, exist_ok=True)
        _current_session_dir = session_dir
        logger.info(f"Session dir set to: {session_dir}")


# ============ 兼容旧 API ============

def get_default_log_root(agent_name: str) -> Path:
    """返回给定 agent 的日志目录：<session_dir>/<agent_name>.
    
    兼容旧代码，但现在会自动放在当前 session 下。
    """
    session_dir = get_current_session_dir()
    return session_dir / agent_name


# ============ 内部工具函数 ============

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


# ============ 日志保存函数 ============

def save_json_log(
    agent_name: str,
    func_name: str,
    payload: Dict[str, Any],
    suffix: Optional[str] = None,
    log_root: Optional[Path] = None,
) -> Path:
    """将数值 / LLM 输入输出等信息保存为 json 日志.

    日志路径结构：
        <session_dir>/<agent_name>/<func_name>/<func_name>[_suffix]_timestamp.json
    
    Returns:
        保存的文件路径
    """
    root = log_root or get_default_log_root(agent_name)
    subdir = root / func_name
    _ensure_dir(subdir)

    ts = _timestamp()
    suffix_str = f"_{suffix}" if suffix else ""
    path = subdir / f"{func_name}{suffix_str}_{ts}.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved JSON log to {path}")
    print(f"Saved JSON log to {path}")
    return path


def save_before_after_image(
    agent_name: str,
    func_name: str,
    before_img: Image.Image,
    after_img: Image.Image,
    suffix: Optional[str] = None,
    log_root: Optional[Path] = None,
) -> Path:
    """将处理前后的图片横向拼接后保存为 PNG.

    日志路径结构：
        <session_dir>/<agent_name>/<func_name>/<func_name>[_suffix]_timestamp.png
    
    Returns:
        保存的文件路径
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
    logger.info(f"Saved image log to {path}")
    print(f"Saved image log to {path}")
    return path


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
        log_root: 日志根目录
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


def save_image(
    agent_name: str,
    func_name: str,
    image: Image.Image,
    suffix: Optional[str] = None,
    log_root: Optional[Path] = None,
) -> Path:
    """保存单张图片到日志目录。

    Args:
        agent_name: agent 名称
        func_name: 函数名称
        image: 要保存的图片
        suffix: 可选的文件名后缀
        log_root: 日志根目录

    Returns:
        保存的文件路径
    """
    root = log_root or get_default_log_root(agent_name)
    subdir = root / func_name
    _ensure_dir(subdir)

    ts = _timestamp()
    suffix_str = f"_{suffix}" if suffix else ""
    path = subdir / f"{func_name}{suffix_str}_{ts}.png"

    image.save(path)
    logger.info(f"Saved image to {path}")
    print(f"Saved image to {path}")
    return path
