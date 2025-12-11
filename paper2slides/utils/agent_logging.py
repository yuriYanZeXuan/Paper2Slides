"""Agent-style logging utilities (rich console), inspired by PosterGen.

- 带有 [ AgentName ] 头部
- 自动附加调用代码位置 (相对路径:行号)
"""

import inspect
from pathlib import Path

from rich.console import Console
from rich.text import Text

console = Console()


def _get_caller_info() -> str:
    """返回调用方的相对文件路径和行号，例如 `paper2slides/agents/zimage_pipeline_agent.py:42`"""
    frame = inspect.currentframe()
    # 向上回溯到第一个非本文件的调用帧
    current_frame = frame.f_back
    while current_frame:
        filepath = Path(current_frame.f_code.co_filename)
        if filepath.name != "agent_logging.py":
            line_number = current_frame.f_lineno
            # 直接按工作目录做相对路径，失败也直接退回文件名
            try:
                relative_path = filepath.relative_to(Path.cwd())
            except ValueError:
                relative_path = filepath.name
            return f"{relative_path}:{line_number}"
        current_frame = current_frame.f_back
    return "unknown:0"


def log(agent_name: str, level: str, message: str, max_width: int = 18, show_location: bool = True) -> None:
    """中心化的 agent 日志输出接口。

    Args:
        agent_name: agent 或模块名称（如 "zimage_agent"）
        level: 日志等级："info" | "warning" | "error" | "success" | "debug"
        message: 日志内容
        max_width: 头部中 agent 名的宽度
        show_location: 是否显示代码位置信息
    """
    display_name = agent_name.replace("_agent", "").replace("_node", "").replace("_", " ").title()

    level_colors = {
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
        "success": "green",
        "debug": "blue",
    }
    level_color = level_colors.get(level.lower(), "white")

    header = Text(f"[ {display_name:^{max_width}} ]", style=f"bold {level_color}")

    if show_location:
        location_info = _get_caller_info()
        location_text = Text(f" [{location_info}] ", style="dim")
        header.append(location_text)

    body = Text(message)
    console.print(header, body)


def log_agent_start(agent_name: str, show_location: bool = True) -> None:
    log(agent_name, "info", f"starting {agent_name}...", show_location=show_location)


def log_agent_success(agent_name: str, message: str, show_location: bool = True) -> None:
    log(agent_name, "success", f"✅ {message}", show_location=show_location)


def log_agent_error(agent_name: str, message: str, show_location: bool = True) -> None:
    log(agent_name, "error", f"❌ {message}", show_location=show_location)


def log_agent_warning(agent_name: str, message: str, show_location: bool = True) -> None:
    log(agent_name, "warning", f"⚠️ {message}", show_location=show_location)


def log_agent_info(agent_name: str, message: str, show_location: bool = True) -> None:
    log(agent_name, "info", message, show_location=show_location)
