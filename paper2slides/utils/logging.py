"""
Project logging helpers.

注意：
- 该文件名为 `logging.py`，在“从 paper2slides/utils 目录直接运行脚本”时可能会遮蔽标准库 `logging`。
- 正确用法是以包方式运行（例如 `python -m paper2slides...`），或者确保工作目录不是 `paper2slides/utils`。
"""

from __future__ import annotations

import logging
from typing import Optional


def setup_logging(level: int = logging.INFO) -> None:
    """Setup basic logging configuration for the project."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a standard library logger."""
    return logging.getLogger(name or __name__)


def log_section(title: str) -> None:
    """Print a readable section divider in logs."""
    logger = get_logger(__name__)
    line = "─" * 60
    logger.info("")
    logger.info(line)
    logger.info(title)
    logger.info(line)


