"""统一的配置加载模块，供所有 agent 工具使用。"""

from pathlib import Path
from typing import Any

import yaml


# 默认配置文件路径（tools 目录下的 config.yml）
_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yml"

# 全局配置缓存
_CONFIG_CACHE: dict[str, dict] = {}

# 当前使用的配置路径
_CURRENT_CONFIG_PATH: Path = _DEFAULT_CONFIG_PATH


def set_config_path(config_path: str | Path) -> None:
    """设置全局配置文件路径，供外部（如 zimage_pipeline_agent）调用。"""
    global _CURRENT_CONFIG_PATH
    _CURRENT_CONFIG_PATH = Path(config_path)
    # 清除缓存以便重新加载
    _CONFIG_CACHE.clear()


def get_config() -> dict:
    """获取当前配置（带缓存）。"""
    cache_key = str(_CURRENT_CONFIG_PATH)
    if cache_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[cache_key]
    
    with open(_CURRENT_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    
    _CONFIG_CACHE[cache_key] = config
    return config


def get_flowedit_config() -> dict[str, Any]:
    """获取 flowedit 配置节。"""
    return get_config().get("flowedit", {})


def get_text_match_config() -> dict[str, Any]:
    """获取 text_match 配置节。"""
    return get_config().get("text_match", {})


def get_glm_image_config() -> dict[str, Any]:
    """获取 glm_image 配置节。"""
    return get_config().get("glm_image", {})


def reload_config() -> dict:
    """强制重新加载配置。"""
    _CONFIG_CACHE.clear()
    return get_config()

