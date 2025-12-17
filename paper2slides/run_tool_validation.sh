#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Configuration
IMAGE_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Paper2Slides/outputs/agent_logs/run_016/outputs/poster.png"  # 替换为实际图片路径
SRC_PROMPT="Black text 'Complhomd&Molivation'"
TAR_PROMPT="Black text 'Background & Motivation'"
BBOX="100,350,321,372"    
OUTPUT_PATH="validation_result.png"

python -m paper2slides.tool_validation \
    --image_path "$IMAGE_PATH" \
    --src_prompt "$SRC_PROMPT" \
    --tar_prompt "$TAR_PROMPT" \
    --bbox "$BBOX" \
    --output_path "$OUTPUT_PATH"
