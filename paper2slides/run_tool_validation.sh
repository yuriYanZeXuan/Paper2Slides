#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Configuration
IMAGE_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Paper2Slides/outputs/agent_logs/run_016/outputs/poster.png"
SRC_PROMPT="Black text 'Complhomd&Molivation'"
TAR_PROMPT="Black text 'Background & Motivation'"
BBOX="100,350,321,390"

# 从环境变量获取 LOCAL_IMAGE_MODEL，如果未设置则使用默认路径
LOCAL_IMAGE_MODEL=${LOCAL_IMAGE_MODEL:-/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image}

# 是否启用 In-Context 模式（可通过环境变量 IN_CONTEXT=1 启用）
IN_CONTEXT_FLAG=""
if [ "${IN_CONTEXT}" = "1" ] || [ "${IN_CONTEXT}" = "true" ]; then
    IN_CONTEXT_FLAG="--in_context"
    echo "=== In-Context Mode ENABLED ==="
fi

# 根据是否启用 in_context 设置输出路径
if [ -n "$IN_CONTEXT_FLAG" ]; then
    OUTPUT_PATH="validation_result_incontext.png"
else
    OUTPUT_PATH="validation_result.png"
fi

echo "Image Path: $IMAGE_PATH"
echo "Source Prompt: $SRC_PROMPT"
echo "Target Prompt: $TAR_PROMPT"
echo "BBox: $BBOX"
echo "Output Path: $OUTPUT_PATH"
echo "Model Path: $LOCAL_IMAGE_MODEL"
echo ""

CMD="python -m paper2slides.tool_validation \
    --image_path \"$IMAGE_PATH\" \
    --src_prompt \"$SRC_PROMPT\" \
    --tar_prompt \"$TAR_PROMPT\" \
    --bbox \"$BBOX\" \
    --output_path \"$OUTPUT_PATH\" \
    --model_path \"$LOCAL_IMAGE_MODEL\" \
    $IN_CONTEXT_FLAG"

echo "Running command:"
echo "$CMD"
echo ""

eval "$CMD"
