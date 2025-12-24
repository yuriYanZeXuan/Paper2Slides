#!/bin/bash
# 对比测试脚本：同时运行有/无 In-Context 模式，方便对比效果

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Configuration
IMAGE_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Paper2Slides/outputs/agent_logs/run_058/outputs/poster.png"
SRC_PROMPT="Black text 'Complhomd&Molivation'"
TAR_PROMPT="Black text 'Background & Motivation'"
BBOX="100,350,321,390"

# 从环境变量获取 LOCAL_IMAGE_MODEL，如果未设置则使用默认路径
LOCAL_IMAGE_MODEL=${LOCAL_IMAGE_MODEL:-/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image}

echo "============================================"
echo "=== FlowEdit In-Context Comparison Test ==="
echo "============================================"
echo ""
echo "Image: $IMAGE_PATH"
echo "Source: $SRC_PROMPT"
echo "Target: $TAR_PROMPT"
echo "BBox: $BBOX"
echo ""

# 1. 运行普通模式（无 In-Context）
echo ">>> [1/2] Running WITHOUT In-Context mode..."
OUTPUT_NORMAL="validation_compare_normal.png"
python -m paper2slides.tool_validation \
    --image_path "$IMAGE_PATH" \
    --src_prompt "$SRC_PROMPT" \
    --tar_prompt "$TAR_PROMPT" \
    --bbox "$BBOX" \
    --output_path "$OUTPUT_NORMAL" \
    --model_path "$LOCAL_IMAGE_MODEL"

echo ""
echo ">>> [2/2] Running WITH In-Context mode..."
OUTPUT_INCONTEXT="validation_compare_incontext.png"
python -m paper2slides.tool_validation \
    --image_path "$IMAGE_PATH" \
    --src_prompt "$SRC_PROMPT" \
    --tar_prompt "$TAR_PROMPT" \
    --bbox "$BBOX" \
    --output_path "$OUTPUT_INCONTEXT" \
    --model_path "$LOCAL_IMAGE_MODEL" \
    --in_context

echo ""
echo "============================================"
echo "=== Comparison Complete ==="
echo "============================================"
echo "Normal output:     $OUTPUT_NORMAL"
echo "In-Context output: $OUTPUT_INCONTEXT"
echo ""
echo "Compare these two images to see the effect of In-Context mode."
echo "In-Context mode should preserve more similarity with the original image."

