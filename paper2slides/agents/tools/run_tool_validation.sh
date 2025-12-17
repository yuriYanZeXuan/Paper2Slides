#!/bin/bash


# ================= 配置参数 =================
# 在这里直接修改测试参数
IMAGE_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Paper2Slides/outputs/agent_logs/run_016/outputs/poster.png"  # 替换为实际图片路径
SRC_PROMPT="Black text 'Complhomd&Molivation'"
TAR_PROMPT="Black text 'Background & Motivation'"
BBOX="100,350,321,372"                  # 可选: x1,y1,x2,y2
OUTPUT_PATH="result.png"
# ===========================================

echo "Running validation with:"
echo "  Image: $IMAGE_PATH"
echo "  Task:  '$SRC_PROMPT' -> '$TAR_PROMPT'"

# 运行 Python 脚本
python tool_validation.py \
    --image_path "$IMAGE_PATH" \
    --src_prompt "$SRC_PROMPT" \
    --tar_prompt "$TAR_PROMPT" \
    --bbox "$BBOX" \
    --output_path "$OUTPUT_PATH"
