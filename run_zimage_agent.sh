#!/usr/bin/env bash

# 使用 Z-Image + Agent 生成 academic 风格的 poster，并使用本地权重
# 权重路径与 run_poster.sh 中的 Z-Image 默认路径保持一致
export LOCAL_IMAGE_HEIGHT=960
export LOCAL_IMAGE_WIDTH=1280

LOCAL_IMAGE_MODEL="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image"

# Qwen Image Edit 权重路径（供 eraser_qwen_editor 工具加载）
export LOCAL_QWEN_EDIT_MODEL="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/qwen_edit_2511"

# Refiner 模式:
#   - pptx: (默认) 新流程 - 擦除模糊文字 + PPTX 渲染清晰文字，输出 PPTX/PDF
#   - legacy: 旧流程 - iterative FlowEdit
REFINER_MODE="legacy"

python -m paper2slides.agents.zimage_pipeline_agent \
  --input /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/postergenparserunit/examples/tusen_1210.md \
  --output poster \
  --style academic \
  --local-image-model "${LOCAL_IMAGE_MODEL}" \
  --refiner-mode "${REFINER_MODE}" \
  --fast \
  --device cuda

