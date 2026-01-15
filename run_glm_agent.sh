#!/usr/bin/env bash

# 使用 GLM-Image + Agent 生成 academic 风格的 poster，并使用本地权重
export LOCAL_IMAGE_HEIGHT=1024
export LOCAL_IMAGE_WIDTH=768

LOCAL_GLM_IMAGE_MODEL="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/GLM-Image"

python -m paper2slides.agents.GLMimage_pipeline_agent \
  --input /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/postergenparserunit/examples/tusen_1210.md \
  --output poster \
  --style academic \
  --local-image-model "${LOCAL_GLM_IMAGE_MODEL}" \
  --fast \
  --device cuda
