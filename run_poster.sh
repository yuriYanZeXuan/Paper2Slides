#!/usr/bin/env bash

# IMAGE_BACKEND 可选：
#   - gemini：远程 Gemini Image（默认）
#   - zimage：本地 Z-Image 模型
#   - qwen  ：本地 Qwen-Image 模型
IMAGE_BACKEND=${IMAGE_BACKEND:-qwen}

# 根据后端选择默认本地权重路径（可通过环境变量 LOCAL_IMAGE_MODEL 覆盖）
if [ "${IMAGE_BACKEND}" = "qwen" ]; then
  LOCAL_IMAGE_MODEL=${LOCAL_IMAGE_MODEL:-/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/qwen_image}
else
  LOCAL_IMAGE_MODEL=${LOCAL_IMAGE_MODEL:-/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image}
fi

python -m paper2slides \
     --input /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/PosterGen/data/Object_Pose_Estimation_with_Statistical_Guarantees_Conformal_Keypoint_Detection_and_Geometric_Uncertainty_Propagation/paper.pdf \
     --output poster \
     --style doraemon \
     --length medium \
     --fast \
     --image-backend "${IMAGE_BACKEND}" \
     --local-image-model "${LOCAL_IMAGE_MODEL}"