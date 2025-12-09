#!/usr/bin/env bash

# IMAGE_BACKEND 可选：gemini（默认）或 zimage（使用本地 Z-Image 模型）
IMAGE_BACKEND=${IMAGE_BACKEND:-gemini}

# 本地 Z-Image 权重路径（可通过环境变量覆盖）
LOCAL_IMAGE_MODEL=${LOCAL_IMAGE_MODEL:-/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image}

python -m paper2slides \
     --input /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/PosterGen/data/Object_Pose_Estimation_with_Statistical_Guarantees_Conformal_Keypoint_Detection_and_Geometric_Uncertainty_Propagation/paper.pdf \
     --output slides \
     --style doraemon \
     --length medium \
     --fast \
     --image-backend "${IMAGE_BACKEND}" \
     --local-image-model "${LOCAL_IMAGE_MODEL}"