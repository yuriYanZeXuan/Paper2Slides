#!/usr/bin/env bash

# 使用 Z-Image + Agent 生成 academic 风格的 poster，并使用本地权重
# 权重路径与 run_poster.sh 中的 Z-Image 默认路径保持一致

LOCAL_IMAGE_MODEL=${LOCAL_IMAGE_MODEL:-/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image}

python -m paper2slides.agents.zimage_pipeline_agent \
  --input /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/PosterGen/data/Object_Pose_Estimation_with_Statistical_Guarantees_Conformal_Keypoint_Detection_and_Geometric_Uncertainty_Propagation/paper.pdf \
  --output poster \
  --style academic \
  --local-image-model "${LOCAL_IMAGE_MODEL}"
