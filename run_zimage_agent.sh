#!/usr/bin/env bash

# 固定示例：使用 Z-Image + Agent 生成 academic 风格的 poster
# 需要时手动把 path/to/paper.pdf 改成真实路径

python -m paper2slides.agents.zimage_pipeline_agent \
  --input /mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/PosterGen/data/Object_Pose_Estimation_with_Statistical_Guarantees_Conformal_Keypoint_Detection_and_Geometric_Uncertainty_Propagation/paper.pdf \
  --output poster \
  --style academic
