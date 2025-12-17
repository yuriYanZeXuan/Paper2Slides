#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Configuration
IMAGE_PATH="/Users/yanzexuan/Downloads/test.jpg"
SRC_PROMPT="original text"
TAR_PROMPT="edited text"
BBOX="100,100,500,500"
OUTPUT_PATH="$SCRIPT_DIR/validation_result.png"

python -m paper2slides.tool_validation \
    --image_path "$IMAGE_PATH" \
    --src_prompt "$SRC_PROMPT" \
    --tar_prompt "$TAR_PROMPT" \
    --bbox "$BBOX" \
    --output_path "$OUTPUT_PATH"

