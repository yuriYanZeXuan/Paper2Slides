#!/usr/bin/env bash
set -euo pipefail

# 便于在远程服务器一键运行 chart_tool.py 的脚本
#
# 最简用法（默认使用本地 OpenAI 兼容接口，与 poster_refiner.py 一致）：
#   bash Paper2Slides/dev/run_chart_tool.sh \
#     --prompt "画一个柱状图：A/B/C=10/20/15，标题：销量" \
#     --out "/tmp/chart.png"
#
# 可选：命令行覆盖（远程/不同 base_url 时使用；注意要带 /v1，避免 /openai vs /openai/v1 404）
#   bash Paper2Slides/dev/run_chart_tool.sh \
#     --prompt "画一个折线图：x=2021,2022,2023,2024；y=10,23,35,50。标题：用户增长" \
#     --out "/tmp/line.png" \
#     --base-url "https://xxx.yyy/v1" \
#     --api-key "your_key" \
#     --model "gpt-4o-mini"
#
#   # 可选参考图（用于配色/风格）
#   bash Paper2Slides/dev/run_chart_tool.sh \
#     --prompt "用参考图风格画一个饼图：研发/销售/运营/行政=40/30/20/10，标题：预算占比" \
#     --image "/path/to/style.png" \
#     --out "/tmp/pie.png"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROMPT=""
IMAGE=""
OUT="chart.png"
# 默认值对齐 paper2slides/agents/poster_refiner.py：写死 base_url，避免路径差异导致 404
BASE_URL="http://127.0.0.1:51958/v1"
API_KEY=""
MODEL=""
PRINT_SPEC="0"

usage() {
  cat <<'EOF'
用法：
  bash Paper2Slides/dev/run_chart_tool.sh --prompt "..." --out /path/out.png [--image /path/style.png]
参数：
  --prompt      必选，文字描述
  --out         可选，输出路径（默认 chart.png）
  --image       可选，参考图片路径
  --base-url    可选，OpenAI 兼容 base_url（默认 http://127.0.0.1:51958/v1）
  --api-key     可选，API key（如你的网关需要鉴权）
  --model       可选，模型名
  --print-spec  可选，打印生成的 ChartSpec JSON
示例：
  bash Paper2Slides/dev/run_chart_tool.sh --prompt "画一个柱状图：A/B/C=10/20/15，标题：销量" --out /tmp/chart.png
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      PROMPT="${2:-}"; shift 2;;
    --image)
      IMAGE="${2:-}"; shift 2;;
    --out)
      OUT="${2:-}"; shift 2;;
    --base-url)
      BASE_URL="${2:-}"; shift 2;;
    --api-key)
      API_KEY="${2:-}"; shift 2;;
    --model)
      MODEL="${2:-}"; shift 2;;
    --print-spec)
      PRINT_SPEC="1"; shift 1;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "未知参数：$1" >&2
      usage
      exit 2;;
  esac
done

if [[ -z "${PROMPT}" ]]; then
  echo "缺少 --prompt" >&2
  usage
  exit 2
fi

cd "${REPO_ROOT}"

cmd=(python3 "Paper2Slides/dev/chart_tool.py" --prompt "${PROMPT}" --out "${OUT}")
if [[ -n "${IMAGE}" ]]; then
  cmd+=(--image "${IMAGE}")
fi
if [[ -n "${BASE_URL}" ]]; then
  cmd+=(--base-url "${BASE_URL}")
fi
if [[ -n "${API_KEY}" ]]; then
  cmd+=(--api-key "${API_KEY}")
fi
if [[ -n "${MODEL}" ]]; then
  cmd+=(--model "${MODEL}")
fi
if [[ "${PRINT_SPEC}" == "1" ]]; then
  cmd+=(--print-spec)
fi

echo "[run_chart_tool.sh] repo=${REPO_ROOT}"
echo "[run_chart_tool.sh] out=${OUT}"
exec "${cmd[@]}"


