#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 dev/chart_tool.py --base-url "http://127.0.0.1:51958/v1" "$@"


