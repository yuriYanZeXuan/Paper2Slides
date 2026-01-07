"""
Demo for chart_tool.py
----------------------
前置：
1) 安装依赖：pip install -r requirements.txt
2) 配置你的 OpenAI 兼容服务（远程/本地均可）：
   - 方式 A：在环境变量里设置（推荐）：
     export OPENAI_BASE_URL="https://xxx.yyy/v1"   # 注意带 /v1
     export OPENAI_API_KEY="你的key"
     export LLM_MODEL="gpt-4o-mini"                # 可选
   - 方式 B：不设置环境变量，直接在调用 llm_chart_spec 时传 base_url/api_key
   （如果你使用本项目的本地代理 gemini_proxy.py，也可以把 base_url 指向 http://127.0.0.1:51958/v1）

运行：
  python Paper2Slides/dev/z_chart_tool_demo.py

输出：
  Paper2Slides/dev/_chart_demo_out/*.png
"""

from __future__ import annotations

from pathlib import Path

from dev.chart_tool import llm_chart_spec, render_chart


def run_one(prompt: str, out_path: Path, image_path: Path | None = None):
    spec = llm_chart_spec(prompt=prompt, image_path=image_path)
    out = render_chart(spec, out_path)
    return out


def main():
    out_dir = Path(__file__).resolve().parent / "_chart_demo_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = [
        (
            "画一个简洁蓝色柱状图：类别=搜索/推荐/广告/其他，对应数值=120/260/180/90。标题：Q4 流量来源（单位：万）",
            out_dir / "bar_q4_traffic.png",
            None,
        ),
        (
            "画一个折线图：x=2021,2022,2023,2024；y=10,23,35,50。标题：用户规模增长；y轴：百万用户",
            out_dir / "line_user_growth.png",
            None,
        ),
        (
            "画一个饼图：研发/销售/运营/行政=40/30/20/10。标题：部门预算占比",
            out_dir / "pie_budget.png",
            None,
        ),
    ]

    outs = []
    for prompt, out_path, img in examples:
        outs.append(run_one(prompt, out_path, img))

    print("Generated:")
    for p in outs:
        print(" -", p)


if __name__ == "__main__":
    main()


