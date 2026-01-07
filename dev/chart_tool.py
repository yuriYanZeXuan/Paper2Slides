"""
Chart Tool Prototype (dev)
-------------------------
输入：可选图片（用于风格参考/配色等）+ 必选文字描述
输出：用 matplotlib 绘制的简洁图表（柱状/折线/饼等）并保存为图片

默认通过 OpenAI 兼容接口（例如本项目的 gemini_proxy.py: http://127.0.0.1:51958/v1/chat/completions）
调用大模型把自然语言解析成结构化 ChartSpec(JSON)，再本地渲染。

Usage:
  python Paper2Slides/dev/chart_tool.py --prompt "画一个柱状图：A/B/C=10/20/15，标题为销量" --out /tmp/chart.png
  python Paper2Slides/dev/chart_tool.py --prompt "画一个简洁折线图：2021-2024=[10,23,35,50]" --image /path/style.png --out /tmp/chart.png
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import matplotlib

# Headless friendly
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Ensure repo root is on sys.path so `import paper2slides` works when running as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "paper2slides").exists() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paper2slides.utils.api_utils import get_openai_client  # noqa: E402


ChartType = Literal["bar", "line", "pie"]


@dataclass
class ChartSpec:
    chart_type: str
    title: str = ""
    subtitle: str = ""
    x_label: str = ""
    y_label: str = ""
    theme: str = "minimal"
    palette: Optional[List[str]] = None
    # data
    categories: Optional[List[str]] = None  # bar
    values: Optional[List[float]] = None  # bar
    x: Optional[List[Union[str, float, int]]] = None  # line
    series: Optional[List[Dict[str, Any]]] = None  # line: [{name, y:[]}]
    labels: Optional[List[str]] = None  # pie
    sizes: Optional[List[float]] = None  # pie
    # misc
    note: str = ""


SYSTEM_PROMPT = """你是一个“图表规格生成器”。用户会给一段中文/英文描述，可能还会给一张参考图片（用于风格/配色）。你的任务：
1) 只输出一个 JSON 对象（不要 Markdown，不要代码块）。
2) JSON 必须满足以下 schema（字段可省略但尽量完整）：
{
  "chart_type": "bar" | "line" | "pie",
  "title": string,
  "subtitle": string,
  "x_label": string,
  "y_label": string,
  "theme": "minimal" | string,
  "palette": [ "#RRGGBB", ... ],  // 可选，尽量从参考图提取 3-6 个色值
  // bar:
  "categories": [string,...],
  "values": [number,...],
  // line:
  "x": [string|number,...],
  "series": [ { "name": string, "y": [number,...] }, ... ],
  // pie:
  "labels": [string,...],
  "sizes": [number,...],
  "note": string
}

约束：
- 图表要“简洁、适合 PPT”，元素尽量少，避免拥挤。
- 如果用户没给出具体数值：你可以合理补全一个小型示例数据（5-8 个点以内），并在 note 里说明是示例。
- 只能返回 JSON。
"""


def _read_image_as_data_url(image_path: Union[str, Path]) -> Tuple[str, str]:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    ext = p.suffix.lower().lstrip(".")
    mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(ext, "application/octet-stream")
    b = p.read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    return mime, f"data:{mime};base64,{b64}"


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort extract a JSON object from model text."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response.")
    # Fast path: pure JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try find first {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"Failed to find JSON object in response: {text[:300]}")
    candidate = m.group(0)
    try:
        obj = json.loads(candidate)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {e}. Snippet: {candidate[:300]}") from e
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object.")
    return obj


def _coerce_float_list(xs: Any) -> Optional[List[float]]:
    if xs is None:
        return None
    if not isinstance(xs, list):
        return None
    out: List[float] = []
    for x in xs:
        if isinstance(x, (int, float)):
            out.append(float(x))
        elif isinstance(x, str):
            try:
                out.append(float(x.strip()))
            except Exception:
                return None
        else:
            return None
    return out


def _coerce_str_list(xs: Any) -> Optional[List[str]]:
    if xs is None:
        return None
    if not isinstance(xs, list):
        return None
    out: List[str] = []
    for x in xs:
        if isinstance(x, str):
            out.append(x)
        else:
            out.append(str(x))
    return out


def _normalize_spec(raw: Dict[str, Any]) -> ChartSpec:
    chart_type = (raw.get("chart_type") or "").strip().lower()
    if chart_type not in ("bar", "line", "pie"):
        # heuristic
        for k in ("bar", "line", "pie"):
            if k in chart_type:
                chart_type = k
                break
    if chart_type not in ("bar", "line", "pie"):
        raise ValueError(f"Unsupported chart_type: {raw.get('chart_type')}")

    spec = ChartSpec(
        chart_type=chart_type,
        title=str(raw.get("title") or ""),
        subtitle=str(raw.get("subtitle") or ""),
        x_label=str(raw.get("x_label") or ""),
        y_label=str(raw.get("y_label") or ""),
        theme=str(raw.get("theme") or "minimal"),
        palette=_coerce_str_list(raw.get("palette")),
        categories=_coerce_str_list(raw.get("categories")),
        values=_coerce_float_list(raw.get("values")),
        labels=_coerce_str_list(raw.get("labels")),
        sizes=_coerce_float_list(raw.get("sizes")),
        note=str(raw.get("note") or ""),
    )

    # line-specific
    x = raw.get("x")
    if isinstance(x, list):
        spec.x = [v if isinstance(v, (int, float, str)) else str(v) for v in x]
    series = raw.get("series")
    if isinstance(series, list):
        cleaned: List[Dict[str, Any]] = []
        for s in series:
            if not isinstance(s, dict):
                continue
            name = str(s.get("name") or "")
            y = _coerce_float_list(s.get("y"))
            if y is None:
                continue
            cleaned.append({"name": name or f"series{len(cleaned)+1}", "y": y})
        spec.series = cleaned or None

    return spec


def _choose_figsize(spec: ChartSpec) -> Tuple[float, float]:
    # PPT-friendly: wide-ish, not too tall
    w, h = 6.4, 3.6  # ~16:9-ish
    if spec.chart_type == "bar":
        n = len(spec.categories or []) or len(spec.values or []) or 5
        w = max(6.0, min(10.0, 4.5 + 0.55 * n))
    elif spec.chart_type == "line":
        n = 0
        if spec.series:
            n = max((len(s.get("y", [])) for s in spec.series), default=0)
        w = max(6.0, min(10.0, 5.5 + 0.35 * n))
    elif spec.chart_type == "pie":
        w, h = 5.2, 4.0
    return w, h


def _apply_minimal_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D0D0",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "grid.color": "#E6E6E6",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
        }
    )


def _render_bar(ax: Any, spec: ChartSpec):
    categories = spec.categories or []
    values = spec.values or []
    if not categories or not values or len(categories) != len(values):
        raise ValueError("bar chart requires equal-length categories and values.")

    colors = spec.palette or ["#2F6FED"]
    color = colors[0]
    ax.bar(categories, values, color=color, edgecolor="none")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    if spec.x_label:
        ax.set_xlabel(spec.x_label)
    if spec.y_label:
        ax.set_ylabel(spec.y_label)
    ax.margins(x=0.02)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # light value labels (only if not too many)
    if len(values) <= 12:
        ymax = max(values) if values else 0
        offset = max(0.01 * ymax, 0.2)
        for i, v in enumerate(values):
            ax.text(i, v + offset, f"{v:g}", ha="center", va="bottom", fontsize=10, color="#333333")


def _render_line(ax: Any, spec: ChartSpec):
    if not spec.series:
        raise ValueError("line chart requires series.")
    x = spec.x
    # fallback: index
    n = max((len(s["y"]) for s in spec.series), default=0)
    if not x:
        x = list(range(1, n + 1))
    if len(x) != n:
        # allow mismatch if all series share min length
        n2 = min([len(s["y"]) for s in spec.series] + [len(x)])
        x = list(x)[:n2]
        spec.series = [{"name": s["name"], "y": s["y"][:n2]} for s in spec.series]

    palette = spec.palette or ["#2F6FED", "#EF6C00", "#2E7D32", "#6A1B9A"]
    for idx, s in enumerate(spec.series):
        y = s["y"]
        color = palette[idx % len(palette)]
        ax.plot(x, y, color=color, linewidth=2.2, marker="o", markersize=4.5, label=s.get("name") or None)

    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    if spec.x_label:
        ax.set_xlabel(spec.x_label)
    if spec.y_label:
        ax.set_ylabel(spec.y_label)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if len(spec.series) > 1:
        ax.legend(frameon=False, loc="best", fontsize=10)


def _render_pie(ax: Any, spec: ChartSpec):
    labels = spec.labels or []
    sizes = spec.sizes or []
    if not labels or not sizes or len(labels) != len(sizes):
        raise ValueError("pie chart requires equal-length labels and sizes.")

    palette = spec.palette or ["#2F6FED", "#EF6C00", "#2E7D32", "#6A1B9A", "#00838F", "#C2185B"]
    colors = [palette[i % len(palette)] for i in range(len(labels))]

    total = float(sum(sizes)) if sizes else 0.0
    # only show pct if meaningful
    def _autopct(pct: float) -> str:
        if pct < 5:
            return ""
        return f"{pct:.0f}%"

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels if len(labels) <= 6 else None,
        colors=colors,
        startangle=90,
        autopct=_autopct if total > 0 else None,
        textprops={"color": "#222222", "fontsize": 10},
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax.axis("equal")

    # If too many labels, show legend
    if len(labels) > 6:
        ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=10)


def render_chart(spec: ChartSpec, out_path: Union[str, Path]) -> Path:
    _apply_minimal_style()
    w, h = _choose_figsize(spec)
    fig, ax = plt.subplots(figsize=(w, h))

    if spec.chart_type == "bar":
        _render_bar(ax, spec)
    elif spec.chart_type == "line":
        _render_line(ax, spec)
    elif spec.chart_type == "pie":
        _render_pie(ax, spec)
    else:
        raise ValueError(f"Unsupported chart_type: {spec.chart_type}")

    title = spec.title.strip()
    if title:
        ax.set_title(title, pad=12)
    if spec.subtitle.strip():
        ax.text(0.5, 1.02, spec.subtitle.strip(), transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="#555555")

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def llm_chart_spec(
    prompt: str,
    image_path: Optional[Union[str, Path]] = None,
    model: str = "",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ChartSpec:
    client = get_openai_client(api_key=api_key, base_url=base_url, key_type="text")

    user_content: Union[str, List[Dict[str, Any]]]
    if image_path:
        mime, data_url = _read_image_as_data_url(image_path)
        _ = mime  # kept for clarity; proxy parses mime from data_url header
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    else:
        user_content = prompt

    used_model = model.strip() or (os.getenv("LLM_MODEL") or "gemini-3-pro")
    resp = client.chat.completions.create(
        model=used_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    text = resp.choices[0].message.content or ""
    raw = _extract_json(text)
    return _normalize_spec(raw)


def main():
    ap = argparse.ArgumentParser(description="Generate a PPT-friendly chart image from text (+ optional style image).")
    ap.add_argument("--prompt", required=True, help="文字描述（必选），例如：画一个柱状图：A/B/C=10/20/15，标题为销量")
    ap.add_argument("--image", default=None, help="可选，参考图片路径（用于风格/配色）")
    ap.add_argument("--out", default="chart.png", help="输出图片路径（png/jpg 等）")
    ap.add_argument("--model", default="", help="模型名（默认读 LLM_MODEL 或 gemini-3-pro）")
    ap.add_argument("--base-url", default=None, help="OpenAI 兼容 base_url（默认 http://127.0.0.1:51958/v1）")
    ap.add_argument("--api-key", default=None, help="可选 API key（多数情况下本地 gemini_proxy 不需要传入）")
    ap.add_argument("--print-spec", action="store_true", help="打印生成的 ChartSpec JSON")
    args = ap.parse_args()

    spec = llm_chart_spec(
        prompt=args.prompt,
        image_path=args.image,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    if args.print_spec:
        print(json.dumps(spec.__dict__, ensure_ascii=False, indent=2))
    out = render_chart(spec, args.out)
    print(f"Saved chart to: {out}")


if __name__ == "__main__":
    main()


