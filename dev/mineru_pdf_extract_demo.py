"""
MinerU PDF 提取 Demo（独立脚本）
================================

目标：
- 给定 PDF 路径，使用 MinerU 解析并导出：
  - <out>/content/raw.md
  - <out>/assets/（图片）
  - <out>/assets/figures.json / tables.json / equations.json

要求：
- 直接在本目录运行：`python mineru_pdf_extract_demo.py --pdf /path/to/a.pdf`
- 不依赖 Paper2Slides / PosterGen2 项目内其它代码文件

依赖（按你实际安装方式二选一即可）：
1) Python 后端（推荐，支持公式/表格等结构化）
   - pip install "mineru[core]" pillow
2) CLI 后端（不需要在当前 python 环境安装 mineru 包）
   - 系统中可执行 `mineru` 命令，或通过 --mineru-cli 指定可执行文件路径

示例：
  python mineru_pdf_extract_demo.py --pdf ./example.pdf
  python mineru_pdf_extract_demo.py --pdf ./example.pdf --out ./out_dir
  python mineru_pdf_extract_demo.py --pdf ./example.pdf --backend cli

GPU/设备控制（可选）：
- 直接指定：MINERU_CUDA_VISIBLE_DEVICES="0" 或 POSTERGEN_MINERU_CUDA_VISIBLE_DEVICES="0"
- 或者指定 device：MINERU_DEVICE="cuda:0" / "cpu"（也兼容 POSTERGEN_MINERU_DEVICE / POSTERGEN_DEVICE）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def _resolve_cuda_visible_devices() -> Optional[str]:
    """
    解析设备配置并转换为 CUDA_VISIBLE_DEVICES：
    优先级：
      1) MINERU_CUDA_VISIBLE_DEVICES / POSTERGEN_MINERU_CUDA_VISIBLE_DEVICES（直接传递）
      2) MINERU_DEVICE / POSTERGEN_MINERU_DEVICE / POSTERGEN_DEVICE：
         - 'cpu'/'none'/'-1' -> ""
         - 'cuda:N' -> "N"
    未提供则返回 None，不做覆盖。
    """
    direct = os.getenv("MINERU_CUDA_VISIBLE_DEVICES")
    if direct is None:
        direct = os.getenv("POSTERGEN_MINERU_CUDA_VISIBLE_DEVICES")
    if direct is not None:
        return direct

    device = (
        os.getenv("MINERU_DEVICE")
        or os.getenv("POSTERGEN_MINERU_DEVICE")
        or os.getenv("POSTERGEN_DEVICE")
        or ""
    ).strip()
    if not device:
        return None

    val = device.lower()
    if val in ("cpu", "none", "-1"):
        return ""
    if val.startswith("cuda:"):
        # cuda:0 -> "0"
        try:
            idx = int(val.split(":", 1)[1])
            return str(idx)
        except Exception:
            return (val.split(":", 1)[1] or "0")
    return None


def _safe_image_size(path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return 0, 0


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_content_item_type(t: Any) -> str:
    return str(t or "").strip().lower()


def _extract_from_content_list(
    content_list: List[Dict[str, Any]],
    out_dir: Path,
    assets_dir: Path,
    include_equations_in_figures: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    将 MinerU 的 content_list 汇总成 figures/tables/equations 元数据。
    同时确保图片存在于 assets_dir（若 content_list 指向 out_dir 下的 assets/，一般无需复制）。
    """
    figures: Dict[str, Any] = {}
    tables: Dict[str, Any] = {}
    equations: Dict[str, Any] = {}

    fig_idx = 0
    tab_idx = 0
    eq_idx = 0

    def _resolve_img_abs(rel_or_abs: str) -> Optional[Path]:
        if not rel_or_abs:
            return None
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        # 常见：'assets/xxx.png'（相对 out_dir）
        cand = (out_dir / p).resolve()
        if cand.exists():
            return cand
        # 兼容：只给文件名
        cand2 = (assets_dir / p.name).resolve()
        if cand2.exists():
            return cand2
        return cand  # 不存在也返回一个路径，便于下游排查

    def _context_for(idx: int, window: int = 3) -> Dict[str, str]:
        try:
            it = content_list[idx]
            page = it.get("page_no") or it.get("page")
            texts_before: List[str] = []
            texts_after: List[str] = []
            nearest_heading = ""

            # 向后
            j = idx + 1
            while j < len(content_list) and len(texts_after) < window:
                tj = content_list[j]
                if (tj.get("page_no") or tj.get("page")) != page:
                    break
                if _normalize_content_item_type(tj.get("type")) in ("text", "paragraph"):
                    cj = tj.get("text") or tj.get("content") or ""
                    if isinstance(cj, str) and cj.strip():
                        texts_after.append(cj.strip())
                j += 1

            # 向前（并找 heading）
            j = idx - 1
            while j >= 0 and len(texts_before) < window:
                tj = content_list[j]
                if (tj.get("page_no") or tj.get("page")) != page:
                    break
                tname = _normalize_content_item_type(tj.get("type"))
                if tname in ("heading", "title", "section_title") and not nearest_heading:
                    nearest_heading = str(tj.get("text") or tj.get("content") or "").strip()
                if tname in ("text", "paragraph"):
                    cj = tj.get("text") or tj.get("content") or ""
                    if isinstance(cj, str) and cj.strip():
                        texts_before.append(cj.strip())
                j -= 1

            ctx = " ".join(list(reversed(texts_before)) + texts_after)
            if nearest_heading and len(nearest_heading) > 120:
                nearest_heading = nearest_heading[:120] + "..."
            return {"context_text": ctx[:1000], "nearest_heading": nearest_heading}
        except Exception:
            return {"context_text": "", "nearest_heading": ""}

    for idx, item in enumerate(content_list):
        t = _normalize_content_item_type(item.get("type"))
        img_path = item.get("img_path") or ""
        if not img_path:
            continue

        abs_img = _resolve_img_abs(str(img_path))
        if abs_img is None:
            continue

        # 若图片不在 assets_dir，尽量复制一份过去（保持输出整洁）
        try:
            assets_dir.mkdir(parents=True, exist_ok=True)
            if abs_img.exists():
                dst = (assets_dir / abs_img.name).resolve()
                if dst != abs_img.resolve():
                    try:
                        shutil.copyfile(abs_img, dst)
                        abs_img = dst
                    except Exception:
                        # 复制失败就用原路径
                        pass
        except Exception:
            pass

        w, h = _safe_image_size(abs_img)
        aspect = (w / h) if h else 1

        if t == "image":
            fig_idx += 1
            caption_list = item.get("image_caption") or []
            caption = caption_list[0] if caption_list else f"Figure {fig_idx}"
            figures[str(fig_idx)] = {
                "caption": caption,
                "path": str(abs_img),
                "width": w,
                "height": h,
                "aspect": aspect,
                "kind": "figure",
            }
        elif t == "table":
            tab_idx += 1
            caption_list = item.get("table_caption") or []
            caption = caption_list[0] if caption_list else f"Table {tab_idx}"
            tables[str(tab_idx)] = {
                "caption": caption,
                "path": str(abs_img),
                "width": w,
                "height": h,
                "aspect": aspect,
                "kind": "table",
            }
        elif t in ("equation", "formula", "latex", "math"):
            eq_idx += 1
            latex_content = item.get("latex") or item.get("text") or ""
            page_no = item.get("page_no") or item.get("page") or None
            bbox = item.get("bbox") or item.get("position") or None
            ctx = _context_for(idx, window=3)

            caption = f"Equation {eq_idx}"
            if isinstance(latex_content, str) and latex_content.strip():
                caption = latex_content.strip()
                if len(caption) > 120:
                    caption = caption[:120] + "..."

            equations[str(eq_idx)] = {
                "caption": caption,
                "latex": latex_content,
                "path": str(abs_img),
                "width": w,
                "height": h,
                "aspect": aspect,
                "page_no": page_no,
                "bbox": bbox,
                "kind": "equation",
                "context_text": ctx.get("context_text", ""),
                "nearest_heading": ctx.get("nearest_heading", ""),
            }

            if include_equations_in_figures:
                fig_idx += 1
                figures[str(fig_idx)] = {
                    "caption": equations[str(eq_idx)]["caption"],
                    "path": str(abs_img),
                    "width": w,
                    "height": h,
                    "aspect": aspect,
                    "kind": "equation",
                    "equation_ref": str(eq_idx),
                    "equation_context": {
                        "latex": latex_content,
                        "context_text": ctx.get("context_text", ""),
                        "nearest_heading": ctx.get("nearest_heading", ""),
                        "page_no": page_no,
                    },
                }

    return figures, tables, equations


def extract_with_mineru_python(pdf_path: Path, out_dir: Path) -> None:
    """
    使用 MinerU Python pipeline 后端解析 PDF。
    产物：
      - out_dir/content/raw.md
      - out_dir/assets/*
      - out_dir/assets/figures.json / tables.json / equations.json
    """
    cvd = _resolve_cuda_visible_devices()
    prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    applied = False
    if cvd is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cvd
        applied = True
        print(f"[mineru(py)] CUDA_VISIBLE_DEVICES='{cvd}'")

    content_dir = out_dir / "content"
    assets_dir = out_dir / "assets"
    content_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    try:
        try:
            from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze  # type: ignore
            from mineru.backend.pipeline.model_json_to_middle_json import (  # type: ignore
                result_to_middle_json as pipeline_result_to_middle_json,
            )
            from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make  # type: ignore
            from mineru.data.data_reader_writer import FileBasedDataWriter  # type: ignore
            from mineru.utils.enum_class import MakeMode  # type: ignore
            from mineru.cli.common import read_fn, convert_pdf_bytes_to_bytes_by_pypdfium2  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "MinerU Python 包导入失败：请先安装 mineru[core]。\n"
                "例如：pip install \"mineru[core]\" pillow\n"
                f"原始错误：{e}"
            ) from e

        pdf_bytes = read_fn(str(pdf_path))
        pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, 0, None)

        # 注意：lang 这里用 'ch' 与你给的参考实现一致；如需英文可改为 'en'
        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
            [pdf_bytes],
            ["ch"],
            parse_method="auto",
            formula_enable=True,
            table_enable=True,
        )

        model_list = infer_results[0]
        images_list = all_image_lists[0]
        pdf_doc = all_pdf_docs[0]
        _lang = lang_list[0]
        _ocr_enable = ocr_enabled_list[0]

        image_writer = FileBasedDataWriter(str(assets_dir))
        middle_json = pipeline_result_to_middle_json(model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, True)

        image_rel_dir = "assets"
        md_content_str = pipeline_union_make(middle_json["pdf_info"], MakeMode.MM_MD, image_rel_dir)
        (content_dir / "raw.md").write_text(md_content_str, encoding="utf-8")

        content_list = pipeline_union_make(middle_json["pdf_info"], MakeMode.CONTENT_LIST, image_rel_dir)
        if not isinstance(content_list, list):
            raise RuntimeError(f"MinerU 返回的 content_list 非 list：{type(content_list)}")

        figures, tables, equations = _extract_from_content_list(
            content_list=content_list,
            out_dir=out_dir,
            assets_dir=assets_dir,
            include_equations_in_figures=True,
        )

        _write_json(assets_dir / "figures.json", figures)
        _write_json(assets_dir / "tables.json", tables)
        _write_json(assets_dir / "equations.json", equations)

        print(f"[mineru(py)] OK: raw.md + assets written to: {out_dir}")
        print(f"[mineru(py)] figures={len(figures)}, tables={len(tables)}, equations={len(equations)}")
    finally:
        if applied:
            if prev_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd
            print("[mineru(py)] restored CUDA_VISIBLE_DEVICES")


def extract_with_mineru_cli(pdf_path: Path, out_dir: Path, mineru_cli: str = "mineru") -> None:
    """
    使用 MinerU CLI 解析 PDF：
      mineru -p <pdf> -o <out_dir/tmp_mineru_cli_output>
    然后把 markdown、content_list 与 images 规范化到 out_dir/{content,assets}。
    """
    content_dir = out_dir / "content"
    assets_dir = out_dir / "assets"
    content_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    mineru_out_dir = out_dir / "tmp_mineru_cli_output"
    mineru_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [mineru_cli, "-p", str(pdf_path), "-o", str(mineru_out_dir)]
    env = os.environ.copy()
    cvd = _resolve_cuda_visible_devices()
    if cvd is not None:
        env["CUDA_VISIBLE_DEVICES"] = cvd
        print(f"[mineru(cli)] CUDA_VISIBLE_DEVICES='{cvd}'")

    print("[mineru(cli)] running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"找不到 mineru CLI 可执行文件：{mineru_cli}\n"
            "请确认已安装 mineru CLI，或用 --mineru-cli 指定路径。"
        ) from e
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode(errors="ignore")
        out = (e.stdout or b"").decode(errors="ignore")
        raise RuntimeError(f"调用 mineru CLI 失败：{err or out}") from e

    pdf_stem = pdf_path.stem
    md_candidates = list(mineru_out_dir.rglob(f"{pdf_stem}.md"))
    if not md_candidates:
        md_candidates = list(mineru_out_dir.rglob("*.md"))
    if not md_candidates:
        raise RuntimeError("未在 MinerU CLI 输出中找到 markdown 文件（*.md）")
    mineru_md = md_candidates[0]

    clist_candidates = list(mineru_out_dir.rglob(f"{pdf_stem}_content_list.json"))
    if not clist_candidates:
        clist_candidates = list(mineru_out_dir.rglob("*_content_list.json"))
    if not clist_candidates:
        raise RuntimeError("未在 MinerU CLI 输出中找到 content_list.json（*_content_list.json）")
    content_list_path = clist_candidates[0]

    images_dir = mineru_out_dir / "images"
    if not images_dir.exists():
        imgs = [p for p in mineru_out_dir.rglob("images") if p.is_dir()]
        if imgs:
            images_dir = imgs[0]

    md_text = mineru_md.read_text(encoding="utf-8")
    (content_dir / "raw.md").write_text(md_text, encoding="utf-8")

    content_list = json.loads(content_list_path.read_text(encoding="utf-8"))
    if not isinstance(content_list, list):
        raise RuntimeError(f"content_list.json 非 list：{type(content_list)}")

    # CLI 输出图片通常在 mineru_out_dir/images 或者 img_path 指向的相对位置；这里尽量复制到 assets_dir
    for item in content_list:
        rel_img = item.get("img_path")
        if not rel_img:
            continue
        src_path = (mineru_out_dir / rel_img).resolve()
        if not src_path.exists() and images_dir.exists():
            src_path2 = images_dir / Path(rel_img).name
            if src_path2.exists():
                src_path = src_path2.resolve()
        if not src_path.exists():
            continue
        dst_path = (assets_dir / src_path.name).resolve()
        if dst_path != src_path:
            try:
                shutil.copyfile(src_path, dst_path)
            except Exception:
                pass

    figures, tables, equations = _extract_from_content_list(
        content_list=content_list,
        out_dir=out_dir,
        assets_dir=assets_dir,
        include_equations_in_figures=True,
    )

    _write_json(assets_dir / "figures.json", figures)
    _write_json(assets_dir / "tables.json", tables)
    _write_json(assets_dir / "equations.json", equations)

    print(f"[mineru(cli)] OK: raw.md + assets written to: {out_dir}")
    print(f"[mineru(cli)] figures={len(figures)}, tables={len(tables)}, equations={len(equations)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", required=True, help="PDF 文件路径")
    p.add_argument(
        "--out",
        default="",
        help="输出目录（默认：./mineru_demo_output/<pdf_stem>）",
    )
    p.add_argument(
        "--backend",
        choices=["python", "cli"],
        default="python",
        help="使用 mineru 的 python 包还是 CLI（默认：python）",
    )
    p.add_argument(
        "--mineru-cli",
        default=os.getenv("MINERU_CLI") or "mineru",
        help="mineru CLI 可执行文件（仅 backend=cli 时使用；也可用环境变量 MINERU_CLI）",
    )
    p.add_argument(
        "--keep-tmp",
        action="store_true",
        help="backend=cli 时保留 tmp_mineru_cli_output（默认会保留；此参数仅用于未来扩展）",
    )
    args = p.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF 不存在：{pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        _eprint(f"警告：输入文件后缀不是 .pdf：{pdf_path.name}")

    if args.out.strip():
        out_dir = Path(args.out).expanduser().resolve()
    else:
        out_dir = (Path.cwd() / "mineru_demo_output" / pdf_path.stem).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("PDF:", pdf_path)
    print("OUT:", out_dir)
    print("BACKEND:", args.backend)

    if args.backend == "cli":
        extract_with_mineru_cli(pdf_path, out_dir, mineru_cli=str(args.mineru_cli))
    else:
        extract_with_mineru_python(pdf_path, out_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _eprint("Interrupted")
        raise

