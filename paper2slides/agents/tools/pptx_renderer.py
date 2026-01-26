"""
PPTX Renderer Tool for Paper2Slides

Renders layout data into PowerPoint presentations using python-pptx.
Supports text, shapes, images, and various styling options.
"""

import json
import subprocess
import platform
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import re
import os
import math

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from lxml import etree

from qwen_agent.tools.base import BaseTool, register_tool
from paper2slides.utils.agent_logging import (
    log_agent_info,
    log_agent_success,
    log_agent_warning,
    log_agent_error,
)
from paper2slides.utils.agent_artifact_logging import save_json_log

# Define namespaces for XML manipulation
NSMAP = {
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
}


class PPTXRenderer:
    """PowerPoint renderer with rich styling support"""
    
    def __init__(self, width: float = 16, height: float = 9):
        """Initialize presentation with custom dimensions (in inches)"""
        self.prs = Presentation()
        self.prs.slide_width = Inches(width)
        self.prs.slide_height = Inches(height)
        self.width = width
        self.height = height
        self._debug_records: list[dict[str, Any]] = []
    
    def _parse_color(self, color_str: str) -> RGBColor:
        """Parse hex color string to RGBColor"""
        hex_color = color_str.lstrip('#')
        r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return RGBColor(r, g, b)

    _MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
    _MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)")
    _MD_CODE_RE = re.compile(r"`([^`]+)`")

    def _normalize_text_content(self, text: str) -> str:
        """清理常见 markdown 符号（避免把 ** ** 等渲染进 PPT）。"""
        s = str(text or "")
        s = self._MD_BOLD_RE.sub(r"\1", s)
        s = self._MD_ITALIC_RE.sub(r"\1", s)
        s = self._MD_CODE_RE.sub(r"\1", s)
        return s

    def _auto_font_size_pt(self, text: str, w_in: float, h_in: float) -> float:
        """按 bbox 尺寸简单估算字号（pt）：用 bbox 高度的 70% 作为字号基准。
        
        核心思路：文本框高度 * 0.7 / 行数 = 每行字号（考虑行距）。
        这种方式简单、稳定，避免过于复杂的迭代逻辑。
        """
        cleaned = self._normalize_text_content(text)
        char_count = len(cleaned.replace(" ", "").replace("\n", ""))
        if char_count == 0:
            return 24.0
        
        # 估算行数：用 bbox 宽度 / 字符数 来粗估每行字符数，再算行数
        # 假设英文字符平均宽度约 0.5em，bbox 宽度对应的 em 数约 w_in * 72 / font_size
        # 简化：先用高度直接算，假设 1~3 行文本
        h_pt = float(h_in) * 72.0
        w_pt = float(w_in) * 72.0
        
        # 粗略估计：假设字符平均宽度 0.5 * font_size，那么一行能放 w_pt / (0.5 * font_size) 个字符
        # 行数 = char_count / (w_pt / (0.5 * font_size)) = char_count * 0.5 * font_size / w_pt
        # 同时行高 ~= 1.2 * font_size，总高度 ~= 行数 * 1.2 * font_size
        # 联立：h_pt = (char_count * 0.5 * font_size / w_pt) * 1.2 * font_size
        #       h_pt = 0.6 * char_count * font_size^2 / w_pt
        #       font_size = sqrt(h_pt * w_pt / (0.6 * char_count))
        
        font_size = math.sqrt(h_pt * w_pt / max(1, 0.6 * char_count))
        
        # 同时用高度做上限：假设至少 1 行，字号不超过 h_pt / 1.2
        font_size = min(font_size, h_pt / 1.2)
        
        # 用宽度做上限：假设最短的一行至少能放下
        min_line_chars = min(20, max(5, char_count // 3))  # 假设最短行至少 5~20 字符
        font_by_w = w_pt / (0.5 * min_line_chars)
        font_size = min(font_size, font_by_w)
        
        return max(10.0, min(200.0, float(font_size)))
    
    def add_slide(self):
        """Add a blank slide"""
        slide_layout = self.prs.slide_layouts[6]  # blank layout
        return self.prs.slides.add_slide(slide_layout)
    
    def _add_shadow_to_shape(self, shape, 
                            blur_radius: int = 50800,
                            distance: int = 38100,
                            direction: int = 2700000,
                            color: str = "#000000",
                            alpha: int = 40):
        """Add outer shadow effect to shape via XML"""
        sp = shape._element
        spPr = sp.find('.//p:spPr', NSMAP)
        if spPr is None:
            return
        
        effectLst = spPr.find('a:effectLst', NSMAP)
        if effectLst is None:
            effectLst = etree.SubElement(spPr, '{%s}effectLst' % NSMAP['a'])
        
        outerShdw = etree.SubElement(effectLst, '{%s}outerShdw' % NSMAP['a'])
        outerShdw.set('blurRad', str(blur_radius))
        outerShdw.set('dist', str(distance))
        outerShdw.set('dir', str(direction))
        outerShdw.set('algn', 'tl')
        outerShdw.set('rotWithShape', '0')
        
        srgbClr = etree.SubElement(outerShdw, '{%s}srgbClr' % NSMAP['a'])
        srgbClr.set('val', color.lstrip('#'))
        alphaElem = etree.SubElement(srgbClr, '{%s}alpha' % NSMAP['a'])
        alphaElem.set('val', f'{alpha * 1000}')
    
    def render_element(self, slide, element: Dict[str, Any], image_map: Optional[Dict[str, str]] = None):
        """Render a single layout element"""
        element_type = element.get("type", "text")
        # Be robust to common aliases from LLMs / other generators
        if isinstance(element_type, str):
            et = element_type.strip().lower()
            alias_map = {
                "textbox": "text",
                "text_box": "text",
                "text-box": "text",
                "textblock": "text",
                "backgroundimage": "background_image",
                "bgimage": "background_image",
            }
            element_type = alias_map.get(et, et)
        
        # Allow treating a normal image as background via flags/role
        if element_type == "image":
            if element.get("is_background") or element.get("as_background") or str(element.get("role") or "").lower() == "background":
                self._render_background_image(slide, element, image_map)
                return

        if element_type == "text":
            self._render_text(slide, element)
        elif element_type == "title":
            self._render_title(slide, element)
        elif element_type == "section_title":
            self._render_section_title(slide, element)
        elif element_type == "shape":
            self._render_shape(slide, element)
        elif element_type == "image":
            self._render_image(slide, element, image_map)
        elif element_type in ("background_image", "bg_image"):
            # Dedicated background image element (full-canvas by default)
            self._render_background_image(slide, element, image_map)
        elif element_type == "background":
            self._render_background(slide, element)
        else:
            log_agent_warning("pptx_renderer", f"unknown element type: {element_type}, treating as text")
            self._render_text(slide, element)

    def _resolve_image_path(self, element: Dict[str, Any], image_map: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Resolve image path from various possible keys, plus image_map."""
        image_id = element.get("image_id", element.get("id", ""))
        image_path = (
            element.get("image_path")
            or element.get("path")
            or element.get("src")
            or element.get("file")
            or element.get("filepath")
        )
        if image_map and image_id:
            image_path = image_map.get(str(image_id), image_path)
        if not image_path:
            return None
        p = Path(str(image_path))
        return str(p) if p.exists() else None

    # ---------- coordinate normalization (robust to agent param noise) ----------
    def _get_box_in_inches(self, element: Dict[str, Any]) -> tuple[float, float, float, float]:
        """Return (x,y,w,h) in inches.

        Supported keys (best-effort):
        - x/y/width/height
        - left/top/w/h
        - bbox: [x0,y0,x1,y1] (interpreted as inches unless pixel conversion info is available)
        - bbox_px: [x0,y0,x1,y1] (requires image_width_px/image_height_px + canvas_width_in/canvas_height_in)
        Pixel conversion context:
        - element.image_width_px / element.image_height_px or element-level image_size_px
        - element.canvas_width_in / element.canvas_height_in
        """
        def _num(v, default=0.0) -> float:
            try:
                return float(v)
            except Exception:
                return float(default)

        # direct inches
        x = element.get("x", element.get("left", None))
        y = element.get("y", element.get("top", None))
        w = element.get("width", element.get("w", None))
        h = element.get("height", element.get("h", None))
        if x is not None and y is not None and w is not None and h is not None:
            return _num(x, 0.0), _num(y, 0.0), _num(w, 4.0), _num(h, 1.0)

        # bbox / bbox_px
        bbox = element.get("bbox_px", None)
        bbox_kind = "px"
        if bbox is None:
            bbox = element.get("bbox", None)
            bbox_kind = "unknown"
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x0, y0, x1, y1 = bbox
            # Try pixel->inch conversion if context exists OR looks like pixels
            img_w_px = element.get("image_width_px", None)
            img_h_px = element.get("image_height_px", None)
            if (img_w_px is None or img_h_px is None) and isinstance(element.get("image_size_px"), dict):
                img_w_px = element["image_size_px"].get("width")
                img_h_px = element["image_size_px"].get("height")
            canvas_w_in = element.get("canvas_width_in", None)
            canvas_h_in = element.get("canvas_height_in", None)
            if canvas_w_in is None:
                canvas_w_in = element.get("slide_width_in", None)
            if canvas_h_in is None:
                canvas_h_in = element.get("slide_height_in", None)

            x0f, y0f, x1f, y1f = _num(x0), _num(y0), _num(x1), _num(y1)
            looks_like_px = max(x0f, y0f, x1f, y1f) > max(self.width, self.height) * 2

            if (bbox_kind == "px" or looks_like_px) and img_w_px and img_h_px and canvas_w_in and canvas_h_in:
                iw = max(1.0, _num(img_w_px, 1.0))
                ih = max(1.0, _num(img_h_px, 1.0))
                cw = max(0.01, _num(canvas_w_in, self.width))
                ch = max(0.01, _num(canvas_h_in, self.height))
                xi = (x0f / iw) * cw
                yi = (y0f / ih) * ch
                wi = ((x1f - x0f) / iw) * cw
                hi = ((y1f - y0f) / ih) * ch
                return xi, yi, wi, hi

            # treat as inches bbox
            return x0f, y0f, max(0.01, x1f - x0f), max(0.01, y1f - y0f)

        # fallback
        return 0.0, 0.0, 4.0, 1.0
    
    def _render_text(self, slide, element: Dict[str, Any]):
        """Render text element.
        
        简化策略：所有文本都用 bbox 规则估算字号，忽略上游传入的 font_size。
        这样可以避免 LLM 乱填字号导致的"字体过小"问题。
        """
        x, y, w, h = self._get_box_in_inches(element)
        
        text = element.get("content", element.get("text", ""))
        if not text:
            return
        text = self._normalize_text_content(text)
        
        textbox = slide.shapes.add_textbox(
            Inches(x), Inches(y), Inches(w), Inches(h)
        )
        tf = textbox.text_frame
        tf.word_wrap = True  # 始终开启自动换行
        tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE  # 不自动缩放，让文字按我们算的字号渲染

        # 减小内边距
        try:
            tf.margin_left = 0
            tf.margin_right = 0
            tf.margin_top = 0
            tf.margin_bottom = 0
        except Exception:
            pass
        
        # 文本属性
        font_name = element.get("font_family", element.get("font_name", "Arial"))
        font_color = element.get("font_color", element.get("color", "#000000"))
        bold = element.get("bold", element.get("font_weight") == "bold")
        italic = element.get("italic", False)
        align = element.get("alignment", "left").lower()
        vertical_align = element.get("vertical_align", element.get("vertical_alignment", "top")).lower()

        # 字号：强制按 bbox 估算，忽略上游 font_size（除非显式 keep_font_size=true）
        provided_fs = None
        try:
            provided_fs = float(element.get("font_size")) if element.get("font_size") is not None else None
        except Exception:
            pass
        
        if element.get("keep_font_size") and provided_fs is not None:
            font_size_f = provided_fs
        else:
            font_size_f = self._auto_font_size_pt(text, w_in=w, h_in=h)
        
        font_size_f = max(10.0, min(200.0, float(font_size_f)))
        
        # 垂直对齐
        anchor_map = {"top": MSO_ANCHOR.TOP, "middle": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}
        tf.vertical_anchor = anchor_map.get(vertical_align, MSO_ANCHOR.TOP)
        
        # 渲染文本
        lines = text.split('\n')
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.font.name = font_name
            p.font.size = Pt(int(round(font_size_f)))
            p.font.color.rgb = self._parse_color(font_color)
            p.font.bold = bold
            p.font.italic = italic
            align_map = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}
            p.alignment = align_map.get(align, PP_ALIGN.LEFT)

        # Debug 记录
        try:
            self._debug_records.append({
                "type": "text",
                "content_preview": text[:100],
                "box_in": {"x": x, "y": y, "w": w, "h": h},
                "font_size_pt_final": float(font_size_f),
                "font_size_pt_provided": float(provided_fs) if provided_fs else None,
            })
        except Exception:
            pass
    
    def _render_title(self, slide, element: Dict[str, Any]):
        """Render title element (bold, auto-sized by bbox)"""
        element_copy = element.copy()
        element_copy["bold"] = True
        self._render_text(slide, element_copy)
    
    def _render_section_title(self, slide, element: Dict[str, Any]):
        """Render section title (bold, auto-sized by bbox)"""
        element_copy = element.copy()
        element_copy["bold"] = True
        self._render_text(slide, element_copy)
    
    def _render_shape(self, slide, element: Dict[str, Any]):
        """Render shape element (rectangle, rounded_rectangle, oval)"""
        x, y, w, h = self._get_box_in_inches(element)
        
        shape_type = element.get("shape_type", "rectangle").lower()
        fill_color = element.get("fill_color", element.get("color", "#FFFFFF"))
        border_color = element.get("border_color", element.get("border", None))
        border_width = element.get("border_width", 1.0)
        shadow = element.get("shadow", False)
        corner_radius = element.get("corner_radius", 0.2)
        
        if shape_type == "rounded_rectangle":
            shape = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                Inches(x), Inches(y), Inches(w), Inches(h)
            )
            # Adjust corner radius via XML
            sp = shape._element
            prstGeom = sp.find('.//a:prstGeom', NSMAP)
            if prstGeom is not None:
                avLst = prstGeom.find('a:avLst', NSMAP)
                if avLst is None:
                    avLst = etree.SubElement(prstGeom, '{%s}avLst' % NSMAP['a'])
                for child in list(avLst):
                    avLst.remove(child)
                radius_val = int(corner_radius * 100000 / min(w, h))
                gd = etree.SubElement(avLst, '{%s}gd' % NSMAP['a'])
                gd.set('name', 'adj')
                gd.set('fmla', f'val {radius_val}')
        elif shape_type == "oval" or shape_type == "circle":
            shape = slide.shapes.add_shape(
                MSO_SHAPE.OVAL,
                Inches(x), Inches(y), Inches(w), Inches(h)
            )
        else:  # rectangle
            shape = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE,
                Inches(x), Inches(y), Inches(w), Inches(h)
            )
        
        # Fill color
        shape.fill.solid()
        shape.fill.fore_color.rgb = self._parse_color(fill_color)
        
        # Border
        if border_color:
            shape.line.color.rgb = self._parse_color(border_color)
            shape.line.width = Pt(border_width)
        else:
            shape.line.fill.background()
        
        # Shadow
        if shadow:
            self._add_shadow_to_shape(shape)
    
    def _render_image(self, slide, element: Dict[str, Any], image_map: Optional[Dict[str, str]] = None):
        """Render image element"""
        x, y, w_in, h_in = self._get_box_in_inches(element)
        # Allow None if not provided (kept for backward compatibility)
        w = element.get("width", element.get("w", w_in if w_in else None))
        h = element.get("height", element.get("h", h_in if h_in else None))
        
        image_path = self._resolve_image_path(element, image_map)
        if not image_path:
            log_agent_warning("pptx_renderer", "image not found (missing or path does not exist)")
            return
        
        left, top = Inches(x), Inches(y)
        
        if w and h:
            slide.shapes.add_picture(
                image_path, left, top,
                width=Inches(w), height=Inches(h)
            )
        elif w:
            slide.shapes.add_picture(
                image_path, left, top, width=Inches(w)
            )
        elif h:
            slide.shapes.add_picture(
                image_path, left, top, height=Inches(h)
            )
        else:
            slide.shapes.add_picture(image_path, left, top)

    def _render_background_image(self, slide, element: Dict[str, Any], image_map: Optional[Dict[str, str]] = None):
        """Render a background image (full-canvas by default) and move it to back."""
        # Fill the slide by default
        element = dict(element)
        element.setdefault("x", 0)
        element.setdefault("y", 0)
        element.setdefault("width", self.width)
        element.setdefault("height", self.height)
        # Use same image resolution path logic as _render_image
        x, y, w, h = self._get_box_in_inches(element)

        image_path = self._resolve_image_path(element, image_map)
        if not image_path:
            log_agent_warning("pptx_renderer", "background image not found (missing or path does not exist)")
            return

        pic = slide.shapes.add_picture(
            image_path, Inches(x), Inches(y), width=Inches(w), height=Inches(h)
        )

        # Move to back
        sp = pic._element
        spTree = sp.getparent()
        spTree.remove(sp)
        spTree.insert(0, sp)
    
    def _render_background(self, slide, element: Dict[str, Any]):
        """Render background element"""
        color = element.get("color", element.get("fill_color", "#FFFFFF"))
        
        bg_shape = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(0), Inches(0),
            Inches(self.width), Inches(self.height)
        )
        bg_shape.fill.solid()
        bg_shape.fill.fore_color.rgb = self._parse_color(color)
        bg_shape.line.fill.background()
        
        # Move to back
        sp = bg_shape._element
        spTree = sp.getparent()
        spTree.remove(sp)
        spTree.insert(0, sp)
    
    def save(self, output_path: str):
        """Save the presentation to a file"""
        self.prs.save(output_path)
        return output_path
    
    def convert_to_pdf(self, pptx_path: str) -> Optional[str]:
        """Convert PPTX to PDF using LibreOffice"""
        pptx_path = Path(pptx_path)
        output_dir = pptx_path.parent
        
        system = platform.system().lower()
        
        if system == "darwin":  # macOS
            libreoffice_paths = [
                "/Applications/LibreOffice.app/Contents/MacOS/soffice",
                "/usr/local/bin/libreoffice",
                "libreoffice",
                "soffice"
            ]
        elif system == "linux":
            libreoffice_paths = [
                "/usr/bin/libreoffice",
                "/usr/local/bin/libreoffice",
                "/snap/bin/libreoffice",
                "libreoffice",
                "soffice"
            ]
        elif system == "windows":
            libreoffice_paths = [
                r"C:\Program Files\LibreOffice\program\soffice.exe",
                r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
                "soffice.exe",
                "libreoffice.exe"
            ]
        else:
            libreoffice_paths = ["libreoffice", "soffice"]
        
        for lo_path in libreoffice_paths:
            try:
                cmd = [
                    lo_path, "--headless", "--convert-to", "pdf",
                    "--outdir", str(output_dir), str(pptx_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    pdf_name = pptx_path.stem + ".pdf"
                    pdf_path = output_dir / pdf_name
                    if pdf_path.exists():
                        return str(pdf_path)
                        
            except (subprocess.SubprocessError, FileNotFoundError):
                continue
        
        return None


def render_layout_to_pptx(
    layout_data: Union[Dict[str, Any], List[Dict[str, Any]]],
    output_path: str,
    width: float = 16,
    height: float = 9,
    image_map: Optional[Dict[str, str]] = None,
    convert_to_pdf: bool = False,
) -> Dict[str, Any]:
    """Render layout data to PPTX file.
    
    Args:
        layout_data: Layout data. Accepted formats:
            1. dict with "slides" key: {"slides": [{"elements": [...]}, ...]}
            2. dict with "elements" key: {"elements": [...]} -> single slide
            3. list of slide dicts: [{"elements": [...]}, ...]
            4. list of element dicts (all on ONE slide): [{"type": "text", ...}, ...]
        output_path: Output PPTX file path
        width: Slide width in inches
        height: Slide height in inches
        image_map: Mapping from image_id to image_path
        convert_to_pdf: Whether to also convert to PDF
    
    Returns:
        Dict with 'pptx_path' and optionally 'pdf_path'
    """
    # Handle different layout data formats and extract dimensions
    global_background = None
    if isinstance(layout_data, dict):
        if "slides" in layout_data:
            slides_data = layout_data["slides"]
        elif "elements" in layout_data:
            # Single slide with elements list
            slides_data = [layout_data]
        else:
            # Treat entire dict as single-element slide
            slides_data = [{"elements": [layout_data]}]
        # Override dimensions if specified in layout_data
        width = layout_data.get("width", width)
        height = layout_data.get("height", height)
        # Optional global background path (applied to every slide)
        global_background = (
            layout_data.get("background_image_path")
            or layout_data.get("bg_image_path")
            or layout_data.get("background_path")
        )
    elif isinstance(layout_data, list):
        if not layout_data:
            slides_data = []
        else:
            # Check if list contains element-like dicts (have "type" but no "elements")
            # If so, treat entire list as elements of ONE slide
            first = layout_data[0]
            if isinstance(first, dict) and "type" in first and "elements" not in first:
                # All items are elements -> wrap into single slide
                slides_data = [{"elements": layout_data}]
            else:
                # Items are slides (each may have "elements" key)
                slides_data = layout_data
    else:
        raise ValueError(f"Invalid layout_data type: {type(layout_data)}")
    
    # -------- infer slide size from element bounds (robust to missing width/height args) --------
    def _num(v, default=0.0) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    def _infer_bounds_from_element(el: Dict[str, Any]) -> tuple[float, float] | None:
        # x/y/width/height or aliases
        x = el.get("x", el.get("left", None))
        y = el.get("y", el.get("top", None))
        w = el.get("width", el.get("w", None))
        h = el.get("height", el.get("h", None))
        if x is not None and y is not None and w is not None and h is not None:
            return _num(x) + max(0.0, _num(w)), _num(y) + max(0.0, _num(h))
        # bbox interpreted as inches
        bbox = el.get("bbox", None)
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x0, y0, x1, y1 = bbox
            x0f, y0f, x1f, y1f = _num(x0), _num(y0), _num(x1), _num(y1)
            return max(x0f, x1f), max(y0f, y1f)
        return None

    inferred_w = float(width)
    inferred_h = float(height)
    for sd in slides_data:
        if isinstance(sd, dict):
            els = sd.get("elements", [])
            if isinstance(els, dict):
                els = [els]
            if isinstance(els, list):
                for el in els:
                    if isinstance(el, dict):
                        b = _infer_bounds_from_element(el)
                        if b:
                            inferred_w = max(inferred_w, float(b[0]))
                            inferred_h = max(inferred_h, float(b[1]))
            # also consider legacy slide-level background dict if present
            bg_obj = sd.get("background")
            if isinstance(bg_obj, dict):
                b = _infer_bounds_from_element(bg_obj)
                if b:
                    inferred_w = max(inferred_w, float(b[0]))
                    inferred_h = max(inferred_h, float(b[1]))
        elif isinstance(sd, list):
            for el in sd:
                if isinstance(el, dict):
                    b = _infer_bounds_from_element(el)
                    if b:
                        inferred_w = max(inferred_w, float(b[0]))
                        inferred_h = max(inferred_h, float(b[1]))

    # If inferred size is significantly larger than provided size, upgrade slide size.
    # (Common when caller forgets to pass width/height but uses 48x36 coordinates.)
    if inferred_w > float(width) * 1.05 or inferred_h > float(height) * 1.05:
        log_agent_info(
            "pptx_renderer",
            f"auto-inferred slide size from elements: {width}x{height} -> {inferred_w:.2f}x{inferred_h:.2f}",
        )
        width, height = inferred_w, inferred_h

    # Create renderer with correct dimensions
    renderer = PPTXRenderer(width=width, height=height)
    
    log_agent_info("pptx_renderer", f"rendering {len(slides_data)} slide(s) to {output_path}")
    
    # Render each slide
    for slide_idx, slide_data in enumerate(slides_data):
        slide = renderer.add_slide()
        
        # Handle slide-level properties
        if isinstance(slide_data, dict):
            elements = slide_data.get("elements", [slide_data])
            # If slide has background info, insert as a background element first
            bg = (
                slide_data.get("background_image_path")
                or slide_data.get("bg_image_path")
                or slide_data.get("background_path")
                or global_background
            )
            # Also support legacy/agent-produced "background": {"type":"image","image_path":...} or {"color":"#fff"}
            bg_obj = slide_data.get("background")
            if isinstance(bg_obj, dict):
                # image background
                if (
                    bg_obj.get("image_path")
                    or bg_obj.get("path")
                    or bg_obj.get("src")
                    or bg_obj.get("file")
                    or bg_obj.get("filepath")
                ):
                    bg = bg_obj.get("image_path") or bg_obj.get("path") or bg_obj.get("src") or bg_obj.get("file") or bg_obj.get("filepath") or bg
                # color background
                elif bg_obj.get("color") or bg_obj.get("fill_color"):
                    # Insert a color background element (rectangle) if present
                    color = bg_obj.get("color", bg_obj.get("fill_color", "#FFFFFF"))
                    elements = [
                        {
                            "type": "background",
                            "color": color,
                            "z_order": -10_000_000,
                        }
                    ] + list(elements if isinstance(elements, list) else [elements])
            elif isinstance(bg_obj, str) and bg_obj.strip():
                # If background is a path string
                bg = bg_obj.strip()

            if bg:
                elements = [
                    {
                        "type": "background_image",
                        "image_path": bg,
                        "x": 0,
                        "y": 0,
                        "width": renderer.width,
                        "height": renderer.height,
                        "z_order": -10_000_000,
                    }
                ] + list(elements if isinstance(elements, list) else [elements])
            # Note: Individual slide dimensions are not supported in python-pptx
            # All slides in a presentation must have the same size
        else:
            elements = [slide_data]
        
        # Sort elements by z-order or priority if available
        if isinstance(elements, list):
            elements = sorted(
                elements,
                key=lambda e: e.get("z_order", e.get("priority", 0))
            )
        
        # Render each element
        for element_idx, element in enumerate(elements):
            try:
                renderer.render_element(slide, element, image_map)
            except Exception as e:
                log_agent_warning(
                    "pptx_renderer",
                    f"failed to render element {element_idx} on slide {slide_idx}: {e}"
                )
    
    # Save PPTX
    pptx_path = renderer.save(output_path)
    log_agent_success("pptx_renderer", f"saved PPTX: {pptx_path}")
    
    result = {"pptx_path": pptx_path}
    # Attach internal debug info (caller/tool wrapper decides whether to persist it)
    try:
        result["_last_renderer_debug"] = {
            "slide_size_in": {"width": float(renderer.width), "height": float(renderer.height)},
            "num_debug_records": len(getattr(renderer, "_debug_records", []) or []),
            "records": getattr(renderer, "_debug_records", []) or [],
        }
    except Exception:
        pass
    
    # Convert to PDF if requested
    if convert_to_pdf:
        pdf_path = renderer.convert_to_pdf(pptx_path)
        if pdf_path:
            result["pdf_path"] = pdf_path
            log_agent_success("pptx_renderer", f"converted to PDF: {pdf_path}")
        else:
            log_agent_warning("pptx_renderer", "PDF conversion failed (LibreOffice not found)")
    
    return result


@register_tool('pptx_renderer')
class PPTXRendererTool(BaseTool):
    """Render layout data into PowerPoint presentations.
    
    Supports text, shapes, images, and various styling options.
    Can optionally convert to PDF if LibreOffice is available.
    """

    description = 'Render layout data into a PowerPoint presentation (PPTX) file. Supports text, shapes, images, and styling.'
    parameters = {
        'type': 'object',
        'properties': {
            'layout_data': {
                'type': ['object', 'array'],
                'description': 'Layout data. Can be a dict with "slides" key or a list of slide elements. Each element should have type, position (x, y, width, height), and content/styling properties.'
            },
            'output_path': {
                'type': 'string',
                'description': 'Output path for the PPTX file.'
            },
            'width': {
                'type': 'number',
                'description': 'Slide width in inches (default: 16).',
                'default': 16
            },
            'height': {
                'type': 'number',
                'description': 'Slide height in inches (default: 9).',
                'default': 9
            },
            'image_map': {
                'type': 'object',
                'description': 'Optional mapping from image_id to image_path for image elements.'
            },
            'convert_to_pdf': {
                'type': 'boolean',
                'description': 'Whether to also convert to PDF (requires LibreOffice).',
                'default': False
            }
        },
        'required': ['layout_data', 'output_path'],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        
        layout_data = params['layout_data']
        output_path = params['output_path']
        width = params.get('width', 16)
        height = params.get('height', 9)
        image_map = params.get('image_map', None)
        convert_to_pdf = params.get('convert_to_pdf', False)
        debug = bool(params.get("debug", False)) or str(os.getenv("PPTX_RENDER_DEBUG", "")).strip() in ("1", "true", "True")
        
        log_agent_info("pptx_renderer", f"starting render: output={output_path}, size={width}x{height}")
        
        # Save input layout data for debugging
        save_json_log(
            agent_name="pptx_renderer",
            func_name="render_input",
            payload={
                "output_path": output_path,
                "width": width,
                "height": height,
                "layout_data": layout_data,
                "image_map": image_map,
                "convert_to_pdf": convert_to_pdf,
            },
        )
        result = render_layout_to_pptx(
            layout_data=layout_data,
            output_path=output_path,
            width=width,
            height=height,
            image_map=image_map,
            convert_to_pdf=convert_to_pdf,
        )

        # Save debug info if enabled
        try:
            if debug and "_last_renderer_debug" in result:
                save_json_log(
                    agent_name="pptx_renderer",
                    func_name="render_debug",
                    payload=result["_last_renderer_debug"],
                )
        except Exception:
            pass
        
        # Save render result
        save_json_log(
            agent_name="pptx_renderer",
            func_name="render_result",
            payload=result,
        )
        
        # Do not expose internal debug payload to agent by default
        if "_last_renderer_debug" in result:
            result = dict(result)
            result.pop("_last_renderer_debug", None)
        return json.dumps(result, ensure_ascii=False)
            

