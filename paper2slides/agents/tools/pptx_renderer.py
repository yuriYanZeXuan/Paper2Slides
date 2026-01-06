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
    
    def _parse_color(self, color_str: str) -> RGBColor:
        """Parse hex color string to RGBColor"""
        hex_color = color_str.lstrip('#')
        r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return RGBColor(r, g, b)
    
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
        elif element_type == "background":
            self._render_background(slide, element)
        else:
            log_agent_warning("pptx_renderer", f"unknown element type: {element_type}, treating as text")
            self._render_text(slide, element)
    
    def _render_text(self, slide, element: Dict[str, Any]):
        """Render text element"""
        x = element.get("x", 0)
        y = element.get("y", 0)
        w = element.get("width", 4)
        h = element.get("height", 1)
        
        text = element.get("content", element.get("text", ""))
        if not text:
            return
        
        textbox = slide.shapes.add_textbox(
            Inches(x), Inches(y), Inches(w), Inches(h)
        )
        tf = textbox.text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        
        # Get text properties
        font_name = element.get("font_family", element.get("font_name", "Arial"))
        font_size = element.get("font_size", 24)
        font_color = element.get("font_color", element.get("color", "#000000"))
        bold = element.get("bold", element.get("font_weight") == "bold")
        italic = element.get("italic", False)
        align = element.get("alignment", "left").lower()
        vertical_align = element.get("vertical_align", element.get("vertical_alignment", "top")).lower()
        
        # Vertical alignment
        anchor_map = {
            "top": MSO_ANCHOR.TOP,
            "middle": MSO_ANCHOR.MIDDLE,
            "bottom": MSO_ANCHOR.BOTTOM,
        }
        tf.vertical_anchor = anchor_map.get(vertical_align, MSO_ANCHOR.TOP)
        
        # Handle multiline text
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if i == 0:
                p = tf.paragraphs[0]
            else:
                p = tf.add_paragraph()
            
            p.text = line
            p.font.name = font_name
            p.font.size = Pt(font_size)
            p.font.color.rgb = self._parse_color(font_color)
            p.font.bold = bold
            p.font.italic = italic
            
            # Horizontal alignment
            align_map = {
                "left": PP_ALIGN.LEFT,
                "center": PP_ALIGN.CENTER,
                "right": PP_ALIGN.RIGHT,
            }
            p.alignment = align_map.get(align, PP_ALIGN.LEFT)
    
    def _render_title(self, slide, element: Dict[str, Any]):
        """Render title element with larger font"""
        element_copy = element.copy()
        element_copy["font_size"] = element.get("font_size", 48)
        element_copy["bold"] = True
        self._render_text(slide, element_copy)
    
    def _render_section_title(self, slide, element: Dict[str, Any]):
        """Render section title"""
        element_copy = element.copy()
        element_copy["font_size"] = element.get("font_size", 36)
        element_copy["bold"] = True
        self._render_text(slide, element_copy)
    
    def _render_shape(self, slide, element: Dict[str, Any]):
        """Render shape element (rectangle, rounded_rectangle, oval)"""
        x = element.get("x", 0)
        y = element.get("y", 0)
        w = element.get("width", 2)
        h = element.get("height", 1)
        
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
        x = element.get("x", 0)
        y = element.get("y", 0)
        w = element.get("width", None)
        h = element.get("height", None)
        
        # Get image path
        image_id = element.get("image_id", element.get("id", ""))
        image_path = element.get("image_path", None)
        
        if image_map and image_id:
            image_path = image_map.get(image_id, image_path)
        
        if not image_path or not Path(image_path).exists():
            log_agent_warning("pptx_renderer", f"image not found: {image_path or image_id}")
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
    
    # Create renderer with correct dimensions
    renderer = PPTXRenderer(width=width, height=height)
    
    log_agent_info("pptx_renderer", f"rendering {len(slides_data)} slide(s) to {output_path}")
    
    # Render each slide
    for slide_idx, slide_data in enumerate(slides_data):
        slide = renderer.add_slide()
        
        # Handle slide-level properties
        if isinstance(slide_data, dict):
            elements = slide_data.get("elements", [slide_data])
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
        
        # Save render result
        save_json_log(
            agent_name="pptx_renderer",
            func_name="render_result",
            payload=result,
        )
        
        return json.dumps(result, ensure_ascii=False)
            

