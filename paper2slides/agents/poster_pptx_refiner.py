"""
Poster PPTX Refiner Agent

新的 refine 流程：
1. 使用 MinerU VLM 解析海报，定位模糊文字区域
2. 使用 Z-Image FlowEdit 擦除模糊文字区域，生成干净背景
3. 使用 text_match 匹配正确的文字内容
4. 使用 PPTX 工具在对应位置渲染清晰文字
5. 将渲染的文字叠加到干净背景上
6. 迭代处理，最终输出 PPTX 文件和 PDF

关键优势：
- 背景：干净的海报底图（无模糊文字）
- 前景：PPTX 渲染的清晰、可编辑文字
- 输出：可编辑的 PPTX 文件 + PDF
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from PIL import Image

from paper2slides.utils.logging import get_logger
from paper2slides.utils.agent_logging import (
    log_agent_start,
    log_agent_info,
    log_agent_success,
    log_agent_warning,
    log_agent_error,
)
from paper2slides.utils.agent_artifact_logging import (
    save_json_log,
    save_before_after_image,
    save_image,
    get_default_log_root,
)

# Import tools
from paper2slides.agents.tools.image_content_grounding import ground_poster_text_regions_with_mineru_vlm
from paper2slides.agents.tools.text_erase_flowedit import erase_multiple_text_regions
from paper2slides.agents.tools.poster_text_match import match_plan_text_for_patch, _load_plan_text_spans
from paper2slides.agents.tools.pptx_renderer import render_layout_to_pptx, PPTXRenderer

logger = get_logger(__name__)


BBox = Tuple[int, int, int, int]
_AGENT_NAME = "poster_pptx_refiner"


class TextRegion:
    """表示一个文字区域及其相关信息"""
    
    def __init__(
        self,
        bbox: BBox,
        text_content: str = "",
        region_type: str = "text",
        matched_text: Optional[str] = None,
        font_size: int = 24,
        font_color: str = "#000000",
        font_family: str = "Arial",
        bold: bool = False,
    ):
        self.bbox = bbox
        self.text_content = text_content
        self.region_type = region_type
        self.matched_text = matched_text
        self.font_size = font_size
        self.font_color = font_color
        self.font_family = font_family
        self.bold = bold
    
    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "text_content": self.text_content,
            "region_type": self.region_type,
            "matched_text": self.matched_text,
            "font_size": self.font_size,
            "font_color": self.font_color,
            "font_family": self.font_family,
            "bold": self.bold,
        }


class PosterPPTXRefiner:
    """Poster PPTX Refiner - 使用擦除+PPTX渲染的新 refine 流程
    
    核心流程：
    1. 解析海报，定位文字区域
    2. 擦除模糊文字，生成干净背景
    3. 匹配并渲染清晰文字
    4. 合成最终结果，输出 PPTX 和 PDF
    """
    
    def __init__(
        self,
        zimage_model_name: str = "Tongyi-MAI/Z-Image-Turbo",
        device: str = None,
        style_name: str = "academic",
        plan_text_spans_path: Optional[str] = None,
    ):
        self.zimage_model_name = zimage_model_name
        self.device = device or "cuda"
        self.style_name = style_name
        self.plan_text_spans_path = plan_text_spans_path
        
        log_agent_start(_AGENT_NAME)
        log_agent_info(_AGENT_NAME, f"initialized with style={style_name}, model={zimage_model_name}")
    
    def _estimate_font_size(self, bbox: BBox, text: str) -> int:
        """根据 bbox 大小和文字长度估算合适的字体大小"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # 简单估算：根据高度和文字行数
        lines = text.count('\n') + 1
        max_font_by_height = int(height / lines * 0.8)
        
        # 根据宽度和字符数估算
        char_count = max(len(text.replace('\n', '')), 1)
        max_font_by_width = int(width / (char_count * 0.6))
        
        # 取两者较小值，并限制范围
        estimated = min(max_font_by_height, max_font_by_width)
        return max(12, min(72, estimated))
    
    def _get_style_font_family(self) -> str:
        """根据样式获取默认字体"""
        style_fonts = {
            "academic": "Arial",
            "doraemon": "Comic Sans MS",
            "totoro": "Georgia",
            "default": "Arial",
        }
        return style_fonts.get(self.style_name, "Arial")
    
    def _convert_bbox_to_inches(
        self,
        bbox: BBox,
        image_width: int,
        image_height: int,
        poster_width: float,
        poster_height: float,
    ) -> Tuple[float, float, float, float]:
        """将像素 bbox 转换为英寸坐标"""
        x0, y0, x1, y1 = bbox
        
        x_inch = (x0 / image_width) * poster_width
        y_inch = (y0 / image_height) * poster_height
        w_inch = ((x1 - x0) / image_width) * poster_width
        h_inch = ((y1 - y0) / image_height) * poster_height
        
        return x_inch, y_inch, w_inch, h_inch
    
    def _create_text_overlay_layout(
        self,
        regions: List[TextRegion],
        image_width: int,
        image_height: int,
        poster_width: float = 48,
        poster_height: float = 36,
    ) -> Dict[str, Any]:
        """创建文字叠加层的 PPTX 布局"""
        elements = []
        
        for i, region in enumerate(regions):
            if not region.matched_text:
                continue
            
            x, y, w, h = self._convert_bbox_to_inches(
                region.bbox,
                image_width,
                image_height,
                poster_width,
                poster_height,
            )
            
            element = {
                "type": "text",
                "id": f"text_region_{i}",
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "content": region.matched_text,
                "font_family": region.font_family,
                "font_size": region.font_size,
                "font_color": region.font_color,
                "bold": region.bold,
                "alignment": "left",
                "z_order": 10 + i,  # 文字层在背景之上
            }
            elements.append(element)
        
        return {
            "width": poster_width,
            "height": poster_height,
            "slides": [{"elements": elements}],
        }
    
    def _render_text_to_image(
        self,
        regions: List[TextRegion],
        image_width: int,
        image_height: int,
        output_path: str,
        poster_width: float = 48,
        poster_height: float = 36,
    ) -> Optional[str]:
        """将文字区域渲染为透明 PNG 图像
        
        使用 PPTX 渲染后转换，或直接使用 PIL 绘制
        """
        from PIL import ImageDraw, ImageFont
        
        # 创建透明背景图像
        overlay = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        for region in regions:
            if not region.matched_text:
                continue
            
            x0, y0, x1, y1 = region.bbox
            
            # 尝试加载字体
            try:
                font = ImageFont.truetype(
                    f"/System/Library/Fonts/{region.font_family}.ttc",
                    region.font_size
                )
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", region.font_size)
                except:
                    font = ImageFont.load_default()
            
            # 解析颜色
            color_hex = region.font_color.lstrip('#')
            r, g, b = (int(color_hex[i:i+2], 16) for i in (0, 2, 4))
            
            # 绘制文字
            draw.text(
                (x0, y0),
                region.matched_text,
                font=font,
                fill=(r, g, b, 255),
            )
        
        overlay.save(output_path)
        return output_path
    
    def _composite_background_and_text(
        self,
        background: Image.Image,
        text_overlay: Image.Image,
    ) -> Image.Image:
        """合成背景和文字叠加层"""
        # 确保尺寸一致
        if text_overlay.size != background.size:
            text_overlay = text_overlay.resize(background.size, Image.LANCZOS)
        
        # 合成
        result = background.copy().convert("RGBA")
        result = Image.alpha_composite(result, text_overlay)
        
        return result.convert("RGB")
    
    def run(
        self,
        image: Image.Image,
        output_dir: str,
        poster_name: str = "refined_poster",
        poster_width: float = 48,
        poster_height: float = 36,
        max_regions: int = 20,
        min_region_size: int = 50,
    ) -> Dict[str, Any]:
        """执行完整的 refine 流程
        
        Args:
            image: 原始海报图像
            output_dir: 输出目录
            poster_name: 海报名称
            poster_width: 海报宽度（英寸）
            poster_height: 海报高度（英寸）
            max_regions: 最大处理区域数
            min_region_size: 最小区域尺寸（像素）
        
        Returns:
            Dict with keys: pptx_path, pdf_path, background_path, final_image_path
        """
        log_agent_info(_AGENT_NAME, f"starting refine: size={image.size}, output_dir={output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        log_root = get_default_log_root(_AGENT_NAME)
        
        image_width, image_height = image.size
        
        # Step 1: 解析海报，定位文字区域
        log_agent_info(_AGENT_NAME, "Step 1: Parsing poster and locating text regions")
        
        bboxes, region_details, ckpt_path = ground_poster_text_regions_with_mineru_vlm(image)
        
        log_agent_info(_AGENT_NAME, f"found {len(bboxes)} text regions")
        
        # 过滤太小的区域
        valid_regions: List[TextRegion] = []
        for i, (bbox, detail) in enumerate(zip(bboxes, region_details)):
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            if width < min_region_size or height < min_region_size:
                log_agent_info(_AGENT_NAME, f"skipping small region {i}: {width}x{height}")
                continue
            
            if len(valid_regions) >= max_regions:
                log_agent_warning(_AGENT_NAME, f"reached max_regions limit: {max_regions}")
                break
            
            region = TextRegion(
                bbox=bbox,
                text_content=detail.get("text_content", ""),
                region_type=detail.get("type", "text"),
                font_family=self._get_style_font_family(),
            )
            valid_regions.append(region)
        
        log_agent_info(_AGENT_NAME, f"processing {len(valid_regions)} valid regions")
        
        if not valid_regions:
            log_agent_warning(_AGENT_NAME, "no valid text regions found, returning original image")
            original_path = os.path.join(output_dir, f"{poster_name}_original.png")
            image.save(original_path)
            return {
                "pptx_path": None,
                "pdf_path": None,
                "background_path": original_path,
                "final_image_path": original_path,
                "num_regions": 0,
            }
        
        # Step 2: 擦除模糊文字，生成干净背景
        log_agent_info(_AGENT_NAME, "Step 2: Erasing text regions to create clean background")
        
        erase_bboxes = [region.bbox for region in valid_regions]
        
        background_description = f"clean {self.style_name} poster background with consistent colors and patterns"
        
        clean_background = erase_multiple_text_regions(
            image=image,
            bboxes=erase_bboxes,
            background_description=background_description,
        )
        
        background_path = os.path.join(output_dir, f"{poster_name}_background.png")
        clean_background.save(background_path)
        log_agent_success(_AGENT_NAME, f"saved clean background: {background_path}")
        
        # 保存 before/after
        save_before_after_image(
            agent_name=_AGENT_NAME,
            func_name="erase_result",
            before_img=image,
            after_img=clean_background,
        )
        
        # Step 3: 匹配正确的文字内容
        log_agent_info(_AGENT_NAME, "Step 3: Matching text content for each region")
        
        plan_text_spans = []
        if self.plan_text_spans_path:
            try:
                plan_text_spans = _load_plan_text_spans(self.plan_text_spans_path)
                log_agent_info(_AGENT_NAME, f"loaded {len(plan_text_spans)} plan text spans")
            except Exception as e:
                log_agent_warning(_AGENT_NAME, f"failed to load plan text spans: {e}")
        
        for i, region in enumerate(valid_regions):
            # 裁剪区域
            patch = image.crop(region.bbox)
            
            # 匹配文字
            matched_text = None
            if plan_text_spans:
                matched_text, meta = match_plan_text_for_patch(
                    patch=patch,
                    bbox=region.bbox,
                    plan_text_spans=plan_text_spans,
                    hint_text=region.text_content,
                    agent_name=_AGENT_NAME,
                )
            
            # 如果没有匹配到，使用 OCR 结果
            if not matched_text:
                matched_text = region.text_content
            
            region.matched_text = matched_text
            region.font_size = self._estimate_font_size(region.bbox, matched_text or "")
            region.bold = region.region_type in ("title", "section_title", "heading")
            
            log_agent_info(
                _AGENT_NAME,
                f"region {i}: matched_text='{(matched_text or '')[:30]}...', font_size={region.font_size}"
            )
        
        # Step 4: 创建 PPTX 布局并渲染
        log_agent_info(_AGENT_NAME, "Step 4: Creating PPTX layout with text overlays")
        
        # 创建完整的 PPTX 布局（背景 + 文字）
        layout_data = {
            "width": poster_width,
            "height": poster_height,
            "slides": [{
                "elements": [
                    # 背景图片
                    {
                        "type": "image",
                        "id": "background",
                        "x": 0,
                        "y": 0,
                        "width": poster_width,
                        "height": poster_height,
                        "image_path": background_path,
                        "z_order": 0,
                    },
                ]
            }],
        }
        
        # 添加文字元素
        for i, region in enumerate(valid_regions):
            if not region.matched_text:
                continue
            
            x, y, w, h = self._convert_bbox_to_inches(
                region.bbox,
                image_width,
                image_height,
                poster_width,
                poster_height,
            )
            
            text_element = {
                "type": "text",
                "id": f"text_{i}",
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "content": region.matched_text,
                "font_family": region.font_family,
                "font_size": region.font_size,
                "font_color": region.font_color,
                "bold": region.bold,
                "alignment": "left",
                "z_order": 10 + i,
            }
            layout_data["slides"][0]["elements"].append(text_element)
        
        # 保存布局数据
        layout_path = os.path.join(output_dir, f"{poster_name}_layout.json")
        with open(layout_path, 'w', encoding='utf-8') as f:
            json.dump(layout_data, f, ensure_ascii=False, indent=2)
        
        save_json_log(
            agent_name=_AGENT_NAME,
            func_name="layout_data",
            payload=layout_data,
        )
        
        # 渲染 PPTX
        pptx_path = os.path.join(output_dir, f"{poster_name}.pptx")
        
        result = render_layout_to_pptx(
            layout_data=layout_data,
            output_path=pptx_path,
            width=poster_width,
            height=poster_height,
            convert_to_pdf=True,
        )
        
        pptx_path = result.get("pptx_path")
        pdf_path = result.get("pdf_path")
        
        log_agent_success(_AGENT_NAME, f"saved PPTX: {pptx_path}")
        if pdf_path:
            log_agent_success(_AGENT_NAME, f"saved PDF: {pdf_path}")
        
        # Step 5: 生成合成的最终图像预览
        log_agent_info(_AGENT_NAME, "Step 5: Generating composite preview image")
        
        # 使用 PIL 直接在背景上绘制文字作为预览
        overlay_path = os.path.join(output_dir, f"{poster_name}_text_overlay.png")
        self._render_text_to_image(
            regions=valid_regions,
            image_width=image_width,
            image_height=image_height,
            output_path=overlay_path,
            poster_width=poster_width,
            poster_height=poster_height,
        )
        
        text_overlay = Image.open(overlay_path).convert("RGBA")
        final_image = self._composite_background_and_text(clean_background, text_overlay)
        
        final_image_path = os.path.join(output_dir, f"{poster_name}_final.png")
        final_image.save(final_image_path)
        
        log_agent_success(_AGENT_NAME, f"saved final composite image: {final_image_path}")
        
        # 保存 before/after
        save_before_after_image(
            agent_name=_AGENT_NAME,
            func_name="final_result",
            before_img=image,
            after_img=final_image,
        )
        
        # 汇总结果
        result_summary = {
            "pptx_path": pptx_path,
            "pdf_path": pdf_path,
            "background_path": background_path,
            "final_image_path": final_image_path,
            "overlay_path": overlay_path,
            "layout_path": layout_path,
            "num_regions": len(valid_regions),
            "regions": [r.to_dict() for r in valid_regions],
        }
        
        save_json_log(
            agent_name=_AGENT_NAME,
            func_name="refine_result",
            payload=result_summary,
        )
        
        log_agent_success(
            _AGENT_NAME,
            f"refine complete: {len(valid_regions)} regions processed, PPTX={pptx_path is not None}, PDF={pdf_path is not None}"
        )
        
        return result_summary

