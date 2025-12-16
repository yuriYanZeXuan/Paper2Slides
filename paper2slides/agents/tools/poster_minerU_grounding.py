"""
使用 MinerU 在海报图片中定位文字区域。

将图片转换为 PDF 后，使用 MinerU 的 OCR 能力提取文字 bbox 和内容。
"""
import io
import json
import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any

from PIL import Image

from qwen_agent.tools.base import BaseTool, register_tool
from paper2slides.utils.agent_logging import log_agent_info, log_agent_warning
from paper2slides.utils.agent_artifact_logging import save_bbox_visualization, save_json_log


BBox = Tuple[int, int, int, int]


def _image_to_pdf_bytes(image: Image.Image) -> bytes:
    """将 PIL Image 转换为 PDF 字节流（单页）。"""
    buf = io.BytesIO()
    # PIL 可以直接将图片保存为 PDF 格式
    image_rgb = image.convert("RGB")
    image_rgb.save(buf, format="PDF")
    return buf.getvalue()


def _resolve_mineru_cuda_visible_devices() -> str | None:
    """
    解析 MinerU 设备配置并转换为 CUDA_VISIBLE_DEVICES：
    优先级：
      1) POSTERGEN_MINERU_CUDA_VISIBLE_DEVICES（直接传递）
      2) POSTERGEN_MINERU_DEVICE / POSTERGEN_DEVICE：
         - 'cpu'/'none'/'-1' -> ""
         - 'cuda:N' -> "N"
    未提供则返回 None，不做覆盖。
    """
    direct = os.getenv("POSTERGEN_MINERU_CUDA_VISIBLE_DEVICES")
    if direct is not None:
        return direct
    device = os.getenv("POSTERGEN_MINERU_DEVICE") or os.getenv("POSTERGEN_DEVICE")
    if not device:
        return None
    val = device.strip().lower()
    if val in ("cpu", "none", "-1"):
        return ""
    if val.startswith("cuda:"):
        try:
            idx = int(val.split(":", 1)[1])
            return str(idx)
        except Exception:
            return val.split(":", 1)[1] or "0"
    return None


def ground_poster_text_regions_with_mineru(image: Image.Image) -> Tuple[List[BBox], List[Dict[str, Any]]]:
    """使用 MinerU 在海报图片中定位文字区域。
    
    返回值：
    - bboxes: 像素级别的 bbox 列表
    - region_details: 完整的 region 信息，包含 text_content
    """
    w, h = image.size
    
    # 在 MinerU 执行期临时覆盖 CUDA_VISIBLE_DEVICES，并在结束后恢复
    prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    cvd = _resolve_mineru_cuda_visible_devices()
    applied = False
    if cvd is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cvd
        applied = True
        log_agent_info("poster_text_grounding", f"mineru: set CUDA_VISIBLE_DEVICES='{cvd}'")
    
    try:
        # 延迟导入 MinerU，避免未安装时影响其他路径
        try:
            from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
            from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
            from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
            from mineru.data.data_reader_writer import FileBasedDataWriter
            from mineru.utils.enum_class import MakeMode
        except ImportError as e:
            raise RuntimeError(f"MinerU 未安装或导入失败，请先安装 mineru[core]。错误: {e}")
        
        # 将图片转换为 PDF 字节流
        pdf_bytes = _image_to_pdf_bytes(image)
        
        log_agent_info("poster_text_grounding", f"converting image ({w}x{h}) to PDF for MinerU processing")
        
        # 创建临时目录用于 MinerU 输出
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_writer = FileBasedDataWriter(str(tmp_path))
            
            # 调用 MinerU pipeline 分析
            infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                [pdf_bytes], 
                ["ch"],  # 语言设置：中英混合
                parse_method="auto", 
                formula_enable=False,  # 不需要公式识别
                table_enable=False,  # 不需要表格识别
            )
            
            model_list = infer_results[0]
            images_list = all_image_lists[0]
            pdf_doc = all_pdf_docs[0]
            _lang = lang_list[0]
            _ocr_enable = ocr_enabled_list[0]
            
            # 生成 middle_json
            middle_json = pipeline_result_to_middle_json(
                model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, True
            )
            
            # 生成 content_list
            content_list = pipeline_union_make(
                middle_json["pdf_info"], 
                MakeMode.CONTENT_LIST, 
                "assets"
            )
        
        log_agent_info(
            "poster_text_grounding",
            f"MinerU extracted {len(content_list)} content items"
        )
        
        # 记录原始响应
        save_json_log(
            agent_name="poster_text_grounding",
            func_name="mineru_grounding_raw",
            payload={
                "image_size": {"width": w, "height": h},
                "content_list_count": len(content_list),
                "content_list": content_list,
            },
        )
        
        # 提取文字区域的 bbox
        bboxes: List[BBox] = []
        region_details: List[Dict[str, Any]] = []
        
        for idx, item in enumerate(content_list):
            item_type = str(item.get("type", "")).lower()
            
            # 只处理文字类型
            if item_type not in ("text", "paragraph", "title", "heading", "section_title"):
                continue
            
            # 获取 bbox
            bbox = item.get("bbox") or item.get("position")
            if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            
            # 获取文字内容
            text_content = item.get("text") or item.get("content") or ""
            if not text_content:
                continue
            
            # 解析坐标 - MinerU 返回的是 PDF 坐标
            # PDF 标准是 72 点/英寸，图片转换为 PDF 时 PIL 使用默认 DPI
            # 需要根据实际 PDF 页面尺寸与图片尺寸的比例进行转换
            x0_raw, y0_raw, x1_raw, y1_raw = map(float, bbox)
            
            # 获取 PDF 页面尺寸（如果可用）
            page_width = middle_json.get("pdf_info", {}).get("page_width", w)
            page_height = middle_json.get("pdf_info", {}).get("page_height", h)
            
            # 如果 PDF 页面尺寸与图片不同，需要缩放坐标
            # 通常 PIL 保存 PDF 时会按 72 DPI 来计算页面尺寸
            scale_x = w / page_width if page_width else 1.0
            scale_y = h / page_height if page_height else 1.0
            
            x0 = int(x0_raw * scale_x)
            y0 = int(y0_raw * scale_y)
            x1 = int(x1_raw * scale_x)
            y1 = int(y1_raw * scale_y)
            
            # 裁剪到图片范围内
            x0 = max(0, min(x0, w))
            x1 = max(0, min(x1, w))
            y0 = max(0, min(y0, h))
            y1 = max(0, min(y1, h))
            
            # 检查有效性
            if x1 <= x0 or y1 <= y0:
                log_agent_warning(
                    "poster_text_grounding",
                    f"invalid bbox after processing: ({x0}, {y0}, {x1}, {y1})"
                )
                continue
            
            bboxes.append((x0, y0, x1, y1))
            region_details.append({
                "idx": len(region_details),
                "raw_bbox": list(bbox),
                "final_bbox": [x0, y0, x1, y1],
                "text_content": text_content,
                "type": item_type,
            })
            
            log_msg = f"region {len(region_details)-1}: type={item_type}, bbox={[x0, y0, x1, y1]}"
            if len(text_content) > 50:
                log_msg += f", text='{text_content[:50]}...'"
            else:
                log_msg += f", text='{text_content}'"
            log_agent_info("poster_text_grounding", log_msg)
        
        # 保存详细的 region 信息到日志
        save_json_log(
            agent_name="poster_text_grounding",
            func_name="mineru_grounding_regions",
            payload={
                "image_size": {"width": w, "height": h},
                "valid_regions_count": len(bboxes),
                "region_details": region_details,
            },
        )
        
        if len(bboxes) == 0:
            log_agent_warning("poster_text_grounding", "No valid text regions found from MinerU")
            raise ValueError(f"No valid text regions found. content_list_count={len(content_list)}, image_size=({w}, {h})")
        
        log_agent_info("poster_text_grounding", f"MinerU grounded {len(bboxes)} text regions: {bboxes}")
        return bboxes, region_details
    
    finally:
        if applied:
            if prev_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd
            log_agent_info("poster_text_grounding", "mineru: restored CUDA_VISIBLE_DEVICES")


@register_tool('poster_text_grounding')
class PosterTextGrounding(BaseTool):
    """在海报中定位文字区域。

    当前实现：使用 MinerU 进行 OCR 和文字定位。
    """

    description = 'Locate text regions in a poster image using MinerU OCR.'
    parameters = {
        'type': 'object',
        'properties': {
            'image_path': {
                'type': 'string',
                'description': 'Path to the poster image file.'
            }
        },
        'required': ['image_path'],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        image_path = params['image_path']
        img = Image.open(image_path).convert("RGB")
        bboxes, region_details = ground_poster_text_regions_with_mineru(img)

        # 保存 bbox 可视化结果
        vis_path = save_bbox_visualization(
            agent_name="poster_text_grounding",
            func_name="mineru_grounding",
            image=img,
            bboxes=bboxes,
        )
        log_agent_info("poster_text_grounding", f"bbox visualization saved to {vis_path}")
      
        return json.dumps({
            'bboxes': bboxes,
            'regions': region_details,
        }, ensure_ascii=False)
