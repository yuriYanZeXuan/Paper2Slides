import json
import os
from typing import List, Tuple, Union, Dict, Any

from PIL import Image

from qwen_agent.tools.base import BaseTool, register_tool
from paper2slides.utils.agent_logging import log_agent_info, log_agent_success, log_agent_warning
from paper2slides.utils.agent_artifact_logging import save_bbox_visualization, save_json_log

from mineru_vl_utils import MinerUClient
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
BBox = Tuple[int, int, int, int]

# 全局缓存 Client，避免重复加载模型
_MINERU_CLIENT = None
# 默认模型路径，与 dev/minerU_vlm.py 保持一致
_DEFAULT_MODEL_PATH = "opendatalab/MinerU2.5-2509-1.2B"


def _get_mineru_client() -> "MinerUClient":
    global _MINERU_CLIENT
    if _MINERU_CLIENT is not None:
        return _MINERU_CLIENT
    model_path = os.getenv("MINERU_MODEL_PATH", _DEFAULT_MODEL_PATH)
    log_agent_info("poster_text_grounding", f"loading MinerU model from {model_path}...")
    
    # 加载模型和处理器
    # 参考 dev/minerU_vlm.py
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        use_fast=True
    )
    
    client = MinerUClient(
        backend="transformers",
        model=model,
        processor=processor
    )
    _MINERU_CLIENT = client
    return client


def ground_poster_text_regions_with_mineru_vlm(image: Image.Image) -> Tuple[List[BBox], List[Dict[str, Any]]]:
    """使用 MinerU VLM (OCR) 在海报图片中定位文字区域。
    
    返回值：
    - bboxes: 像素级别的 bbox 列表 [x0, y0, x1, y1]
    - region_details: 完整的 region 信息，包含 text_content
    """
    w, h = image.size
    client = _get_mineru_client()
    
    log_agent_info("poster_text_grounding", f"extracting text from image ({w}x{h}) using MinerU VLM")
    
    # 调用 two_step_extract 进行提取
    # 返回 list[ContentBlock]，其中 ContentBlock 包含 type, bbox(normalized), content
    extracted_blocks = client.two_step_extract(image)
    
    # 记录原始结果
    # ContentBlock 对象不是 JSON 可序列化的，需要转换
    raw_blocks_dump = []
    for idx, block in enumerate(extracted_blocks):
        raw_blocks_dump.append({
            "id": idx,  # 增加 id 方便后续索引
            "type": getattr(block, "type", ""),
            "bbox": getattr(block, "bbox", []),
            "content": getattr(block, "content", ""),
            "angle": getattr(block, "angle", 0)
        })
        
    ckpt_path = save_json_log(
        agent_name="poster_text_grounding",
        func_name="mineru_vlm_raw",
        payload={
            "image_size": {"width": w, "height": h},
            "blocks_count": len(extracted_blocks),
            "raw_blocks": raw_blocks_dump
        },
    )
    
    bboxes: List[BBox] = []
    region_details: List[Dict[str, Any]] = []
    
    # 感兴趣的类型
    target_types = {"title", "text", "list", "section_title", "heading"}
    
    for idx, block in enumerate(extracted_blocks):
        b_type = getattr(block, "type", "").lower()
        
        # 过滤类型
        if b_type not in target_types:
            continue
            
        # 获取内容
        content = getattr(block, "content", "") or ""
        if not content.strip():
            continue
            
        # 获取 bbox (normalized [x0, y0, x1, y1])
        # 注意：user sample 显示 x0, y0, x1, y1
        # MinerUClient _convert_bbox 返回的是 [x1, y1, x2, y2] normalized
        norm_bbox = getattr(block, "bbox", [])
        if not norm_bbox or len(norm_bbox) != 4:
            continue
            
        nx0, ny0, nx1, ny1 = norm_bbox
        
        # 转换为像素坐标
        x0 = int(nx0 * w)
        y0 = int(ny0 * h)
        x1 = int(nx1 * w)
        y1 = int(ny1 * h)
        
        # 边界检查
        x0 = max(0, min(x0, w))
        y0 = max(0, min(y0, h))
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        
        if x1 <= x0 or y1 <= y0:
            continue
            
        bboxes.append((x0, y0, x1, y1))
        region_details.append({
            "idx": len(region_details),
            "id": idx, # 对应原始 list 的索引
            "bbox": [x0, y0, x1, y1],
            "text_content": content,
            "type": b_type,
            "angle": getattr(block, "angle", 0)
        })
        
    log_agent_info(
        "poster_text_grounding", 
        f"MinerU VLM grounded {len(bboxes)} text regions"
    )
    
    # 返回值增加 ckpt_path
    return bboxes, region_details, str(ckpt_path)


@register_tool('poster_text_grounding')
class PosterTextGrounding(BaseTool):
    """在海报中定位文字区域。

    当前实现：使用 MinerU VLM 进行 OCR 和文字定位。
    """

    description = 'Locate text regions in a poster image using MinerU VLM OCR.'
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
        
        bboxes, region_details, ckpt_path = ground_poster_text_regions_with_mineru_vlm(img)
        
        # 保存 bbox 可视化结果
        vis_path = save_bbox_visualization(
            agent_name="poster_text_grounding",
            func_name="mineru_vlm_grounding",
            image=img,
            bboxes=bboxes,
        )
        log_agent_success("poster_text_grounding", f"bbox visualization saved to {vis_path}")
        
        # 简化返回给 Agent 的内容，去除冗长的 text_content，仅保留 ID、Type、BBox
        # 并提供 grounding_ckpt_path 供后续 lookup
        agent_regions = []
        for r in region_details:
            agent_regions.append({
                "id": r["id"],
                "idx": r["idx"],
                "bbox": r["bbox"],
                "type": r["type"],
                # "text_content": ... (OMITTED to save context)
            })

        return json.dumps({
            'grounding_ckpt_path': ckpt_path,
            'bboxes': bboxes,
            'regions': agent_regions,
            'note': "Full text content is saved in grounding_ckpt_path. Use it in subsequent tools."
        }, ensure_ascii=False)
