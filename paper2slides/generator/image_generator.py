"""
Image Generator

Generate poster/slides images from ContentPlan.
"""
import os
import json
import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from openai import OpenAI

from .config import GenerationInput
from .content_planner import ContentPlan, Section
from ..prompts.image_generation import (
    STYLE_PROCESS_PROMPT,
    FORMAT_POSTER,
    FORMAT_SLIDE,
    POSTER_STYLE_HINTS,
    SLIDE_STYLE_HINTS,
    SLIDE_LAYOUTS_ACADEMIC,
    SLIDE_LAYOUTS_DORAEMON,
    SLIDE_LAYOUTS_DEFAULT,
    SLIDE_COMMON_STYLE_RULES,
    POSTER_COMMON_STYLE_RULES,
    VISUALIZATION_HINTS,
    CONSISTENCY_HINT,
    SLIDE_FIGURE_HINT,
    POSTER_FIGURE_HINT,
)


@dataclass
class GeneratedImage:
    """Generated image result."""
    section_id: str
    image_data: bytes
    mime_type: str


@dataclass
class ProcessedStyle:
    """Processed custom style from LLM."""
    style_name: str       # e.g., "Cyberpunk sci-fi style with high-tech aesthetic"
    color_tone: str       # e.g., "dark background with neon accents"
    special_elements: str # e.g., "Characters appear as guides" or ""
    decorations: str      # e.g., "subtle grid pattern" or ""
    valid: bool
    error: Optional[str] = None


def process_custom_style(client: OpenAI, user_style: str, model: str = None) -> ProcessedStyle:
    """Process user's custom style request with LLM."""
    model = model or os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": STYLE_PROCESS_PROMPT.format(user_style=user_style)}],
            # extra_body logic removed here as it's not standard for text generation and handled by client wrapper if needed
            response_format={"type": "json_object"},
        )
        
        # Handle wrapper response object which might differ slightly from raw SDK response
        if hasattr(response, 'choices') and len(response.choices) > 0:
             content = response.choices[0].message.content
        else:
             raise ValueError("Empty response from LLM")

        result = json.loads(content)
        return ProcessedStyle(
            style_name=result.get("style_name", ""),
            color_tone=result.get("color_tone", ""),
            special_elements=result.get("special_elements", ""),
            decorations=result.get("decorations", ""),
            valid=result.get("valid", False),
            error=result.get("error"),
        )
    except Exception as e:
        return ProcessedStyle(style_name="", color_tone="", special_elements="", decorations="", valid=False, error=str(e))


class ImageGenerator:
    """Generate poster/slides images from ContentPlan."""
    
    # Default native endpoint (matches PosterGen2/eval/gemini_image_gen.py)
    DEFAULT_GEMINI_NATIVE_URL = "https://runway.devops.rednote.life/openai/google/v1:generateContent"
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = "google/gemini-3-pro-image-preview",
        throttle_ms: int = None,
        backend: str = None,
        local_model: str = None,
    ):
        """
        Args:
            backend: 'gemini' 使用远程 Gemini/OpenAI 网关（默认），
                     'zimage' 使用本地 Z-Image 模型（diffusers）。
            local_model: 本地 Z-Image 模型的 repo id 或路径，默认 `Tongyi-MAI/Z-Image-Turbo`。
        """
        from ..utils.api_utils import load_env_api_key, get_api_base_url, get_openai_client
        
        # 后端选择 & 本地模型配置
        self.backend = (backend or os.getenv("P2S_IMAGE_BACKEND", "gemini")).lower()
        self.local_model = local_model or os.getenv("P2S_LOCAL_IMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")
        self.local_device = os.getenv("P2S_LOCAL_IMAGE_DEVICE", "cuda")
        self._local_pipe = None  # 延迟加载 Z-ImagePipeline
        self._local_torch = None
        self.model = model
        
        # Throttle between image generations to avoid 429; default from env
        env_throttle_ms = os.getenv("IMAGE_GEN_THROTTLE_MS")
        self.throttle_seconds = (throttle_ms or (int(env_throttle_ms) if env_throttle_ms else 0)) / 1000.0
        
        # Load keys specifically for "image" usage
        self.api_key = api_key or load_env_api_key("image")
        
        # 若 .env 未提供 IMAGE_GEN_BASE_URL，则使用与 PosterGen2 相同的默认原生 URL
        env_base_url = get_api_base_url("image")
        self.base_url = base_url or env_base_url or self.DEFAULT_GEMINI_NATIVE_URL
        
        # Use key_type="image" to ensure correct logging/client selection
        self.client = get_openai_client(api_key=self.api_key, base_url=self.base_url, key_type="image")
    
    def generate(
        self,
        plan: ContentPlan,
        gen_input: GenerationInput,
    ) -> List[GeneratedImage]:
        """
        Generate images from ContentPlan.
        
        Args:
            plan: ContentPlan from ContentPlanner
            gen_input: GenerationInput with config and origin
        
        Returns:
            List of GeneratedImage (1 for poster, N for slides)
        """
        figure_images = self._load_figure_images(plan, gen_input.origin.base_path)
        style_name = gen_input.config.style.value
        custom_style = gen_input.config.custom_style
        
        # Process custom style with LLM if needed
        processed_style = None
        if style_name == "custom" and custom_style:
            processed_style = process_custom_style(self.client, custom_style)
            if not processed_style.valid:
                raise ValueError(f"Invalid custom style: {processed_style.error}")
        
        all_sections_md = self._format_sections_markdown(plan)
        all_images = self._filter_images(plan.sections, figure_images)
        
        if plan.output_type == "poster":
            return self._generate_poster(style_name, processed_style, all_sections_md, all_images)
        else:
            return self._generate_slides(plan, style_name, processed_style, all_sections_md, figure_images)
    
    def _generate_poster(self, style_name, processed_style: Optional[ProcessedStyle], sections_md, images) -> List[GeneratedImage]:
        """Generate 1 poster image."""
        prompt = self._build_poster_prompt(
            format_prefix=FORMAT_POSTER,
            style_name=style_name,
            processed_style=processed_style,
            sections_md=sections_md,
        )
        
        image_data, mime_type = self._call_model(prompt, images)
        return [GeneratedImage(section_id="poster", image_data=image_data, mime_type=mime_type)]
    
    def _generate_slides(self, plan, style_name, processed_style: Optional[ProcessedStyle], all_sections_md, figure_images) -> List[GeneratedImage]:
        """Generate N slide images."""
        results = []
        total = len(plan.sections)
        
        # Select layout rules based on style
        # Custom styles use default layout (LLM only generates style hints, not layouts)
        if style_name == "custom":
            layouts = SLIDE_LAYOUTS_DEFAULT
        elif style_name == "doraemon":
            layouts = SLIDE_LAYOUTS_DORAEMON
        else:
            layouts = SLIDE_LAYOUTS_ACADEMIC
        
        style_ref_image = None  # Store 2nd slide as reference for all subsequent slides
        
        for i, section in enumerate(plan.sections):
            section_md = self._format_single_section_markdown(section, plan)
            layout_rule = layouts.get(section.section_type, layouts["content"])
            
            prompt = self._build_slide_prompt(
                style_name=style_name,
                processed_style=processed_style,
                sections_md=section_md,
                layout_rule=layout_rule,
                slide_info=f"Slide {i+1} of {total}",
                context_md=all_sections_md,
            )
            
            # Collect reference images
            section_images = self._filter_images([section], figure_images)
            reference_images = []
            
            # Use 2nd slide as reference for all subsequent slides (all styles)
            if style_ref_image:
                reference_images.append(style_ref_image)
            
            reference_images.extend(section_images)
            
            image_data, mime_type = self._call_model(prompt, reference_images)
            
            # Save 2nd slide (i=1) as the style reference for all styles
            if i == 1:
                style_ref_image = {
                    "figure_id": "Reference Slide",
                    "caption": "STRICTLY MAINTAIN: same background color, same accent color, same font style, same chart/icon style. Keep visual consistency.",
                    "base64": base64.b64encode(image_data).decode("utf-8"),
                    "mime_type": mime_type,
                }
            
            results.append(GeneratedImage(section_id=section.id, image_data=image_data, mime_type=mime_type))

            # Throttle to avoid hitting rate limits (429)
            if self.throttle_seconds > 0 and i != len(plan.sections) - 1:
                import time
                time.sleep(self.throttle_seconds)
        
        return results
    
    def _format_custom_style_for_poster(self, ps: ProcessedStyle) -> str:
        """Format ProcessedStyle into style hints string for poster."""
        parts = [
            ps.style_name + ".",
            "English text only.",
            "Use ROUNDED sans-serif fonts for ALL text.",
            "Characters should react to or interact with the content, with appropriate poses/actions and sizes - not just decoration."
            f"LIMITED COLOR PALETTE (3-4 colors max): {ps.color_tone}.",
            POSTER_COMMON_STYLE_RULES,
        ]
        if ps.special_elements:
            parts.append(ps.special_elements + ".")
        return " ".join(parts)
    
    def _format_custom_style_for_slide(self, ps: ProcessedStyle) -> str:
        """Format ProcessedStyle into style hints string for slide."""
        parts = [
            ps.style_name + ".",
            "English text only.",
            "Use ROUNDED sans-serif fonts for ALL text.",
            "Characters should react to or interact with the content, with appropriate poses/actions and sizes - not just decoration.",
            f"LIMITED COLOR PALETTE (3-4 colors max): {ps.color_tone}.",
            SLIDE_COMMON_STYLE_RULES,
        ]
        if ps.special_elements:
            parts.append(ps.special_elements + ".")
        return " ".join(parts)
    
    def _build_poster_prompt(self, format_prefix, style_name, processed_style: Optional[ProcessedStyle], sections_md) -> str:
        """Build prompt for poster."""
        parts = [format_prefix]
        
        if style_name == "custom" and processed_style:
            parts.append(f"Style: {self._format_custom_style_for_poster(processed_style)}")
            if processed_style.decorations:
                parts.append(f"Decorations: {processed_style.decorations}")
        else:
            parts.append(POSTER_STYLE_HINTS.get(style_name, POSTER_STYLE_HINTS["academic"]))
        
        parts.append(VISUALIZATION_HINTS)
        parts.append(POSTER_FIGURE_HINT)
        parts.append(f"---\nContent:\n{sections_md}")
        
        return "\n\n".join(parts)
    
    def _build_slide_prompt(self, style_name, processed_style: Optional[ProcessedStyle], sections_md, layout_rule, slide_info, context_md) -> str:
        """Build prompt for slide with layout rules and consistency."""
        parts = [FORMAT_SLIDE]
        
        if style_name == "custom" and processed_style:
            parts.append(f"Style: {self._format_custom_style_for_slide(processed_style)}")
        else:
            parts.append(SLIDE_STYLE_HINTS.get(style_name, SLIDE_STYLE_HINTS["academic"]))
        
        # Add layout rule, then decorations if custom style
        parts.append(layout_rule)
        if style_name == "custom" and processed_style and processed_style.decorations:
            parts.append(f"Decorations: {processed_style.decorations}")
        
        parts.append(VISUALIZATION_HINTS)
        parts.append(CONSISTENCY_HINT)
        parts.append(SLIDE_FIGURE_HINT)
        
        parts.append(slide_info)
        parts.append(f"---\nFull presentation context:\n{context_md}")
        parts.append(f"---\nThis slide content:\n{sections_md}")
        
        return "\n\n".join(parts)
    
    def _format_sections_markdown(self, plan: ContentPlan) -> str:
        """Format all sections as markdown."""
        parts = []
        for section in plan.sections:
            parts.append(self._format_single_section_markdown(section, plan))
        return "\n\n---\n\n".join(parts)
    
    def _format_single_section_markdown(self, section: Section, plan: ContentPlan) -> str:
        """Format a single section as markdown."""
        lines = [f"## {section.title}", "", section.content]
        
        for ref in section.tables:
            table = plan.tables_index.get(ref.table_id)
            if table:
                focus_str = f" (focus: {ref.focus})" if ref.focus else ""
                lines.append("")
                lines.append(f"**{ref.table_id}**{focus_str}:")
                lines.append(ref.extract if ref.extract else table.html_content)
        
        for ref in section.figures:
            fig = plan.figures_index.get(ref.figure_id)
            if fig:
                focus_str = f" (focus: {ref.focus})" if ref.focus else ""
                caption = f": {fig.caption}" if fig.caption else ""
                lines.append("")
                lines.append(f"**{ref.figure_id}**{focus_str}{caption}")
                lines.append("[Image attached]")
        
        return "\n".join(lines)
    
    def _load_figure_images(self, plan: ContentPlan, base_path: str) -> List[dict]:
        """Load figure images as base64."""
        images = []
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"
        }
        
        for fig_id, fig in plan.figures_index.items():
            if base_path:
                img_path = Path(base_path) / fig.image_path
            else:
                img_path = Path(fig.image_path)
            
            if not img_path.exists():
                continue
            
            mime_type = mime_map.get(img_path.suffix.lower(), "image/jpeg")
            
            try:
                with open(img_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode("utf-8")
                images.append({
                    "figure_id": fig_id,
                    "caption": fig.caption,
                    "base64": img_data,
                    "mime_type": mime_type,
                })
            except Exception:
                continue
        
        return images
    
    def _filter_images(self, sections: List[Section], figure_images: List[dict]) -> List[dict]:
        """Filter images used in given sections."""
        used_ids = set()
        for section in sections:
            for ref in section.figures:
                used_ids.add(ref.figure_id)
        return [img for img in figure_images if img.get("figure_id") in used_ids]
    
    def _call_model(self, prompt: str, reference_images: List[dict]) -> tuple:
        """Call the image generation model."""
        # 若选择本地 Z-Image 后端，则直接走本地推理分支
        if getattr(self, "backend", "gemini") == "zimage":
            return self._call_zimage_local(prompt, reference_images)
        
        # Check if we should use native Gemini API (based on model name or config)
        if "gemini" in self.model.lower() and "preview" in self.model.lower():
            try:
                return self._call_gemini_native(prompt, reference_images)
            except Exception as e:
                # 如果使用的是原生 gemini endpoint，避免回退到 chat/completions 以免 404
                if "google/v1" in (self.base_url or "") or ":generateContent" in (self.base_url or ""):
                    raise
                print(f"Gemini native call failed: {e}. Falling back to OpenAI SDK.")
        
        content = [{"type": "text", "text": prompt}]
        
        # Add each image with figure_id and caption label
        for img in reference_images:
            if img.get("base64") and img.get("mime_type"):
                fig_id = img.get("figure_id", "Figure")
                caption = img.get("caption", "")
                label = f"[{fig_id}]: {caption}" if caption else f"[{fig_id}]"
                content.append({"type": "text", "text": label})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{img['mime_type']};base64,{img['base64']}"}
                })
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            extra_body={"modalities": ["image", "text"]}
        )
        
        # 防御：若没有返回 choices，直接报错而不是 IndexError
        if not hasattr(response, "choices") or not response.choices:
            raise RuntimeError("Image generation failed: empty choices from fallback client.")
        
        message = response.choices[0].message
        if hasattr(message, 'images') and message.images:
            image_url = message.images[0]['image_url']['url']
            if image_url.startswith('data:'):
                header, base64_data = image_url.split(',', 1)
                mime_type = header.split(':')[1].split(';')[0]
                return base64.b64decode(base64_data), mime_type
        
        raise RuntimeError("Image generation failed")
    
    def _get_zimage_pipeline(self):
        """Lazy-load ZImagePipeline 模型，避免在未选择本地后端时加载大模型。"""
        if self._local_pipe is not None:
            return self._local_pipe
        
        try:
            import torch
            from diffusers import ZImagePipeline
        except Exception as e:
            raise RuntimeError(f"Failed to import local Z-Image dependencies (torch/diffusers): {e}")
        
        pipe = ZImagePipeline.from_pretrained(
            self.local_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        pipe.to(self.local_device)
        
        self._local_pipe = pipe
        self._local_torch = torch
        return self._local_pipe
    
    def _call_zimage_local(self, prompt: str, reference_images: List[dict]) -> tuple:
        """
        使用本地 ZImagePipeline 生成图片，返回 (bytes, mime_type)。
        当前忽略 reference_images，仅基于文本 prompt 生成。
        """
        pipe = self._get_zimage_pipeline()
        torch = self._local_torch
        
        # 简单的尺寸设定：后续可以根据 prompt / config 调整比例
        height = 1024
        width = 1024
        
        generator = torch.Generator(device=self.local_device).manual_seed(42)
        
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=9,  # 与 dev/Z_image.py 一致
            guidance_scale=0.0,     # Turbo 模型推荐 0
            generator=generator,
        ).images[0]
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue(), "image/png"

    def _call_gemini_native(self, prompt: str, reference_images: List[dict]) -> tuple:
        """
        Call Gemini native API directly (ported from PosterGen2).
        Handles raw HTTP request for image generation.
        """
        import requests
        
        # 固定使用与 PosterGen2 一致的原生 URL，避免分支和拼接误差
        endpoint = "https://runway.devops.rednote.life/openai/google/v1:generateContent"
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
        
        print(f"[DEBUG] Gemini Native Endpoint (Fixed): {endpoint}")

        # Build parts list
        parts = [{"text": prompt}]
        
        for img in reference_images:
            if img.get("base64") and img.get("mime_type"):
                fig_id = img.get("figure_id", "Figure")
                caption = img.get("caption", "")
                label = f"[{fig_id}]: {caption}" if caption else f"[{fig_id}]"
                
                # Add label text part
                parts.append({"text": label})
                
                # Add image part (inlineData format)
                parts.append({
                    "inlineData": {
                        "mimeType": img['mime_type'],
                        "data": img['base64']
                    }
                })

        # Config exactly as in eval_gemini_poster.py
        generation_config = {
            "temperature": 0.6,
            "maxOutputTokens": 7000,
            "topP": 1,
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {
                "aspectRatio": "3:4" if "poster" in prompt.lower() else "16:9", # Auto-detect aspect ratio
                "imageSize": "1K",
                "imageOutputOptions": {
                    "mimeType": "image/png"
                },
                "personGeneration": "ALLOW_ALL"
            }
        }

        payload = {
            "contents": [{
                "role": "user",
                "parts": parts
            }],
            "generationConfig": generation_config,
            "safetySettings": [
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"}
            ]
        }

        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        
        if response.status_code != 200:
             raise RuntimeError(f"Gemini native API error ({response.status_code}): {response.text}")

        result = response.json()
        
        # Parse response to find image
        if result and "candidates" in result and result["candidates"]:
            candidate = result["candidates"][0]
            # Check finishReason
            finish_reason = candidate.get("finishReason")
            if finish_reason and finish_reason != "STOP":
                 print(f"Warning: Gemini finishReason is {finish_reason}. Full candidate: {candidate}")
            
            candidate_parts = candidate.get("content", {}).get("parts", [])
            for part in candidate_parts:
                inline_data = part.get("inlineData")
                if inline_data:
                    b64_data = inline_data.get("data")
                    mime_type = inline_data.get("mimeType", "image/png")
                    if b64_data:
                        return base64.b64decode(b64_data), mime_type
        
        # If we got here, something is wrong with the response content
        error_msg = f"No image data in Gemini native response. "
        if "error" in result:
             error_msg += f"API Error: {result['error']}"
        else:
             error_msg += f"Full response: {json.dumps(result)[:500]}"
        raise RuntimeError(error_msg)


def save_images_as_pdf(images: List[GeneratedImage], output_path: str):
    """
    Save generated images as a single PDF file.
    
    Args:
        images: List of GeneratedImage from ImageGenerator.generate()
        output_path: Output PDF file path
    """
    from PIL import Image
    import io
    
    pdf_images = []
    
    for img in images:
        # Load image from bytes
        pil_img = Image.open(io.BytesIO(img.image_data))
        
        # Convert RGBA to RGB (PDF doesn't support alpha)
        if pil_img.mode == 'RGBA':
            pil_img = pil_img.convert('RGB')
        elif pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        pdf_images.append(pil_img)
    
    if pdf_images:
        # Save first image and append the rest
        pdf_images[0].save(
            output_path,
            save_all=True,
            append_images=pdf_images[1:] if len(pdf_images) > 1 else [],
            resolution=100.0,
        )
        print(f"PDF saved: {output_path}")
