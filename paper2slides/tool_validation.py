import argparse
import os
import sys
from typing import List, Optional

from PIL import Image

# Explicit absolute import based on project structure
from paper2slides.agents.tools.zimage_flowedit_tool import ZImageFlowEdit


def resize_image_longest_side(image: Image.Image, size: int = 1024) -> Image.Image:
    """Resize image so that the longest side is at most `size`."""
    width, height = image.size
    if max(width, height) <= size:
        return image
    
    scale = size / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def main(
    image_path: str,
    src_prompt: str,
    tar_prompt: str,
    bbox: Optional[List[int]] = None,
    output_path: str = "output.png",
    resize: bool = True
):
    """
    Main function to edit an image using ZImageFlowEdit tool.
    """
    # 1. Input Validation
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Source image not found at: {image_path}")
    
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Processing Image: {image_path}")
    print(f"Output Path: {output_path}")

    # 2. Load and Preprocess
    original_img = Image.open(image_path).convert("RGB")
    
    if bbox:
        if len(bbox) != 4:
            raise ValueError("bbox must be [x1, y1, x2, y2]")
        x1, y1, x2, y2 = bbox
        img_to_edit = original_img.crop((x1, y1, x2, y2))
        print(f"Cropped region: {bbox}")
    else:
        img_to_edit = original_img.copy()
        print("Editing full image")
        
    if resize:
        img_to_edit = resize_image_longest_side(img_to_edit, 1024)
        print(f"Resized for editing to: {img_to_edit.size}")

    # 3. Prepare Tool Execution
    # Save intermediate files locally instead of temp directory for debugging
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    intermediate_input_path = os.path.join(output_dir, f"{base_name}_input_to_tool.png")
    intermediate_output_path = os.path.join(output_dir, f"{base_name}_tool_output.png")
    
    img_to_edit.save(intermediate_input_path)
    print(f"Saved input to tool at: {intermediate_input_path}")
    
    tool = ZImageFlowEdit()
    
    params = {
        "src_image_path": intermediate_input_path,
        "src_prompt": src_prompt,
        "tar_prompt": tar_prompt,
        "output_path": intermediate_output_path,
    }
    
    print("Invoking ZImageFlowEdit tool...")
    # Direct call, letting exceptions propagate
    tool.call(params)
    
    # 4. Process Result
    if not os.path.exists(intermediate_output_path):
        raise RuntimeError(f"Tool finished but output file was not found at: {intermediate_output_path}")
        
    edited_img = Image.open(intermediate_output_path).convert("RGB")
    print(f"Tool execution successful. Saved intermediate output to: {intermediate_output_path}")
    print(f"Edited image size: {edited_img.size}")
    
    if bbox:
        # Resize edited crop back to original bbox dimensions if necessary
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        if edited_img.size != (bbox_w, bbox_h):
                edited_img = edited_img.resize((bbox_w, bbox_h), Image.Resampling.LANCZOS)
        
        final_img = original_img.copy()
        final_img.paste(edited_img, (x1, y1))
    else:
        final_img = edited_img
        
    # 5. Save Final Output
    final_img.save(output_path)
    print(f"Successfully saved final result to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool Validation Script for ZImageFlowEdit")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--src_prompt", type=str, required=True, help="Source prompt")
    parser.add_argument("--tar_prompt", type=str, required=True, help="Target prompt")
    parser.add_argument("--bbox", type=str, help="Bounding box as x1,y1,x2,y2 (e.g., '100,100,300,300')")
    parser.add_argument("--output_path", type=str, default="output.png", help="Output path")
    parser.add_argument("--no_resize", action="store_true", help="Disable resizing to 1024 longest side")
    
    args = parser.parse_args()
    
    bbox_list = None
    if args.bbox:
        bbox_list = [int(x.strip()) for x in args.bbox.split(",")]
        
    main(
        image_path=args.image_path,
        src_prompt=args.src_prompt,
        tar_prompt=args.tar_prompt,
        bbox=bbox_list,
        output_path=args.output_path,
        resize=not args.no_resize
    )
