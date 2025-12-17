import argparse
import os
import tempfile
from typing import List, Optional

from PIL import Image

from .zimage_flowedit_tool import ZImageFlowEdit


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
    
    Args:
        image_path: Path to the source image.
        src_prompt: Source prompt describing the image/region.
        tar_prompt: Target prompt describing the desired edit.
        bbox: Optional bounding box [x1, y1, x2, y2]. If provided, only this region is edited.
        output_path: Path to save the final result.
        resize: Whether to resize the input (or crop) to 1024 longest side before editing.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Load original image
    original_img = Image.open(image_path).convert("RGB")
    
    # Logic for handling bbox
    if bbox:
        # Validate bbox
        if len(bbox) != 4:
            raise ValueError("bbox must be [x1, y1, x2, y2]")
        
        x1, y1, x2, y2 = bbox
        # Crop the region defined by bbox
        img_to_edit = original_img.crop((x1, y1, x2, y2))
    else:
        # If no bbox, edit the whole image
        img_to_edit = original_img.copy()
        
    # Resize if requested (default to 1024 longest side)
    if resize:
        img_to_edit = resize_image_longest_side(img_to_edit, 1024)
        
    # Save to temp file because the tool expects a file path
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_input:
        tmp_input_path = tmp_input.name
        img_to_edit.save(tmp_input_path)
        
    # Temporary output path for the tool result
    tmp_output_path = tmp_input_path.replace(".png", "_out.png")
    
    # Initialize the tool
    tool = ZImageFlowEdit()
    
    # Prepare parameters for the tool
    params = {
        "src_image_path": tmp_input_path,
        "src_prompt": src_prompt,
        "tar_prompt": tar_prompt,
        "output_path": tmp_output_path,
        # Default model and device will be used if not specified
    }
    
    print(f"Running ZImageFlowEdit with params: {params}")
    # Call the tool
    tool.call(params)
    
    # Process the result
    edited_img = Image.open(tmp_output_path).convert("RGB")
    
    if bbox:
        # If we had a bbox, we need to paste the result back
        
        # 1. Resize edited crop back to original bbox dimensions if necessary
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        if edited_img.size != (bbox_w, bbox_h):
                edited_img = edited_img.resize((bbox_w, bbox_h), Image.Resampling.LANCZOS)
        
        # 2. Paste back into a copy of the original image
        final_img = original_img.copy()
        final_img.paste(edited_img, (x1, y1))
    else:
        # If no bbox, the result is the whole image
        final_img = edited_img
        
    # Save final result
    final_img.save(output_path)
    print(f"Saved result to {output_path}")
            

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
        try:
            bbox_list = [int(x.strip()) for x in args.bbox.split(",")]
        except ValueError:
            print("Error: bbox must be four comma-separated integers.")
            exit(1)
        
    main(
        image_path=args.image_path,
        src_prompt=args.src_prompt,
        tar_prompt=args.tar_prompt,
        bbox=bbox_list,
        output_path=args.output_path,
        resize=not args.no_resize
    )

