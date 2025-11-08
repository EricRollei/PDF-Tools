"""
Eric Rectangle Detector Node

Description: Rectangle and bounding box detection node using AI vision models
Author: Eric Hiss (GitHub: EricRollei)
Contact: eric@historic.camera, eric@rollei.us
License: Dual License (Non-Commercial and Commercial Use)
Copyright (c) 2025 Eric Hiss. All rights reserved.

Dual License:
1. Non-Commercial Use: This software is licensed under the terms of the
   Creative Commons Attribution-NonCommercial 4.0 International License.
   To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
   
2. Commercial Use: For commercial use, a separate license is required.
   Please contact Eric Hiss at eric@historic.camera or eric@rollei.us for licensing options.

Dependencies:
This code depends on several third-party libraries, each with its own license.
See CREDITS.md for a comprehensive list of dependencies and their licenses.

Third-party code:
- See CREDITS.md for complete list of dependencies
"""

"""
Florence2 Rectangle Detector Node for ComfyUI
Uses the new clean Florence2RectangleDetector implementation

Updated with smart full-page filtering and natural crop sizes
"""

import torch
import numpy as np
import os
from PIL import Image, ImageDraw
import torchvision.transforms.functional as F
from typing import Tuple, List, Optional

# Import the new detector
from PDF_tools import Florence2RectangleDetector, BoundingBox

# ComfyUI folder paths
try:
    import folder_paths
    # Get the ComfyUI base path
    COMFYUI_BASE_PATH = os.path.dirname(folder_paths.models_dir)
except ImportError:
    COMFYUI_BASE_PATH = "."
    print("ComfyUI folder_paths not available, using current directory")


class Florence2RectangleDetectorNode:
    """
    ComfyUI Node for detecting rectangular objects using Florence2.
    
    Features:
    - Smart full-page detection filtering
    - Natural crop sizes (no white padding)
    - Individual masks for crops
    - Debug visualization
    """
    
    @classmethod
    def INPUT_TYPES(s):
        # Get available models from ComfyUI/models/LLM directory
        llm_models_dir = os.path.join(COMFYUI_BASE_PATH, "models", "LLM")
        available_models = []
        
        if os.path.exists(llm_models_dir):
            for item in os.listdir(llm_models_dir):
                if os.path.isdir(os.path.join(llm_models_dir, item)):
                    available_models.append(item)
        
        # Default models if none found
        if not available_models:
            available_models = [
                "CogFlorence-2.2-Large",
                "microsoft/Florence-2-large", 
                "microsoft/Florence-2-base"
            ]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "text_prompt": ("STRING", {
                    "default": "rectangular images in page",
                    "multiline": False,
                    "tooltip": "What to detect. Full-page detections are automatically filtered unless they're the only detection."
                }),
                "model_name": (available_models, {
                    "default": available_models[0] if available_models else "CogFlorence-2.2-Large"
                }),
                "min_box_area": ("INT", {
                    "default": 1000,
                    "min": 100,
                    "max": 50000,
                    "step": 100
                }),
            },
            "optional": {
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "return_cropped_images": ("BOOLEAN", {"default": False}),
                "crop_padding": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "tooltip": "Extra pixels around crops"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("annotated_image", "detection_mask", "detection_info", "cropped_images")
    OUTPUT_IS_LIST = (False, False, False, True)  # Only cropped_images is a list
    FUNCTION = "detect_rectangles"
    CATEGORY = "Florence2/Detection"
    
    def __init__(self):
        self.detector = None
        self.current_model = None
    

    def filter_nested_boxes(bounding_boxes, image_area):
        """
        Remove boxes that are inside a larger box, but only if:
        - The larger box is <= 75% of the page area
        - The smaller box is fully inside the larger box
        - The smaller box is < 50% of the larger box's area
        """
        keep = [True] * len(bounding_boxes)
        for i, big in enumerate(bounding_boxes):
            big_area = big.area
            if big_area > 0.75 * image_area:
                continue  # Don't apply logic for near-full-page boxes
            for j, small in enumerate(bounding_boxes):
                if i == j or not keep[j]:
                    continue
                # Check if small is inside big
                if (small.x1 >= big.x1 and small.y1 >= big.y1 and
                    small.x2 <= big.x2 and small.y2 <= big.y2):
                    if small.area < 0.5 * big_area:
                        keep[j] = False  # Exclude the smaller box
        return [box for k, box in zip(keep, bounding_boxes) if k]
        
    def detect_rectangles(self, 
                         image: torch.Tensor,
                         text_prompt: str,
                         model_name: str,
                         min_box_area: int,
                         keep_model_loaded: bool = True,
                         return_cropped_images: bool = False,
                         crop_padding: int = 0) -> Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
        """
        Detect rectangular objects in the input image.
        
        Args:
            image: Input image tensor from ComfyUI (B, H, W, C)
            text_prompt: Text description of what to detect
            model_name: Florence2 model to use
            min_box_area: Minimum area for bounding boxes
            keep_model_loaded: Whether to keep model in memory
            return_cropped_images: Whether to return cropped regions
            crop_padding: Extra pixels around crops
            
        Returns:
            Tuple of (annotated_image, mask, info_string, cropped_images)
        """
        
        # Initialize or update detector if needed
        if self.detector is None or self.current_model != model_name:
            print(f"Initializing Florence2RectangleDetector with model: {model_name}")
            
            self.detector = Florence2RectangleDetector(
                model_name=model_name,
                comfyui_base_path=COMFYUI_BASE_PATH,
                min_box_area=min_box_area
            )
            self.current_model = model_name
        else:
            # Update settings if different
            self.detector.update_settings(min_box_area=min_box_area)
        
        # Process each image in the batch
        batch_size = image.shape[0]
        annotated_images = []
        masks = []
        all_info = []
        all_cropped = []  # This will be a list of individual crop tensors
        
        for i in range(batch_size):
            # Convert ComfyUI image tensor to PIL
            img_tensor = image[i]  # (H, W, C)
            pil_image = F.to_pil_image(img_tensor.permute(2, 0, 1))  # Convert to (C, H, W) for torchvision
            
            print(f"Processing image {i+1}/{batch_size} - Size: {pil_image.size}")
            
            try:
                # Detect rectangles
                bounding_boxes, mask_image = self.detector.detect_rectangles(
                    image=pil_image,
                    text_input=text_prompt,
                    return_mask=True,
                    keep_model_loaded=keep_model_loaded
                )
                
                # Apply smart full-page filtering
                image_area = pil_image.width * pil_image.height
                filtered_boxes = []
                full_page_boxes = []
                
                for box in bounding_boxes:
                    # Consider a box "full page" if it covers >90% of the image area
                    coverage_ratio = box.area / image_area
                    if coverage_ratio > 0.90:
                        full_page_boxes.append(box)
                        print(f"  Detected full-page box: {box} (coverage: {coverage_ratio:.2%})")
                    else:
                        filtered_boxes.append(box)
                
                # After you build filtered_boxes and full_page_boxes:
                filtered_boxes = self.filter_nested_boxes(filtered_boxes, image_area)

                # Decide which boxes to use
                if filtered_boxes:
                    # Use non-full-page boxes 
                    boxes_to_show = filtered_boxes
                    print(f"  Using {len(filtered_boxes)} non-full-page detections")
                elif full_page_boxes:
                    # Only full-page detections found - use them
                    boxes_to_show = full_page_boxes
                    print(f"  Only full-page detections found - using {len(full_page_boxes)} boxes")
                else:
                    boxes_to_show = []
                
                # Create annotated image with smart visualization
                if boxes_to_show or full_page_boxes:
                    annotated_pil = pil_image.copy()
                    draw = ImageDraw.Draw(annotated_pil)
                    
                    # Draw boxes to show in green
                    for j, box in enumerate(boxes_to_show):
                        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline='green', width=3)
                        draw.text((box.x1, box.y1 - 20), f"{j+1}: {box.label}", fill='green')
                    
                    # Draw filtered full-page boxes in red (if any were filtered)
                    if filtered_boxes and full_page_boxes:
                        for j, box in enumerate(full_page_boxes):
                            draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline='red', width=2)
                            draw.text((box.x1, box.y1 - 40), f"FILTERED: {box.label}", fill='red')
                else:
                    annotated_pil = pil_image
                
                # Convert back to ComfyUI tensor format
                annotated_tensor = F.to_tensor(annotated_pil).permute(1, 2, 0)  # (H, W, C)
                annotated_images.append(annotated_tensor)
                
                # Create detection mask from filtered boxes only
                if boxes_to_show:
                    detection_mask = Image.new('L', pil_image.size, 0)  # Black background
                    mask_draw = ImageDraw.Draw(detection_mask)
                    for box in boxes_to_show:
                        mask_draw.rectangle([box.x1, box.y1, box.x2, box.y2], fill=255)  # White regions
                    
                    # Convert PIL mask to tensor properly
                    mask_tensor = F.to_tensor(detection_mask).squeeze(0)  # (H, W)
                    # Ensure mask values are 0-1 float
                    mask_tensor = mask_tensor.float()
                    print(f"  Created mask with {len(boxes_to_show)} regions, mask range: {mask_tensor.min():.3f}-{mask_tensor.max():.3f}")
                else:
                    # Create empty mask
                    mask_tensor = torch.zeros(img_tensor.shape[:2], dtype=torch.float32)
                    print(f"  Created empty mask: {mask_tensor.shape}")
                
                masks.append(mask_tensor)
                
                # Create detailed info string
                if boxes_to_show:
                    info_lines = [f"Detected {len(boxes_to_show)} valid rectangles:"]
                    for j, box in enumerate(boxes_to_show):
                        info_lines.append(f"  {j+1}: {box.label} - Area: {box.area} - Box: ({box.x1}, {box.y1}, {box.x2}, {box.y2})")
                    
                    # Add filtering info
                    if filtered_boxes and full_page_boxes:
                        info_lines.append(f"\nFiltered out {len(full_page_boxes)} full-page detection(s)")
                    elif full_page_boxes and not filtered_boxes:
                        info_lines.append(f"\nUsed full-page detection(s) - no other regions found")
                    
                    info_str = "\n".join(info_lines)
                else:
                    if full_page_boxes:
                        info_str = f"Only full-page detections found (filtered out). Try a more specific prompt."
                    else:
                        info_str = "No rectangles detected"
                
                all_info.append(info_str)
                
                # Handle cropped images if requested - Add to list (no stacking/padding)
                if return_cropped_images and boxes_to_show:
                    for box in boxes_to_show:
                        # Apply padding if specified
                        if crop_padding > 0:
                            padded_box = BoundingBox(
                                x1=max(0, box.x1 - crop_padding),
                                y1=max(0, box.y1 - crop_padding),
                                x2=min(pil_image.width, box.x2 + crop_padding),
                                y2=min(pil_image.height, box.y2 + crop_padding),
                                label=box.label
                            )
                            cropped_pil = pil_image.crop(padded_box.to_tuple())
                        else:
                            cropped_pil = pil_image.crop(box.to_tuple())
                        
                        # Convert to tensor and add batch dimension - KEEP ORIGINAL SIZE
                        cropped_tensor = F.to_tensor(cropped_pil).permute(1, 2, 0).unsqueeze(0)  # (1, H, W, C)
                        all_cropped.append(cropped_tensor)
                        
                        print(f"  Added crop {len(all_cropped)}: {cropped_pil.size}")
                
                print(f"  Found {len(boxes_to_show)} valid rectangles from {len(bounding_boxes)} total detections")
                
            except Exception as e:
                print(f"Error processing image {i+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Create fallback outputs
                annotated_images.append(img_tensor)
                masks.append(torch.zeros(img_tensor.shape[:2], dtype=torch.float32))
                all_info.append(f"Error: {str(e)}")
        
        # Stack results
        if annotated_images:
            output_images = torch.stack(annotated_images, dim=0)
        else:
            output_images = image.clone()
        
        if masks:
            output_masks = torch.stack(masks, dim=0)
        else:
            output_masks = torch.zeros((batch_size, image.shape[1], image.shape[2]), dtype=torch.float32)
        
        # Combine info strings
        combined_info = "\n\n".join(all_info)
        
        # Handle cropped images output - Return as list, ensuring at least one element
        if all_cropped:
            print(f"Returning {len(all_cropped)} individual crops as list")
            cropped_output = all_cropped
        else:
            # No crops - return list with one empty image (ComfyUI list outputs need at least one element)
            empty_crop = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            cropped_output = [empty_crop]
            print("No crops to output - returning list with empty image")
        
        return (output_images, output_masks, combined_info, cropped_output)


class Florence2ModelInfoNode:
    """
    Node to display information about available Florence2 models.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "model_name": ("STRING", {"default": "CogFlorence-2.2-Large"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_info",)
    FUNCTION = "get_model_info"
    CATEGORY = "Florence2/Utils"
    
    def get_model_info(self, model_name: str = "CogFlorence-2.2-Large") -> Tuple[str]:
        """Get information about Florence2 model configuration."""
        
        try:
            detector = Florence2RectangleDetector(
                model_name=model_name,
                comfyui_base_path=COMFYUI_BASE_PATH,
                min_box_area=1000
            )
            
            info = detector.get_model_info()
            
            info_lines = [
                f"Florence2 Model Information:",
                f"  Model Name: {info['model_name']}",
                f"  Model Path: {info['model_path']}",
                f"  Path Exists: {os.path.exists(info['model_path'])}",
                f"  Precision: {info['precision']}",
                f"  Attention: {info['attention']}",
                f"  Min Box Area: {info['min_box_area']}",
                f"  Model Loaded: {info['loaded']}",
                f"  Transformers Version: {info['transformers_version']}",
                f"  ComfyUI Available: {info['comfyui_available']}",
            ]
            
            # Check for required files
            model_path = info['model_path']
            if os.path.exists(model_path):
                files = os.listdir(model_path)
                info_lines.append(f"  Model Files: {', '.join(files)}")
            
            info_str = "\n".join(info_lines)
            
        except Exception as e:
            info_str = f"Error getting model info: {str(e)}"
        
        return (info_str,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Florence2RectangleDetector": Florence2RectangleDetectorNode,
    "Florence2ModelInfo": Florence2ModelInfoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Florence2RectangleDetector": "Florence2 Rectangle Detector",
    "Florence2ModelInfo": "Florence2 Model Info",
}