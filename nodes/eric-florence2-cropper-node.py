"""
Eric-Florence2-Cropper-Node

Description: Florence2 vision model integration for object detection, image analysis, and caption generation
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
- Uses Florence-2 models (MIT License) by Microsoft: https://huggingface.co/microsoft/Florence-2-large
- See CREDITS.md for complete list of all dependencies
"""

"""
Florence2 Image Cropping Node for ComfyUI
Focused on detecting and cropping rectangular regions from input images

Primary use case: Scan of a magazine/book page with multiple images -> 
Detect the images -> Crop them out -> Output individual cropped images

Key Features:
- Automatically filters out full-page detections (unless only detection)
- Returns crops at their natural size (no white padding)
- Handles variable-sized crops intelligently
- Optional debug visualization showing cropped vs filtered regions
- Individual masks for each cropped region
"""

import torch
import numpy as np
import os
from PIL import Image
import torchvision.transforms.functional as F
from typing import Tuple, List, Optional

# Import the detector from main package
from PDF_tools import Florence2RectangleDetector, BoundingBox

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_BASE_PATH = os.path.dirname(folder_paths.models_dir)
except ImportError:
    COMFYUI_BASE_PATH = "."
    print("ComfyUI folder_paths not available, using current directory")


class Florence2ImageCroppingNode:
    """
    Detects rectangular regions in an input image and crops them out.
    
    Perfect for: Scanned pages with multiple images -> Extract individual images
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
                "detection_prompt": ("STRING", {
                    "default": "rectangular images in page",
                    "multiline": False,
                    "tooltip": "What to detect and crop (e.g., 'images', 'photos', 'rectangular objects'). Full-page detections are automatically filtered unless they're the only detection."
                }),
                "min_crop_area": ("INT", {
                    "default": 1000,
                    "min": 100,
                    "max": 50000,
                    "step": 100,
                    "tooltip": "Minimum area for detected regions to be cropped"
                }),
            },
            "optional": {
                "model_name": (available_models, {
                    "default": available_models[0] if available_models else "CogFlorence-2.2-Large"
                }),
                "return_debug_image": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Return annotated image showing detected regions"
                }),
                "return_masks": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Return masks for detected regions"
                }),
                "crop_padding": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "tooltip": "Extra pixels to add around detected regions"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("cropped_images", "debug_image", "masks", "detection_info")
    OUTPUT_IS_LIST = (True, False, True, False)  # cropped_images and masks are lists
    FUNCTION = "crop_detected_regions"
    CATEGORY = "Florence2/Cropping"
    
    def __init__(self):
        self.detector = None
        self.current_model = None
    
    def crop_detected_regions(self, 
                            image: torch.Tensor,
                            detection_prompt: str,
                            min_crop_area: int,
                            model_name: str = "CogFlorence-2.2-Large",
                            return_debug_image: bool = False,
                            return_masks: bool = False,
                            crop_padding: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Crop detected rectangular regions from input image.
        
        Returns:
            - cropped_images: Individual detected regions as separate images
            - debug_image: Original image with bounding boxes drawn (if requested)
            - masks: Masks for detected regions (if requested)  
            - detection_info: Text description of what was found
        """
        
        # Initialize detector if needed
        if self.detector is None or self.current_model != model_name:
            print(f"Initializing Florence2 detector with model: {model_name}")
            try:
                self.detector = Florence2RectangleDetector(
                    model_name=model_name,
                    comfyui_base_path=COMFYUI_BASE_PATH,
                    min_box_area=min_crop_area
                )
                self.current_model = model_name
            except Exception as e:
                error_msg = f"Failed to initialize detector: {str(e)}"
                print(error_msg)
                return self._create_fallback_outputs(image, error_msg)
        else:
            # Update settings
            self.detector.update_settings(min_box_area=min_crop_area)
        
        try:
            # Process first image (assuming single image input for cropping)
            input_tensor = image[0]  # (H, W, C)
            pil_image = F.to_pil_image(input_tensor.permute(2, 0, 1))
            
            print(f"Processing image - Size: {pil_image.size}, Prompt: '{detection_prompt}'")
            
            # Detect rectangular regions
            bounding_boxes, mask_image = self.detector.detect_rectangles(
                image=pil_image,
                text_input=detection_prompt,
                return_mask=return_masks,
                keep_model_loaded=True
            )
            
            print(f"Detected {len(bounding_boxes)} regions")
            
            # Filter out full-page detections (unless it's the only detection)
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
            
            # Decide which boxes to crop
            if filtered_boxes:
                # Use non-full-page boxes
                boxes_to_crop = filtered_boxes
                print(f"  Using {len(filtered_boxes)} non-full-page detections")
            elif full_page_boxes:
                # Only full-page detections found - use them (might be intentional)
                boxes_to_crop = full_page_boxes
                print(f"  Only full-page detections found - using {len(full_page_boxes)} boxes")
            else:
                boxes_to_crop = []

            # Create info string about what was actually cropped
            if boxes_to_crop:
                info_lines = [f"Successfully cropped {len(boxes_to_crop)} regions:"]
                for i, box in enumerate(boxes_to_crop):
                    info_lines.append(f"  Region {i+1}: {box.label} - Size: {box.width}x{box.height} (Area: {box.area})")
                
                # Add info about filtering
                if filtered_boxes and full_page_boxes:
                    info_lines.append(f"\nFiltered out {len(full_page_boxes)} full-page detection(s)")
                elif full_page_boxes and not filtered_boxes:
                    info_lines.append(f"\nUsed full-page detection(s) - no other regions found")
                    
                info_string = "\n".join(info_lines)
            else:
                if full_page_boxes:
                    info_string = f"Only full-page detections found (filtered out). Try a more specific prompt."
                else:
                    info_string = f"No regions detected with prompt: '{detection_prompt}'"
            
            
            # Crop the selected regions - build lists (no stacking)
            cropped_images = []
            crop_masks = []
            
            if boxes_to_crop:
                for i, box in enumerate(boxes_to_crop):
                    # Apply padding if specified
                    if crop_padding > 0:
                        padded_box = BoundingBox(
                            x1=max(0, box.x1 - crop_padding),
                            y1=max(0, box.y1 - crop_padding),
                            x2=min(pil_image.width, box.x2 + crop_padding),
                            y2=min(pil_image.height, box.y2 + crop_padding),
                            label=box.label
                        )
                        cropped = pil_image.crop(padded_box.to_tuple())
                        crop_box = padded_box
                    else:
                        cropped = pil_image.crop(box.to_tuple())
                        crop_box = box
                    
                    # Convert to tensor with batch dimension - KEEP ORIGINAL CROP SIZE
                    # Use exact same approach as working Florence2RectangleDetector
                    cropped_tensor = F.to_tensor(cropped).permute(1, 2, 0).unsqueeze(0)  # (1, H, W, C)
                    cropped_images.append(cropped_tensor)
                    
                    # Create individual mask for this crop - simplified approach
                    crop_mask = Image.new('L', (cropped.width, cropped.height), 255)  # White mask
                    crop_mask_tensor = F.to_tensor(crop_mask).squeeze(0).unsqueeze(0)  # (1, H, W)
                    crop_masks.append(crop_mask_tensor)
                    
                    print(f"  Cropped region {i+1}: {cropped.size} from box {crop_box}")
            
            # Handle outputs - Return lists directly (no stacking/padding)
            if cropped_images:
                print(f"Returning {len(cropped_images)} individual crops as list")
                cropped_output = cropped_images  # Return list directly
            else:
                # No valid crops - return empty list
                cropped_output = []
                print("No valid regions to crop - returning empty list")
            
            # Create debug image with bounding boxes
            if return_debug_image:
                if boxes_to_crop or full_page_boxes:
                    # Create custom visualization showing cropped vs filtered boxes
                    debug_pil = pil_image.copy()
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(debug_pil)
                    
                    # Draw cropped boxes in green
                    for i, box in enumerate(boxes_to_crop):
                        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline='green', width=3)
                        draw.text((box.x1, box.y1 - 20), f"CROP {i+1}: {box.label}", fill='green')
                    
                    # Draw filtered full-page boxes in red (if any were filtered)
                    if filtered_boxes and full_page_boxes:
                        for i, box in enumerate(full_page_boxes):
                            draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline='red', width=2)
                            draw.text((box.x1, box.y1 - 40), f"FILTERED: {box.label}", fill='red')
                    
                    debug_tensor = F.to_tensor(debug_pil).permute(1, 2, 0).unsqueeze(0)
                else:
                    debug_tensor = input_tensor.unsqueeze(0)
            else:
                # Return original image
                debug_tensor = input_tensor.unsqueeze(0)
            
            # Handle masks - Return list directly
            if return_masks and crop_masks:
                masks_output = crop_masks  # Return list directly
            else:
                # Return empty list
                masks_output = []
            
            print(f"Cropping complete: {len(boxes_to_crop)} regions cropped from {len(bounding_boxes)} total detections")
            
            return (cropped_output, debug_tensor, masks_output, info_string)
            
        except Exception as e:
            error_msg = f"Error during detection/cropping: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return self._create_fallback_outputs(image, error_msg)
    
    def _create_fallback_outputs(self, image: torch.Tensor, error_msg: str):
        """Create safe fallback outputs when detection fails - same as working node."""
        input_tensor = image[0]
        empty_crop = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
        return (
            [empty_crop],  # List with one empty crop
            input_tensor.unsqueeze(0),  # Original image as debug
            [empty_mask],  # List with one empty mask
            error_msg  # Error info
        )


class Florence2RegionVisualizerNode:
    """
    Utility node for visualizing detected regions without cropping.
    Useful for debugging detection prompts and settings.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_prompt": ("STRING", {
                    "default": "rectangular images in page",
                    "multiline": False
                }),
                "min_area": ("INT", {
                    "default": 1000,
                    "min": 100,
                    "max": 50000,
                    "step": 100
                }),
            },
            "optional": {
                "model_name": ("STRING", {"default": "CogFlorence-2.2-Large"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("annotated_image", "detection_mask", "detection_info")
    FUNCTION = "visualize_detections"
    CATEGORY = "Florence2/Debug"
    
    def __init__(self):
        self.detector = None
        self.current_model = None
    
    def visualize_detections(self, 
                           image: torch.Tensor,
                           detection_prompt: str,
                           min_area: int,
                           model_name: str = "CogFlorence-2.2-Large") -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Visualize detected regions with bounding boxes for debugging."""
        
        # Initialize detector
        if self.detector is None or self.current_model != model_name:
            self.detector = Florence2RectangleDetector(
                model_name=model_name,
                comfyui_base_path=COMFYUI_BASE_PATH,
                min_box_area=min_area
            )
            self.current_model = model_name
        
        try:
            # Process image
            input_tensor = image[0]  # (H, W, C)
            pil_image = F.to_pil_image(input_tensor.permute(2, 0, 1))
            
            # Detect regions
            boxes, mask = self.detector.detect_rectangles(
                image=pil_image,
                text_input=detection_prompt,
                return_mask=True,
                keep_model_loaded=True
            )
            
            # Create annotated image
            if boxes:
                annotated_pil = self.detector.visualize_detections(pil_image, boxes, show_labels=True)
                annotated_tensor = F.to_tensor(annotated_pil).permute(1, 2, 0).unsqueeze(0)
                
                # Create info
                info_lines = [f"Detected {len(boxes)} regions:"]
                for i, box in enumerate(boxes):
                    info_lines.append(f"  {i+1}: {box.label} - {box.width}x{box.height} (Area: {box.area})")
                info_string = "\n".join(info_lines)
            else:
                annotated_tensor = input_tensor.unsqueeze(0)
                info_string = f"No regions detected with prompt: '{detection_prompt}'"
            
            # Handle mask
            if mask:
                mask_tensor = F.to_tensor(mask.convert('L')).squeeze(0).unsqueeze(0)
            else:
                mask_tensor = torch.zeros((1, input_tensor.shape[0], input_tensor.shape[1]), dtype=torch.float32)
            
            return (annotated_tensor, mask_tensor, info_string)
            
        except Exception as e:
            error_msg = f"Visualization error: {str(e)}"
            return (input_tensor.unsqueeze(0), 
                   torch.zeros((1, input_tensor.shape[0], input_tensor.shape[1]), dtype=torch.float32),
                   error_msg)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Florence2ImageCropper": Florence2ImageCroppingNode,
    "Florence2RegionVisualizer": Florence2RegionVisualizerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Florence2ImageCropper": "Florence2 Image Cropper",
    "Florence2RegionVisualizer": "Florence2 Region Visualizer (Debug)",
}