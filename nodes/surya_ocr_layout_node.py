"""
Surya Ocr Layout Node

Description: Surya OCR integration for multilingual text recognition and layout analysis
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
- Uses Surya OCR (GNU GPL v3) by Vik Paruchuri: https://github.com/VikParuchuri/surya
- See CREDITS.md for complete list of all dependencies
"""

"""
Surya OCR & Layout Detection Node for ComfyUI
Provides layout detection, text detection, and OCR functionality
Built for Surya OCR v0.17.0+
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Lazy imports for Surya components
SURYA_AVAILABLE = False
SURYA_IMPORT_ERROR = None

try:
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.layout import LayoutPredictor
    from surya.recognition import RecognitionPredictor
    SURYA_AVAILABLE = True
except Exception as e:
    SURYA_IMPORT_ERROR = e


class SuryaOCRLayoutNode:
    """
    ComfyUI node for Surya OCR with layout detection capabilities.
    
    Outputs:
    - Annotated image with bounding boxes
    - Extracted text (OCR)
    - Layout data (JSON) - images, text blocks, headers, etc.
    - Text bounding boxes (for cropping)
    - Image bounding boxes (for cropping)
    - Full JSON with all data
    - Status message
    """
    
    def __init__(self):
        self.device = self._get_device()
        self.foundation_predictor = None
        self.foundation_predictor_cpu = None  # Separate CPU predictor for layout
        self.detection_predictor = None
        self.recognition_predictor = None
        self.layout_predictor = None
        self.layout_predictor_cpu = None  # Separate CPU predictor for layout
        
    @staticmethod
    def _get_device() -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            # Use first available GPU
            device = torch.device("cuda:0")
            print(f"✅ Surya OCR: Using GPU - {torch.cuda.get_device_name(0)}")
            return device
        print("⚠️ Surya OCR: CUDA not available, using CPU (will be slow)")
        return torch.device("cpu")
    
    def _init_predictors(self, force_reinit: bool = False, cpu_layout: bool = False):
        """Initialize Surya predictors lazily."""
        if not SURYA_AVAILABLE:
            error_msg = "surya-ocr is not installed or failed to import"
            if SURYA_IMPORT_ERROR:
                error_msg += f": {SURYA_IMPORT_ERROR}"
            raise ImportError(error_msg)
        
        # Skip if already initialized
        if not force_reinit and self.foundation_predictor is not None:
            # Check if we need CPU layout predictor
            if cpu_layout and self.layout_predictor_cpu is None:
                self._init_cpu_layout_predictor()
            return
        
        device_str = "cuda" if self.device.type == "cuda" else "cpu"
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        print(f"🔧 Initializing Surya predictors on {device_str}...")
        
        # Initialize foundation predictor (required for recognition and layout)
        try:
            self.foundation_predictor = FoundationPredictor(
                device=device_str,
                dtype=dtype
            )
            print("  ✓ Foundation predictor initialized")
        except Exception as e:
            print(f"  ✗ Foundation predictor failed: {e}")
            raise
        
        # Initialize detection predictor
        try:
            self.detection_predictor = DetectionPredictor(
                device=device_str,
                dtype=dtype
            )
            print("  ✓ Detection predictor initialized")
        except Exception as e:
            print(f"  ✗ Detection predictor failed: {e}")
            raise
        
        # Initialize recognition predictor
        try:
            self.recognition_predictor = RecognitionPredictor(
                foundation_predictor=self.foundation_predictor
            )
            print("  ✓ Recognition predictor initialized")
        except Exception as e:
            print(f"  ✗ Recognition predictor failed: {e}")
            raise
        
        # Initialize layout predictor (GPU or CPU)
        if cpu_layout:
            self._init_cpu_layout_predictor()
        else:
            try:
                self.layout_predictor = LayoutPredictor(
                    foundation_predictor=self.foundation_predictor
                )
                print("  ✓ Layout predictor initialized (GPU)")
            except Exception as e:
                print(f"  ⚠ Layout predictor failed on GPU: {e}")
                print("  → Falling back to CPU for layout detection...")
                self._init_cpu_layout_predictor()
        
        print("✅ All Surya predictors initialized successfully")
    
    def _init_cpu_layout_predictor(self):
        """Initialize a separate CPU-based layout predictor to avoid Flash Attention issues."""
        print("  🔧 Initializing CPU-based layout predictor...")
        try:
            # Create CPU foundation predictor if not exists
            if self.foundation_predictor_cpu is None:
                self.foundation_predictor_cpu = FoundationPredictor(
                    device="cpu",
                    dtype=torch.float32
                )
                print("    ✓ CPU Foundation predictor initialized")
            
            # Create CPU layout predictor
            self.layout_predictor_cpu = LayoutPredictor(
                foundation_predictor=self.foundation_predictor_cpu
            )
            print("  ✓ Layout predictor initialized (CPU - avoids Flash Attention issues)")
        except Exception as e:
            print(f"  ✗ CPU Layout predictor also failed: {e}")
            raise
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["layout_only", "ocr_only", "layout_and_ocr"], {
                    "default": "layout_and_ocr"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
            },
            "optional": {
                "show_labels": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show label text on annotated image"
                }),
                "batch_size": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 32,
                    "tooltip": "Batch size for processing (lower if OOM errors)"
                }),
                "force_cpu_layout": ("BOOLEAN", {
                    "default": True,  # Changed to True - required for newer GPUs
                    "tooltip": "Force CPU for layout detection (required for RTX 40XX/Blackwell GPUs due to Flash Attention incompatibility)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "annotated_image",
        "extracted_text", 
        "layout_json",
        "text_bboxes_json",
        "image_bboxes_json",
        "full_data_json",
        "status"
    )
    FUNCTION = "process"
    CATEGORY = "text/ocr"
    
    def process(
        self,
        image: torch.Tensor,
        mode: str = "layout_and_ocr",
        confidence_threshold: float = 0.5,
        show_labels: bool = True,
        batch_size: int = 2,
        force_cpu_layout: bool = False
    ) -> Tuple[torch.Tensor, str, str, str, str, str, str]:
        """
        Main processing function.
        
        Args:
            image: ComfyUI image tensor [B, H, W, C] in range 0-1
            mode: Processing mode
            confidence_threshold: Minimum confidence for detections
            show_labels: Whether to show labels on annotated image
            batch_size: Batch size for processing
            force_cpu_layout: Force CPU for layout (avoids Flash Attention issues)
            
        Returns:
            Tuple of (annotated_image, text, layout_json, text_bboxes, image_bboxes, full_json, status)
        """
        try:
            # Initialize predictors
            self._init_predictors(cpu_layout=force_cpu_layout)
            
            # Convert to PIL images
            pil_images = self._tensor_to_pil(image)
            
            # Process based on mode
            results = {}
            
            if mode in ["layout_only", "layout_and_ocr"]:
                print(f"🔍 Running layout detection on {len(pil_images)} image(s)...")
                
                # Use CPU layout predictor if forced or if GPU predictor unavailable
                layout_pred = self.layout_predictor_cpu if (force_cpu_layout or self.layout_predictor is None) else self.layout_predictor
                
                if layout_pred is None:
                    raise RuntimeError("Layout predictor not available. Try enabling force_cpu_layout option.")
                
                try:
                    layout_results = layout_pred(pil_images, batch_size=batch_size)
                    results['layout'] = layout_results
                    print(f"  ✓ Found {sum(len(r.bboxes) for r in layout_results)} layout boxes")
                except RuntimeError as e:
                    # If GPU layout fails with Flash Attention error, auto-switch to CPU
                    if "CUDA error" in str(e) or "PTX" in str(e) or "flash_attn" in str(e).lower():
                        print(f"  ⚠️ GPU layout failed (Flash Attention issue), auto-switching to CPU...")
                        if self.layout_predictor_cpu is None:
                            self._init_cpu_layout_predictor()
                        layout_results = self.layout_predictor_cpu(pil_images, batch_size=batch_size)
                        results['layout'] = layout_results
                        print(f"  ✓ Found {sum(len(r.bboxes) for r in layout_results)} layout boxes (CPU)")
                    else:
                        raise
            
            if mode in ["ocr_only", "layout_and_ocr"]:
                print(f"📝 Running OCR on {len(pil_images)} image(s)...")
                ocr_results = self.recognition_predictor(
                    pil_images,
                    det_predictor=self.detection_predictor,
                    detection_batch_size=batch_size,
                    recognition_batch_size=batch_size
                )
                results['ocr'] = ocr_results
                print(f"  ✓ Extracted {sum(len(r.text_lines) for r in ocr_results)} text lines")
            
            # Filter by confidence and extract outputs
            filtered_results = self._filter_by_confidence(results, confidence_threshold)
            
            # Generate outputs
            annotated_image = self._create_annotated_image(
                pil_images[0], 
                filtered_results, 
                show_labels,
                image.device
            )
            
            extracted_text = self._extract_text(filtered_results)
            layout_json = self._extract_layout_json(filtered_results)
            text_bboxes_json = self._extract_text_bboxes(filtered_results)
            image_bboxes_json = self._extract_image_bboxes(filtered_results)
            full_json = self._create_full_json(filtered_results)
            
            status = self._create_status_message(filtered_results, mode)
            
            # Clear GPU cache
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            return (
                annotated_image,
                extracted_text,
                layout_json,
                text_bboxes_json,
                image_bboxes_json,
                full_json,
                status
            )
            
        except Exception as e:
            error_msg = f"❌ Surya OCR Error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            # Clear GPU cache on error
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            # Return error state
            empty_image = torch.zeros_like(image)
            error_json = json.dumps({"error": str(e)}, indent=2)
            
            return (
                empty_image,
                f"ERROR: {str(e)}",
                error_json,
                error_json,
                error_json,
                error_json,
                error_msg
            )
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> List[Image.Image]:
        """Convert ComfyUI tensor to PIL images."""
        # Move to CPU for PIL conversion
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        
        # Handle batch dimension
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        pil_images = []
        for i in range(tensor.shape[0]):
            # Convert to numpy and scale to 0-255
            img_array = (tensor[i].numpy() * 255).astype(np.uint8)
            
            # Convert to RGB
            if img_array.shape[2] == 3:
                pil_image = Image.fromarray(img_array, 'RGB')
            elif img_array.shape[2] == 4:
                pil_image = Image.fromarray(img_array, 'RGBA').convert('RGB')
            else:
                raise ValueError(f"Unsupported channel count: {img_array.shape[2]}")
            
            pil_images.append(pil_image)
        
        return pil_images
    
    def _filter_by_confidence(self, results: Dict, threshold: float) -> Dict:
        """Filter all results by confidence threshold."""
        filtered = {}
        
        # Filter layout results
        if 'layout' in results:
            filtered_layout = []
            for layout_result in results['layout']:
                filtered_bboxes = [
                    bbox for bbox in layout_result.bboxes
                    if bbox.confidence >= threshold
                ]
                # Create a new result with filtered bboxes
                filtered_layout.append({
                    'bboxes': filtered_bboxes,
                    'image_bbox': layout_result.image_bbox,
                    'sliced': layout_result.sliced
                })
            filtered['layout'] = filtered_layout
        
        # Filter OCR results
        if 'ocr' in results:
            filtered_ocr = []
            for ocr_result in results['ocr']:
                # OCR text_lines don't have confidence, so we keep all
                filtered_ocr.append({
                    'text_lines': ocr_result.text_lines,
                    'image_bbox': ocr_result.image_bbox
                })
            filtered['ocr'] = filtered_ocr
        
        return filtered
    
    def _create_annotated_image(
        self, 
        pil_image: Image.Image, 
        results: Dict,
        show_labels: bool,
        target_device: torch.device
    ) -> torch.Tensor:
        """Create annotated image with bounding boxes."""
        # Create a copy for drawing
        annotated = pil_image.copy()
        draw = ImageDraw.Draw(annotated)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            small_font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        # Color map for different layout types
        color_map = {
            'Image': '#FF0000',      # Red
            'Table': '#00FF00',      # Green
            'Text': '#0000FF',       # Blue
            'Title': '#FF00FF',      # Magenta
            'Section-header': '#FFFF00',  # Yellow
            'List': '#00FFFF',       # Cyan
            'Caption': '#FFA500',    # Orange
            'Footnote': '#800080',   # Purple
            'Formula': '#FFC0CB',    # Pink
            'Page-header': '#A52A2A',  # Brown
            'Page-footer': '#808080',  # Gray
        }
        
        # Draw layout boxes
        if 'layout' in results:
            for page_result in results['layout']:
                for bbox in page_result['bboxes']:
                    label = bbox.label
                    polygon = bbox.polygon
                    confidence = bbox.confidence
                    
                    # Get color for this label type
                    color = color_map.get(label, '#FFFFFF')
                    
                    # Draw polygon
                    if polygon and len(polygon) >= 3:
                        # Flatten polygon coordinates for PIL
                        flat_polygon = [coord for point in polygon for coord in point]
                        draw.polygon(flat_polygon, outline=color, width=3)
                        
                        # Draw label if enabled
                        if show_labels and polygon:
                            text = f"{label} ({confidence:.2f})"
                            # Position text at top-left of polygon
                            text_pos = (int(polygon[0][0]), max(0, int(polygon[0][1]) - 20))
                            
                            # Draw text background
                            bbox_coords = draw.textbbox(text_pos, text, font=small_font)
                            draw.rectangle(bbox_coords, fill=color)
                            draw.text(text_pos, text, fill='black', font=small_font)
        
        # Draw OCR text boxes (in a different color to distinguish)
        if 'ocr' in results:
            for page_result in results['ocr']:
                for text_line in page_result['text_lines']:
                    # Get polygon or bbox
                    polygon = getattr(text_line, 'polygon', None)
                    bbox = getattr(text_line, 'bbox', None)
                    
                    if polygon and len(polygon) >= 3:
                        # Flatten polygon coordinates for PIL
                        flat_polygon = [coord for point in polygon for coord in point]
                        draw.polygon(flat_polygon, outline='#00FF00', width=2)
                    elif bbox and len(bbox) >= 4:
                        draw.rectangle(bbox, outline='#00FF00', width=2)
        
        # Convert back to tensor
        img_array = np.array(annotated).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array).unsqueeze(0)
        
        # Move to target device
        if target_device.type != "cpu":
            tensor = tensor.to(target_device)
        
        return tensor
    
    def _extract_text(self, results: Dict) -> str:
        """Extract all text from OCR results."""
        if 'ocr' not in results:
            return ""
        
        text_lines = []
        for page_result in results['ocr']:
            for text_line in page_result['text_lines']:
                text = getattr(text_line, 'text', '')
                if text:
                    text_lines.append(text)
        
        return '\n'.join(text_lines)
    
    def _extract_layout_json(self, results: Dict) -> str:
        """Extract layout data as JSON."""
        if 'layout' not in results:
            return json.dumps({"layout": []}, indent=2)
        
        layout_data = []
        for page_idx, page_result in enumerate(results['layout']):
            page_data = {
                'page': page_idx,
                'boxes': []
            }
            
            for bbox in page_result['bboxes']:
                box_data = {
                    'label': bbox.label,
                    'confidence': float(bbox.confidence),
                    'polygon': bbox.polygon,
                    'position': bbox.position
                }
                page_data['boxes'].append(box_data)
            
            layout_data.append(page_data)
        
        return json.dumps({'layout': layout_data}, indent=2)
    
    def _extract_text_bboxes(self, results: Dict) -> str:
        """Extract text bounding boxes for cropping."""
        text_boxes = []
        
        # Get text boxes from layout if available
        if 'layout' in results:
            for page_idx, page_result in enumerate(results['layout']):
                for bbox in page_result['bboxes']:
                    if bbox.label in ['Text', 'Title', 'Section-header', 'Caption', 'List']:
                        text_boxes.append({
                            'page': page_idx,
                            'type': bbox.label,
                            'polygon': bbox.polygon,
                            'confidence': float(bbox.confidence)
                        })
        
        # Also get from OCR results
        if 'ocr' in results:
            for page_idx, page_result in enumerate(results['ocr']):
                for text_line in page_result['text_lines']:
                    polygon = getattr(text_line, 'polygon', None)
                    bbox = getattr(text_line, 'bbox', None)
                    
                    text_boxes.append({
                        'page': page_idx,
                        'type': 'OCR_line',
                        'polygon': polygon if polygon else bbox,
                        'text': getattr(text_line, 'text', '')
                    })
        
        return json.dumps({'text_boxes': text_boxes}, indent=2)
    
    def _extract_image_bboxes(self, results: Dict) -> str:
        """Extract image bounding boxes for cropping."""
        if 'layout' not in results:
            return json.dumps({"image_boxes": []}, indent=2)
        
        image_boxes = []
        for page_idx, page_result in enumerate(results['layout']):
            for bbox in page_result['bboxes']:
                if bbox.label == 'Image':
                    image_boxes.append({
                        'page': page_idx,
                        'polygon': bbox.polygon,
                        'confidence': float(bbox.confidence),
                        'position': bbox.position
                    })
        
        return json.dumps({'image_boxes': image_boxes}, indent=2)
    
    def _create_full_json(self, results: Dict) -> str:
        """Create comprehensive JSON with all data."""
        full_data = {}
        
        # Add layout data
        if 'layout' in results:
            layout_list = []
            for page_idx, page_result in enumerate(results['layout']):
                page_data = {
                    'page': page_idx,
                    'image_bbox': page_result.get('image_bbox'),
                    'sliced': page_result.get('sliced', False),
                    'boxes': []
                }
                
                for bbox in page_result['bboxes']:
                    box_data = {
                        'label': bbox.label,
                        'confidence': float(bbox.confidence),
                        'polygon': bbox.polygon,
                        'position': bbox.position,
                        'top_k': getattr(bbox, 'top_k', None)
                    }
                    page_data['boxes'].append(box_data)
                
                layout_list.append(page_data)
            
            full_data['layout'] = layout_list
        
        # Add OCR data
        if 'ocr' in results:
            ocr_list = []
            for page_idx, page_result in enumerate(results['ocr']):
                page_data = {
                    'page': page_idx,
                    'image_bbox': page_result.get('image_bbox'),
                    'text_lines': []
                }
                
                for text_line in page_result['text_lines']:
                    line_data = {
                        'text': getattr(text_line, 'text', ''),
                        'polygon': getattr(text_line, 'polygon', None),
                        'bbox': getattr(text_line, 'bbox', None),
                    }
                    page_data['text_lines'].append(line_data)
                
                ocr_list.append(page_data)
            
            full_data['ocr'] = ocr_list
        
        return json.dumps(full_data, indent=2)
    
    def _create_status_message(self, results: Dict, mode: str) -> str:
        """Create status message with processing summary."""
        status_lines = [f"✅ Surya OCR completed in '{mode}' mode"]
        
        if 'layout' in results:
            total_boxes = sum(len(page['bboxes']) for page in results['layout'])
            status_lines.append(f"  📐 Layout: {total_boxes} boxes detected")
            
            # Count by type
            type_counts = {}
            for page in results['layout']:
                for bbox in page['bboxes']:
                    label = bbox.label
                    type_counts[label] = type_counts.get(label, 0) + 1
            
            for label, count in sorted(type_counts.items()):
                status_lines.append(f"    • {label}: {count}")
        
        if 'ocr' in results:
            total_lines = sum(len(page['text_lines']) for page in results['ocr'])
            total_chars = sum(
                len(getattr(line, 'text', ''))
                for page in results['ocr']
                for line in page['text_lines']
            )
            status_lines.append(f"  📝 OCR: {total_lines} text lines, {total_chars} characters")
        
        return '\n'.join(status_lines)


# Register node with ComfyUI
NODE_CLASS_MAPPINGS = {
    "SuryaOCRLayout": SuryaOCRLayoutNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SuryaOCRLayout": "Surya OCR & Layout Detection"
}
