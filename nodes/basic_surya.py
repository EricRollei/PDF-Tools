"""
Basic Surya

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
ComfyUI Surya OCR No e Template
Optimized for RTX 4090 24GB VRAM
"""

import os
import torch
import numpy as np
from PIL import Image
import folder_paths
from typing import Tuple, List, Dict, Any, Optional

# Set optimal environment variables for RTX 4090 before importing surya
def setup_surya_environment():
    """Configure optimal settings for RTX 4090 24GB VRAM"""
    env_vars = {
        'RECOGNITION_BATCH_SIZE': '768',    # ~24GB VRAM usage
        'DETECTOR_BATCH_SIZE': '54',        # ~24GB VRAM usage  
        'LAYOUT_BATCH_SIZE': '48',          # ~24GB VRAM usage
        'TABLE_REC_BATCH_SIZE': '96',       # ~24GB VRAM usage
        # Prefer an explicit CUDA index string to avoid torch.set_device errors
        'TORCH_DEVICE': 'cuda:0',           # Force CUDA device 0 by default
        'COMPILE_ALL': 'true',              # Enable model compilation for speed
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:  # Don't override existing settings
            os.environ[key] = value

# Setup environment before importing surya
setup_surya_environment()

# Now import surya modules
try:
    from surya.foundation import FoundationPredictor  # NEW: Required for Surya 0.17+
    from surya.detection import DetectionPredictor
    from surya.recognition import RecognitionPredictor
    from surya.layout import LayoutPredictor
    SURYA_AVAILABLE = True
except ImportError as e:
    print(f"Surya import error: {e}")
    SURYA_AVAILABLE = False

class SuryaOCRNode:
    """
    ComfyUI Node for Surya OCR with optimized settings
    """
    
    def __init__(self):
        self.foundation_predictor = None  # NEW: Required for Surya 0.17+
        self.detection_predictor = None
        self.recognition_predictor = None
        self.layout_predictor = None
        self._initialize_predictors()
    
    def _initialize_predictors(self):
        """Lazy initialization of predictors"""
        if not SURYA_AVAILABLE:
            raise ImportError("Surya OCR is not available. Please install surya-ocr.")
        
        try:
            # Initialize predictors with GPU optimization
            print("Initializing Surya OCR predictors...")
            
            # NEW API (Surya 0.17+): Create FoundationPredictor first
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            print(f"🔧 Initializing FoundationPredictor (device: {device}, dtype: {dtype})")
            self.foundation_predictor = FoundationPredictor(
                checkpoint=None,  # Use default checkpoint
                device=device,
                dtype=dtype,
                attention_implementation="sdpa"  # Use scaled dot product attention
            )
            
            # Now create other predictors with foundation predictor
            self.detection_predictor = DetectionPredictor()
            self.recognition_predictor = RecognitionPredictor(foundation_predictor=self.foundation_predictor)
            
            # Optional: Initialize layout predictor for advanced features
            # self.layout_predictor = LayoutPredictor()
            
            print("✅ Surya OCR predictors initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Surya predictors: {e}")
            raise

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "task_mode": (["ocr_with_boxes", "ocr_without_boxes", "detection_only"], {
                    "default": "ocr_with_boxes"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
            "optional": {
                "enable_layout": ("BOOLEAN", {"default": False}),
                "batch_size_override": ("INT", {
                    "default": 0,  # 0 = use environment defaults
                    "min": 0,
                    "max": 1024,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "LIST")
    RETURN_NAMES = ("text_output", "json_output", "annotated_image", "detection_data")
    FUNCTION = "process_ocr"
    CATEGORY = "text/ocr"
    
    def process_ocr(self, image: torch.Tensor, task_mode: str = "ocr_with_boxes", 
                   confidence_threshold: float = 0.5, enable_layout: bool = False,
                   batch_size_override: int = 0) -> Tuple[str, str, torch.Tensor, List[Dict]]:
        """
        Main OCR processing function
        
        Args:
            image: Input image tensor from ComfyUI
            task_mode: Type of OCR task to perform
            confidence_threshold: Minimum confidence for text detection
            enable_layout: Whether to perform layout analysis
            batch_size_override: Override default batch sizes (0 = use defaults)
            
        Returns:
            Tuple of (text_output, json_output, annotated_image, detection_data)
        """
        
        try:
            # Override batch sizes if specified
            if batch_size_override > 0:
                original_recognition_batch = os.environ.get('RECOGNITION_BATCH_SIZE')
                original_detector_batch = os.environ.get('DETECTOR_BATCH_SIZE')
                
                os.environ['RECOGNITION_BATCH_SIZE'] = str(batch_size_override)
                os.environ['DETECTOR_BATCH_SIZE'] = str(min(batch_size_override // 10, 64))
            
            # Convert ComfyUI tensor to PIL Image
            pil_images = self._tensor_to_pil_batch(image)
            
            # Process based on task mode
            if task_mode == "detection_only":
                results = self._process_detection_only(pil_images, confidence_threshold)
            elif task_mode in ["ocr_with_boxes", "ocr_without_boxes"]:
                results = self._process_full_ocr(pil_images, task_mode, confidence_threshold)
            else:
                raise ValueError(f"Unknown task mode: {task_mode}")
            
            # Optional layout analysis
            if enable_layout and self.layout_predictor:
                layout_results = self._process_layout(pil_images)
                results['layout'] = layout_results
            
            # Generate outputs
            text_output = self._extract_text(results)
            json_output = self._format_json_output(results)
            annotated_image = self._create_annotated_image(pil_images[0], results)
            detection_data = self._format_detection_data(results)
            
            return (text_output, json_output, annotated_image, detection_data)
            
        except Exception as e:
            error_msg = f"OCR processing failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Return error state
            empty_image = torch.zeros_like(image)
            return (error_msg, f'{{"error": "{error_msg}"}}', empty_image, [])
        
        finally:
            # Restore original batch sizes
            if batch_size_override > 0:
                if original_recognition_batch:
                    os.environ['RECOGNITION_BATCH_SIZE'] = original_recognition_batch
                if original_detector_batch:
                    os.environ['DETECTOR_BATCH_SIZE'] = original_detector_batch
    
    def _tensor_to_pil_batch(self, tensor: torch.Tensor) -> List[Image.Image]:
        """Convert ComfyUI image tensor to PIL Images"""
        # ComfyUI format: [batch, height, width, channels] in 0-1 range
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        pil_images = []
        for i in range(tensor.shape[0]):
            # Convert to numpy and scale to 0-255
            img_array = (tensor[i].cpu().numpy() * 255).astype(np.uint8)
            
            # Ensure proper RGB format
            if img_array.shape[2] == 3:  # RGB
                pil_image = Image.fromarray(img_array, 'RGB')
            elif img_array.shape[2] == 4:  # RGBA
                pil_image = Image.fromarray(img_array, 'RGBA').convert('RGB')
            else:
                raise ValueError(f"Unsupported number of channels: {img_array.shape[2]}")
            
            pil_images.append(pil_image)
        
        return pil_images
    
    def _process_detection_only(self, images: List[Image.Image], confidence_threshold: float) -> Dict:
        """Process text detection only"""
        try:
            detections = self.detection_predictor(images)
            
            # Convert results to dictionaries and filter by confidence
            filtered_detections = []
            for i, detection in enumerate(detections):                
                # Access object attributes, not dictionary keys
                detection_dict = {
                    'bboxes': [],
                    'vertical_lines': getattr(detection, 'vertical_lines', []),
                    'page': getattr(detection, 'page', 0),
                    'image_bbox': getattr(detection, 'image_bbox', None)
                }
                
                # Process bboxes
                bboxes = getattr(detection, 'bboxes', [])
                
                for j, bbox in enumerate(bboxes):
                    # Convert bbox object to dict
                    bbox_dict = {
                        'bbox': getattr(bbox, 'bbox', None),
                        'polygon': getattr(bbox, 'polygon', None),
                        'confidence': getattr(bbox, 'confidence', 1.0)
                    }
                    
                    if bbox_dict['confidence'] >= confidence_threshold:
                        detection_dict['bboxes'].append(bbox_dict)
                
                filtered_detections.append(detection_dict)
            
            print(f"✅ Detection complete: {sum(len(d['bboxes']) for d in filtered_detections)} boxes found")
            return {'detections': filtered_detections}
            
        except Exception as e:
            print(f"Error in _process_detection_only: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _process_full_ocr(self, images: List[Image.Image], task_mode: str, confidence_threshold: float) -> Dict:
        """Process full OCR (detection + recognition)"""
        try:
            # Run OCR with detection predictor
            ocr_results = self.recognition_predictor(images, det_predictor=self.detection_predictor)
            
            # Convert results to dictionaries and filter by confidence
            filtered_results = []
            total_text_lines = 0
            
            for i, result in enumerate(ocr_results):
                # Access object attributes, not dictionary keys
                result_dict = {
                    'text_lines': [],
                    'page': getattr(result, 'page', 0),
                    'image_bbox': getattr(result, 'image_bbox', None)
                }
                
                # Process text lines
                text_lines = getattr(result, 'text_lines', [])
                
                for j, line in enumerate(text_lines):
                    # Convert line object to dict - handle all nested objects
                    line_dict = {
                        'text': getattr(line, 'text', ''),
                        'confidence': getattr(line, 'confidence', 1.0),
                        'polygon': getattr(line, 'polygon', None),
                        'bbox': getattr(line, 'bbox', None),
                    }
                    
                    # Convert chars to serializable format
                    chars = getattr(line, 'chars', [])
                    line_dict['chars'] = []
                    for char in chars:
                        char_dict = {
                            'text': getattr(char, 'text', ''),
                            'bbox': getattr(char, 'bbox', None),
                            'polygon': getattr(char, 'polygon', None),
                            'confidence': getattr(char, 'confidence', 1.0),
                            'bbox_valid': getattr(char, 'bbox_valid', True)
                        }
                        line_dict['chars'].append(char_dict)
                    
                    # Convert words to serializable format
                    words = getattr(line, 'words', [])
                    line_dict['words'] = []
                    for word in words:
                        word_dict = {
                            'text': getattr(word, 'text', ''),
                            'bbox': getattr(word, 'bbox', None),
                            'polygon': getattr(word, 'polygon', None),
                            'confidence': getattr(word, 'confidence', 1.0),
                            'bbox_valid': getattr(word, 'bbox_valid', True)
                        }
                        line_dict['words'].append(word_dict)
                    
                    if line_dict['confidence'] >= confidence_threshold:
                        result_dict['text_lines'].append(line_dict)
                        total_text_lines += 1
                
                filtered_results.append(result_dict)
            
            print(f"✅ OCR complete: {total_text_lines} text lines extracted")
            return {'ocr_results': filtered_results, 'task_mode': task_mode}
            
        except Exception as e:
            print(f"Error in _process_full_ocr: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _process_layout(self, images: List[Image.Image]) -> Dict:
        """Process layout analysis"""
        if not self.layout_predictor:
            # NEW API (Surya 0.17+): Pass foundation predictor to LayoutPredictor
            self.layout_predictor = LayoutPredictor(foundation_predictor=self.foundation_predictor)
        
        layout_results = self.layout_predictor(images)
        return layout_results
    
    def _extract_text(self, results: Dict) -> str:
        """Extract plain text from results"""
        text_lines = []
        
        if 'ocr_results' in results:
            for page_result in results['ocr_results']:
                for line in page_result.get('text_lines', []):
                    text_lines.append(line.get('text', ''))
        
        return '\n'.join(text_lines)
    
    def _format_json_output(self, results: Dict) -> str:
        """Format results as JSON string with safe serialization"""
        import json
        
        def make_json_serializable(obj):
            """Recursively convert objects to JSON-serializable format"""
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return list(obj)  # Convert tuples to lists
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif hasattr(obj, '__dict__'):
                # Convert custom objects to dictionaries
                return make_json_serializable(obj.__dict__)
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            else:
                # Fallback: convert to string
                return str(obj)
        
        try:
            # Make results completely serializable
            safe_results = make_json_serializable(results)
            return json.dumps(safe_results, indent=2, ensure_ascii=False)
        except Exception as e:
            error_msg = f"Failed to serialize results: {str(e)}"
            print(f"JSON serialization error: {error_msg}")
            return f'{{"error": "{error_msg}", "partial_results": {{"text_count": {len(results.get("ocr_results", []))}}}}}'
    
    def _create_annotated_image(self, original_image: Image.Image, results: Dict) -> torch.Tensor:
        """Create annotated image with bounding boxes"""
        try:
            from PIL import ImageDraw, ImageFont
            
            # Create copy for annotation
            annotated = original_image.copy()
            draw = ImageDraw.Draw(annotated)
            
            # Try to load a font, fallback to default
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            # Draw bounding boxes
            if 'ocr_results' in results:
                for page_result in results['ocr_results']:
                    for line in page_result.get('text_lines', []):
                        bbox = line.get('bbox')
                        if bbox and len(bbox) >= 4:
                            # Draw rectangle
                            draw.rectangle(bbox, outline='red', width=2)
                            
                            # Draw text if available
                            text = line.get('text', '')[:50]  # Truncate long text
                            if text:
                                # Position text above the box
                                text_y = max(5, bbox[1] - 15)
                                draw.text((bbox[0], text_y), text, fill='red', font=font)
            
            elif 'detections' in results:
                for page_result in results['detections']:
                    for bbox_info in page_result.get('bboxes', []):
                        bbox = bbox_info.get('bbox')
                        if bbox and len(bbox) >= 4:
                            draw.rectangle(bbox, outline='blue', width=2)
                            
                            # Draw confidence score
                            confidence = bbox_info.get('confidence', 1.0)
                            conf_text = f"{confidence:.2f}"
                            text_y = max(5, bbox[1] - 15)
                            draw.text((bbox[0], text_y), conf_text, fill='blue', font=font)
            
            # Convert back to tensor
            img_array = np.array(annotated).astype(np.float32) / 255.0
            return torch.from_numpy(img_array).unsqueeze(0)
            
        except Exception as e:
            print(f"Failed to create annotated image: {e}")
            # Return original image as tensor
            img_array = np.array(original_image).astype(np.float32) / 255.0
            return torch.from_numpy(img_array).unsqueeze(0)
    
    def _format_detection_data(self, results: Dict) -> List[Dict]:
        """Format detection data for downstream nodes"""
        detection_data = []
        
        if 'ocr_results' in results:
            for page_result in results['ocr_results']:
                for line in page_result.get('text_lines', []):
                    detection_data.append({
                        'text': line.get('text', ''),
                        'bbox': line.get('bbox'),
                        'confidence': line.get('confidence', 1.0),
                        'polygon': line.get('polygon'),
                    })
        
        return detection_data

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SuryaOCR": SuryaOCRNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SuryaOCR": "Surya OCR (RTX 4090 Optimized)"
}

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']