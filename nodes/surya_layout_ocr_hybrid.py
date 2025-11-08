"""
Surya Layout Ocr Hybrid

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
Surya Layout + Hybrid OCR Node for ComfyUI
Uses Surya for layout detection, with choice of Surya OCR or Tesseract for text extraction
Optimized for systems with multiple GPUs and Flash Attention compatibility issues
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Lazy imports for Surya
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

# Lazy imports for Tesseract
TESSERACT_AVAILABLE = False
TESSERACT_IMPORT_ERROR = None

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception as e:
    TESSERACT_IMPORT_ERROR = e


class SuryaLayoutOCRHybrid:
    """
    Hybrid OCR node that combines:
    - Surya layout detection (CPU - always works)
    - Choice of Surya OCR (GPU) or Tesseract (CPU) for text extraction
    
    Designed for systems with Flash Attention compatibility issues.
    """
    
    def __init__(self):
        self.cpu_device = torch.device("cpu")
        self.gpu_device = None
        self.foundation_predictor_cpu = None
        self.layout_predictor = None
        self.detection_predictor = None
        self.recognition_predictor = None
        
    @staticmethod
    def _get_available_gpus() -> List[str]:
        """Get list of available GPU devices."""
        devices = ["cpu"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                devices.append(f"cuda:{i} ({gpu_name})")
        return devices
    
    def _init_layout_predictor(self):
        """Initialize CPU-based layout predictor."""
        if not SURYA_AVAILABLE:
            raise ImportError(f"surya-ocr not installed: {SURYA_IMPORT_ERROR}")
        
        if self.layout_predictor is not None:
            return
        
        print("🔧 Initializing Surya layout predictor (CPU)...")
        
        # Create CPU foundation predictor
        if self.foundation_predictor_cpu is None:
            self.foundation_predictor_cpu = FoundationPredictor(
                device="cpu",
                dtype=torch.float32
            )
            print("  ✓ CPU Foundation predictor initialized")
        
        # Create layout predictor
        self.layout_predictor = LayoutPredictor(
            foundation_predictor=self.foundation_predictor_cpu
        )
        print("  ✓ Layout predictor initialized (CPU)")
    
    def _init_surya_ocr(self, gpu_id: int = 1):
        """Initialize Surya OCR on specified GPU."""
        if not SURYA_AVAILABLE:
            raise ImportError(f"surya-ocr not installed: {SURYA_IMPORT_ERROR}")
        
        if self.detection_predictor is not None and self.recognition_predictor is not None:
            return
        
        device_str = f"cuda:{gpu_id}"
        self.gpu_device = torch.device(device_str)
        
        print(f"🔧 Initializing Surya OCR on {device_str} ({torch.cuda.get_device_name(gpu_id)})...")
        
        try:
            # Set the GPU
            torch.cuda.set_device(gpu_id)
            
            # Create GPU foundation predictor
            foundation_predictor_gpu = FoundationPredictor(
                device=device_str,
                dtype=torch.float16
            )
            print("  ✓ GPU Foundation predictor initialized")
            
            # Create detection predictor
            self.detection_predictor = DetectionPredictor(
                device=device_str,
                dtype=torch.float16
            )
            print("  ✓ Detection predictor initialized")
            
            # Create recognition predictor
            self.recognition_predictor = RecognitionPredictor(
                foundation_predictor=foundation_predictor_gpu
            )
            print("  ✓ Recognition predictor initialized")
            print(f"✅ Surya OCR ready on GPU {gpu_id}")
            
        except Exception as e:
            print(f"❌ Failed to initialize Surya OCR on GPU {gpu_id}: {e}")
            self.detection_predictor = None
            self.recognition_predictor = None
            raise
    
    @classmethod
    def INPUT_TYPES(cls):
        available_gpus = cls._get_available_gpus()
        
        return {
            "required": {
                "image": ("IMAGE",),
                "ocr_engine": (["tesseract", "surya"], {
                    "default": "tesseract"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
            },
            "optional": {
                "enable_layout": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable layout detection (finds images, tables, text blocks, etc.)"
                }),
                "enable_ocr": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable text extraction"
                }),
                "surya_gpu": (["cuda:1 (RTX A4000)", "cuda:0", "cpu"], {
                    "default": "cuda:1 (RTX A4000)",
                    "tooltip": "GPU to use for Surya OCR (ignored if using Tesseract)"
                }),
                "tesseract_lang": ("STRING", {
                    "default": "eng",
                    "tooltip": "Tesseract language (e.g., eng, fra, deu, spa)"
                }),
                "show_labels": ("BOOLEAN", {
                    "default": True,
                }),
                "batch_size": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 32,
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
        ocr_engine: str = "tesseract",
        confidence_threshold: float = 0.5,
        enable_layout: bool = True,
        enable_ocr: bool = True,
        surya_gpu: str = "cuda:1 (RTX A4000)",
        tesseract_lang: str = "eng",
        show_labels: bool = True,
        batch_size: int = 2
    ) -> Tuple[torch.Tensor, str, str, str, str, str, str]:
        """Process image with layout detection and OCR."""
        
        try:
            # Convert to PIL
            pil_images = self._tensor_to_pil(image)
            results = {}
            
            # Layout detection (always on CPU)
            if enable_layout:
                print("🔍 Running layout detection (CPU)...")
                self._init_layout_predictor()
                layout_results = self.layout_predictor(pil_images, batch_size=batch_size)
                results['layout'] = layout_results
                print(f"  ✓ Found {sum(len(r.bboxes) for r in layout_results)} layout boxes")
            
            # OCR
            if enable_ocr:
                if ocr_engine == "surya":
                    results['ocr'] = self._run_surya_ocr(pil_images, surya_gpu, batch_size)
                else:  # tesseract
                    results['ocr'] = self._run_tesseract_ocr(pil_images, tesseract_lang)
            
            # Filter and generate outputs
            filtered_results = self._filter_by_confidence(results, confidence_threshold)
            
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
            status = self._create_status(filtered_results, ocr_engine)
            
            # Cleanup
            if torch.cuda.is_available():
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
            error_msg = f"❌ Error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
    
    def _run_surya_ocr(self, pil_images: List[Image.Image], surya_gpu: str, batch_size: int) -> List[Dict]:
        """Run Surya OCR on specified GPU."""
        print(f"📝 Running Surya OCR...")
        
        # Parse GPU ID from string like "cuda:1 (RTX A4000)"
        gpu_id = 1  # default
        if "cuda:" in surya_gpu:
            try:
                gpu_id = int(surya_gpu.split("cuda:")[1].split()[0])
            except:
                pass
        
        # Initialize if needed
        if self.detection_predictor is None or self.recognition_predictor is None:
            self._init_surya_ocr(gpu_id)
        
        # Run OCR
        ocr_results = self.recognition_predictor(
            pil_images,
            det_predictor=self.detection_predictor,
            detection_batch_size=batch_size,
            recognition_batch_size=batch_size
        )
        
        # Convert to dict format
        ocr_list = []
        for ocr_result in ocr_results:
            ocr_list.append({
                'text_lines': ocr_result.text_lines,
                'image_bbox': ocr_result.image_bbox,
                'engine': 'surya'
            })
        
        print(f"  ✓ Extracted {sum(len(r['text_lines']) for r in ocr_list)} text lines (Surya)")
        return ocr_list
    
    def _run_tesseract_ocr(self, pil_images: List[Image.Image], lang: str) -> List[Dict]:
        """Run Tesseract OCR."""
        if not TESSERACT_AVAILABLE:
            raise ImportError(f"pytesseract not installed: {TESSERACT_IMPORT_ERROR}")
        
        print(f"📝 Running Tesseract OCR (language: {lang})...")
        
        ocr_list = []
        for img in pil_images:
            # Get OCR data with bounding boxes
            data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
            
            # Group by line
            text_lines = []
            current_line = []
            current_line_num = -1
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) < 0:  # Skip invalid detections
                    continue
                
                line_num = data['line_num'][i]
                text = data['text'][i].strip()
                
                if not text:
                    continue
                
                if line_num != current_line_num:
                    # Save previous line
                    if current_line:
                        text_lines.append(self._merge_tesseract_line(current_line))
                    current_line = []
                    current_line_num = line_num
                
                current_line.append({
                    'text': text,
                    'bbox': [data['left'][i], data['top'][i], 
                            data['left'][i] + data['width'][i], 
                            data['top'][i] + data['height'][i]],
                    'conf': float(data['conf'][i]) / 100.0
                })
            
            # Save last line
            if current_line:
                text_lines.append(self._merge_tesseract_line(current_line))
            
            ocr_list.append({
                'text_lines': text_lines,
                'image_bbox': [0, 0, img.width, img.height],
                'engine': 'tesseract'
            })
        
        total_lines = sum(len(r['text_lines']) for r in ocr_list)
        print(f"  ✓ Extracted {total_lines} text lines (Tesseract)")
        return ocr_list
    
    def _merge_tesseract_line(self, words: List[Dict]) -> Any:
        """Merge Tesseract words into a line object."""
        if not words:
            return None
        
        # Create a simple object with text and bbox
        class TextLine:
            def __init__(self, text, bbox, confidence):
                self.text = text
                self.bbox = bbox
                self.confidence = confidence
                self.polygon = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], 
                               [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
        
        text = ' '.join(w['text'] for w in words)
        
        # Merge bboxes
        x1 = min(w['bbox'][0] for w in words)
        y1 = min(w['bbox'][1] for w in words)
        x2 = max(w['bbox'][2] for w in words)
        y2 = max(w['bbox'][3] for w in words)
        bbox = [x1, y1, x2, y2]
        
        # Average confidence
        conf = sum(w['conf'] for w in words) / len(words)
        
        return TextLine(text, bbox, conf)
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> List[Image.Image]:
        """Convert ComfyUI tensor to PIL images."""
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        pil_images = []
        for i in range(tensor.shape[0]):
            img_array = (tensor[i].numpy() * 255).astype(np.uint8)
            
            if img_array.shape[2] == 3:
                pil_image = Image.fromarray(img_array, 'RGB')
            elif img_array.shape[2] == 4:
                pil_image = Image.fromarray(img_array, 'RGBA').convert('RGB')
            else:
                raise ValueError(f"Unsupported channel count: {img_array.shape[2]}")
            
            pil_images.append(pil_image)
        
        return pil_images
    
    def _filter_by_confidence(self, results: Dict, threshold: float) -> Dict:
        """Filter results by confidence."""
        filtered = {}
        
        if 'layout' in results:
            filtered_layout = []
            for layout_result in results['layout']:
                filtered_bboxes = [
                    bbox for bbox in layout_result.bboxes
                    if bbox.confidence >= threshold
                ]
                filtered_layout.append({
                    'bboxes': filtered_bboxes,
                    'image_bbox': layout_result.image_bbox,
                    'sliced': layout_result.sliced
                })
            filtered['layout'] = filtered_layout
        
        if 'ocr' in results:
            filtered_ocr = []
            for ocr_result in results['ocr']:
                # Filter text lines by confidence
                filtered_lines = [
                    line for line in ocr_result['text_lines']
                    if getattr(line, 'confidence', 1.0) >= threshold
                ]
                filtered_ocr.append({
                    'text_lines': filtered_lines,
                    'image_bbox': ocr_result['image_bbox'],
                    'engine': ocr_result.get('engine', 'unknown')
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
        annotated = pil_image.copy()
        draw = ImageDraw.Draw(annotated)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            small_font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        color_map = {
            'Image': '#FF0000', 'Table': '#00FF00', 'Text': '#0000FF',
            'Title': '#FF00FF', 'Section-header': '#FFFF00', 'List': '#00FFFF',
            'Caption': '#FFA500', 'Footnote': '#800080', 'Formula': '#FFC0CB',
            'Page-header': '#A52A2A', 'Page-footer': '#808080', 'Form': '#FF1493',
        }
        
        # Draw layout boxes
        if 'layout' in results:
            for page_result in results['layout']:
                for bbox in page_result['bboxes']:
                    label = bbox.label
                    polygon = bbox.polygon
                    confidence = bbox.confidence
                    color = color_map.get(label, '#FFFFFF')
                    
                    if polygon and len(polygon) >= 3:
                        flat_polygon = [coord for point in polygon for coord in point]
                        draw.polygon(flat_polygon, outline=color, width=3)
                        
                        if show_labels and polygon:
                            text = f"{label} ({confidence:.2f})"
                            text_pos = (int(polygon[0][0]), max(0, int(polygon[0][1]) - 20))
                            bbox_coords = draw.textbbox(text_pos, text, font=small_font)
                            draw.rectangle(bbox_coords, fill=color)
                            draw.text(text_pos, text, fill='black', font=small_font)
        
        # Draw OCR boxes
        if 'ocr' in results:
            for page_result in results['ocr']:
                for text_line in page_result['text_lines']:
                    polygon = getattr(text_line, 'polygon', None)
                    bbox = getattr(text_line, 'bbox', None)
                    
                    if polygon and len(polygon) >= 3:
                        flat_polygon = [coord for point in polygon for coord in point]
                        draw.polygon(flat_polygon, outline='#00FF00', width=2)
                    elif bbox and len(bbox) >= 4:
                        draw.rectangle(bbox, outline='#00FF00', width=2)
        
        # Convert to tensor
        img_array = np.array(annotated).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array).unsqueeze(0)
        
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
        """Extract text bounding boxes."""
        text_boxes = []
        
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
        """Extract image bounding boxes."""
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
                        'position': bbox.position
                    }
                    page_data['boxes'].append(box_data)
                
                layout_list.append(page_data)
            
            full_data['layout'] = layout_list
        
        if 'ocr' in results:
            ocr_list = []
            for page_idx, page_result in enumerate(results['ocr']):
                page_data = {
                    'page': page_idx,
                    'image_bbox': page_result.get('image_bbox'),
                    'engine': page_result.get('engine', 'unknown'),
                    'text_lines': []
                }
                
                for text_line in page_result['text_lines']:
                    line_data = {
                        'text': getattr(text_line, 'text', ''),
                        'polygon': getattr(text_line, 'polygon', None),
                        'bbox': getattr(text_line, 'bbox', None),
                        'confidence': getattr(text_line, 'confidence', None)
                    }
                    page_data['text_lines'].append(line_data)
                
                ocr_list.append(page_data)
            
            full_data['ocr'] = ocr_list
        
        return json.dumps(full_data, indent=2)
    
    def _create_status(self, results: Dict, ocr_engine: str) -> str:
        """Create status message."""
        status_lines = ["✅ Processing complete"]
        
        if 'layout' in results:
            total_boxes = sum(len(page['bboxes']) for page in results['layout'])
            status_lines.append(f"  📐 Layout: {total_boxes} boxes detected")
            
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
            status_lines.append(f"  📝 OCR ({ocr_engine}): {total_lines} lines, {total_chars} chars")
        
        return '\n'.join(status_lines)


# Register node
NODE_CLASS_MAPPINGS = {
    "SuryaLayoutOCRHybrid": SuryaLayoutOCRHybrid
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SuryaLayoutOCRHybrid": "Surya Layout + OCR (Hybrid)"
}
