"""
Enhanced Layout Parser V06

Description: Enhanced layout parsing node for ComfyUI using Surya OCR, Florence2 vision models, and advanced analysis
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

import os

# REDUCED batch sizes for multi-model usage (Surya + Florence2 + LayoutLMv3)
os.environ['RECOGNITION_BATCH_SIZE'] = '64'   # Much smaller for shared VRAM
os.environ['DETECTOR_BATCH_SIZE'] = '8'       # Much smaller for shared VRAM
os.environ['LAYOUT_BATCH_SIZE'] = '6'         # Much smaller for shared VRAM
os.environ['TABLE_REC_BATCH_SIZE'] = '12'     # Much smaller for shared VRAM
os.environ['TORCH_DEVICE'] = 'cuda'

import torch
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Any, Tuple, Optional
import cv2
from PIL import ImageFont

# ComfyUI imports
try:
    from comfy.utils import common_upscale
    from comfy.model_management import get_torch_device
    import folder_paths
    COMFYUI_BASE_PATH = os.path.dirname(folder_paths.models_dir)
    
    def tensor_to_PIL(tensor):
        return Image.fromarray(np.clip(255. * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
    def PIL_to_tensor(pil_image):
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)
    
except ImportError:
    print("ComfyUI imports not available, using fallback")
    COMFYUI_BASE_PATH = "."
    
    def tensor_to_PIL(tensor):
        if isinstance(tensor, torch.Tensor):
            return Image.fromarray(np.clip(255. * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        return tensor
    
    def PIL_to_tensor(pil_image):
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

# Handle dependencies gracefully
try:
    from surya.foundation import FoundationPredictor  # NEW: Required for Surya 0.17+
    from surya.layout import LayoutPredictor
    SURYA_LAYOUT_AVAILABLE = True
except ImportError:
    print("⚠️ Surya Layout not available - install with: pip install surya-ocr")
    SURYA_LAYOUT_AVAILABLE = False

try:
    from PDF_tools import Florence2RectangleDetector
    FLORENCE2_AVAILABLE = True
except ImportError:
    try:
        from ..florence2_rectangle_detector import Florence2RectangleDetector
        FLORENCE2_AVAILABLE = True
    except ImportError:
        print("⚠️ Florence2RectangleDetector not found - some features will be disabled")
        FLORENCE2_AVAILABLE = False

class EnhancedLayoutParserNode:
    """
    Streamlined layout analysis with Surya Layout + Tesseract:
    - Primary: Surya Layout detection for semantic regions
    - OCR: Tesseract for text extraction
    - Backup: Florence2 for images and emergency text fallback
    """
    
    @classmethod  
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "primary_ocr": (["tesseract", "paddleocr", "easyocr"], {"default": "tesseract"}),
                "enable_florence2": ("BOOLEAN", {"default": True}),
                "extract_images": ("BOOLEAN", {"default": True}),
                "extract_text": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "florence2_image_prompt": ("STRING", {
                    "default": "rectangular images in page",
                    "multiline": True
                }),
                "florence2_text_prompt": ("STRING", {
                    "default": "text, caption, paragraph, title",
                    "multiline": True
                }),
                "min_image_area": ("INT", {"default": 10000, "min": 1000, "max": 100000}),
                "min_text_area": ("INT", {"default": 500, "min": 100, "max": 10000}),
                "surya_confidence": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.05}),
                "florence2_confidence": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "text_extraction_strategy": ([
                    "surya_layout_primary",           # Surya Layout → Tesseract (primary)
                    "florence2_backup_only"          # Florence2 → Tesseract (alternative)
                ], {"default": "surya_layout_primary"}),
                "include_text_recognition": ("BOOLEAN", {"default": True}),
                "enable_debug_logging": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LIST", "LIST", "LIST", "LIST", "STRING", "DICT")
    RETURN_NAMES = ("overlay_image", "image_boxes", "text_boxes", "image_crops", "text_crops", "extracted_text", "analysis_results")

    FUNCTION = "analyze_layout"
    CATEGORY = "Enhanced Layout/Analysis"
    
    def __init__(self):
        self.florence2_detector = None
        self.current_image = None
        
        print("🚀 Enhanced Layout Parser v06 - Streamlined Edition")
        
    def _clear_cuda_cache(self):
        """Clear CUDA cache to free up memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_surya_layout_detection_minimal(self, image: Image.Image, confidence_threshold: float = 0.3) -> List[Dict]:
        """Minimal Surya layout detection (proven working approach)"""
        try:
            print("🔍 Running Surya layout detection...")

            if not SURYA_LAYOUT_AVAILABLE:
                print("❌ Surya Layout not available")
                return []

            # Clear environment interference 
            import os
            env_vars_to_remove = [
                'TORCHDYNAMO_DISABLE', 'TORCH_COMPILE_DISABLE', 'PYTORCH_DISABLE_DYNAMO',
                'TORCHDYNAMO_VERBOSE', 'TORCH_LOGS'
            ]
            
            original_env = {}
            for var in env_vars_to_remove:
                if var in os.environ:
                    original_env[var] = os.environ[var]
                    del os.environ[var]
            
            # Fresh torch import
            import torch
            if hasattr(torch._dynamo, 'reset'):
                torch._dynamo.reset()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Initialize layout predictor (exactly like console) with NEW API (Surya 0.17+)
            from surya.foundation import FoundationPredictor
            from surya.layout import LayoutPredictor
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            # Create foundation predictor first
            foundation_predictor = FoundationPredictor(
                checkpoint=None,
                device=device,
                dtype=dtype,
                attention_implementation="sdpa"
            )
            
            # Create layout predictor with foundation predictor
            layout_predictor = LayoutPredictor(foundation_predictor=foundation_predictor)
            layout_predictions = layout_predictor([image])
            
            # Process results
            layout_regions = []
            for page_idx, page_result in enumerate(layout_predictions):
                if hasattr(page_result, 'bboxes'):
                    bboxes = page_result.bboxes
                    
                    for layout_box in bboxes:
                        if hasattr(layout_box, 'bbox') and hasattr(layout_box, 'label'):
                            bbox = layout_box.bbox
                            label = layout_box.label
                            confidence = getattr(layout_box, 'confidence', 0.9)
                            position = getattr(layout_box, 'position', -1)
                            
                            if bbox and confidence >= confidence_threshold:
                                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                layout_regions.append({
                                    'text': '',
                                    'bbox': bbox,
                                    'confidence': confidence,
                                    'area': area,
                                    'source': 'surya_layout',
                                    'semantic_label': label,
                                    'reading_position': position
                                })
            
            # Sort by reading order
            layout_regions.sort(key=lambda x: x['reading_position'] if x['reading_position'] >= 0 else 999)
            
            print(f"✅ Surya layout found {len(layout_regions)} semantic regions")
            
            if layout_regions:
                label_counts = {}
                for region in layout_regions:
                    label = region['semantic_label']
                    label_counts[label] = label_counts.get(label, 0) + 1
                print(f"   Layout elements: {dict(label_counts)}")
            
            # Restore environment
            for var, value in original_env.items():
                os.environ[var] = value
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return layout_regions
            
        except Exception as e:
            print(f"❌ Layout detection failed: {e}")
            try:
                for var, value in original_env.items():
                    os.environ[var] = value
            except:
                pass
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return []

    def _init_florence2(self):
        """Initialize Florence2 for image detection"""
        if not FLORENCE2_AVAILABLE:
            return False

        try:
            self._clear_cuda_cache()
            self.florence2_detector = Florence2RectangleDetector(
                model_name="CogFlorence-2.2-Large",
                comfyui_base_path=COMFYUI_BASE_PATH,
                min_box_area=1000
            )
            print("✅ Florence2 initialized")
            return True
            
        except Exception as e:
            print(f"❌ Florence2 initialization failed: {e}")
            try:
                self.florence2_detector = Florence2RectangleDetector(
                    model_name="microsoft/Florence-2-base",
                    comfyui_base_path=COMFYUI_BASE_PATH,
                    min_box_area=1000
                )
                print("✅ Florence2 initialized with base model")
                return True
            except Exception as e2:
                print(f"❌ Florence2 base model also failed: {e2}")
                return False

    def _run_florence2_detection(self, image: Image.Image, 
                               image_prompt: str = "rectangular images in page",
                               text_prompt: str = "text, caption, paragraph, title") -> Tuple[List[Dict], List[Dict]]:
        """Run Florence2 detection"""
        try:
            if not self.florence2_detector:
                if not self._init_florence2():
                    return [], []
            
            self._clear_cuda_cache()
            
            # Detect images
            image_boxes, _ = self.florence2_detector.detect_rectangles(
                image=image,
                text_input=image_prompt,
                return_mask=False,
                keep_model_loaded=True
            )
            
            florence2_images = []
            for box in image_boxes:
                bbox = [int(box.x1), int(box.y1), int(box.x2), int(box.y2)]
                florence2_images.append({
                    'bbox': bbox,
                    'label': box.label,
                    'confidence': box.confidence if box.confidence else 0.8,
                    'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    'source': 'florence2'
                })
            
            # Detect text regions
            text_boxes, _ = self.florence2_detector.detect_rectangles(
                image=image,
                text_input=text_prompt,
                return_mask=False,
                keep_model_loaded=True
            )
            
            florence2_text = []
            for box in text_boxes:
                bbox = [int(box.x1), int(box.y1), int(box.x2), int(box.y2)]
                florence2_text.append({
                    'bbox': bbox,
                    'label': box.label,
                    'confidence': box.confidence if box.confidence else 0.8,
                    'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    'source': 'florence2'
                })
            
            print(f"✅ Florence2 found {len(florence2_images)} images, {len(florence2_text)} text regions")
            self._clear_cuda_cache()
            
            return florence2_images, florence2_text
            
        except Exception as e:
            print(f"❌ Florence2 detection failed: {e}")
            self._clear_cuda_cache()
            return [], []

    def _extract_text_from_crop(self, image: Image.Image, bbox: List[int], ocr_engine: str = "tesseract") -> Dict:
        """Extract text using Tesseract with enhanced preprocessing"""
        try:
            # Crop with padding
            padding = 10
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding) 
            x2 = min(image.width, x2 + padding)
            y2 = min(image.height, y2 + padding)
            
            cropped = image.crop((x1, y1, x2, y2))
            
            if cropped.width < 50 or cropped.height < 20:
                return {"text": "", "confidence": 0.0, "method": "too_small"}
            
            # Enhanced preprocessing for Tesseract
            enhanced = self._enhance_image_for_tesseract_ocr(cropped)
            
            if ocr_engine == "tesseract":
                return self._extract_with_tesseract(enhanced)
            else:
                return {"text": "", "confidence": 0.0, "method": f"unsupported_engine_{ocr_engine}"}
                
        except Exception as e:
            print(f"❌ Text extraction failed: {e}")
            return {"text": "", "confidence": 0.0, "method": "error"}

    def _enhance_image_for_tesseract_ocr(self, image: Image.Image) -> Image.Image:
        """Apply Tesseract-optimized enhancements"""
        if image.mode != 'L':
            image = image.convert('L')
        
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.8)
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        
        if image.width < 300:
            scale = 400 / image.width
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image

    def _extract_with_tesseract(self, image: Image.Image) -> Dict:
        """Extract text with Tesseract"""
        try:
            import pytesseract
            
            tesseract_cmd = self._find_tesseract_path()
            if not tesseract_cmd:
                return {"text": "", "confidence": 0.0, "method": "tesseract_not_found"}
            
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
            # Try different PSM modes
            psm_modes = [6, 3, 8, 11, 13]
            best_text = ""
            best_length = 0
            best_psm = 6
            
            for psm in psm_modes:
                try:
                    config = f'--oem 3 --psm {psm}'
                    text = pytesseract.image_to_string(image, config=config).strip()
                    
                    if len(text) > best_length and len(text) > 5:
                        best_text = text
                        best_length = len(text)
                        best_psm = psm
                except Exception:
                    continue
            
            if best_text:
                return {
                    "text": best_text,
                    "confidence": 0.8,
                    "method": f"tesseract_psm_{best_psm}",
                    "word_count": len(best_text.split()),
                    "best_psm": best_psm
                }
            else:
                return {"text": "", "confidence": 0.0, "method": "tesseract_no_results"}
                
        except ImportError:
            return {"text": "", "confidence": 0.0, "method": "tesseract_library_not_available"}
        except Exception as e:
            return {"text": "", "confidence": 0.0, "method": "tesseract_error"}

    def _find_tesseract_path(self) -> str:
        """Find Tesseract installation"""
        try:
            import pytesseract
            
            common_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract', 
                r'C:\Users\Public\Tesseract-OCR\tesseract',
                r'A:\Tesseract-OCR\tesseract',
                'tesseract'
            ]
            
            for path in common_paths:
                try:
                    pytesseract.pytesseract.tesseract_cmd = path
                    version = pytesseract.get_tesseract_version()
                    print(f"✅ Found Tesseract at: {path} (version: {version})")
                    return path
                except:
                    continue
            
            return None
            
        except ImportError:
            return None

    def _calculate_overlap_ratio(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        union_area = bbox1_area + bbox2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def _create_structured_text_output(self, text_regions: List[Dict]) -> str:
        """Create structured multiline text with semantic labels"""
        if not text_regions:
            return ""
        
        sorted_regions = sorted(text_regions, key=lambda x: x.get('reading_position', x.get('position', 999)))
        structured_lines = []
        
        for region in sorted_regions:
            text_content = region.get('text', '').strip()
            if not text_content:
                continue
            
            semantic_label = region.get('semantic_label', 'text')
            
            label_mapping = {
                'Text': 'text',
                'SectionHeader': 'header', 
                'PageHeader': 'page_header',
                'PageFooter': 'page_footer',
                'Caption': 'caption',
                'ListItem': 'list_item',
                'TableOfContents': 'toc',
                'Title': 'title',
                'Handwriting': 'handwriting',
                'Footnote': 'footnote',
                'Formula': 'formula',
                'Form': 'form',
                'TextInlineMath': 'math',
                'Picture': 'image_caption',
                'Figure': 'figure_caption',
                'Table': 'table_caption'
            }
            
            display_label = label_mapping.get(semantic_label, semantic_label.lower())
            
            if '\n' in text_content:
                lines = text_content.split('\n')
                structured_lines.append(f"{display_label}: {lines[0]}")
                for line in lines[1:]:
                    if line.strip():
                        structured_lines.append(f"  {line.strip()}")
            else:
                structured_lines.append(f"{display_label}: {text_content}")
        
        return '\n'.join(structured_lines)

    def analyze_layout(self, image, primary_ocr="tesseract", enable_florence2=True, 
                    extract_images=True, extract_text=True,
                    florence2_image_prompt=None, florence2_text_prompt=None,
                    min_image_area=10000, min_text_area=500, 
                    surya_confidence=0.3, florence2_confidence=0.5,
                    text_extraction_strategy="surya_layout_primary",
                    include_text_recognition=True, enable_debug_logging=True):
        """Streamlined layout analysis with Surya Layout + Tesseract"""
        
        # Convert tensor to PIL
        pil_image = tensor_to_PIL(image)
        self.current_image = pil_image
        
        if enable_debug_logging:
            print(f"🔍 Streamlined Layout Analysis:")
            print(f"  Image size: {pil_image.size}")
            print(f"  Strategy: {text_extraction_strategy}")
        
        # 1. RUN SURYA LAYOUT DETECTION
        layout_results = self._run_surya_layout_detection_minimal(pil_image, surya_confidence)
        
        # Define semantic labels
        text_like_labels = [
            'Text', 'SectionHeader', 'PageHeader', 'PageFooter', 'Caption', 
            'ListItem', 'TableOfContents', 'Title', 'Handwriting', 'Footnote', 
            'Formula', 'Form', 'TextInlineMath'
        ]
        
        image_like_labels = [
            'Picture', 'Figure', 'Table'
        ]
        
        layout_text_regions = [r for r in layout_results if r['semantic_label'] in text_like_labels]
        layout_image_regions = [r for r in layout_results if r['semantic_label'] in image_like_labels]
        
        print(f"✅ Layout detection: {len(layout_text_regions)} text regions, {len(layout_image_regions)} image regions")
        
        # 2. TEXT EXTRACTION STRATEGY
        text_regions_to_process = []
        extraction_method_used = ""
        
        if text_extraction_strategy == "surya_layout_primary":
            if layout_text_regions:
                text_regions_to_process = layout_text_regions
                extraction_method_used = "surya_layout"
                print(f"✅ Using Surya Layout: {len(text_regions_to_process)} text regions")
                
                # Show semantic breakdown
                semantic_breakdown = {}
                for region in text_regions_to_process:
                    label = region.get('semantic_label', 'unknown')
                    semantic_breakdown[label] = semantic_breakdown.get(label, 0) + 1
                print(f"   📋 Semantic breakdown: {dict(semantic_breakdown)}")
            else:
                # Fallback to Florence2
                if enable_florence2:
                    print("🔧 Florence2 emergency fallback...")
                    _, florence2_text = self._run_florence2_detection(pil_image, 
                        florence2_image_prompt or "rectangular images in page", 
                        florence2_text_prompt or "text, caption, paragraph, title")
                    text_regions_to_process = florence2_text
                    extraction_method_used = "florence2_emergency"
                    print(f"✅ Florence2 emergency: {len(text_regions_to_process)} regions")
                else:
                    text_regions_to_process = []
                    extraction_method_used = "no_text_detected"
        
        elif text_extraction_strategy == "florence2_backup_only":
            if enable_florence2:
                _, florence2_text = self._run_florence2_detection(pil_image, 
                    florence2_image_prompt or "rectangular images in page", 
                    florence2_text_prompt or "text, caption, paragraph, title")
                text_regions_to_process = florence2_text
                extraction_method_used = "florence2_only"
                print(f"✅ Florence2 only: {len(text_regions_to_process)} regions")
            else:
                text_regions_to_process = []
                extraction_method_used = "florence2_disabled"
        
        # 3. EXTRACT TEXT USING TESSERACT
        final_text_regions = []
        if include_text_recognition and text_regions_to_process:
            print(f"🔍 Extracting text using {primary_ocr} from {len(text_regions_to_process)} regions...")
            
            successful_extractions = 0
            total_characters = 0
            
            for i, region in enumerate(text_regions_to_process):
                extraction_result = self._extract_text_from_crop(pil_image, region['bbox'], primary_ocr)
                
                if extraction_result['text'] and len(extraction_result['text']) > 2:
                    enhanced_region = {
                        **region,
                        'text': extraction_result['text'],
                        'extraction_confidence': extraction_result['confidence'],
                        'extraction_method': extraction_result['method'],
                        'text_quality': 'high' if extraction_result['confidence'] > 0.7 else 'medium',
                        'detection_methods': [extraction_method_used, primary_ocr],
                        'confidence_sources': [region['confidence'], extraction_result['confidence']],
                        'confidence': (region['confidence'] + extraction_result['confidence']) / 2
                    }
                    
                    final_text_regions.append(enhanced_region)
                    successful_extractions += 1
                    total_characters += len(extraction_result['text'])
                    
                    if enable_debug_logging:
                        semantic_label = region.get('semantic_label', 'text')
                        preview = extraction_result['text'][:60]
                        print(f"   ✅ {semantic_label}: '{preview}...'")
                
                else:
                    failed_region = {
                        **region,
                        'text': "",
                        'extraction_confidence': 0.0,
                        'extraction_method': extraction_result['method'],
                        'text_quality': 'failed',
                        'detection_methods': [extraction_method_used],
                        'confidence_sources': [region['confidence']],
                        'confidence': region['confidence'] * 0.5
                    }
                    final_text_regions.append(failed_region)
            
            print(f"📝 Text extraction: {successful_extractions}/{len(text_regions_to_process)} successful, {total_characters} characters")
        else:
            final_text_regions = text_regions_to_process
            for region in final_text_regions:
                region['detection_methods'] = [extraction_method_used]
                region['confidence_sources'] = [region['confidence']]
        
        # 4. HANDLE IMAGES
        final_image_regions = []
        florence2_images = []
        
        if enable_florence2:
            florence2_images, _ = self._run_florence2_detection(pil_image, 
                florence2_image_prompt or "rectangular images in page", 
                florence2_text_prompt or "text, caption, paragraph, title")
        
        if extract_images:
            # Add Florence2 images
            for f2_img in florence2_images:
                if f2_img.get('area', 0) >= min_image_area and f2_img['confidence'] >= florence2_confidence:
                    enhanced_f2_img = {
                        **f2_img,
                        'detection_methods': ['florence2'],
                        'confidence_sources': [f2_img['confidence']],
                        'semantic_label': 'image'
                    }
                    final_image_regions.append(enhanced_f2_img)
            
            # Add layout images
            for layout_img in layout_image_regions:
                if layout_img.get('area', 0) >= min_image_area:
                    has_overlap = False
                    for f2_img in final_image_regions:
                        overlap_ratio = self._calculate_overlap_ratio(layout_img['bbox'], f2_img['bbox'])
                        if overlap_ratio > 0.3:
                            has_overlap = True
                            f2_img['semantic_label'] = layout_img['semantic_label']
                            f2_img['detection_methods'].append('surya_layout')
                            f2_img['confidence_sources'].append(layout_img['confidence'])
                            break
                    
                    if not has_overlap:
                        enhanced_layout_img = {
                            **layout_img,
                            'detection_methods': ['surya_layout'],
                            'confidence_sources': [layout_img['confidence']]
                        }
                        final_image_regions.append(enhanced_layout_img)
        
        # 5. FINAL FILTERING
        image_boxes = [img for img in final_image_regions if img.get('area', 0) >= min_image_area] if extract_images else []
        text_boxes = [txt for txt in final_text_regions if txt.get('area', 0) >= min_text_area] if extract_text else []
        
        print(f"🎯 Final results: {len(image_boxes)} images, {len(text_boxes)} text regions")
        
        # 6. CREATE OUTPUTS
        extracted_text = ""
        if include_text_recognition and text_boxes:
            extracted_text = self._create_structured_text_output(text_boxes)
        
        overlay = self._create_enhanced_overlay(pil_image, image_boxes, text_boxes)
        overlay_tensor = PIL_to_tensor(overlay)
        
        # 7. EXTRACT CROPS
        image_crops = []
        text_crops = []
        
        if extract_images:
            for img_box in image_boxes:
                try:
                    crop = pil_image.crop(img_box['bbox'])
                    image_crops.append(PIL_to_tensor(crop))
                except Exception as e:
                    print(f"⚠️ Error cropping image: {e}")
        
        if extract_text:
            for txt_box in text_boxes:
                try:
                    crop = pil_image.crop(txt_box['bbox'])
                    text_crops.append(PIL_to_tensor(crop))
                except Exception as e:
                    print(f"⚠️ Error cropping text: {e}")
        
        # 8. ANALYSIS RESULTS
        analysis_results = {
            "total_detections": len(image_boxes) + len(text_boxes),
            "image_detections": len(image_boxes),
            "text_detections": len(text_boxes),
            "extraction_strategy": "surya_layout_streamlined",
            "text_extraction_strategy_used": extraction_method_used,
            "text_extraction_strategy_requested": text_extraction_strategy,
            "text_recognition_enabled": include_text_recognition,
            "extracted_text_length": len(extracted_text),
            "primary_ocr_engine": primary_ocr,
            "confidence_thresholds": {
                "surya": surya_confidence,
                "florence2": florence2_confidence
            },
            "models_used": {
                "surya_layout": len(layout_results) > 0,
                "florence2": len(florence2_images) > 0,
                "tesseract": include_text_recognition,
            },
            "extraction_statistics": {
                "layout_text_regions": len(layout_text_regions),
                "layout_image_regions": len(layout_image_regions),
                "florence2_images": len(florence2_images),
                "successful_text_extractions": len([txt for txt in text_boxes if txt.get('text', '').strip()]),
                "total_extracted_characters": len(extracted_text)
            },
            "semantic_labels": {
                label: len([r for r in text_boxes if r.get('semantic_label') == label])
                for label in set([r.get('semantic_label', 'unknown') for r in text_boxes])
            }
        }
        
        if enable_debug_logging:
            print(f"📊 Analysis complete:")
            print(f"   🖼️  Images: {len(image_boxes)}")
            print(f"   📝 Text regions: {len(text_boxes)}")
            print(f"   📊 Total characters: {len(extracted_text)}")
            print(f"   🏷️  Method: {extraction_method_used}")
        
        return (overlay_tensor, image_boxes, text_boxes, image_crops, text_crops, extracted_text, analysis_results)

    def _create_enhanced_overlay(self, image: Image.Image, image_boxes: List[Dict], text_boxes: List[Dict]) -> Image.Image:
        """Create enhanced visualization"""
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        colors = {
            'florence2': (255, 0, 0),          # Red
            'surya_layout': (0, 255, 100),     # Bright Green
            'tesseract': (50, 255, 50),        # Light Green
            'multi': (255, 165, 0),            # Orange
            'SectionHeader': (255, 100, 0),    # Orange
            'PageHeader': (255, 150, 0),       # Light Orange  
            'PageFooter': (200, 100, 0),       # Dark Orange
            'Caption': (150, 255, 150),        # Light Green
            'ListItem': (100, 200, 255),       # Light Blue
            'TableOfContents': (200, 0, 200),  # Purple
            'Title': (255, 0, 100),            # Pink
            'Handwriting': (100, 100, 255),    # Blue
        }
        
        # Draw text boxes
        for i, box in enumerate(text_boxes):
            bbox = [int(coord) for coord in box['bbox']]
            methods = box.get('detection_methods', ['unknown'])
            semantic_label = box.get('semantic_label', '')
            
            if len(methods) > 1:
                color = colors['multi']
                method_label = "+".join(methods)
            else:
                primary_method = methods[0]
                if semantic_label in colors:
                    color = colors[semantic_label]
                else:
                    color = colors.get(primary_method, (128, 128, 128))
                method_label = primary_method
            
            self._draw_dashed_rectangle(draw, bbox, color, width=2)
            
            label = f"TXT{i+1}: {method_label} ({box.get('confidence', 0):.2f})"
            if semantic_label:
                label += f" [{semantic_label}]"
            
            if box.get('text_quality'):
                quality_indicator = {"high": "🟢", "medium": "🟡", "failed": "🔴"}.get(box['text_quality'], "")
                label += f" {quality_indicator}"
            
            draw.text((bbox[0], max(0, bbox[1] - 20)), label, fill=color, font=font)
        
        # Draw image boxes
        for i, box in enumerate(image_boxes):
            bbox = [int(coord) for coord in box['bbox']]
            methods = box['detection_methods']
            
            if len(methods) > 1:
                color = colors['multi']
                method_label = "+".join(methods)
            else:
                color = colors.get(methods[0], (128, 128, 128))
                method_label = methods[0]
            
            draw.rectangle(bbox, outline=color, width=3)
            
            label = f"IMG{i+1}: {method_label} ({box['confidence']:.2f})"
            if box.get('semantic_label'):
                label += f" [{box['semantic_label']}]"
            draw.text((bbox[0], max(0, bbox[1] - 20)), label, fill=color)
        
        return overlay
    
    def _draw_dashed_rectangle(self, draw, bbox, color, width=2, dash_length=5):
        """Draw a dashed rectangle"""
        x1, y1, x2, y2 = map(int, bbox)
        
        for x in range(x1, x2, dash_length * 2):
            draw.line([(x, y1), (min(x + dash_length, x2), y1)], fill=color, width=width)
        
        for x in range(x1, x2, dash_length * 2):
            draw.line([(x, y2), (min(x + dash_length, x2), y2)], fill=color, width=width)
        
        for y in range(y1, y2, dash_length * 2):
            draw.line([(x1, y), (x1, min(y + dash_length, y2))], fill=color, width=width)
        
        for y in range(y1, y2, dash_length * 2):
            draw.line([(x2, y), (x2, min(y + dash_length, y2))], fill=color, width=width)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "EnhancedLayoutParser_v06_Streamlined": EnhancedLayoutParserNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedLayoutParser_v06_Streamlined": "Enhanced Layout Parser (Streamlined) v06",
}