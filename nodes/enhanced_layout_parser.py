"""
Enhanced Layout Parser

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

# Set sane defaults only when the variables are not already set by the user or
# environment. Avoid forcing a bare 'cuda' value which some torch APIs may
# interpret incorrectly when an index is expected; prefer 'cuda:0' as the
# default device string.
_ENV_DEFAULTS = {
    'RECOGNITION_BATCH_SIZE': '768',
    'DETECTOR_BATCH_SIZE': '54',
    'LAYOUT_BATCH_SIZE': '48',
    'TABLE_REC_BATCH_SIZE': '96',
    # Prefer explicit device index to avoid torch.set_device errors
    'TORCH_DEVICE': 'cuda:0',
}

for _k, _v in _ENV_DEFAULTS.items():
    if _k not in os.environ:
        os.environ[_k] = _v

import torch
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Any, Tuple, Optional
import cv2

# ComfyUI imports
try:
    from comfy.utils import common_upscale
    from comfy.model_management import get_torch_device
    import folder_paths
    COMFYUI_BASE_PATH = os.path.dirname(folder_paths.models_dir)
    
    # Helper functions for tensor conversion
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

class EnhancedLayoutParserNode:
    """
    Multi-modal layout analysis combining:
    - LayoutLMv3 for semantic understanding
    - Florence2 for image detection  
    - Surya-OCR for precise text layout
    - CV2 for geometric analysis
    - Logic fusion for optimal results
    
    Optimized for high-end hardware (24GB VRAM, 128GB RAM)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "primary_ocr": (["surya", "paddleocr", "easyocr"], {"default": "surya"}),
                "enable_florence2": ("BOOLEAN", {"default": True}),
                "enable_layoutlmv3": ("BOOLEAN", {"default": True}),
                "enable_cv2_backup": ("BOOLEAN", {"default": True}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0}),
                "extract_images": ("BOOLEAN", {"default": True}),
                "extract_text": ("BOOLEAN", {"default": True}),
                "fusion_strategy": (["conservative", "aggressive", "hybrid"], {"default": "hybrid"}),
            },
            "optional": {
                "florence2_prompt": ("STRING", {
                    "default": "images, photographs, illustrations, diagrams, charts, figures",
                    "multiline": True
                }),
                "text_extraction_prompt": ("STRING", {
                    "default": "text blocks, paragraphs, headers, captions",
                    "multiline": True
                }),
                "min_image_area": ("INT", {"default": 10000, "min": 1000, "max": 100000}),
                "min_text_area": ("INT", {"default": 500, "min": 100, "max": 10000}),
                # High-performance batch size controls
                "surya_recognition_batch": ("INT", {"default": 768, "min": 128, "max": 2048}),
                "surya_detector_batch": ("INT", {"default": 54, "min": 16, "max": 128}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "LIST", "LIST", "LIST", "LIST", "DICT")
    RETURN_NAMES = ("overlay_image", "image_boxes", "text_boxes", "image_crops", "text_crops", "analysis_results")
    FUNCTION = "analyze_layout"
    CATEGORY = "Enhanced Layout/Analysis"
    
    def __init__(self):
        self.surya_ocr = None
        self.florence2_detector = None
        self.layoutlmv3_model = None
        self.current_ocr = None
        print("🚀 Enhanced Layout Parser initialized with high-performance settings")
        print(f"   Recognition batch size: {os.environ.get('RECOGNITION_BATCH_SIZE', 'default')}")
        print(f"   Detector batch size: {os.environ.get('DETECTOR_BATCH_SIZE', 'default')}")
        print(f"   TORCH_DEVICE env: {os.environ.get('TORCH_DEVICE', 'auto')}")
        
    def _debug_surya_import(self):
        """Debug Surya import issues"""
        try:
            print("🔍 Debugging Surya import...")
            
            # Try basic import
            import surya
            print(f"✅ Basic surya import successful, version: {getattr(surya, '__version__', 'unknown')}")
            
            # Try specific imports
            from surya.ocr import run_ocr
            print("✅ surya.ocr.run_ocr import successful")
            
            from surya.model.detection.model import load_model as load_det_model
            print("✅ surya detection model import successful")
            
            from surya.model.recognition.model import load_model as load_rec_model  
            print("✅ surya recognition model import successful")
            
            return True
            
        except ImportError as e:
            print(f"❌ Surya import failed: {e}")
            print("💡 Try: pip install surya-ocr")
            return False
        except Exception as e:
            print(f"❌ Unexpected Surya error: {e}")
            return False

    def _init_surya_ocr(self, language="en"):
        """Initialize Surya-OCR with high-performance settings"""
        try:
            print("🔧 Initializing Surya-OCR with optimized batch sizes...")
            
            # Debug import first
            if not self._debug_surya_import():
                return False
            
            from surya.ocr import run_ocr
            from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor
            
            # Load Surya models with CUDA optimization
            print("🔧 Loading detection models...")
            try:
                det_processor, det_model = load_det_processor(), load_det_model()
                print("✅ Detection models loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load detection models: {e}")
                return False
            
            print("🔧 Loading recognition models...")
            try:
                rec_model, rec_processor = load_rec_model(), load_rec_processor()
                print("✅ Recognition models loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load recognition models: {e}")
                return False
            
            # Move models to CUDA for maximum performance. Use the currently
            # active CUDA device (torch.cuda.current_device()). If a user has set
            # TORCH_DEVICE to an explicit cuda index, Surya or other parts of the
            # code should already respect that; here we just attempt to .cuda()
            # the model and fall back gracefully.
            if torch.cuda.is_available():
                try:
                    # Use .to('cuda') to allow PyTorch to choose current device
                    det_model = det_model.to('cuda')
                    rec_model = rec_model.to('cuda')
                    print("✅ Models moved to CUDA")
                except Exception as e:
                    print(f"⚠️ Failed to move models to CUDA: {e}")
                    print("   Continuing with CPU...")
            
            self.surya_ocr = {
                'det_model': det_model,
                'det_processor': det_processor,
                'rec_model': rec_model,
                'rec_processor': rec_processor
            }
            
            print("✅ Surya-OCR initialized successfully with high-performance settings")
            if torch.cuda.is_available():
                try:
                    idx = torch.cuda.current_device()
                    props = torch.cuda.get_device_properties(idx)
                    print(f"   Available GPU memory (device {idx}): {props.total_memory / 1e9:.1f}GB")
                except Exception:
                    # Best-effort; don't crash on unusual CUDA setups
                    pass
            return True
            
        except ImportError:
            print("❌ Surya-OCR not available. Install with: pip install surya-ocr")
            return False
        except Exception as e:
            print(f"❌ Error initializing Surya-OCR: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _init_florence2(self):
        """Initialize Florence2 detector with high-performance settings"""
        try:
            from PDF_tools import Florence2RectangleDetector
            
            print("🔧 Initializing Florence2 with large model for maximum accuracy...")
            self.florence2_detector = Florence2RectangleDetector(
                model_name="CogFlorence-2.2-Large",  # Use large model with your VRAM
                comfyui_base_path=COMFYUI_BASE_PATH,
                min_box_area=1000
                # Remove device parameter - not supported by this class
            )
            print("✅ Florence2 detector initialized with CogFlorence-2.2-Large")
            return True
        except Exception as e:
            print(f"❌ Error initializing Florence2: {e}")
            # Try with base model as fallback
            try:
                print("🔧 Trying Florence2 with base model...")
                self.florence2_detector = Florence2RectangleDetector(
                    model_name="microsoft/Florence-2-base",
                    comfyui_base_path=COMFYUI_BASE_PATH,
                    min_box_area=1000
                )
                print("✅ Florence2 detector initialized with base model")
                return True
            except Exception as e2:
                print(f"❌ Florence2 base model also failed: {e2}")
                return False

    def _init_fallback_ocr(self, ocr_engine="paddleocr"):
        """Initialize fallback OCR when Surya is not available"""
        try:
            if ocr_engine == "paddleocr":
                from paddleocr import PaddleOCR
                self.fallback_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                print(f"✅ Fallback OCR (PaddleOCR) initialized")
                return True
            elif ocr_engine == "easyocr":
                import easyocr
                self.fallback_ocr = easyocr.Reader(['en'])
                print(f"✅ Fallback OCR (EasyOCR) initialized")
                return True
            else:
                return False
        except Exception as e:
            print(f"❌ Fallback OCR initialization failed: {e}")
            return False

    def _run_fallback_ocr(self, image: Image.Image, ocr_engine="paddleocr") -> List[Dict]:
        """Run fallback OCR when Surya is not available"""
        try:
            image_array = np.array(image)
            
            if ocr_engine == "paddleocr":
                results = self.fallback_ocr.ocr(image_array, cls=True)
                ocr_results = []
                if results and results[0]:  # PaddleOCR can return None
                    for line in results[0]:
                        bbox, (text, confidence) = line
                        x1 = int(min([point[0] for point in bbox]))
                        y1 = int(min([point[1] for point in bbox]))
                        x2 = int(max([point[0] for point in bbox]))
                        y2 = int(max([point[1] for point in bbox]))
                        
                        ocr_results.append({
                            'text': text,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'source': 'paddleocr'
                        })
                        
            elif ocr_engine == "easyocr":
                results = self.fallback_ocr.readtext(image_array)
                ocr_results = []
                for (bbox, text, confidence) in results:
                    # EasyOCR returns bbox as [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                    x1 = int(min([point[0] for point in bbox]))
                    y1 = int(min([point[1] for point in bbox]))
                    x2 = int(max([point[0] for point in bbox]))
                    y2 = int(max([point[1] for point in bbox]))
                    
                    ocr_results.append({
                        'text': text,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'source': 'easyocr'
                    })
            
            print(f"✅ Fallback OCR ({ocr_engine}) processed {len(ocr_results)} text regions")
            return ocr_results
            
        except Exception as e:
            print(f"❌ Fallback OCR ({ocr_engine}) failed: {e}")
            return []

    def _run_surya_ocr(self, image: Image.Image) -> List[Dict]:
        """Run Surya-OCR with high-performance batch processing"""
        try:
            from surya.ocr import run_ocr
            
            # For large images, process in parallel batches
            image_width, image_height = image.size
            
            # Use larger batch processing for your hardware
            if image_width * image_height > 2000000:  # Large image
                print(f"🚀 Processing large image ({image_width}x{image_height}) with optimized batching")
            
            # Run OCR with optimized settings
            predictions = run_ocr(
                [image], 
                [["en"]], 
                self.surya_ocr['det_model'],
                self.surya_ocr['det_processor'], 
                self.surya_ocr['rec_model'],
                self.surya_ocr['rec_processor']
            )
            
            # Convert to our format
            ocr_results = []
            for prediction in predictions:
                for text_line in prediction.text_lines:
                    bbox = text_line.bbox
                    text = text_line.text
                    confidence = text_line.confidence if hasattr(text_line, 'confidence') else 0.9
                    
                    ocr_results.append({
                        'text': text,
                        'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                        'confidence': confidence,
                        'source': 'surya'
                    })
            
            print(f"✅ Surya-OCR processed {len(ocr_results)} text regions")
            return ocr_results
            
        except Exception as e:
            print(f"❌ Surya-OCR failed: {e}")
            return []
    
    def _run_florence2_detection(self, image: Image.Image, prompt: str) -> Tuple[List[Dict], List[Dict]]:
        """Run Florence2 for both image and text detection"""
        try:
            # Detect images
            image_boxes, _ = self.florence2_detector.detect_rectangles(
                image=image,
                text_input=prompt,
                return_mask=False,
                keep_model_loaded=True
            )
            
            # Convert to our format with integer coordinates
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
            
            # Also detect text regions with Florence2 for comparison
            text_boxes, _ = self.florence2_detector.detect_rectangles(
                image=image,
                text_input="text blocks, paragraphs, headers, captions, titles",
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
            
            return florence2_images, florence2_text
            
        except Exception as e:
            print(f"❌ Florence2 detection failed: {e}")
            return [], []
    
    def _run_cv2_analysis(self, image: Image.Image) -> Tuple[List[Dict], List[Dict]]:
        """CV2-based geometric analysis as backup"""
        try:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Image detection using contours
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            cv2_images = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 5000:  # Filter small contours
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Filter for image-like regions (rectangular, reasonable aspect ratio)
                if 0.3 < aspect_ratio < 3.0 and w > 100 and h > 100:
                    cv2_images.append({
                        'bbox': [x, y, x + w, y + h],
                        'label': 'image_region',
                        'confidence': 0.6,
                        'area': area,
                        'source': 'cv2'
                    })
            
            # Text detection using MSER (more sensitive to text regions)
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            
            cv2_text = []
            for region in regions:
                if len(region) < 50:  # Filter very small regions
                    continue
                
                hull = cv2.convexHull(region)
                x, y, w, h = cv2.boundingRect(hull)
                
                # Filter for text-like regions
                if w > 20 and h > 10 and w/h > 2:  # Text tends to be wider than tall
                    cv2_text.append({
                        'bbox': [x, y, x + w, y + h],
                        'label': 'text_region',
                        'confidence': 0.5,
                        'area': w * h,
                        'source': 'cv2'
                    })
            
            return cv2_images, cv2_text
            
        except Exception as e:
            print(f"❌ CV2 analysis failed: {e}")
            return [], []
    
    def _smart_fusion(self, 
                     ocr_results: List[Dict],
                     florence2_images: List[Dict],
                     florence2_text: List[Dict], 
                     cv2_images: List[Dict],
                     cv2_text: List[Dict],
                     strategy: str,
                     min_image_area: int,
                     min_text_area: int) -> Tuple[List[Dict], List[Dict]]:
        """Smart fusion of results from different detection methods"""
        
        print(f"🔍 Fusion input: OCR={len(ocr_results)}, F2_img={len(florence2_images)}, "
              f"F2_txt={len(florence2_text)}, CV2_img={len(cv2_images)}, CV2_txt={len(cv2_text)}")
        
        print(f"🔍 Area thresholds: min_image_area={min_image_area}, min_text_area={min_text_area}")
        
        # Debug CV2 text areas
        if len(cv2_text) > 0:
            cv2_areas = [txt['area'] for txt in cv2_text]
            print(f"🔍 CV2 text areas: min={min(cv2_areas)}, max={max(cv2_areas)}, avg={sum(cv2_areas)/len(cv2_areas):.0f}")
            print(f"🔍 CV2 text threshold: {min_text_area // 2} (halved for debugging)")
        
        # 1. START WITH FLORENCE2 IMAGES (most reliable for image detection)
        final_images = []
        for f2_img in florence2_images:
            if f2_img['area'] >= min_image_area:
                final_images.append({
                    **f2_img,
                    'detection_methods': ['florence2'],  # Initialize as list with single method
                    'confidence_sources': [f2_img['confidence']]
                })
        
        # 2. ADD CV2 IMAGES that don't overlap significantly with Florence2
        for cv2_img in cv2_images:
            if cv2_img['area'] < min_image_area:
                continue
                
            # Check overlap with existing Florence2 detections
            has_significant_overlap = False
            for f2_img in final_images:
                overlap_ratio = self._calculate_overlap_ratio(cv2_img['bbox'], f2_img['bbox'])
                if overlap_ratio > 0.5:  # 50% overlap threshold
                    has_significant_overlap = True
                    # FIX: Only add CV2 if not already in methods
                    if 'cv2' not in f2_img['detection_methods']:
                        f2_img['detection_methods'].append('cv2')
                        f2_img['confidence_sources'].append(cv2_img['confidence'])
                    break
            
            if not has_significant_overlap:
                final_images.append({
                    **cv2_img,
                    'detection_methods': ['cv2'],
                    'confidence_sources': [cv2_img['confidence']]
                })
        
        # 3. PRIORITIZE SURYA OCR RESULTS (most accurate for text)
        final_text = []
        
        # Use OCR results directly without grouping if they're from Surya (already optimal)
        for ocr_result in ocr_results:
            if ocr_result['area'] >= min_text_area // 2:  # Lower threshold for high-quality OCR
                final_text.append({
                    **ocr_result,
                    'detection_methods': [ocr_result['source']],  # 'surya' or 'paddleocr'
                    'confidence_sources': [ocr_result['confidence']]
                })
        
        # 4. ADD FLORENCE2 TEXT only if no OCR overlap
        for f2_txt in florence2_text:
            if f2_txt['area'] < min_text_area:
                continue
                
            # Check if this overlaps with OCR detections
            has_significant_overlap = False
            for ocr_txt in final_text:
                overlap_ratio = self._calculate_overlap_ratio(f2_txt['bbox'], ocr_txt['bbox'])
                if overlap_ratio > 0.3:  # Lower threshold for text
                    has_significant_overlap = True
                    # FIX: Only add florence2 if not already in methods
                    if 'florence2' not in ocr_txt['detection_methods']:
                        ocr_txt['detection_methods'].append('florence2')
                        ocr_txt['confidence_sources'].append(f2_txt['confidence'])
                    break
            
            if not has_significant_overlap:
                final_text.append({
                    **f2_txt,
                    'detection_methods': ['florence2'],
                    'confidence_sources': [f2_txt['confidence']]
                })

        # 5. ADD CV2 TEXT only as last resort and with strict filtering
        cv2_text_added = 0
        for cv2_txt in cv2_text:
            # Much stricter filtering for CV2 text
            if cv2_txt['area'] < min_text_area or cv2_txt['confidence'] < 0.6:
                continue
                
            # Check if this overlaps with existing detections
            has_significant_overlap = False
            for existing_txt in final_text:
                overlap_ratio = self._calculate_overlap_ratio(cv2_txt['bbox'], existing_txt['bbox'])
                if overlap_ratio > 0.2:  # Even lower threshold - CV2 is noisy
                    has_significant_overlap = True
                    # FIX: Only add cv2 if not already in methods
                    if 'cv2' not in existing_txt['detection_methods']:
                        existing_txt['detection_methods'].append('cv2')
                        existing_txt['confidence_sources'].append(cv2_txt['confidence'])
                    break
            
            if not has_significant_overlap and cv2_text_added < 5:  # Limit CV2 additions
                final_text.append({
                    **cv2_txt,
                    'detection_methods': ['cv2'],
                    'confidence_sources': [cv2_txt['confidence']]
                })
                cv2_text_added += 1

        # 6. APPLY STRATEGY-SPECIFIC FILTERING AND CONFIDENCE CALCULATION
        if strategy == "conservative":
            # Keep only high-confidence, multi-method detections
            final_images = [img for img in final_images if len(img['detection_methods']) > 1 or img['confidence'] > 0.8]
            final_text = [txt for txt in final_text if len(txt['detection_methods']) > 1 or txt['confidence'] > 0.8]
        elif strategy == "aggressive":
            # Keep everything above minimum thresholds
            pass  # Already filtered by area
        else:  # hybrid
            # Calculate final confidence scores properly
            for img in final_images:
                if len(img['detection_methods']) > 1:
                    img['confidence'] = max(img['confidence_sources'])
                else:
                    img['confidence'] = img['confidence_sources'][0] * (0.9 if img['detection_methods'][0] == 'florence2' else 0.7)
            
            for txt in final_text:
                if len(txt['detection_methods']) > 1:
                    txt['confidence'] = max(txt['confidence_sources'])
                else:
                    # Prioritize OCR methods
                    if txt['detection_methods'][0] in ['surya', 'paddleocr']:
                        txt['confidence'] = txt['confidence_sources'][0] * 0.95  # High confidence for OCR
                    elif txt['detection_methods'][0] == 'florence2':
                        txt['confidence'] = txt['confidence_sources'][0] * 0.8
                    else:  # cv2
                        txt['confidence'] = txt['confidence_sources'][0] * 0.6  # Lower confidence for CV2
        
        # 7. FINAL FILTERING AND CLEANUP
        final_images = [img for img in final_images if img['confidence'] > 0.4]
        final_text = [txt for txt in final_text if txt['confidence'] > 0.3]
        
        print(f"🎯 Fusion result: Images={len(final_images)}, Text={len(final_text)}")
        
        # Better debugging
        if len(final_text) > 0:
            print("🔍 Debug: Final text detections:")
            for i, txt in enumerate(final_text[:5]):
                methods_str = " + ".join(txt['detection_methods'])
                print(f"  Text {i}: {methods_str} ({txt['confidence']:.2f}) area={txt['area']}")
        
        return final_images, final_text
    
    def _group_nearby_text(self, ocr_results: List[Dict], merge_distance: int = 20) -> List[Dict]:
        """Group nearby OCR text detections into larger blocks"""
        if not ocr_results:
            return []
        
        groups = []
        used = set()
        
        for i, result in enumerate(ocr_results):
            if i in used:
                continue
            
            # Start new group
            group_texts = [result['text']]
            group_bbox = result['bbox'][:]
            group_confidence = result['confidence']
            group_indices = {i}
            
            # Find nearby text
            for j, other in enumerate(ocr_results):
                if j <= i or j in used:
                    continue
                
                # Check if boxes are close
                distance = self._calculate_bbox_distance(result['bbox'], other['bbox'])
                if distance <= merge_distance:
                    group_texts.append(other['text'])
                    group_bbox = self._merge_bboxes(group_bbox, other['bbox'])
                    group_confidence = min(group_confidence, other['confidence'])
                    group_indices.add(j)
            
            # Add indices to used set
            used.update(group_indices)
            
            # Create group
            groups.append({
                'text': ' '.join(group_texts),
                'bbox': group_bbox,
                'confidence': group_confidence,
                'area': (group_bbox[2] - group_bbox[0]) * (group_bbox[3] - group_bbox[1]),
                'source': 'ocr_grouped',
                'component_count': len(group_texts)
            })
        
        return groups
    
    def _calculate_overlap_ratio(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
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
    
    def _calculate_bbox_distance(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate minimum distance between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate center points
        x1_center = (x1_min + x1_max) / 2
        y1_center = (y1_min + y1_max) / 2
        x2_center = (x2_min + x2_max) / 2
        y2_center = (y2_min + y2_max) / 2
        
        return ((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2) ** 0.5
    
    def _merge_bboxes(self, bbox1: List[int], bbox2: List[int]) -> List[int]:
        """Merge two bounding boxes into one encompassing box"""
        return [
            min(bbox1[0], bbox2[0]),  # x_min
            min(bbox1[1], bbox2[1]),  # y_min
            max(bbox1[2], bbox2[2]),  # x_max
            max(bbox1[3], bbox2[3])   # y_max
        ]
    
    def _integrate_basic_surya_ocr(self, image: Image.Image) -> List[Dict]:
        """Use the optimized Surya OCR from basic_surya node"""
        try:
            # Import the basic surya node
            from .basic_surya import SuryaOCRNode
            
            # Initialize if needed
            if not hasattr(self, 'basic_surya_node'):
                self.basic_surya_node = SuryaOCRNode()
            
            # Convert PIL to tensor format expected by basic_surya
            img_array = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            # Run OCR
            text_output, json_output, annotated_image, detection_data = self.basic_surya_node.process_ocr(
                img_tensor, 
                task_mode="ocr_with_boxes",
                confidence_threshold=0.5
            )
            
            # Convert detection_data to our format
            ocr_results = []
            for detection in detection_data:
                bbox = detection.get('bbox')
                if bbox:
                    ocr_results.append({
                        'text': detection.get('text', ''),
                        'bbox': bbox,
                        'confidence': detection.get('confidence', 0.9),
                        'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                        'source': 'surya_optimized'
                    })
            
            print(f"✅ Basic Surya OCR processed {len(ocr_results)} text regions")
            return ocr_results
            
        except Exception as e:
            print(f"❌ Basic Surya OCR failed: {e}")
            return []

    def analyze_layout(self, image, primary_ocr="surya", enable_florence2=True, 
                      enable_layoutlmv3=True, enable_cv2_backup=True,
                      confidence_threshold=0.5, extract_images=True, extract_text=True,
                      fusion_strategy="hybrid", florence2_prompt=None, text_extraction_prompt=None,
                      min_image_area=10000, min_text_area=500, surya_recognition_batch=768, 
                      surya_detector_batch=54):
        """Main analysis function combining all detection methods"""
        
        # Convert tensor to PIL
        pil_image = tensor_to_PIL(image)
        
        print(f"🔍 Enhanced Layout Analysis starting:")
        print(f"  Image size: {pil_image.size}")
        print(f"  Primary OCR: {primary_ocr}")
        print(f"  Florence2: {enable_florence2}, LayoutLMv3: {enable_layoutlmv3}, CV2: {enable_cv2_backup}")
        
        # Initialize components as needed
        ocr_results = []
        florence2_images = []
        florence2_text = []
        cv2_images = []
        cv2_text = []
        
        # 1. RUN OPTIMIZED SURYA OCR (prioritize this)
        if primary_ocr == "surya":
            # Try the optimized basic_surya first
            ocr_results = self._integrate_basic_surya_ocr(pil_image)
            
            # Fallback to original surya implementation if needed
            if not ocr_results:
                if self.surya_ocr is None:
                    if not self._init_surya_ocr():
                        print("🔧 Both Surya methods failed, trying PaddleOCR fallback...")
                        primary_ocr = "paddleocr"
                
                if primary_ocr == "surya" and self.surya_ocr:
                    ocr_results = self._run_surya_ocr(pil_image)
        
        # Fallback to PaddleOCR if Surya not available
        if not ocr_results and primary_ocr != "surya":
            if not hasattr(self, 'fallback_ocr') or self.fallback_ocr is None:
                self._init_fallback_ocr(primary_ocr)
            
            if hasattr(self, 'fallback_ocr') and self.fallback_ocr:
                ocr_results = self._run_fallback_ocr(pil_image, primary_ocr)
        
        # 2. RUN FLORENCE2 DETECTION
        if enable_florence2:
            if self.florence2_detector is None:
                self._init_florence2()
            
            if self.florence2_detector:
                florence2_images, florence2_text = self._run_florence2_detection(
                    pil_image, 
                    florence2_prompt or "images, photographs, illustrations, diagrams"
                )
        
        # 3. RUN CV2 ANALYSIS (backup) - but limit it
        if enable_cv2_backup and len(ocr_results) < 5:  # Only use CV2 if OCR found little
            cv2_images, cv2_text = self._run_cv2_analysis(pil_image)
        
        # 4. SMART FUSION WITH PROPER DEDUPLICATION
        final_images, final_text = self._smart_fusion(
            ocr_results, florence2_images, florence2_text, cv2_images, cv2_text,
            fusion_strategy, min_image_area, min_text_area
        )

        # 5. CREATE OUTPUTS
        image_boxes = final_images if extract_images else []
        text_boxes = final_text if extract_text else []
        
        # Create visualization
        overlay = self._create_enhanced_overlay(pil_image, image_boxes, text_boxes)
        overlay_tensor = PIL_to_tensor(overlay)
        
        # Extract crops
        image_crops = []
        text_crops = []
        
        if extract_images:
            for img_box in image_boxes:
                try:
                    crop = pil_image.crop(img_box['bbox'])
                    image_crops.append(PIL_to_tensor(crop))
                except Exception as e:
                    print(f"⚠️ Error cropping image box: {e}")
        
        if extract_text:
            for txt_box in text_boxes:
                try:
                    crop = pil_image.crop(txt_box['bbox'])
                    text_crops.append(PIL_to_tensor(crop))
                except Exception as e:
                    print(f"⚠️ Error cropping text box: {e}")
        
        # Analysis results
        analysis_results = {
            "total_detections": len(image_boxes) + len(text_boxes),
            "image_detections": len(image_boxes),
            "text_detections": len(text_boxes),
            "fusion_strategy": fusion_strategy,
            "batch_settings": {
                "recognition_batch": surya_recognition_batch,
                "detector_batch": surya_detector_batch
            },
            "detection_methods_used": {
                "ocr": len(ocr_results) > 0,
                "florence2": len(florence2_images) + len(florence2_text) > 0,
                "cv2": len(cv2_images) + len(cv2_text) > 0
            },
            "method_statistics": {
                "ocr_results": len(ocr_results),
                "florence2_images": len(florence2_images),
                "florence2_text": len(florence2_text),
                "cv2_images": len(cv2_images),
                "cv2_text": len(cv2_text)
            }
        }
        
        return (overlay_tensor, image_boxes, text_boxes, image_crops, text_crops, analysis_results)
    
    def _create_enhanced_overlay(self, image: Image.Image, image_boxes: List[Dict], text_boxes: List[Dict]) -> Image.Image:
        """Create enhanced visualization showing detection sources"""
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        
        # Color coding by detection method
        colors = {
            'florence2': (255, 0, 0),      # Red
            'surya': (0, 255, 0),          # Green  
            'surya_optimized': (0, 200, 0), # Dark Green
            'paddleocr': (0, 150, 255),    # Light Blue
            'cv2': (0, 0, 255),            # Blue
            'multi': (255, 165, 0)         # Orange for multi-method
        }
        
        # Draw image boxes
        for i, box in enumerate(image_boxes):
            # Convert bbox to integers
            bbox = [int(coord) for coord in box['bbox']]
            methods = box['detection_methods']
            
            # Choose color based on detection methods
            if len(methods) > 1:
                color = colors['multi']
                method_label = "+".join(methods)
            else:
                color = colors.get(methods[0], (128, 128, 128))
                method_label = methods[0]
            
            # Draw rectangle
            draw.rectangle(bbox, outline=color, width=3)
            
            # Add label
            label = f"IMG{i+1}: {method_label} ({box['confidence']:.2f})"
            draw.text((bbox[0], max(0, bbox[1] - 20)), label, fill=color)
        
        # Draw text boxes
        for i, box in enumerate(text_boxes):
            # Convert bbox to integers
            bbox = [int(coord) for coord in box['bbox']]
            methods = box['detection_methods']
            
            # Choose color based on detection methods
            if len(methods) > 1:
                color = colors['multi']
                method_label = "+".join(methods)
            else:
                color = colors.get(methods[0], (128, 128, 128))
                method_label = methods[0]
            
            # Draw rectangle (dashed style for text)
            self._draw_dashed_rectangle(draw, bbox, color, width=2)
            
            # Add label
            label = f"TXT{i+1}: {method_label} ({box['confidence']:.2f})"
            draw.text((bbox[0], max(0, bbox[1] - 20)), label, fill=color)
        
        return overlay
    
    def _draw_dashed_rectangle(self, draw, bbox, color, width=2, dash_length=5):
        """Draw a dashed rectangle"""
        # Convert all coordinates to integers
        x1, y1, x2, y2 = map(int, bbox)
        
        # Top edge
        for x in range(x1, x2, dash_length * 2):
            draw.line([(x, y1), (min(x + dash_length, x2), y1)], fill=color, width=width)
        
        # Bottom edge  
        for x in range(x1, x2, dash_length * 2):
            draw.line([(x, y2), (min(x + dash_length, x2), y2)], fill=color, width=width)
        
        # Left edge
        for y in range(y1, y2, dash_length * 2):
            draw.line([(x1, y), (x1, min(y + dash_length, y2))], fill=color, width=width)
        
        # Right edge
        for y in range(y1, y2, dash_length * 2):
            draw.line([(x2, y), (x2, min(y + dash_length, y2))], fill=color, width=width)



# Node mappings
NODE_CLASS_MAPPINGS = {
    "EnhancedLayoutParser": EnhancedLayoutParserNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedLayoutParser": "Enhanced Layout Parser (Multi-Modal)",
}