"""
Analysis Engine

Description: ComfyUI custom node for PDF tools and media processing
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
Shared Analysis Engine v1.0
Multi-modal content analysis combining Surya Layout + Florence2 + OCR
Used by: Enhanced Layout Parser, PDF Extractor, Rectangle Cropper, etc.
"""

import os
import sys
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple
import torch
import importlib.util


# Add the current directory and parent to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# ComfyUI path detection with fallback
try:
    import folder_paths
    COMFYUI_BASE_PATH = os.path.dirname(folder_paths.models_dir)
    print(f"✅ ComfyUI base path detected: {COMFYUI_BASE_PATH}")
except ImportError:
    # Smart fallback for test scripts
    # Look for ComfyUI directory structure
    possible_comfy_paths = [
        # Try to find ComfyUI from current location
        os.path.abspath(os.path.join(current_dir, "..", "..", "..")),  # Go up from PDF_tools/florence2_scripts
        os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..")),  # Go up one more level
        "A:\\Comfy_Dec\\ComfyUI",  # Your specific path
        "C:\\ComfyUI",  # Common location
        "D:\\ComfyUI",  # Another common location
    ]
    
    COMFYUI_BASE_PATH = "."  # Default fallback
    
    for path in possible_comfy_paths:
        models_dir = os.path.join(path, "models", "LLM")
        if os.path.exists(models_dir):
            COMFYUI_BASE_PATH = path
            print(f"✅ ComfyUI base path auto-detected: {COMFYUI_BASE_PATH}")
            break
    
    if COMFYUI_BASE_PATH == ".":
        print("⚠️ ComfyUI folder_paths not available, using current directory")
        print(f"   Checked paths: {possible_comfy_paths}")

# Surya imports
try:
    from surya.layout import LayoutPredictor
    SURYA_LAYOUT_AVAILABLE = True
    print("✅ Surya layout import successful")
except ImportError as e:
    SURYA_LAYOUT_AVAILABLE = False
    print(f"❌ Surya layout import failed: {e}")

# Florence2 imports with enhanced import strategy
print("🔍 Checking import availability...")

FLORENCE2_AVAILABLE = False
Florence2RectangleDetector = None
BoundingBox = None

def import_florence2_modules():
    """Import Florence2 modules with multiple fallback strategies"""
    global Florence2RectangleDetector, BoundingBox
    
    try:
        # Strategy 1: Direct import (works when everything is in sys.modules)
        try:
            from florence2_detector import Florence2RectangleDetector, BoundingBox
            print("✅ Florence2 import from local florence2_detector successful")
            return True
        except ImportError:
            pass
            
        # Strategy 2: Load florence2_detector with proper module dependencies
        import importlib.util
        import sys
        
        # First, ensure the required modules are loaded
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load configuration_florence2 if not already loaded
        if 'configuration_florence2' not in sys.modules:
            config_path = os.path.join(current_dir, "configuration_florence2.py")
            if os.path.exists(config_path):
                spec = importlib.util.spec_from_file_location("configuration_florence2", config_path)
                config_module = importlib.util.module_from_spec(spec)
                sys.modules['configuration_florence2'] = config_module
                spec.loader.exec_module(config_module)
                print("✅ Pre-loaded configuration_florence2")
        
        # Load modeling_florence2 if not already loaded
        if 'modeling_florence2' not in sys.modules:
            modeling_path = os.path.join(current_dir, "modeling_florence2.py")
            if os.path.exists(modeling_path):
                spec = importlib.util.spec_from_file_location("modeling_florence2", modeling_path)
                modeling_module = importlib.util.module_from_spec(spec)
                sys.modules['modeling_florence2'] = modeling_module
                spec.loader.exec_module(modeling_module)
                print("✅ Pre-loaded modeling_florence2")
        
        # Now load florence2_detector
        detector_path = os.path.join(current_dir, "florence2_detector.py")
        if os.path.exists(detector_path):
            spec = importlib.util.spec_from_file_location("florence2_detector", detector_path)
            detector_module = importlib.util.module_from_spec(spec)
            sys.modules['florence2_detector'] = detector_module
            spec.loader.exec_module(detector_module)
            
            Florence2RectangleDetector = detector_module.Florence2RectangleDetector
            BoundingBox = detector_module.BoundingBox
            print("✅ Florence2 import via importlib successful")
            return True
        else:
            print(f"❌ florence2_detector.py not found at {detector_path}")
            return False
            
    except Exception as e:
        print(f"❌ Florence2 import failed: {e}")
        print(f"   Current dir: {current_dir}")
        if os.path.exists(current_dir):
            py_files = [f for f in os.listdir(current_dir) if f.endswith('.py')]
            print(f"   Available .py files: {py_files}")
        return False

# Try to import Florence2
try:
    FLORENCE2_AVAILABLE = import_florence2_modules()
except Exception as e:
    print(f"❌ Florence2 import completely failed: {e}")
    FLORENCE2_AVAILABLE = False

print(f"📊 Import Summary:")
print(f"   Surya Available: {SURYA_LAYOUT_AVAILABLE}")
print(f"   Florence2 Available: {FLORENCE2_AVAILABLE}")


class ContentAnalysisEngine:
    """Shared engine for multi-modal content analysis"""
    
    def __init__(self, enable_surya=True, enable_florence2=True, enable_ocr=True, debug_mode=False):
        self.debug_mode = debug_mode
        
        if debug_mode:
            print(f"🏗️ Initializing ContentAnalysisEngine...")
            print(f"   Surya requested: {enable_surya}, available: {SURYA_LAYOUT_AVAILABLE}")
            print(f"   Florence2 requested: {enable_florence2}, available: {FLORENCE2_AVAILABLE}")
            print(f"   OCR requested: {enable_ocr}")
        
        self.enable_surya = enable_surya and SURYA_LAYOUT_AVAILABLE
        self.enable_florence2 = enable_florence2 and FLORENCE2_AVAILABLE
        self.enable_ocr = enable_ocr
        
        # Initialize models
        self.surya_predictor = None
        self.florence2_detector = None
        
        if self.enable_surya:
            if debug_mode:
                print("🔄 Initializing Surya...")
            success = self._init_surya()
            if not success and debug_mode:
                print("❌ Surya initialization failed")
        
        if self.enable_florence2:
            if debug_mode:
                print("🔄 Initializing Florence2...")
            success = self._init_florence2()
            if not success and debug_mode:
                print("❌ Florence2 initialization failed")
        
        self._report_capabilities()
    
    def _init_surya(self):
        """Initialize Surya Layout predictor with new API"""
        try:
            # Clear environment interference
            import os
            env_vars_to_remove = ['TORCHDYNAMO_DISABLE', 'TORCH_COMPILE_DISABLE', 'PYTORCH_DISABLE_DYNAMO']
            original_env = {}
            
            for var in env_vars_to_remove:
                if var in os.environ:
                    original_env[var] = os.environ[var]
                    del os.environ[var]
            
            # Initialize Surya with new API (requires FoundationPredictor)
            from surya.foundation import FoundationPredictor
            from surya.layout import LayoutPredictor
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            if self.debug_mode:
                print(f"   🔧 Initializing FoundationPredictor (device: {device}, dtype: {dtype})")
            
            # Create foundation predictor first
            foundation_predictor = FoundationPredictor(
                checkpoint=None,  # Use default checkpoint
                device=device,
                dtype=dtype,
                attention_implementation="sdpa"  # Use scaled dot product attention
            )
            
            # Now create layout predictor with foundation predictor
            self.surya_predictor = LayoutPredictor(foundation_predictor=foundation_predictor)
            
            if self.debug_mode:
                print("✅ Surya Layout initialized successfully")
            
            # Restore environment variables
            for var, value in original_env.items():
                os.environ[var] = value
            
            return True  
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Surya Layout initialization failed: {e}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
                print("   📋 Surya layout features will be disabled")
            
            self.surya_predictor = None
            self.enable_surya = False
            return False
    

    def _init_florence2(self):
        """Initialize Florence2 detector with better error handling"""
        if not FLORENCE2_AVAILABLE or Florence2RectangleDetector is None:
            if self.debug_mode:
                print("❌ Florence2 not available - import failed")
            return False
            
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if self.debug_mode:
                print(f"🔍 Attempting Florence2 initialization...")
                print(f"   Base path: {COMFYUI_BASE_PATH}")
                
                # Show available models
                available_models = get_available_florence2_models()
                if available_models:
                    print(f"   Available Florence2 models: {available_models}")
                else:
                    print(f"   No Florence2 models found")
            
            # Try models in order of preference
            model_priorities = [
                "CogFlorence-2.2-Large",
                "Florence-2-base", 
                "microsoft--Florence-2-base",  # Sometimes downloaded with -- instead of /
            ]
            
            # Add any available models that contain 'florence'
            available_models = get_available_florence2_models()
            for model in available_models:
                if model not in model_priorities:
                    model_priorities.append(model)
            
            for model_name in model_priorities:
                try:
                    if self.debug_mode:
                        print(f"   🔄 Trying model: {model_name}")
                    
                    # Use the imported class
                    self.florence2_detector = Florence2RectangleDetector(
                        model_name=model_name,
                        comfyui_base_path=COMFYUI_BASE_PATH,
                        min_box_area=1000
                    )
                    
                    if self.debug_mode:
                        print(f"   ✅ Florence2 {model_name} initialized successfully")
                    return True
                    
                except Exception as e:
                    if self.debug_mode:
                        print(f"   ❌ {model_name} failed: {e}")
                    continue
            
            # If we get here, all models failed
            if self.debug_mode:
                print(f"   ❌ All Florence2 models failed to initialize")
                models_dir = os.path.join(COMFYUI_BASE_PATH, "models", "LLM")
                if os.path.exists(models_dir):
                    all_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
                    print(f"   📂 All directories in models/LLM: {all_dirs}")
                
            self.florence2_detector = None
            self.enable_florence2 = False
            return False
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Florence2 initialization completely failed: {e}")
            self.florence2_detector = None
            self.enable_florence2 = False
            return False
        
    def _report_capabilities(self):
        """Report available capabilities"""
        capabilities = []
        if self.enable_surya:
            capabilities.append("🎯 Surya Layout (semantic detection)")
        if self.enable_florence2:
            capabilities.append("📦 Florence2 (rectangle detection)")
        if self.enable_ocr:
            capabilities.append("📝 OCR (text extraction)")
        
        if self.debug_mode:
            print(f"🚀 Content Analysis Engine ready:")
            for cap in capabilities:
                print(f"   {cap}")
    
    def analyze_image_comprehensive(self, image: Image.Image, 
                                surya_confidence=0.3,
                                florence2_image_prompt="rectangular images in page",
                                florence2_confidence=0.5) -> Dict[str, Any]:
        """Comprehensive image analysis using all available methods"""
        
        results = {
            "surya_layout": [],  # Initialize as empty list instead of None
            "florence2_rectangles": [],  # Initialize as empty list instead of None
            "semantic_regions": {
                "text_regions": [],
                "image_regions": [],
                "caption_regions": [],
                "header_regions": [],
                "other_regions": []
            },
            "analysis_summary": {}
        }
        
        # 1. Surya Layout Analysis
        if self.enable_surya:
            surya_results = self._run_surya_layout(image, surya_confidence)
            results["surya_layout"] = surya_results if surya_results else []
            results["semantic_regions"] = self._categorize_surya_regions(results["surya_layout"])
        
        # 2. Florence2 Rectangle Detection
        if self.enable_florence2:
            florence2_results = self._run_florence2_detection(
                image, florence2_image_prompt, florence2_confidence
            )
            results["florence2_rectangles"] = florence2_results if florence2_results else []
        
        # 3. Create Analysis Summary
        results["analysis_summary"] = self._create_analysis_summary(results)
        
        return results
    
    def _run_surya_layout(self, image: Image.Image, confidence_threshold: float) -> List[Dict]:
        """Run Surya layout detection (proven minimal approach)"""
        if not self.surya_predictor:
            return []
        
        try:
            # FIX: Resize very large images to avoid embedding size mismatch
            # Surya has a maximum size it can handle - resize if needed
            max_dimension = 2048  # Safe maximum for Surya
            original_size = image.size
            needs_resize = False
            
            if max(image.size) > max_dimension:
                needs_resize = True
                # Calculate new size maintaining aspect ratio
                ratio = max_dimension / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
                layout_predictions = self.surya_predictor([resized_image])
                
                # Scale factor to convert resized coordinates back to original
                scale_x = original_size[0] / new_size[0]
                scale_y = original_size[1] / new_size[1]
            else:
                layout_predictions = self.surya_predictor([image])
                scale_x = scale_y = 1.0
            layout_regions = []
            
            for page_result in layout_predictions:
                if hasattr(page_result, 'bboxes'):
                    for layout_box in page_result.bboxes:
                        if hasattr(layout_box, 'bbox') and hasattr(layout_box, 'label'):
                            bbox = layout_box.bbox
                            label = layout_box.label
                            confidence = getattr(layout_box, 'confidence', 0.9)
                            position = getattr(layout_box, 'position', -1)
                            
                            if confidence >= confidence_threshold:
                                # Scale bbox back to original size if image was resized
                                if needs_resize:
                                    bbox = [
                                        bbox[0] * scale_x,
                                        bbox[1] * scale_y,
                                        bbox[2] * scale_x,
                                        bbox[3] * scale_y
                                    ]
                                
                                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                layout_regions.append({
                                    'bbox': bbox,
                                    'confidence': confidence,
                                    'area': area,
                                    'semantic_label': label,
                                    'reading_position': position,
                                    'source': 'surya_layout'
                                })
            
            # Sort by reading order
            layout_regions.sort(key=lambda x: x['reading_position'] if x['reading_position'] >= 0 else 999)
            
            if self.debug_mode:
                print(f"📋 Surya Layout: {len(layout_regions)} regions detected")
            
            return layout_regions
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Surya layout detection failed: {e}")
                # Check if it's a cv2 import issue
                if "cv2" in str(e) or "opencv" in str(e).lower():
                    print("   💡 This may be an OpenCV compatibility issue with Surya")
                    print("   Try: pip install opencv-python-headless --upgrade")
            return []
    
    def _run_florence2_detection(self, image: Image.Image, prompt: str, confidence_threshold: float) -> List[Dict]:
        """Run Florence2 rectangle detection"""
        if not self.florence2_detector or not self.enable_florence2:
            if self.debug_mode:
                print("📦 Florence2 detection skipped (not available)")
            return []
        
        try:
            bounding_boxes, _ = self.florence2_detector.detect_rectangles(
                image=image,
                text_input=prompt,
                return_mask=False,
                keep_model_loaded=True
            )
            
            florence2_regions = []
            for box in bounding_boxes:
                # Handle missing confidence values
                confidence = getattr(box, 'confidence', None)
                if confidence is None:
                    confidence = 1.0  # Default high confidence if not provided
                    
                # Apply confidence threshold
                if confidence >= confidence_threshold:
                    florence2_regions.append({
                        'bbox': box.to_tuple(),
                        'confidence': confidence,
                        'area': box.area,
                        'label': getattr(box, 'label', 'rectangular_image'),  # Safe label access
                        'source': 'florence2'
                    })
            
            if self.debug_mode:
                print(f"📦 Florence2: {len(florence2_regions)} rectangles detected (threshold: {confidence_threshold})")
                for i, region in enumerate(florence2_regions):
                    print(f"    📦 Region {i+1}: {region['bbox']}, conf: {region['confidence']:.2f}")
            
            return florence2_regions
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Florence2 detection failed: {e}")
                import traceback
                traceback.print_exc()
            return []
    
    def _categorize_surya_regions(self, layout_regions: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize Surya regions by semantic type"""
        categorized = {
            "text_regions": [],
            "image_regions": [],
            "caption_regions": [],
            "header_regions": [],
            "other_regions": []
        }
        
        text_labels = ['Text', 'Footnote', 'Formula', 'TextInlineMath']
        image_labels = ['Picture', 'Figure', 'Table']
        caption_labels = ['Caption']
        header_labels = ['SectionHeader', 'PageHeader', 'PageFooter', 'Title']
        
        for region in layout_regions:
            label = region['semantic_label']
            if label in text_labels:
                categorized["text_regions"].append(region)
            elif label in image_labels:
                categorized["image_regions"].append(region)
            elif label in caption_labels:
                categorized["caption_regions"].append(region)
            elif label in header_labels:
                categorized["header_regions"].append(region)
            else:
                categorized["other_regions"].append(region)
        
        return categorized
    
    def _create_analysis_summary(self, results: Dict) -> Dict[str, Any]:
        """Create summary of analysis results"""
        surya_regions = results.get("surya_layout", [])
        florence2_regions = results.get("florence2_rectangles", [])
        semantic_regions = results.get("semantic_regions", {})
        
        # Ensure we have lists, not None values
        if surya_regions is None:
            surya_regions = []
        if florence2_regions is None:
            florence2_regions = []
        
        return {
            "total_surya_regions": len(surya_regions),
            "total_florence2_regions": len(florence2_regions),
            "text_regions_count": len(semantic_regions.get("text_regions", [])),
            "image_regions_count": len(semantic_regions.get("image_regions", [])),
            "caption_regions_count": len(semantic_regions.get("caption_regions", [])),
            "header_regions_count": len(semantic_regions.get("header_regions", [])),
            "has_semantic_data": len(surya_regions) > 0,
            "has_rectangle_data": len(florence2_regions) > 0,
            "analysis_methods": [
                method for method, enabled in [
                    ("surya_layout", self.enable_surya and len(surya_regions) > 0),
                    ("florence2", self.enable_florence2 and len(florence2_regions) > 0),
                    ("ocr", self.enable_ocr)
                ] if enabled
            ]
        }
        
    def extract_text_from_regions(self, image: Image.Image, text_regions: List[Dict], 
                                 ocr_engine: str = "tesseract") -> List[Dict]:
        """Extract text from detected regions using OCR"""
        if not (self.enable_ocr and text_regions):
            return []
        
        extracted_texts = []
        
        for region in text_regions:
            bbox = region['bbox']
            text_result = self._extract_text_from_bbox(image, bbox, ocr_engine)
            
            if text_result['text'].strip():
                extracted_texts.append({
                    **region,
                    'text': text_result['text'],
                    'ocr_confidence': text_result['confidence'],
                    'ocr_method': text_result['method']
                })
        
        return extracted_texts
    
    def _extract_text_from_bbox(self, image: Image.Image, bbox: List[int], 
                               ocr_engine: str) -> Dict[str, Any]:
        """Extract text from bounding box using specified OCR engine"""
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
            
            # Enhance for OCR
            enhanced = self._enhance_for_ocr(cropped)
            
            if ocr_engine == "tesseract":
                return self._tesseract_ocr(enhanced)
            else:
                return {"text": "", "confidence": 0.0, "method": f"unsupported_{ocr_engine}"}
                
        except Exception as e:
            return {"text": "", "confidence": 0.0, "method": "extraction_error"}
    
    def _enhance_for_ocr(self, image: Image.Image) -> Image.Image:
        """Enhance image for better OCR results"""
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
    
    def _tesseract_ocr(self, image: Image.Image) -> Dict[str, Any]:
        """Extract text using Tesseract"""
        try:
            import pytesseract
            
            # Try different PSM modes for best results
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
                    "method": f"tesseract_psm_{best_psm}"
                }
            else:
                return {"text": "", "confidence": 0.0, "method": "tesseract_no_results"}
                
        except ImportError:
            return {"text": "", "confidence": 0.0, "method": "tesseract_not_available"}
        except Exception as e:
            return {"text": "", "confidence": 0.0, "method": "tesseract_error"}
    
    def get_smart_crop_recommendations(self, image: Image.Image, 
                                     avoid_captions=True, 
                                     prefer_images=True) -> List[Dict[str, Any]]:
        """Get smart cropping recommendations based on content analysis"""
        analysis = self.analyze_image_comprehensive(image)
        recommendations = []
        
        image_regions = analysis["semantic_regions"]["image_regions"]
        caption_regions = analysis["semantic_regions"]["caption_regions"]
        
        for img_region in image_regions:
            crop_recommendation = {
                "bbox": img_region["bbox"],
                "confidence": img_region["confidence"],
                "reason": "detected_image_region",
                "semantic_label": img_region["semantic_label"],
                "avoid_regions": []
            }
            
            # Check for nearby captions to avoid
            if avoid_captions:
                for caption in caption_regions:
                    if self._regions_nearby(img_region["bbox"], caption["bbox"], threshold=50):
                        crop_recommendation["avoid_regions"].append({
                            "bbox": caption["bbox"],
                            "type": "caption",
                            "reason": "avoid_cropping_with_image"
                        })
            
            recommendations.append(crop_recommendation)
        
        return recommendations
    
    def _regions_nearby(self, bbox1: List[int], bbox2: List[int], threshold: int) -> bool:
        """Check if two regions are nearby (within threshold pixels)"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate minimum distance between boxes
        if x1_max < x2_min:  # bbox1 is to the left
            horizontal_dist = x2_min - x1_max
        elif x2_max < x1_min:  # bbox1 is to the right
            horizontal_dist = x1_min - x2_max
        else:  # overlapping horizontally
            horizontal_dist = 0
        
        if y1_max < y2_min:  # bbox1 is above
            vertical_dist = y2_min - y1_max
        elif y2_max < y1_min:  # bbox1 is below
            vertical_dist = y1_min - y2_max
        else:  # overlapping vertically
            vertical_dist = 0
        
        distance = (horizontal_dist**2 + vertical_dist**2)**0.5
        return distance <= threshold



def set_comfyui_base_path(path: str):
    """Override ComfyUI base path for testing purposes"""
    global COMFYUI_BASE_PATH
    old_path = COMFYUI_BASE_PATH
    COMFYUI_BASE_PATH = path
    print(f"🔧 ComfyUI base path changed from {old_path} to {COMFYUI_BASE_PATH}")
    
    # Verify the path has models
    models_dir = os.path.join(COMFYUI_BASE_PATH, "models", "LLM")
    if os.path.exists(models_dir):
        available_models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        print(f"✅ Found {len(available_models)} models: {available_models[:3]}...")
        return True
    else:
        print(f"❌ Models directory not found: {models_dir}")
        return False

def get_available_florence2_models():
    """Get list of available Florence2 models"""
    models_dir = os.path.join(COMFYUI_BASE_PATH, "models", "LLM")
    if not os.path.exists(models_dir):
        return []
    
    available_models = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path) and "florence" in item.lower():
            available_models.append(item)
    
    return available_models


# Factory function for easy instantiation
def create_content_analyzer(enable_surya=True, enable_florence2=True, enable_ocr=True, debug_mode=False):
    """Factory function to create a content analyzer instance"""
    return ContentAnalysisEngine(
        enable_surya=enable_surya,
        enable_florence2=enable_florence2, 
        enable_ocr=enable_ocr,
        debug_mode=debug_mode
    )

# Convenience functions for common use cases
def analyze_for_layout_parsing(image: Image.Image, debug_mode=False) -> Dict[str, Any]:
    """Optimized for layout parsing nodes"""
    analyzer = create_content_analyzer(enable_surya=True, enable_florence2=False, debug_mode=debug_mode)
    return analyzer.analyze_image_comprehensive(image)

def analyze_for_pdf_extraction(image: Image.Image, debug_mode=False) -> Dict[str, Any]:
    """Optimized for PDF extraction (needs both Surya + Florence2)"""
    analyzer = create_content_analyzer(enable_surya=True, enable_florence2=True, debug_mode=debug_mode)
    return analyzer.analyze_image_comprehensive(image)

def analyze_for_smart_cropping(image: Image.Image, debug_mode=False) -> List[Dict[str, Any]]:
    """Optimized for smart cropping (avoid captions, find images)"""
    analyzer = create_content_analyzer(enable_surya=True, enable_florence2=True, debug_mode=debug_mode)
    return analyzer.get_smart_crop_recommendations(image, avoid_captions=True, prefer_images=True)

# Add this to your analysis_engine.py for testing:

def test_analysis_engine():
    """Comprehensive test of the analysis engine"""
    print("🧪 Testing Analysis Engine...")
    
    # Test imports
    print(f"   Surya available: {SURYA_LAYOUT_AVAILABLE}")
    print(f"   Florence2 available: {FLORENCE2_AVAILABLE}")
    
    # Create test image
    test_image = Image.new('RGB', (800, 600), color='white')
    
    # Draw some test rectangles
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_image)
    draw.rectangle([100, 100, 300, 200], outline='red', width=3)
    draw.rectangle([400, 150, 600, 350], outline='blue', width=3)
    draw.text((110, 110), "Test text region", fill='black')
    
    # Test analysis
    try:
        analyzer = create_content_analyzer(debug_mode=True)
        result = analyzer.analyze_image_comprehensive(test_image)
        
        print(f"📊 Test Results:")
        print(f"   Surya regions: {len(result.get('surya_layout', []))}")
        print(f"   Florence2 rectangles: {len(result.get('florence2_rectangles', []))}")
        print(f"   Analysis methods: {result.get('analysis_summary', {}).get('analysis_methods', [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_analysis_engine()

