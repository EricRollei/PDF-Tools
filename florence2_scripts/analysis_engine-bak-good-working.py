"""
Analysis Engine-Bak-Good-Working

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
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import torch
import numpy as np

# Set reduced batch sizes for shared usage
os.environ['RECOGNITION_BATCH_SIZE'] = '64'
os.environ['DETECTOR_BATCH_SIZE'] = '8' 
os.environ['LAYOUT_BATCH_SIZE'] = '6'

try:
    import folder_paths
    COMFYUI_BASE_PATH = os.path.dirname(folder_paths.models_dir)
    print(f"✅ ComfyUI base path detected: {COMFYUI_BASE_PATH}")
except ImportError:
    COMFYUI_BASE_PATH = "."
    print("⚠️ ComfyUI folder_paths not available, using current directory")


# Import availability checks
try:
    from surya.layout import LayoutPredictor
    SURYA_LAYOUT_AVAILABLE = True
except ImportError:
    SURYA_LAYOUT_AVAILABLE = False

try:
    from PDF_tools import Florence2RectangleDetector, BoundingBox
    FLORENCE2_AVAILABLE = True
except ImportError:
    FLORENCE2_AVAILABLE = False

class ContentAnalysisEngine:
    """Shared engine for multi-modal content analysis"""
    
    def __init__(self, enable_surya=True, enable_florence2=True, enable_ocr=True, debug_mode=False):
        self.debug_mode = debug_mode
        self.enable_surya = enable_surya and SURYA_LAYOUT_AVAILABLE
        self.enable_florence2 = enable_florence2 and FLORENCE2_AVAILABLE
        self.enable_ocr = enable_ocr
        
        # Initialize models
        self.surya_predictor = None
        self.florence2_detector = None
        
        if self.enable_surya:
            self._init_surya()
        
        if self.enable_florence2:
            self._init_florence2()
        
        self._report_capabilities()
    
    def _init_surya(self):
        """Initialize Surya Layout predictor with minimal approach"""
        try:
            # Clear environment interference
            import os
            env_vars_to_remove = ['TORCHDYNAMO_DISABLE', 'TORCH_COMPILE_DISABLE', 'PYTORCH_DISABLE_DYNAMO']
            original_env = {}
            for var in env_vars_to_remove:
                if var in os.environ:
                    original_env[var] = os.environ[var]
                    del os.environ[var]
            
            # Fresh torch import and reset
            import torch
            if hasattr(torch._dynamo, 'reset'):
                torch._dynamo.reset()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Initialize predictor
            self.surya_predictor = LayoutPredictor()
            
            # Restore environment
            for var, value in original_env.items():
                os.environ[var] = value
            
            if self.debug_mode:
                print("✅ Surya Layout initialized successfully")
                
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Surya Layout initialization failed: {e}")
            self.surya_predictor = None
            self.enable_surya = False
    
    def _init_florence2(self):
        """Initialize Florence2 detector with correct path handling"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Use the same approach as enhanced_layout_parser_v06
            self.florence2_detector = Florence2RectangleDetector(
                model_name="CogFlorence-2.2-Large",
                comfyui_base_path=COMFYUI_BASE_PATH,
                min_box_area=1000
            )
            
            if self.debug_mode:
                print("✅ Florence2 CogFlorence-2.2-Large initialized successfully")
            return True
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Florence2 CogFlorence-2.2-Large failed: {e}")
                print(f"   🔍 Attempted path: {COMFYUI_BASE_PATH}")
                
            # Try base model fallback
            try:
                self.florence2_detector = Florence2RectangleDetector(
                    model_name="microsoft/Florence-2-base",
                    comfyui_base_path=COMFYUI_BASE_PATH,
                    min_box_area=1000
                )
                if self.debug_mode:
                    print("✅ Florence2 base model initialized successfully")
                return True
                
            except Exception as e2:
                if self.debug_mode:
                    print(f"❌ Florence2 base model also failed: {e2}")
                    print("   📋 Florence2 features will be disabled")
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
            layout_predictions = self.surya_predictor([image])
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