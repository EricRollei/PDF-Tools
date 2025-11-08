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
    from surya.detection import DetectionPredictor
    from surya.recognition import RecognitionPredictor
    SURYA_AVAILABLE = True
except ImportError:
    print("⚠️ Surya not available - install with: pip install surya-ocr")
    SURYA_AVAILABLE = False

try:
    from surya.layout import LayoutPredictor
    SURYA_LAYOUT_AVAILABLE = True
except ImportError:
    print("⚠️ Surya Layout not available - install with: pip install surya-ocr")
    SURYA_LAYOUT_AVAILABLE = False

try:
    from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
    LAYOUTLMV3_AVAILABLE = True
except ImportError:
    print("⚠️ LayoutLMv3 not available - install with: pip install transformers")
    LAYOUTLMV3_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    print("⚠️ PaddleOCR not available - install with: pip install paddlepaddle paddleocr")
    PADDLEOCR_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    print("⚠️ EasyOCR not available - install with: pip install easyocr")
    EASYOCR_AVAILABLE = False

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
    Self-contained multi-modal layout analysis:
    - Direct Surya integration (no external dependencies)
    - Florence2 for image detection  
    - LayoutLMv3 for semantic understanding
    - Optimized for shared VRAM usage
    """
    
    @classmethod  
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "primary_ocr": (["tesseract", "surya", "paddleocr", "easyocr"], {"default": "tesseract"}),
                "enable_florence2": ("BOOLEAN", {"default": True}),
                "enable_layoutlmv3": ("BOOLEAN", {"default": True}),
                "extract_images": ("BOOLEAN", {"default": True}),
                "extract_text": ("BOOLEAN", {"default": True}),
                "fusion_strategy": (["conservative", "aggressive", "hybrid"], {"default": "aggressive"}),
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
                
                # NEW: Simplified text extraction strategy
                "text_extraction_strategy": ([
                    "surya_layout_primary",           # Surya Layout → Tesseract (primary)
                    "surya_detection_fallback",      # Surya Detection → Tesseract (if layout fails)
                    "florence2_backup_only"          # Florence2 → Tesseract (if both Surya methods fail)
                ], {"default": "surya_layout_primary"}),
                
                "include_text_recognition": ("BOOLEAN", {"default": True}),
                "enable_debug_logging": ("BOOLEAN", {"default": True}),
                
                # Keep these for fine-tuning (but only used in fallback detection grouping)
                "text_merge_distance": ("INT", {"default": 25, "min": 5, "max": 100, "step": 5}),
                "smart_grouping_enabled": ("BOOLEAN", {"default": True}),
                "vertical_merge_threshold": ("INT", {"default": 15, "min": 5, "max": 50}),
                "horizontal_merge_threshold": ("INT", {"default": 10, "min": 5, "max": 30}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LIST", "LIST", "LIST", "LIST", "STRING", "DICT")
    RETURN_NAMES = ("overlay_image", "image_boxes", "text_boxes", "image_crops", "text_crops", "extracted_text", "analysis_results")

    FUNCTION = "analyze_layout"
    CATEGORY = "Enhanced Layout/Analysis"
    
    def __init__(self):
        # Direct Surya integration
        self.surya_detection_predictor = None
        self.surya_recognition_predictor = None
        
        # Other models
        self.florence2_detector = None
        self.layoutlmv3_model = None
        self.layoutlmv3_processor = None
        self.fallback_ocr = None
        
        self.current_image = None
        
        print("🚀 Enhanced Layout Parser v04 - Self-Contained")
        print(f"   VRAM-optimized batch sizes:")
        print(f"   Recognition: {os.environ.get('RECOGNITION_BATCH_SIZE', 'default')}")
        print(f"   Detection: {os.environ.get('DETECTOR_BATCH_SIZE', 'default')}")
        
    def _clear_cuda_cache(self):
        """Clear CUDA cache to free up memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 Cleared CUDA cache")

    def _init_surya_direct(self):
        """Initialize Surya predictors directly with better error recovery"""
        if not SURYA_AVAILABLE:
            print("❌ Surya not available")
            return False
            
        try:
            print("🔧 Initializing Surya predictors directly...")
            
            # Clear cache before loading
            self._clear_cuda_cache()
            
            # Try to initialize with error recovery
            try:
                self.surya_detection_predictor = DetectionPredictor()
            except Exception as e:
                print(f"⚠️ Detection predictor failed: {e}")
                self._clear_cuda_cache()
                # Try again after clearing cache
                self.surya_detection_predictor = DetectionPredictor()
            
            try:
                self.surya_recognition_predictor = RecognitionPredictor()
            except Exception as e:
                print(f"⚠️ Recognition predictor failed: {e}")
                self._clear_cuda_cache()
                # Try again after clearing cache  
                self.surya_recognition_predictor = RecognitionPredictor()
            
            print("✅ Surya predictors initialized successfully")
            
            if torch.cuda.is_available():
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                allocated_gb = torch.cuda.memory_allocated() / 1e9
                print(f"   GPU memory: {memory_gb:.1f}GB total, {allocated_gb:.1f}GB allocated")
                
            return True
            
        except Exception as e:
            print(f"❌ Surya initialization failed: {e}")
            self._clear_cuda_cache()
            return False

    def _run_surya_detection_direct(self, image: Image.Image, confidence_threshold: float = 0.3) -> List[Dict]:
        """Run Surya detection directly with VRAM management"""
        try:
            print("🔍 Running Surya detection (direct)...")
            
            if not self.surya_detection_predictor:
                if not self._init_surya_direct():
                    return []
            
            # Clear cache before processing
            self._clear_cuda_cache()
            
            # Run detection
            detections = self.surya_detection_predictor([image])
            
            # Convert to our format
            ocr_results = []
            for detection in detections:
                # Handle Surya detection object attributes
                bboxes = getattr(detection, 'bboxes', [])
                
                for bbox_obj in bboxes:
                    bbox = getattr(bbox_obj, 'bbox', None)
                    confidence = getattr(bbox_obj, 'confidence', 0.9)
                    
                    if bbox and confidence >= confidence_threshold:
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        ocr_results.append({
                            'text': '',  # Detection only
                            'bbox': bbox,
                            'confidence': confidence,
                            'area': area,
                            'source': 'surya_detection_direct'
                        })
            
            print(f"✅ Surya detection found {len(ocr_results)} regions")
            
            # Clear cache after processing
            self._clear_cuda_cache()
            
            return ocr_results
            
        except torch.OutOfMemoryError:
            print("❌ CUDA Out of Memory in Surya detection")
            self._clear_cuda_cache()
            return []
        except Exception as e:
            print(f"❌ Surya detection failed: {e}")
            self._clear_cuda_cache()
            return []

    def _run_surya_ocr_direct(self, image: Image.Image, confidence_threshold: float = 0.3) -> List[Dict]:
        """Run full Surya OCR directly with VRAM management"""
        try:
            print("🔍 Running Surya OCR (direct)...")
            
            if not self.surya_recognition_predictor or not self.surya_detection_predictor:
                if not self._init_surya_direct():
                    return []
            
            # Clear cache before processing
            self._clear_cuda_cache()
            
            # Run OCR with detection predictor
            ocr_results_obj = self.surya_recognition_predictor([image], det_predictor=self.surya_detection_predictor)
            
            # Convert to our format
            ocr_results = []
            for result in ocr_results_obj:
                text_lines = getattr(result, 'text_lines', [])
                
                for line_obj in text_lines:
                    text = getattr(line_obj, 'text', '').strip()
                    bbox = getattr(line_obj, 'bbox', None)
                    confidence = getattr(line_obj, 'confidence', 0.9)
                    
                    if text and bbox and confidence >= confidence_threshold:
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        ocr_results.append({
                            'text': text,
                            'bbox': bbox,
                            'confidence': confidence,
                            'area': area,
                            'source': 'surya_ocr_direct'
                        })
            
            print(f"✅ Surya OCR extracted {len(ocr_results)} text regions")
            
            # Clear cache after processing
            self._clear_cuda_cache()
            
            return ocr_results
            
        except torch.OutOfMemoryError:
            print("❌ CUDA Out of Memory in Surya OCR")
            self._clear_cuda_cache()
            return []
        except Exception as e:
            print(f"❌ Surya OCR failed: {e}")
            self._clear_cuda_cache()
            return []

    def _init_florence2(self):
        """Initialize Florence2 with VRAM-conscious settings and correct model path"""
        if not FLORENCE2_AVAILABLE:
            print("❌ Florence2 not available")
            return False

        try:
            print("🔧 Initializing Florence2...")
            
            # Clear cache before loading
            self._clear_cuda_cache()
            
            # Use the same approach as the working PDF extractor
            self.florence2_detector = Florence2RectangleDetector(
                model_name="CogFlorence-2.2-Large",  # Just the model name, not full path
                comfyui_base_path=COMFYUI_BASE_PATH,
                min_box_area=1000
            )
            
            print("✅ Florence2 initialized with CogFlorence-2.2-Large")
            return True
            
        except Exception as e:
            print(f"❌ Florence2 initialization failed: {e}")
            
            # Fallback to base model like the PDF extractor does
            try:
                print("🔧 Trying Florence2 with base model...")
                self.florence2_detector = Florence2RectangleDetector(
                    model_name="microsoft/Florence-2-base",
                    comfyui_base_path=COMFYUI_BASE_PATH,
                    min_box_area=1000
                )
                print("✅ Florence2 initialized with base model")
                return True
            except Exception as e2:
                print(f"❌ Florence2 base model also failed: {e2}")
                self._clear_cuda_cache()
                return False

    def _init_layoutlmv3(self):
        """Initialize LayoutLMv3 with VRAM-conscious settings"""
        if not LAYOUTLMV3_AVAILABLE:
            print("❌ LayoutLMv3 not available")
            return False

        try:
            print("🔧 Initializing LayoutLMv3...")
            
            # Clear cache before loading
            self._clear_cuda_cache()
            
            self.layoutlmv3_processor = LayoutLMv3Processor.from_pretrained(
                "microsoft/layoutlmv3-base",
                apply_ocr=False
            )
            
            self.layoutlmv3_model = LayoutLMv3ForTokenClassification.from_pretrained(
                "microsoft/layoutlmv3-base"
            )
            
            if torch.cuda.is_available():
                self.layoutlmv3_model = self.layoutlmv3_model.cuda()
                print("✅ LayoutLMv3 initialized and moved to CUDA")
            
            return True
            
        except Exception as e:
            print(f"❌ LayoutLMv3 initialization failed: {e}")
            return False

    def _init_fallback_ocr(self, ocr_engine="paddleocr"):
        """Initialize fallback OCR"""
        try:
            if ocr_engine == "paddleocr" and PADDLEOCR_AVAILABLE:
                self.fallback_ocr = PaddleOCR(use_angle_cls=True, lang='en')
                print("✅ PaddleOCR fallback initialized")
                return True
            elif ocr_engine == "easyocr" and EASYOCR_AVAILABLE:
                import easyocr
                self.fallback_ocr = easyocr.Reader(['en'])
                print("✅ EasyOCR fallback initialized")
                return True
            else:
                print(f"❌ {ocr_engine} not available")
                return False
        except Exception as e:
            print(f"❌ Fallback OCR initialization failed: {e}")
            return False

    def _run_florence2_detection(self, image: Image.Image, 
                               image_prompt: str = "rectangular images in page",
                               text_prompt: str = "text, caption, paragraph, title") -> Tuple[List[Dict], List[Dict]]:
        """Run Florence2 with VRAM management"""
        try:
            if not self.florence2_detector:
                if not self._init_florence2():
                    return [], []
            
            # Clear cache before processing
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
            
            # Clear cache after processing
            self._clear_cuda_cache()
            
            return florence2_images, florence2_text
            
        except torch.OutOfMemoryError:
            print("❌ CUDA Out of Memory in Florence2")
            self._clear_cuda_cache()
            return [], []
        except Exception as e:
            print(f"❌ Florence2 detection failed: {e}")
            self._clear_cuda_cache()
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


    def _run_layoutlmv3_analysis(self, image: Image.Image, text_boxes: List[Dict]) -> Dict[str, List[Dict]]:
        """Use LayoutLMv3 to classify and confirm text/image regions semantically"""
        try:
            if self.layoutlmv3_model is None:
                if not self._init_layoutlmv3():
                    return {"confirmed_text": [], "confirmed_images": [], "semantic_labels": []}
            
            # Prepare OCR data for LayoutLMv3
            words = []
            boxes = []
            
            for text_box in text_boxes:
                if text_box.get('text', '').strip():
                    # Split text into words and assign same bbox to all words
                    text_words = text_box['text'].split()
                    for word in text_words:
                        words.append(word)
                        boxes.append(text_box['bbox'])  # [x1, y1, x2, y2]
            
            if not words:
                return {"confirmed_text": [], "confirmed_images": [], "semantic_labels": []}
            
            # Process with LayoutLMv3
            encoding = self.layoutlmv3_processor(
                image, 
                words, 
                boxes=boxes, 
                return_tensors="pt",
                padding="max_length",
                truncation=True
            )
            
            # Move to CUDA if available
            if torch.cuda.is_available():
                encoding = {k: v.cuda() for k, v in encoding.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.layoutlmv3_model(**encoding)
                predictions = outputs.logits.argmax(-1).squeeze().tolist()
            
            # Map predictions to semantic labels
            # LayoutLMv3 typically outputs: 0=Other, 1=Header, 2=Question, 3=Answer, etc.
            label_map = {
                0: "other",
                1: "header", 
                2: "title",
                3: "text",
                4: "list",
                5: "table",
                6: "figure"
            }
            
            # Group results by semantic meaning
            confirmed_text = []
            confirmed_images = []
            semantic_labels = []
            
            current_idx = 0
            for text_box in text_boxes:
                if text_box.get('text', '').strip():
                    text_words = text_box['text'].split()
                    
                    # Get most common prediction for this text box
                    box_predictions = predictions[current_idx:current_idx + len(text_words)]
                    most_common_pred = max(set(box_predictions), key=box_predictions.count)
                    semantic_label = label_map.get(most_common_pred, "other")
                    
                    # Enhance the text box with semantic information
                    enhanced_box = {
                        **text_box,
                        'semantic_label': semantic_label,
                        'semantic_confidence': box_predictions.count(most_common_pred) / len(box_predictions)
                    }
                    
                    # Classify as text or potential figure based on semantic label
                    if semantic_label in ["figure", "table"]:
                        confirmed_images.append(enhanced_box)
                    else:
                        confirmed_text.append(enhanced_box)
                    
                    semantic_labels.append({
                        'bbox': text_box['bbox'],
                        'label': semantic_label,
                        'confidence': enhanced_box['semantic_confidence']
                    })
                    
                    current_idx += len(text_words)

            self._clear_cuda_cache()

            print(f"✅ LayoutLMv3 analyzed {len(semantic_labels)} regions")
            return {
                "confirmed_text": confirmed_text,
                "confirmed_images": confirmed_images, 
                "semantic_labels": semantic_labels
            }
            

        except Exception as e:
            print(f"❌ LayoutLMv3 analysis failed: {e}")
            return {"confirmed_text": [], "confirmed_images": [], "semantic_labels": []}

    def _run_fallback_ocr(self, image: Image.Image, ocr_engine: str) -> List[Dict]:
        """Run fallback OCR"""
        try:
            image_array = np.array(image)
            ocr_results = []
            
            if ocr_engine == "paddleocr" and self.fallback_ocr:
                results = self.fallback_ocr.ocr(image_array, cls=True)
                if results and results[0]:
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
                            'area': (x2-x1)*(y2-y1),
                            'source': 'paddleocr'
                        })
            
            elif ocr_engine == "easyocr" and self.fallback_ocr:
                results = self.fallback_ocr.readtext(image_array)
                for (bbox, text, confidence) in results:
                    x1 = int(min([point[0] for point in bbox]))
                    y1 = int(min([point[1] for point in bbox]))
                    x2 = int(max([point[0] for point in bbox]))
                    y2 = int(max([point[1] for point in bbox]))
                    
                    ocr_results.append({
                        'text': text,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'area': (x2-x1)*(y2-y1),
                        'source': 'easyocr'
                    })
            
            print(f"✅ Fallback OCR ({ocr_engine}) found {len(ocr_results)} text regions")
            return ocr_results
            
        except Exception as e:
            print(f"❌ Fallback OCR failed: {e}")
            return []

    def _smart_fusion_with_layoutlmv3(self, 
                                    ocr_results: List[Dict],
                                    florence2_images: List[Dict],
                                    florence2_text: List[Dict], 
                                    cv2_images: List[Dict],
                                    cv2_text: List[Dict],
                                    strategy: str,
                                    min_image_area: int,
                                    min_text_area: int,
                                    surya_confidence: float = 0.3,
                                    florence2_confidence: float = 0.5,
                                    cv2_confidence: float = 0.7,
                                    text_grouping_mode: str = "smart_hybrid",
                                    enable_layoutlmv3: bool = True) -> Tuple[List[Dict], List[Dict]]:
        """Enhanced fusion with LayoutLMv3 validation of Florence2 grouping"""
        
        print(f"🧠 === FUSION WITH LAYOUTLMV3 DEBUG ===")
        print(f"🧠 Input counts: OCR={len(ocr_results)}, F2_img={len(florence2_images)}, F2_txt={len(florence2_text)}")
        
        # First, do the basic fusion with enhanced Florence2 filtering
        final_images, final_text = self._smart_fusion(
            ocr_results, florence2_images, florence2_text, cv2_images, cv2_text,
            strategy, min_image_area, min_text_area,
            surya_confidence, florence2_confidence, cv2_confidence, text_grouping_mode
        )
        
        print(f"🧠 After basic fusion: Images={len(final_images)}, Text={len(final_text)}")
        
        # Then use LayoutLMv3 to validate and enhance Florence2's grouping decisions
        if enable_layoutlmv3 and final_text:
            print("🧠 Running LayoutLMv3 to validate Florence2 text grouping...")
            
            # FIX: Separate Florence2 text boxes for special validation
            florence2_text_boxes = [txt for txt in final_text if 'florence2' in txt.get('detection_methods', [])]
            other_text_boxes = [txt for txt in final_text if 'florence2' not in txt.get('detection_methods', [])]
            
            print(f"🧠 LayoutLMv3 validation targets: {len(florence2_text_boxes)} Florence2 boxes, "
                f"{len(other_text_boxes)} other boxes")
            
            # Filter text boxes that have actual text content for LayoutLMv3
            text_with_content = [txt for txt in final_text if txt.get('text', '').strip()]
            
            if text_with_content:
                layoutlmv3_results = self._run_layoutlmv3_analysis(self.current_image, text_with_content)
                
                semantic_confirmed_text = layoutlmv3_results["confirmed_text"]
                semantic_confirmed_images = layoutlmv3_results["confirmed_images"]
                semantic_labels = layoutlmv3_results["semantic_labels"]
                
                print(f"🧠 LayoutLMv3 results:")
                print(f"   - confirmed_text: {len(semantic_confirmed_text)}")
                print(f"   - confirmed_images: {len(semantic_confirmed_images)}")
                print(f"   - semantic_labels: {len(semantic_labels)}")
                
                # SPECIAL VALIDATION FOR FLORENCE2 TEXT GROUPING
                florence2_validated = 0
                florence2_rejected = 0
                
                for txt in final_text:
                    if 'florence2' in txt.get('detection_methods', []):
                        # Find corresponding semantic result
                        semantic_validation = None
                        for sem_txt in semantic_confirmed_text:
                            overlap_ratio = self._calculate_overlap_ratio(txt['bbox'], sem_txt['bbox'])
                            if overlap_ratio > 0.3:  # Reasonable overlap
                                semantic_validation = sem_txt
                                break
                        
                        if semantic_validation:
                            semantic_label = semantic_validation.get('semantic_label', 'text')
                            semantic_confidence = semantic_validation.get('semantic_confidence', 0.5)
                            
                            # Validate Florence2's grouping decision
                            if semantic_label in ['text', 'header', 'title', 'list'] and semantic_confidence > 0.6:
                                # LayoutLMv3 confirms this is good text grouping
                                txt['semantic_label'] = semantic_label
                                txt['semantic_confidence'] = semantic_confidence
                                txt['layoutlmv3_validation'] = 'confirmed'
                                
                                # Boost confidence for validated Florence2 grouping
                                old_confidence = txt['confidence']
                                txt['confidence'] = min(txt['confidence'] * 1.3, 1.0)  # Bigger boost
                                florence2_validated += 1
                                
                                if 'layoutlmv3' not in txt.get('detection_methods', []):
                                    txt['detection_methods'].append('layoutlmv3')
                                    txt['confidence_sources'].append(semantic_confidence)
                                
                                print(f"🧠 ✅ Validated Florence2 grouping: {semantic_label} "
                                    f"conf {old_confidence:.2f}→{txt['confidence']:.2f}")
                            
                            elif semantic_label in ['figure', 'table']:
                                # Florence2 grouped text that's actually an image region
                                txt['layoutlmv3_validation'] = 'reclassified_as_image'
                                txt['semantic_label'] = semantic_label
                                
                                # Move to images if it meets size criteria
                                if txt.get('area', 0) >= min_image_area:
                                    final_images.append({
                                        **txt,
                                        'detection_methods': ['florence2', 'layoutlmv3'],
                                        'confidence_sources': [txt['confidence'], semantic_confidence],
                                        'source': 'florence2_reclassified',
                                        'reclassification_reason': f'LayoutLMv3 identified as {semantic_label}'
                                    })
                                    print(f"🧠 🔄 Reclassified Florence2 text as {semantic_label}")
                                
                                # Mark for removal from text
                                txt['layoutlmv3_validation'] = 'remove_from_text'
                                florence2_rejected += 1
                            
                            else:
                                # Low confidence or unknown label
                                txt['layoutlmv3_validation'] = 'uncertain'
                                txt['semantic_label'] = semantic_label
                                txt['semantic_confidence'] = semantic_confidence
                        
                        else:
                            # No semantic validation found - might be problematic Florence2 grouping
                            txt['layoutlmv3_validation'] = 'no_semantic_match'
                            # Reduce confidence for unvalidated Florence2 text
                            txt['confidence'] = txt['confidence'] * 0.8
                
                # Remove Florence2 text boxes that were reclassified
                final_text = [txt for txt in final_text if txt.get('layoutlmv3_validation') != 'remove_from_text']
                
                # Process other text boxes normally
                text_enhanced_count = 0
                text_confidence_boosted = 0
                
                for txt in final_text:
                    if 'florence2' not in txt.get('detection_methods', []):  # Non-Florence2 text
                        for sem_txt in semantic_confirmed_text:
                            overlap_ratio = self._calculate_overlap_ratio(txt['bbox'], sem_txt['bbox'])
                            if overlap_ratio > 0.5:
                                txt['semantic_label'] = sem_txt.get('semantic_label', 'text')
                                txt['semantic_confidence'] = sem_txt.get('semantic_confidence', 0.5)
                                text_enhanced_count += 1
                                
                                if txt['semantic_confidence'] > 0.7:
                                    old_confidence = txt['confidence']
                                    txt['confidence'] = min(txt['confidence'] * 1.2, 1.0)
                                    text_confidence_boosted += 1
                                    
                                    if 'layoutlmv3' not in txt.get('detection_methods', []):
                                        txt['detection_methods'].append('layoutlmv3')
                                        txt['confidence_sources'].append(txt['semantic_confidence'])
                                break
                
                # Add semantically detected images
                images_added_by_semantic = 0
                for sem_img in semantic_confirmed_images:
                    if (sem_img['semantic_label'] in ['figure', 'table'] and 
                        sem_img.get('area', 0) >= min_image_area):
                        
                        has_overlap = False
                        for existing_img in final_images:
                            overlap_ratio = self._calculate_overlap_ratio(sem_img['bbox'], existing_img['bbox'])
                            if overlap_ratio > 0.3:
                                has_overlap = True
                                break
                        
                        if not has_overlap:
                            final_images.append({
                                **sem_img,
                                'detection_methods': ['layoutlmv3'],
                                'confidence_sources': [sem_img.get('semantic_confidence', 0.8)],
                                'source': 'layoutlmv3_semantic'
                            })
                            images_added_by_semantic += 1
                
                print(f"🧠 LayoutLMv3 FLORENCE2 VALIDATION SUMMARY:")
                print(f"   - Florence2 text boxes validated: {florence2_validated}")
                print(f"   - Florence2 text boxes rejected/reclassified: {florence2_rejected}")
                print(f"   - Other text boxes enhanced: {text_enhanced_count}")
                print(f"   - Confidence boosts applied: {text_confidence_boosted}")
                print(f"   - Images added by semantic detection: {images_added_by_semantic}")
                
            else:
                print("🧠 No text content available for LayoutLMv3 analysis")
        
        print(f"🧠 Final result: Images={len(final_images)}, Text={len(final_text)}")
        print(f"🧠 === END LAYOUTLMV3 FUSION DEBUG ===")
        return final_images, final_text
        

    def _smart_fusion(self, 
                     ocr_results: List[Dict],
                     florence2_images: List[Dict],
                     florence2_text: List[Dict], 
                     cv2_images: List[Dict],
                     cv2_text: List[Dict],
                     strategy: str,
                     min_image_area: int,
                     min_text_area: int,
                     surya_confidence: float = 0.3,
                     florence2_confidence: float = 0.5,
                     cv2_confidence: float = 0.7,
                     text_grouping_mode: str = "smart_hybrid") -> Tuple[List[Dict], List[Dict]]:
        """Smart fusion with enhanced Florence2 text filtering"""
        
        print(f"🔍 === SMART FUSION DEBUG ===")
        print(f"🔍 Fusion input: OCR={len(ocr_results)}, F2_img={len(florence2_images)}, "
              f"F2_txt={len(florence2_text)}, CV2_img={len(cv2_images)}, CV2_txt={len(cv2_text)}")
        
        # Calculate image dimensions for filtering
        image_width = getattr(self.current_image, 'width', 1000)
        image_height = getattr(self.current_image, 'height', 1000)
        total_image_area = image_width * image_height
        
        print(f"🔍 Image dimensions: {image_width}x{image_height}, total area: {total_image_area}")
        
        # 1. PRIORITIZE FLORENCE2 IMAGES (best performer)
        final_images = []
        florence2_images_added = 0
        for f2_img in florence2_images:
            if (f2_img['area'] >= min_image_area and 
                f2_img['confidence'] >= florence2_confidence):
                final_images.append({
                    **f2_img,
                    'detection_methods': ['florence2'],
                    'confidence_sources': [f2_img['confidence']]
                })
                florence2_images_added += 1
        
        print(f"🔍 Florence2 images added: {florence2_images_added}/{len(florence2_images)}")
        
        # 2. PROCESS TEXT BASED ON GROUPING MODE with enhanced Florence2 filtering
        final_text = []
        surya_text_added = 0
        florence2_text_added = 0
        florence2_text_filtered = 0
        
        # Get grouping parameters
        merge_distance = getattr(self, 'text_merge_distance', 25)
        smart_grouping = getattr(self, 'smart_grouping_enabled', True)
        v_threshold = getattr(self, 'vertical_merge_threshold', 15)
        h_threshold = getattr(self, 'horizontal_merge_threshold', 10)
        
        if text_grouping_mode == "florence2_priority":
            print(f"🔍 Florence2 Priority mode with smart filtering...")
            
            # ENHANCED FLORENCE2 TEXT FILTERING
            filtered_florence2_text = []
            for f2_txt in florence2_text:
                # Calculate relative size to image
                relative_area = f2_txt['area'] / total_image_area
                
                # Filter criteria
                too_large = relative_area > 0.8  # More than 80% of image
                too_small = f2_txt['area'] < min_text_area
                low_confidence = f2_txt['confidence'] < florence2_confidence
                
                # Aspect ratio check (whole page boxes tend to be very tall or very wide)
                bbox = f2_txt['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                aspect_ratio = width / height if height > 0 else 1
                unusual_aspect = aspect_ratio > 5 or aspect_ratio < 0.1  # Very wide or very tall
                
                # Position check (whole page boxes often start near 0,0)
                near_origin = bbox[0] < 50 and bbox[1] < 50
                covers_most_width = width > image_width * 0.9
                covers_most_height = height > image_height * 0.9
                
                if too_large or too_small or low_confidence:
                    florence2_text_filtered += 1
                    print(f"🔍 Filtered Florence2 text: area={f2_txt['area']} ({relative_area:.1%}), "
                          f"conf={f2_txt['confidence']:.2f}, reasons: "
                          f"{'too_large ' if too_large else ''}"
                          f"{'too_small ' if too_small else ''}"
                          f"{'low_conf ' if low_confidence else ''}")
                    continue
                
                if unusual_aspect and (near_origin or covers_most_width or covers_most_height):
                    florence2_text_filtered += 1
                    print(f"🔍 Filtered Florence2 text (whole-page): aspect={aspect_ratio:.2f}, "
                          f"pos=({bbox[0]},{bbox[1]}), size=({width}x{height})")
                    continue
                
                # If it passes all filters, it's good Florence2 text
                filtered_florence2_text.append(f2_txt)
            
            print(f"🔍 Florence2 text filtering: {len(florence2_text)} → {len(filtered_florence2_text)} "
                  f"(filtered out {florence2_text_filtered})")
            
            # Add filtered Florence2 text first
            for f2_txt in filtered_florence2_text:
                final_text.append({
                    **f2_txt,
                    'detection_methods': ['florence2'],
                    'confidence_sources': [f2_txt['confidence']],
                    'text': '',  # Florence2 doesn't provide actual text content
                    'text_extraction_method': 'florence2_bbox_only'
                })
                florence2_text_added += 1
            
            # Then supplement with Surya grouped text for actual content and gaps
            grouped_ocr = self._group_nearby_text(
                ocr_results, 
                merge_distance=merge_distance,
                smart_grouping=smart_grouping,
                vertical_threshold=v_threshold,
                horizontal_threshold=h_threshold
            )
            
            for group in grouped_ocr:
                if (group.get('area', 0) >= min_text_area // 2 and 
                    group['confidence'] >= surya_confidence):
                    
                    # Check if this overlaps significantly with Florence2 text
                    best_florence_match = None
                    best_overlap = 0
                    
                    for f2_txt in final_text:
                        if 'florence2' in f2_txt['detection_methods']:
                            overlap_ratio = self._calculate_overlap_ratio(group['bbox'], f2_txt['bbox'])
                            if overlap_ratio > best_overlap:
                                best_overlap = overlap_ratio
                                best_florence_match = f2_txt
                    
                    if best_overlap > 0.3:  # Significant overlap with Florence2
                        # Enhance Florence2 detection with Surya text content
                        if best_florence_match and not best_florence_match.get('text'):
                            best_florence_match['text'] = group['text']
                            best_florence_match['detection_methods'].append('surya_grouped')
                            best_florence_match['confidence_sources'].append(group['confidence'])
                            best_florence_match['text_extraction_method'] = 'florence2_bbox_surya_text'
                            print(f"🔍 Enhanced Florence2 with Surya text: '{group['text'][:50]}...'")
                    else:
                        # Add as separate Surya text (fills gaps)
                        final_text.append({
                            **group,
                            'detection_methods': ['surya_grouped'],
                            'confidence_sources': [group['confidence']],
                            'text_extraction_method': 'surya_only'
                        })
                        surya_text_added += 1
        
        elif text_grouping_mode == "individual_lines":
            # Individual Surya detections for precise boundaries
            for ocr_result in ocr_results:
                if (ocr_result.get('area', 0) >= min_text_area // 2 and 
                    ocr_result['confidence'] >= surya_confidence):
                    final_text.append({
                        **ocr_result,
                        'detection_methods': [ocr_result['source']],
                        'confidence_sources': [ocr_result['confidence']],
                        'text_extraction_method': 'surya_individual'
                    })
                    surya_text_added += 1
        
        elif text_grouping_mode == "grouped_blocks":
            # Surya grouped blocks only
            grouped_ocr = self._group_nearby_text(
                ocr_results, 
                merge_distance=merge_distance,
                smart_grouping=smart_grouping,
                vertical_threshold=v_threshold,
                horizontal_threshold=h_threshold
            )
            for group in grouped_ocr:
                if (group.get('area', 0) >= min_text_area and 
                    group['confidence'] >= surya_confidence):
                    final_text.append({
                        **group,
                        'detection_methods': ['surya_grouped'],
                        'confidence_sources': [group['confidence']],
                        'text_extraction_method': 'surya_grouped'
                    })
                    surya_text_added += 1
        
        else:  # smart_hybrid
            # Use strategy-based approach
            if strategy == "conservative":
                # High confidence individual lines
                for ocr_result in ocr_results:
                    if (ocr_result.get('area', 0) >= min_text_area and 
                        ocr_result['confidence'] >= surya_confidence + 0.2):
                        final_text.append({
                            **ocr_result,
                            'detection_methods': [ocr_result['source']],
                            'confidence_sources': [ocr_result['confidence']],
                            'text_extraction_method': 'surya_conservative'
                        })
                        surya_text_added += 1
            else:
                # Grouped blocks for aggressive/hybrid
                grouped_ocr = self._group_nearby_text(
                    ocr_results, 
                    merge_distance=merge_distance,
                    smart_grouping=smart_grouping,
                    vertical_threshold=v_threshold,
                    horizontal_threshold=h_threshold
                )
                for group in grouped_ocr:
                    if (group.get('area', 0) >= min_text_area // 2 and 
                        group['confidence'] >= surya_confidence):
                        final_text.append({
                            **group,
                            'detection_methods': ['surya_grouped'],
                            'confidence_sources': [group['confidence']],
                            'text_extraction_method': 'surya_hybrid'
                        })
                        surya_text_added += 1
        
        print(f"🔍 Text results: Florence2={florence2_text_added}, Surya={surya_text_added}")
        print(f"🔍 Florence2 filtering removed {florence2_text_filtered} problematic boxes")
        
        # 3. APPLY STRATEGY-SPECIFIC FILTERING
        pre_filter_images = len(final_images)
        pre_filter_text = len(final_text)
        
        if strategy == "conservative":
            final_images = [img for img in final_images if 
                          len(img['detection_methods']) > 1 or img['confidence'] > 0.8]
            final_text = [txt for txt in final_text if 
                         len(txt['detection_methods']) > 1 or txt['confidence'] > 0.8]
        elif strategy == "aggressive":
            pass  # Keep most detections
        else:  # hybrid
            for img in final_images:
                if len(img['detection_methods']) > 1:
                    img['confidence'] = max(img['confidence_sources'])
                else:
                    img['confidence'] = img['confidence_sources'][0] * 0.9
            
            for txt in final_text:
                if len(txt['detection_methods']) > 1:
                    # Florence2 + Surya combo gets boost
                    txt['confidence'] = max(txt['confidence_sources'])
                else:
                    if 'surya' in txt['detection_methods'][0]:
                        txt['confidence'] = txt['confidence_sources'][0] * 0.95
                    else:
                        txt['confidence'] = txt['confidence_sources'][0] * 0.8
        
        # 4. FINAL FILTERING
        final_images = [img for img in final_images if img['confidence'] > 0.4]
        final_text = [txt for txt in final_text if txt['confidence'] > 0.3]
        
        print(f"🔍 Strategy filtering: Images {pre_filter_images}→{len(final_images)}, Text {pre_filter_text}→{len(final_text)}")
        print(f"🎯 Final fusion result: Images={len(final_images)}, Text={len(final_text)}")
        print(f"🔍 === END SMART FUSION DEBUG ===")
        
        return final_images, final_text

    def _extract_text_from_crop(self, image: Image.Image, bbox: List[int], ocr_engine: str = "tesseract") -> Dict:
        """Extract text using Tesseract with enhanced preprocessing and path detection"""
        try:
            # Crop the region with padding
            padding = 10
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding) 
            x2 = min(image.width, x2 + padding)
            y2 = min(image.height, y2 + padding)
            
            cropped = image.crop((x1, y1, x2, y2))
            
            # Ensure minimum size for OCR
            if cropped.width < 50 or cropped.height < 20:
                return {"text": "", "confidence": 0.0, "method": "too_small"}
            
            # Enhanced preprocessing for Tesseract (based on our successful test)
            enhanced = self._enhance_image_for_tesseract_ocr(cropped)
            
            if ocr_engine == "tesseract":
                return self._extract_with_tesseract_multimode(enhanced)
            elif ocr_engine == "paddleocr":
                return self._extract_with_paddleocr_fallback(enhanced)
            elif ocr_engine == "easyocr":
                return self._extract_with_easyocr_fallback(enhanced)
            else:
                return {"text": "", "confidence": 0.0, "method": f"unsupported_engine_{ocr_engine}"}
                
        except Exception as e:
            print(f"❌ Text extraction failed: {e}")
            return {"text": "", "confidence": 0.0, "method": "error"}

    def _enhance_image_for_tesseract_ocr(self, image: Image.Image) -> Image.Image:
        """Apply Tesseract-optimized enhancements (based on our test results)"""
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Enhance contrast (proven effective in tests)
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.8)
        
        # Enhance sharpness (proven effective in tests)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        
        # Scale up small text (Tesseract works better with larger text)
        if image.width < 300:
            scale = 400 / image.width
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image

    def _extract_with_tesseract_multimode(self, image: Image.Image) -> Dict:
        """Extract text with multiple PSM modes and path detection"""
        try:
            import pytesseract
            
            # Configure Tesseract path with detection logic
            tesseract_cmd = self._find_tesseract_path()
            if not tesseract_cmd:
                return {"text": "", "confidence": 0.0, "method": "tesseract_not_found"}
            
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
            # Try different PSM modes for best results (from our test)
            psm_modes = [6, 3, 8, 11, 13]  # 6 worked best in our test
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
                except Exception as e:
                    continue
            
            if best_text:
                return {
                    "text": best_text,
                    "confidence": 0.8,  # Tesseract doesn't provide confidence
                    "method": f"tesseract_psm_{best_psm}",
                    "word_count": len(best_text.split()),
                    "best_psm": best_psm
                }
            else:
                return {"text": "", "confidence": 0.0, "method": "tesseract_no_results"}
                
        except ImportError:
            print("❌ Tesseract library not available")
            return {"text": "", "confidence": 0.0, "method": "tesseract_library_not_available"}
        except Exception as e:
            print(f"❌ Tesseract extraction failed: {e}")
            return {"text": "", "confidence": 0.0, "method": "tesseract_error"}

    def _find_tesseract_path(self) -> str:
        """Find Tesseract installation with multiple path detection"""
        try:
            import pytesseract
            
            # Common Tesseract installation paths
            common_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract', 
                r'C:\Users\Public\Tesseract-OCR\tesseract',
                r'A:\Tesseract-OCR\tesseract',
                'tesseract'  # System PATH
            ]
            
            # Test each path
            for path in common_paths:
                try:
                    pytesseract.pytesseract.tesseract_cmd = path
                    version = pytesseract.get_tesseract_version()
                    print(f"✅ Found Tesseract at: {path} (version: {version})")
                    return path
                except pytesseract.TesseractNotFoundError:
                    continue
                except Exception:
                    continue
            
            print("❌ Tesseract not found in any common location")
            return None
            
        except ImportError:
            print("❌ pytesseract library not installed")
            return None

    def _extract_with_paddleocr_fallback(self, image: Image.Image) -> Dict:
        """PaddleOCR fallback extraction"""
        try:
            if not PADDLEOCR_AVAILABLE:
                return {"text": "", "confidence": 0.0, "method": "paddleocr_not_available"}
            
            # Use PaddleOCR with correct API
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(lang='en', show_log=False)  # Correct API
            
            img_array = np.array(image)
            ocr_results = ocr.ocr(img_array, cls=True)
            
            if ocr_results and ocr_results[0]:
                texts = []
                confidences = []
                for line in ocr_results[0]:
                    bbox_pts, (text, conf) = line
                    if text.strip() and conf > 0.3:
                        texts.append(text.strip())
                        confidences.append(conf)
                
                if texts:
                    combined_text = ' '.join(texts)
                    avg_confidence = sum(confidences) / len(confidences)
                    return {
                        "text": combined_text,
                        "confidence": avg_confidence,
                        "method": "paddleocr_fallback",
                        "word_count": len(combined_text.split())
                    }
            
            return {"text": "", "confidence": 0.0, "method": "paddleocr_no_results"}
            
        except Exception as e:
            print(f"❌ PaddleOCR fallback failed: {e}")
            return {"text": "", "confidence": 0.0, "method": "paddleocr_error"}

    def _extract_with_easyocr_fallback(self, image: Image.Image) -> Dict:
        """EasyOCR fallback extraction"""
        try:
            if not EASYOCR_AVAILABLE:
                return {"text": "", "confidence": 0.0, "method": "easyocr_not_available"}
            
            import easyocr
            reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            
            img_array = np.array(image)
            results = reader.readtext(img_array, paragraph=False)
            
            if results:
                texts = []
                confidences = []
                for result in results:
                    if len(result) == 3:  # (bbox, text, confidence)
                        _, text, conf = result
                        if text.strip() and conf > 0.3:
                            texts.append(text.strip())
                            confidences.append(conf)
                
                if texts:
                    combined_text = ' '.join(texts)
                    avg_confidence = sum(confidences) / len(confidences)
                    return {
                        "text": combined_text,
                        "confidence": avg_confidence,
                        "method": "easyocr_fallback",
                        "word_count": len(combined_text.split())
                    }
            
            return {"text": "", "confidence": 0.0, "method": "easyocr_no_results"}
            
        except Exception as e:
            print(f"❌ EasyOCR fallback failed: {e}")
            return {"text": "", "confidence": 0.0, "method": "easyocr_error"}
            
            
    def _integrate_basic_surya_detection(self, image: Image.Image, confidence_threshold: float = 0.3) -> List[Dict]:
        """Use basic Surya detection for text regions"""
        try:
            # Use direct Surya detection
            return self._run_surya_detection_direct(image, confidence_threshold)
            
        except Exception as e:
            print(f"❌ Basic Surya detection failed: {e}")
            return []

    def _integrate_proper_surya_ocr_v3(self, image: Image.Image, confidence_threshold: float = 0.3) -> List[Dict]:
        """Enhanced Surya OCR with multiple strategies and fallbacks"""
        try:
            print("🔍 Enhanced Surya OCR with multiple strategies...")
            
            # Strategy 1: Direct Surya OCR (best quality)
            print("🔍 Strategy 1: Direct Surya OCR")
            try:
                ocr_results = self._run_surya_ocr_direct(image, confidence_threshold)
                if ocr_results:
                    print(f"✅ Direct Surya OCR succeeded: {len(ocr_results)} results")
                    return ocr_results
            except torch.OutOfMemoryError:
                print("❌ CUDA Out of Memory in Strategy 1")
                self._clear_cuda_cache()
            except Exception as e:
                print(f"❌ Direct Surya OCR failed: {e}")
            
            # Strategy 2: Detection only + text estimation
            print("🔍 Strategy 2: Detection only with text placeholders")
            try:
                detection_results = self._run_surya_detection_direct(image, confidence_threshold)
                if detection_results:
                    # Add placeholder text for detection-only results
                    for i, result in enumerate(detection_results):
                        if not result.get('text'):
                            result['text'] = f"[Text region {i+1}]"
                    print(f"✅ Detection-only strategy succeeded: {len(detection_results)} results")
                    return detection_results
            except torch.OutOfMemoryError:
                print("❌ CUDA Out of Memory in Strategy 2")
                self._clear_cuda_cache()
            except Exception as e:
                print(f"❌ Detection-only strategy failed: {e}")
            
            # Strategy 3: PaddleOCR hybrid (most reliable fallback)
            print("🔍 Strategy 3: PaddleOCR hybrid")
            try:
                if PADDLEOCR_AVAILABLE:
                    if not hasattr(self, 'fallback_ocr') or self.fallback_ocr is None:
                        self._init_fallback_ocr("paddleocr")
                    
                    if hasattr(self, 'fallback_ocr') and self.fallback_ocr:
                        ocr_results = self._run_fallback_ocr(image, "paddleocr")
                        if ocr_results:
                            print(f"✅ PaddleOCR hybrid succeeded: {len(ocr_results)} results")
                            return ocr_results
            except Exception as e:
                print(f"❌ PaddleOCR hybrid failed: {e}")
            
            print("❌ All Surya OCR strategies failed")
            return []
            
        except Exception as e:
            print(f"❌ Enhanced Surya OCR failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _detect_surya_images(self, image: Image.Image) -> List[Dict]:
        """Use Surya for image detection (alternative to Florence2)"""
        try:
            print("🔍 Trying Surya for image detection...")
            
            # Use detection and look for image-like regions
            detection_results = self._run_surya_detection_direct(image, 0.5)
            
            # Filter detections that might be images based on aspect ratio and size
            image_regions = []
            image_width, image_height = image.size
            total_area = image_width * image_height
            
            for detection in detection_results:
                bbox = detection['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
                aspect_ratio = width / height if height > 0 else 1
                
                # Heuristics for image-like regions
                relative_area = area / total_area
                is_large_enough = area > 10000  # Minimum area for images
                is_reasonable_aspect = 0.2 < aspect_ratio < 5.0  # Not too narrow or wide
                is_not_whole_page = relative_area < 0.9  # Not the entire page
                
                if is_large_enough and is_reasonable_aspect and is_not_whole_page:
                    image_regions.append({
                        'bbox': bbox,
                        'confidence': detection['confidence'] * 0.8,  # Lower confidence for heuristic
                        'area': area,
                        'source': 'surya_image_detection',
                        'label': 'image_region'
                    })
            
            print(f"✅ Surya image detection found {len(image_regions)} potential image regions")
            return image_regions
            
        except Exception as e:
            print(f"❌ Surya image detection failed: {e}")
            return []

    def _run_surya_layout_detection(self, image: Image.Image, confidence_threshold: float = 0.3) -> List[Dict]:
        """Run Surya layout detection with correct API structure"""
        try:
            print("🔍 Running Surya layout detection...")

            if not SURYA_LAYOUT_AVAILABLE:
                print("❌ Surya Layout not available")
                return []

            # Import layout predictor
            from surya.layout import LayoutPredictor
            
            # Clear cache before processing
            self._clear_cuda_cache()
            
            # BYPASS COMPILATION ISSUES
            import torch
            import os
            # Disable compilation to avoid cl.exe issues
            torch._dynamo.config.suppress_errors = True
            os.environ['TORCHDYNAMO_DISABLE'] = '1'
            os.environ['TORCH_COMPILE_DISABLE'] = '1'
            
            print("🔧 Disabled torch compilation to avoid cl.exe issues")
            
            # Initialize layout predictor
            layout_predictor = LayoutPredictor()
            
            # layout_predictions is a list of dicts, one per image
            layout_predictions = layout_predictor([image])
            
            # Convert to our format with semantic labels
            layout_regions = []
            
            # Process each page (should be just one for single image)
            for page_idx, page_data in enumerate(layout_predictions):
                print(f"🔍 Processing page {page_idx}")
                
                # Extract bboxes from the page data
                bboxes = page_data.get('bboxes', [])
                print(f"🔍 Found {len(bboxes)} bboxes in page data")
                
                for bbox_data in bboxes:
                    # Extract the correct fields based on Surya documentation
                    bbox = bbox_data.get('bbox')  # (x1, y1, x2, y2) format
                    label = bbox_data.get('label', 'Text')  # One of the documented labels
                    position = bbox_data.get('position', -1)  # Reading order
                    top_k = bbox_data.get('top_k', {})  # Alternative labels with confidences
                    
                    # Calculate confidence from top_k or use default
                    if top_k and label in top_k:
                        confidence = top_k[label]
                    else:
                        confidence = 0.9  # Default high confidence for layout detection
                    
                    print(f"   🔍 Layout bbox: {label} at {bbox} (conf: {confidence:.3f}, pos: {position})")
                    
                    if bbox and confidence >= confidence_threshold:
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        layout_regions.append({
                            'text': '',  # Layout detection only provides structure, not text
                            'bbox': bbox,
                            'confidence': confidence,
                            'area': area,
                            'source': 'surya_layout',
                            'semantic_label': label,
                            'reading_position': position,
                            'polygon': bbox_data.get('polygon'),  # Keep original polygon if available
                            'top_k_labels': top_k  # Keep alternative labels
                        })
            
            # Sort by reading order (position)
            layout_regions.sort(key=lambda x: x['reading_position'] if x['reading_position'] >= 0 else 999)
            
            print(f"✅ Surya layout found {len(layout_regions)} semantic regions")
            
            # Debug: show region types and their confidence
            if layout_regions:
                label_counts = {}
                for region in layout_regions:
                    label = region['semantic_label']
                    label_counts[label] = label_counts.get(label, 0) + 1
                
                print(f"   Layout elements: {dict(label_counts)}")
                
                # Show top 5 regions with details
                print(f"   Top 5 regions:")
                for i, region in enumerate(layout_regions[:5]):
                    print(f"     {i+1}. {region['semantic_label']} at pos {region['reading_position']} "
                        f"(conf: {region['confidence']:.3f}, area: {region['area']})")
            
            # Clear cache after processing
            self._clear_cuda_cache()
            
            return layout_regions
            
        except Exception as e:
            print(f"❌ Surya layout detection failed: {e}")
            import traceback
            traceback.print_exc()
            self._clear_cuda_cache()
            return []
            
            
                    
    def _group_nearby_text(self, ocr_results: List[Dict], merge_distance: int = 25, smart_grouping: bool = True, vertical_threshold: int = 15, horizontal_threshold: int = 10) -> List[Dict]:
        """Enhanced text grouping with smarter distance calculations"""
        if not ocr_results:
            return []
        
        print(f"🔍 Grouping {len(ocr_results)} OCR results with merge_distance={merge_distance}")
        
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
            group_areas = [result.get('area', 0)]
            
            # Find nearby text with smart grouping
            for j, other in enumerate(ocr_results):
                if j <= i or j in used:
                    continue
                
                if smart_grouping:
                    # Use different thresholds for vertical vs horizontal merging
                    should_merge = self._smart_should_merge(result['bbox'], other['bbox'], 
                                                          vertical_threshold, horizontal_threshold, merge_distance)
                else:
                    # Use simple distance
                    distance = self._calculate_bbox_distance(result['bbox'], other['bbox'])
                    should_merge = distance <= merge_distance
                
                if should_merge:
                    group_texts.append(other['text'])
                    group_bbox = self._merge_bboxes(group_bbox, other['bbox'])
                    group_confidence = min(group_confidence, other['confidence'])
                    group_indices.add(j)
                    group_areas.append(other.get('area', 0))
            
            # Add indices to used set
            used.update(group_indices)
            
            # Create group with enhanced metadata
            total_area = sum(group_areas)
            groups.append({
                'text': ' '.join(group_texts),
                'bbox': group_bbox,
                'confidence': group_confidence,
                'area': total_area,
                'source': 'surya_grouped',
                'component_count': len(group_texts),
                'merge_distance_used': merge_distance,
                'grouping_method': 'smart' if smart_grouping else 'simple'
            })
        
        print(f"✅ Grouped into {len(groups)} text blocks (reduction: {len(ocr_results)} → {len(groups)})")
        
        # Debug top groups
        for i, group in enumerate(groups[:3]):
            print(f"   Group {i}: {group['component_count']} components, area={group['area']}, text='{group['text'][:50]}...'")
        
        return groups

    def _create_structured_text_output(self, text_regions: List[Dict]) -> str:
        """Create structured multiline text with semantic labels"""
        
        if not text_regions:
            return ""
        
        # Sort by reading position to maintain document flow
        sorted_regions = sorted(text_regions, key=lambda x: x.get('reading_position', x.get('position', 999)))
        
        structured_lines = []
        
        for region in sorted_regions:
            text_content = region.get('text', '').strip()
            if not text_content:
                continue
            
            # Get semantic label (from layout) or fallback to source info
            semantic_label = region.get('semantic_label', 'text')
            
            # Enhanced label mapping for all Surya Layout labels
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
                'Picture': 'image_caption',  # If text is found in image regions
                'Figure': 'figure_caption',
                'Table': 'table_caption'
            }
            
            display_label = label_mapping.get(semantic_label, semantic_label.lower())
            
            # Format the line with label and content
            if '\n' in text_content:
                # Multi-line content - add each line with indentation
                lines = text_content.split('\n')
                structured_lines.append(f"{display_label}: {lines[0]}")
                for line in lines[1:]:
                    if line.strip():
                        structured_lines.append(f"  {line.strip()}")
            else:
                # Single line content
                structured_lines.append(f"{display_label}: {text_content}")
        
        return '\n'.join(structured_lines)
        

        
    def _smart_should_merge(self, bbox1: List[int], bbox2: List[int], 
                           vertical_threshold: int, horizontal_threshold: int, 
                           max_distance: int) -> bool:
        """Smart merging logic that considers text layout patterns"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate centers and dimensions
        x1_center = (x1_min + x1_max) / 2
        y1_center = (y1_min + y1_max) / 2
        x2_center = (x2_min + x2_max) / 2
        y2_center = (y2_min + y2_max) / 2
        
        h1 = y1_max - y1_min
        h2 = y2_max - y2_min
        w1 = x1_max - x1_min
        w2 = x2_max - x2_min

        # Add safety checks for zero dimensions
        if h1 <= 0 or h2 <= 0 or w1 <= 0 or w2 <= 0:
            return False
        
        # Calculate distances
        vertical_distance = abs(y1_center - y2_center)
        horizontal_distance = abs(x1_center - x2_center)
        
        # Check for vertical alignment (same column of text)
        horizontal_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        vertical_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        
        avg_width = (w1 + w2) / 2
        avg_height = (h1 + h2) / 2
        
        # Case 1: Vertically aligned text (same column)
        if horizontal_overlap > avg_width * 0.3:  # Significant horizontal overlap
            return vertical_distance <= vertical_threshold + avg_height * 0.5
        
        # Case 2: Horizontally aligned text (same line)  
        if vertical_overlap > avg_height * 0.3:  # Significant vertical overlap
            return horizontal_distance <= horizontal_threshold + avg_width * 0.2
        
        # Case 3: General proximity
        total_distance = (vertical_distance ** 2 + horizontal_distance ** 2) ** 0.5
        return total_distance <= max_distance


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
    

    def analyze_layout(self, image, primary_ocr="tesseract", enable_florence2=True, 
                    enable_layoutlmv3=True, extract_images=True, extract_text=True,
                    fusion_strategy="aggressive", florence2_image_prompt=None, florence2_text_prompt=None,
                    min_image_area=10000, min_text_area=500, 
                    surya_confidence=0.3, florence2_confidence=0.5,
                    text_extraction_strategy="surya_layout_primary", text_merge_distance=25,
                    include_text_recognition=True, enable_debug_logging=True,
                    smart_grouping_enabled=True, vertical_merge_threshold=15, horizontal_merge_threshold=10):
        """Updated analysis with complete Surya Layout label support"""
        
        # Convert tensor to PIL
        pil_image = tensor_to_PIL(image)
        self.current_image = pil_image
        
        # Store grouping parameters for use in fusion
        self.text_merge_distance = text_merge_distance
        self.smart_grouping_enabled = smart_grouping_enabled
        self.vertical_merge_threshold = vertical_merge_threshold
        self.horizontal_merge_threshold = horizontal_merge_threshold
        
        # Check image size and optimize for VRAM
        image_size = pil_image.width * pil_image.height
        if image_size > 10000000:  # > 10MP
            print(f"⚠️ Large image detected ({pil_image.width}x{pil_image.height}). Processing may be slow.")
            if image_size > 50000000:  # > 50MP
                print("🔧 Resizing image for VRAM efficiency...")
                max_dimension = 4000
                if pil_image.width > max_dimension or pil_image.height > max_dimension:
                    pil_image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                    print(f"✅ Resized to {pil_image.width}x{pil_image.height}")
                    self.current_image = pil_image
        
        if enable_debug_logging:
            print(f"🔍 Enhanced Layout Analysis starting (Surya Layout + Tesseract):")
            print(f"  Image size: {pil_image.size}")
            print(f"  Primary OCR: {primary_ocr}")
            print(f"  Models: Florence2={enable_florence2}, LayoutLMv3={enable_layoutlmv3}")
            print(f"  Text extraction strategy: {text_extraction_strategy}")
            print(f"  Text recognition: {include_text_recognition}")
        
        # Initialize result containers
        layout_results = []
        surya_backup_text = []
        florence2_images = []
        florence2_text = []
        
        # 1. ALWAYS RUN SURYA LAYOUT DETECTION FIRST (our primary strategy)
        print("🔍 Running Surya layout detection (primary strategy)...")
        layout_results = self._run_surya_layout_detection(pil_image, surya_confidence)
        
        # COMPLETE SURYA LAYOUT LABELS (based on your testing + documentation)
        text_like_labels = [
            'Text',              # Main text content
            'SectionHeader',     # Section headers
            'PageHeader',        # Page headers
            'PageFooter',        # Page footers  
            'Caption',           # Image/figure captions
            'ListItem',          # List items
            'TableOfContents',   # Table of contents
            'Title',             # Document titles (from docs)
            'Handwriting',       # Handwritten text
            'Footnote',          # Footnotes (from docs)
            'Formula',           # Mathematical formulas (from docs)
            'Form',              # Form fields (from docs)
            'TextInlineMath'     # Inline math text (from docs)
        ]
        
        image_like_labels = [
            'Picture',           # Images/photos
            'Figure',            # Figures/diagrams
            'Table',             # Tables
        ]
        
        layout_text_regions = [r for r in layout_results if r['semantic_label'] in text_like_labels]
        layout_image_regions = [r for r in layout_results if r['semantic_label'] in image_like_labels]
        
        print(f"✅ Layout detection: {len(layout_text_regions)} text regions, {len(layout_image_regions)} image regions")
        
        # DEBUG: Show what labels we actually got
        if enable_debug_logging and layout_results:
            actual_labels = [r['semantic_label'] for r in layout_results]
            label_counts = {}
            for label in actual_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            print(f"🔍 Actual layout labels found: {dict(label_counts)}")
            print(f"🔍 Text regions after filtering: {len(layout_text_regions)}")
            print(f"🔍 Image regions after filtering: {len(layout_image_regions)}")
            
            # Show which labels were used
            text_labels_found = [r['semantic_label'] for r in layout_text_regions]
            image_labels_found = [r['semantic_label'] for r in layout_image_regions]
            print(f"🔍 Text labels matched: {set(text_labels_found)}")
            print(f"🔍 Image labels matched: {set(image_labels_found)}")
        
        # 2. DETERMINE TEXT EXTRACTION APPROACH based on strategy
        text_regions_to_process = []
        extraction_method_used = ""
        
        if text_extraction_strategy == "surya_layout_primary":
            if layout_text_regions:
                # SUCCESS: Use layout regions
                text_regions_to_process = layout_text_regions
                extraction_method_used = "surya_layout"
                print(f"✅ Using Surya Layout regions: {len(text_regions_to_process)} regions")
                print(f"   📋 Layout region types: {[r.get('semantic_label', 'unknown') for r in text_regions_to_process[:5]]}")
            else:
                # FALLBACK: Use detection only if layout completely failed
                print("🔧 No layout regions found, falling back to detection...")
                surya_backup_text = self._run_surya_detection_direct(pil_image, surya_confidence)
                text_regions_to_process = surya_backup_text
                extraction_method_used = "surya_detection_fallback"
                print(f"✅ Using Surya Detection fallback: {len(text_regions_to_process)} regions")
        
        elif text_extraction_strategy == "surya_detection_fallback":
            # Force use of detection instead of layout (user choice)
            print("🔧 Using Surya Detection by user choice...")
            surya_backup_text = self._run_surya_detection_direct(pil_image, surya_confidence)
            text_regions_to_process = surya_backup_text
            extraction_method_used = "surya_detection_forced"
            print(f"✅ Using Surya Detection: {len(text_regions_to_process)} regions")
        
        elif text_extraction_strategy == "florence2_backup_only":
            # Skip Surya entirely, use Florence2 (user choice)
            print("🔧 Using Florence2 by user choice...")
            if enable_florence2:
                florence2_images, florence2_text = self._run_florence2_detection(pil_image, 
                    florence2_image_prompt or "rectangular images in page", 
                    florence2_text_prompt or "text, caption, paragraph, title")
                text_regions_to_process = florence2_text
                extraction_method_used = "florence2_forced"
                print(f"✅ Using Florence2: {len(text_regions_to_process)} regions")
            else:
                print("❌ Florence2 disabled, no text regions available")
                text_regions_to_process = []
                extraction_method_used = "none"
        
        # 3. FINAL FALLBACK: Only if everything else failed
        if not text_regions_to_process and text_extraction_strategy != "florence2_backup_only":
            print("🔧 No regions found with primary strategy, trying Florence2 backup...")
            if enable_florence2:
                florence2_images, florence2_text = self._run_florence2_detection(pil_image, 
                    florence2_image_prompt or "rectangular images in page", 
                    florence2_text_prompt or "text, caption, paragraph, title")
                text_regions_to_process = florence2_text
                extraction_method_used = extraction_method_used + "_then_florence2_backup"
                print(f"✅ Florence2 backup: {len(text_regions_to_process)} regions")
        
        # 4. EXTRACT TEXT USING TESSERACT
        final_text_regions = []
        if include_text_recognition and text_regions_to_process:
            print(f"🔍 Extracting text using {primary_ocr} from {len(text_regions_to_process)} regions...")
            print(f"   Method used: {extraction_method_used}")
            
            successful_extractions = 0
            total_characters = 0
            
            for i, region in enumerate(text_regions_to_process):
                # Extract text using Tesseract
                extraction_result = self._extract_text_from_crop(pil_image, region['bbox'], primary_ocr)
                
                if extraction_result['text'] and len(extraction_result['text']) > 2:
                    # Success - add enhanced region with CORRECT detection method labels
                    enhanced_region = {
                        **region,
                        'text': extraction_result['text'],
                        'extraction_confidence': extraction_result['confidence'],
                        'extraction_method': extraction_result['method'],
                        'text_quality': 'high' if extraction_result['confidence'] > 0.7 else 'medium',
                        # FIX: Use the actual extraction method, not hardcoded values
                        'detection_methods': [extraction_method_used, primary_ocr],
                        'confidence_sources': [region['confidence'], extraction_result['confidence']],
                        'confidence': (region['confidence'] + extraction_result['confidence']) / 2
                    }
                    
                    final_text_regions.append(enhanced_region)
                    successful_extractions += 1
                    total_characters += len(extraction_result['text'])
                    
                    # Debug high-quality extractions
                    if enable_debug_logging and extraction_result['confidence'] > 0.7:
                        semantic_label = region.get('semantic_label', 'text')
                        preview = extraction_result['text'][:100]
                        print(f"   ✅ {semantic_label}: '{preview}...' (method: {extraction_method_used} + {primary_ocr})")
                
                else:
                    # Failed extraction - keep region but mark as no text
                    failed_region = {
                        **region,
                        'text': "",
                        'extraction_confidence': 0.0,
                        'extraction_method': extraction_result['method'],
                        'text_quality': 'failed',
                        # FIX: Use actual extraction method for failed regions too
                        'detection_methods': [extraction_method_used],
                        'confidence_sources': [region['confidence']],
                        'confidence': region['confidence'] * 0.5
                    }
                    final_text_regions.append(failed_region)
            
            print(f"📝 Tesseract extraction summary:")
            print(f"   ✅ Successful: {successful_extractions}/{len(text_regions_to_process)} regions")
            print(f"   📊 Total characters extracted: {total_characters}")
            print(f"   🎯 Success rate: {successful_extractions/len(text_regions_to_process)*100:.1f}%")
            print(f"   🏷️  Detection method used: {extraction_method_used}")
        
        else:
            # No text recognition - just use detection regions
            final_text_regions = text_regions_to_process
            for region in final_text_regions:
                # FIX: Use correct method labels even for no-text-recognition mode
                region['detection_methods'] = [extraction_method_used]
                region['confidence_sources'] = [region['confidence']]
            
        # 5. RUN FLORENCE2 FOR IMAGES (if not already done)
        if enable_florence2 and not florence2_images:
            print("🔍 Running Florence2 for image detection...")
            florence2_images, _ = self._run_florence2_detection(pil_image, 
                florence2_image_prompt or "rectangular images in page", 
                florence2_text_prompt or "text, caption, paragraph, title")
        
        # 6. PROCESS IMAGES: Florence2 primary, Surya layout backup
        final_image_regions = []
        
        # Add Florence2 images (highest quality)
        if extract_images and florence2_images:
            for f2_img in florence2_images:
                if f2_img.get('area', 0) >= min_image_area and f2_img['confidence'] >= florence2_confidence:
                    enhanced_f2_img = {
                        **f2_img,
                        'detection_methods': ['florence2'],
                        'confidence_sources': [f2_img['confidence']],
                        'semantic_label': 'image'
                    }
                    final_image_regions.append(enhanced_f2_img)
            
            # Add Surya layout images (if not overlapping)
            for layout_img in layout_image_regions:
                if layout_img.get('area', 0) >= min_image_area:
                    # Check for overlap with Florence2 images
                    has_overlap = False
                    for f2_img in final_image_regions:
                        overlap_ratio = self._calculate_overlap_ratio(layout_img['bbox'], f2_img['bbox'])
                        if overlap_ratio > 0.3:
                            has_overlap = True
                            # Enhance Florence2 with Surya semantic info
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
        
        # 7. FILTER REGIONS BY MINIMUM AREAS
        image_boxes = [img for img in final_image_regions if img.get('area', 0) >= min_image_area] if extract_images else []
        text_boxes = [txt for txt in final_text_regions if txt.get('area', 0) >= min_text_area] if extract_text else []
        
        print(f"🎯 Final results: {len(image_boxes)} images, {len(text_boxes)} text regions")
        
        # 8. CREATE STRUCTURED TEXT OUTPUT
        extracted_text = ""
        if include_text_recognition and text_boxes:
            # Create structured output with semantic labels
            extracted_text = self._create_structured_text_output(text_boxes)
            
            print(f"📝 Created structured text output: {len(extracted_text)} characters")
            
            if enable_debug_logging and extracted_text:
                print("📝 Sample structured text:")
                sample_lines = extracted_text.split('\n')[:5]
                for i, line in enumerate(sample_lines):
                    print(f"   Line {i}: '{line}'")
        
        # 9. CREATE VISUALIZATION
        overlay = self._create_enhanced_overlay(pil_image, image_boxes, text_boxes)
        overlay_tensor = PIL_to_tensor(overlay)
        
        # 10. EXTRACT CROPS
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
        
        # 11. ANALYSIS RESULTS
        analysis_results = {
            "total_detections": len(image_boxes) + len(text_boxes),
            "image_detections": len(image_boxes),
            "text_detections": len(text_boxes),
            "extraction_strategy": "surya_layout_tesseract",
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
                "surya_detection_backup": len(surya_backup_text) > 0,
                "florence2": len(florence2_images) + len(florence2_text) > 0,
                "tesseract": include_text_recognition,
                "layoutlmv3": enable_layoutlmv3
            },
            "extraction_statistics": {
                "layout_text_regions": len(layout_text_regions),
                "layout_image_regions": len(layout_image_regions),
                "florence2_images": len(florence2_images),
                "florence2_text_backup": len(florence2_text),
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
            print(f"   🎯 Success rate: {analysis_results['extraction_statistics']['successful_text_extractions']}/{len(text_boxes) if text_boxes else 0}")
        
        return (overlay_tensor, image_boxes, text_boxes, image_crops, text_crops, extracted_text, analysis_results)

    
    def _create_enhanced_overlay(self, image: Image.Image, image_boxes: List[Dict], text_boxes: List[Dict]) -> Image.Image:
        """Create enhanced visualization showing detection sources"""
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        
        # Initialize font
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        # Enhanced color coding for all Surya Layout labels
        colors = {
            'florence2': (255, 0, 0),          # Red
            'surya_layout': (0, 255, 100),     # Bright Green - Main Surya Layout
            'surya_detection_fallback': (0, 200, 0),  # Dark Green - Fallback detection
            'surya_detection_forced': (0, 180, 0),    # Darker Green - User chose detection
            'tesseract': (50, 255, 50),        # Light Green
            'paddleocr': (0, 150, 255),        # Light Blue
            'easyocr': (100, 150, 255),        # Lighter Blue
            'layoutlmv3': (255, 0, 255),       # Magenta
            'multi': (255, 165, 0),            # Orange for multi-method
            
            # Semantic label specific colors (for when we want to show label types)
            'SectionHeader': (255, 100, 0),    # Orange
            'PageHeader': (255, 150, 0),       # Light Orange  
            'PageFooter': (200, 100, 0),       # Dark Orange
            'Caption': (150, 255, 150),        # Light Green
            'ListItem': (100, 200, 255),       # Light Blue
            'TableOfContents': (200, 0, 200),  # Purple
            'Title': (255, 0, 100),            # Pink
            'Handwriting': (100, 100, 255),    # Blue
        }
        
        # Draw text boxes with enhanced labeling
        for i, box in enumerate(text_boxes):
            # Convert bbox to integers
            bbox = [int(coord) for coord in box['bbox']]
            methods = box.get('detection_methods', ['unknown'])
            semantic_label = box.get('semantic_label', '')
            
            # Choose color based on detection methods (primary) or semantic label (secondary)
            if len(methods) > 1:
                color = colors['multi']
                method_label = "+".join(methods)
            else:
                primary_method = methods[0]
                # Use semantic color if available, otherwise method color
                if semantic_label in colors:
                    color = colors[semantic_label]
                else:
                    color = colors.get(primary_method, (128, 128, 128))
                method_label = primary_method
            
            # Draw rectangle (dashed style for text)
            self._draw_dashed_rectangle(draw, bbox, color, width=2)
            
            # Enhanced label with semantic info
            label = f"TXT{i+1}: {method_label} ({box.get('confidence', 0):.2f})"
            if semantic_label:
                label += f" [{semantic_label}]"
            
            # Show text quality indicator
            if box.get('text_quality'):
                quality_indicator = {"high": "🟢", "medium": "🟡", "failed": "🔴"}.get(box['text_quality'], "")
                label += f" {quality_indicator}"
            
            draw.text((bbox[0], max(0, bbox[1] - 20)), label, fill=color, font=font)
        
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
            
            # Add label with semantic info if available
            label = f"IMG{i+1}: {method_label} ({box['confidence']:.2f})"
            if box.get('semantic_label'):
                label += f" [{box['semantic_label']}]"
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
    "EnhancedLayoutParser_v04": EnhancedLayoutParserNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedLayoutParser_v04": "Enhanced Layout Parser (Multi-Modal) v04",
}