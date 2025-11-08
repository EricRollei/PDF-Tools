import os

# Surya OCR performance optimizations for high-end hardware (24GB VRAM, 128GB RAM)
os.environ['RECOGNITION_BATCH_SIZE'] = '768'
os.environ['DETECTOR_BATCH_SIZE'] = '54'
os.environ['LAYOUT_BATCH_SIZE'] = '48'
os.environ['TABLE_REC_BATCH_SIZE'] = '96'
os.environ['TORCH_DEVICE'] = 'cuda'

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
                "enable_cv2_backup": ("BOOLEAN", {"default": False}),
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
                # Separate confidence controls for each method
                "surya_confidence": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.05}),
                "florence2_confidence": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "cv2_confidence": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.05}),
                # Text grouping controls - ENHANCED
                "text_grouping_mode": (["individual_lines", "grouped_blocks", "smart_hybrid", "florence2_priority"], {"default": "florence2_priority"}),
                "text_merge_distance": ("INT", {"default": 25, "min": 5, "max": 100, "step": 5}),  # Made more granular
                # Add text recognition control
                "include_text_recognition": ("BOOLEAN", {"default": True}),
                # Enhanced debugging
                "enable_debug_logging": ("BOOLEAN", {"default": True}),
                # High-performance batch size controls
                "surya_recognition_batch": ("INT", {"default": 768, "min": 128, "max": 2048}),
                "surya_detector_batch": ("INT", {"default": 54, "min": 16, "max": 128}),
                # ADVANCED TEXT GROUPING CONTROLS
                "smart_grouping_enabled": ("BOOLEAN", {"default": True}),
                "vertical_merge_threshold": ("INT", {"default": 15, "min": 5, "max": 50}),  # For line spacing
                "horizontal_merge_threshold": ("INT", {"default": 10, "min": 5, "max": 30}),  # For word spacing
            }
        }

    RETURN_TYPES = ("IMAGE", "LIST", "LIST", "LIST", "LIST", "STRING", "DICT")
    RETURN_NAMES = ("overlay_image", "image_boxes", "text_boxes", "image_crops", "text_crops", "extracted_text", "analysis_results")

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
        print(f"   CUDA device: {os.environ.get('TORCH_DEVICE', 'auto')}")
        
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
            
            # Move models to CUDA for maximum performance
            if torch.cuda.is_available():
                try:
                    det_model = det_model.cuda()
                    rec_model = rec_model.cuda()
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
                print(f"   Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
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

    def _init_layoutlmv3(self):
        """Initialize LayoutLMv3 for semantic document understanding"""
        try:
            from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
            
            print("🔧 Initializing LayoutLMv3 for semantic analysis...")
            
            # Load LayoutLMv3 model for document understanding
            self.layoutlmv3_processor = LayoutLMv3Processor.from_pretrained(
                "microsoft/layoutlmv3-base",
                apply_ocr=False  # We'll provide our own OCR results
            )
            
            self.layoutlmv3_model = LayoutLMv3ForTokenClassification.from_pretrained(
                "microsoft/layoutlmv3-base"
            )
            
            # Move to CUDA if available
            if torch.cuda.is_available():
                self.layoutlmv3_model = self.layoutlmv3_model.cuda()
                print("✅ LayoutLMv3 moved to CUDA")
            
            print("✅ LayoutLMv3 initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ LayoutLMv3 initialization failed: {e}")
            return False

    def _init_fallback_ocr(self, ocr_engine="paddleocr"):
        """Initialize fallback OCR when Surya is not available"""
        try:
            if ocr_engine == "paddleocr":
                from paddleocr import PaddleOCR
                # Remove show_log parameter - it's not supported in newer versions
                self.fallback_ocr = PaddleOCR(use_angle_cls=True, lang='en')
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
    
    def _run_florence2_detection(self, image: Image.Image, 
                               image_prompt: str = "rectangular images in page",
                               text_prompt: str = "text, caption, paragraph, title") -> Tuple[List[Dict], List[Dict]]:
        """Run Florence2 with optimized prompts for both image and text detection"""
        try:
            # Detect images with optimized prompt
            image_boxes, _ = self.florence2_detector.detect_rectangles(
                image=image,
                text_input=image_prompt,
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
            
            # Detect text regions with optimized prompt
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
            
            print(f"✅ LayoutLMv3 analyzed {len(semantic_labels)} regions")
            return {
                "confirmed_text": confirmed_text,
                "confirmed_images": confirmed_images, 
                "semantic_labels": semantic_labels
            }
            
        except Exception as e:
            print(f"❌ LayoutLMv3 analysis failed: {e}")
            return {"confirmed_text": [], "confirmed_images": [], "semantic_labels": []}

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
            
            # Separate Florence2 text boxes for special validation
            florence2_text_boxes = [txt for txt in final_text if 'florence2' in txt['detection_methods']]
            other_text_boxes = [txt for txt in final_text if 'florence2' not in txt['detection_methods']]
            
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
                    if 'florence2' in txt['detection_methods']:
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
                                
                                if 'layoutlmv3' not in txt['detection_methods']:
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
                    if 'florence2' not in txt['detection_methods']:  # Non-Florence2 text
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
                                    
                                    if 'layoutlmv3' not in txt['detection_methods']:
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
    
    def _integrate_basic_surya_detection(self, image: Image.Image, confidence_threshold: float = 0.3) -> List[Dict]:
        """Use DetectionPredictor directly for better text detection coverage"""
        try:
            # Import the basic surya node
            from .basic_surya import SuryaOCRNode
            
            # Initialize if needed
            if not hasattr(self, 'basic_surya_node'):
                self.basic_surya_node = SuryaOCRNode()
            
            # Convert PIL to tensor format expected by basic_surya
            img_array = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            # Run DETECTION ONLY for better coverage
            text_output, json_output, annotated_image, detection_data = self.basic_surya_node.process_ocr(
                img_tensor, 
                task_mode="detection_only",  # This is the key change!
                confidence_threshold=confidence_threshold
            )
            
            print(f"🔍 Debug - Surya Detection returned: text_output={bool(text_output)}, json_output={bool(json_output)}")
            print(f"🔍 Debug - Detection data type: {type(detection_data)}")
            
            # Convert detection_data to our format
            ocr_results = []
            
            # Method 1: Try detection_data directly
            if detection_data and isinstance(detection_data, list):
                print(f"🔍 Debug - Processing {len(detection_data)} detection items")
                for i, detection in enumerate(detection_data):
                    print(f"🔍 Debug - Detection {i}: keys={detection.keys() if isinstance(detection, dict) else 'not dict'}")
                    bbox = detection.get('bbox')
                    confidence = detection.get('confidence', 0.9)
                    
                    if bbox and confidence >= confidence_threshold:
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        ocr_results.append({
                            'text': '',  # Detection only, no text recognition
                            'bbox': bbox,
                            'confidence': confidence,
                            'area': area,
                            'source': 'surya_detection'
                        })
            
            # Method 2: Try JSON parsing
            elif json_output:
                print(f"🔍 Debug - Trying JSON parsing...")
                import json
                try:
                    results_dict = json.loads(json_output)
                    print(f"🔍 Debug - JSON structure: {list(results_dict.keys())}")
                    
                    detections = results_dict.get('detections', [])
                    if detections:
                        print(f"🔍 Debug - Found {len(detections)} detection pages")
                        for page_idx, page_detections in enumerate(detections):
                            if isinstance(page_detections, dict):
                                bboxes = page_detections.get('bboxes', [])
                                print(f"🔍 Debug - Page {page_idx} has {len(bboxes)} bboxes")
                                for bbox_idx, bbox_info in enumerate(bboxes):
                                    bbox = bbox_info.get('bbox')
                                    confidence = bbox_info.get('confidence', 0.9)
                                    
                                    if bbox and confidence >= confidence_threshold:
                                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                        ocr_results.append({
                                            'text': '',
                                            'bbox': bbox,
                                            'confidence': confidence,
                                            'area': area,
                                            'source': 'surya_detection'
                                        })
                                        if bbox_idx < 3:  # Debug first few
                                            print(f"🔍 Debug - Added detection {bbox_idx}: bbox={bbox}, conf={confidence:.2f}")
                    
                    # Also try alternative structures
                    if 'bboxes' in results_dict:
                        print("🔍 Debug - Found direct bboxes in JSON")
                        bboxes = results_dict['bboxes']
                        for bbox_info in bboxes:
                            bbox = bbox_info.get('bbox')
                            confidence = bbox_info.get('confidence', 0.9)
                            if bbox and confidence >= confidence_threshold:
                                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                ocr_results.append({
                                    'text': '',
                                    'bbox': bbox,
                                    'confidence': confidence,
                                    'area': area,
                                    'source': 'surya_detection'
                                })
                        
                except Exception as e:
                    print(f"⚠️ Failed to parse detection JSON: {e}")
                    print(f"🔍 Debug - JSON sample: {json_output[:300]}...")
            
            print(f"✅ Surya Detection found {len(ocr_results)} text regions")
            return ocr_results
            
        except Exception as e:
            print(f"❌ Surya Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _integrate_basic_surya_ocr_with_text(self, image: Image.Image, confidence_threshold: float = 0.3) -> List[Dict]:
        """Use basic_surya for both detection and recognition to get actual text content"""
        try:
            # Import the basic surya node
            from .basic_surya import SuryaOCRNode
            
            # Initialize if needed
            if not hasattr(self, 'basic_surya_node'):
                self.basic_surya_node = SuryaOCRNode()
            
            # Convert PIL to tensor format expected by basic_surya
            img_array = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            # STRATEGY: First get detections, then run OCR for text content
            print("🔍 Strategy: Detection first, then OCR for text content...")
            
            # Step 1: Get high-quality detections
            det_text_output, det_json_output, det_annotated_image, detection_data = self.basic_surya_node.process_ocr(
                img_tensor, 
                task_mode="detection_only",  # Get the best detections first
                confidence_threshold=confidence_threshold
            )
            
            print(f"🔍 Detection step: found {len(detection_data) if detection_data else 0} text regions")
            
            # Step 2: Run full OCR for text recognition
            ocr_text_output, ocr_json_output, ocr_annotated_image, ocr_data = self.basic_surya_node.process_ocr(
                img_tensor, 
                task_mode="ocr_with_boxes",  # Get text content
                confidence_threshold=confidence_threshold
            )
            
            print(f"🔍 OCR step: text_output={bool(ocr_text_output)}, json_output={bool(ocr_json_output)}")
            
            # Step 3: Parse the OCR results to get text content
            ocr_results_with_text = []
            
            # Try to parse OCR JSON first
            if ocr_json_output:
                import json
                try:
                    ocr_dict = json.loads(ocr_json_output)
                    print(f"🔍 OCR JSON structure: {list(ocr_dict.keys())}")
                    
                    detections = ocr_dict.get('detections', [])
                    if detections and len(detections) > 0:
                        page_detections = detections[0]  # First page
                        if isinstance(page_detections, dict):
                            bboxes = page_detections.get('bboxes', [])
                            print(f"🔍 OCR found {len(bboxes)} bboxes with text")
                            
                            for bbox_info in bboxes:
                                bbox = bbox_info.get('bbox')
                                confidence = bbox_info.get('confidence', 0.9)
                                text = bbox_info.get('text', '').strip()
                                
                                if bbox and confidence >= confidence_threshold and text:
                                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                    ocr_results_with_text.append({
                                        'text': text,
                                        'bbox': bbox,
                                        'confidence': confidence,
                                        'area': area,
                                        'source': 'surya_ocr'
                                    })
                                    
                except Exception as e:
                    print(f"⚠️ Failed to parse OCR JSON: {e}")
            
            # Step 4: If OCR parsing failed, use detection bboxes and extract text manually
            if not ocr_results_with_text and detection_data:
                print("🔧 OCR parsing failed, using detection bboxes + manual text extraction...")
                
                # Parse detection data to get high-quality bounding boxes
                detection_bboxes = []
                if isinstance(detection_data, list):
                    for detection in detection_data:
                        bbox = detection.get('bbox')
                        confidence = detection.get('confidence', 0.9)
                        if bbox and confidence >= confidence_threshold:
                            detection_bboxes.append((bbox, confidence))
                
                elif det_json_output:
                    # Parse detection JSON
                    import json
                    try:
                        det_dict = json.loads(det_json_output)
                        detections = det_dict.get('detections', [])
                        if detections and len(detections) > 0:
                            page_detections = detections[0]
                            if isinstance(page_detections, dict):
                                bboxes = page_detections.get('bboxes', [])
                                for bbox_info in bboxes:
                                    bbox = bbox_info.get('bbox')
                                    confidence = bbox_info.get('confidence', 0.9)
                                    if bbox and confidence >= confidence_threshold:
                                        detection_bboxes.append((bbox, confidence))
                    except Exception as e:
                        print(f"⚠️ Failed to parse detection JSON: {e}")
                
                print(f"🔍 Found {len(detection_bboxes)} detection bboxes for text extraction")
                
                # Use the text from full OCR output if available
                if ocr_text_output:
                    # Split text content and assign to detection boxes
                    text_lines = [line.strip() for line in ocr_text_output.strip().split('\n') if line.strip()]
                    print(f"🔍 Found {len(text_lines)} text lines to match with {len(detection_bboxes)} boxes")
                    
                    # Match text lines to detection boxes (simple approach)
                    for i, (bbox, confidence) in enumerate(detection_bboxes):
                        text = text_lines[i] if i < len(text_lines) else f"Text region {i+1}"
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        
                        ocr_results_with_text.append({
                            'text': text,
                            'bbox': bbox,
                            'confidence': confidence,
                            'area': area,
                            'source': 'surya_ocr_hybrid'
                        })
                else:
                    # No text content available, use detection boxes with placeholder text
                    print("🔧 No text content available, using detection boxes only...")
                    for i, (bbox, confidence) in enumerate(detection_bboxes):
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        ocr_results_with_text.append({
                            'text': f"[Text region {i+1}]",  # Placeholder text
                            'bbox': bbox,
                            'confidence': confidence,
                            'area': area,
                            'source': 'surya_detection_only'
                        })
            
            # Step 5: Final fallback - use just the text output with fake bboxes
            if not ocr_results_with_text and ocr_text_output:
                print("🔧 Final fallback: using text output with estimated bboxes...")
                lines = ocr_text_output.strip().split('\n')
                line_height = image.height // max(len(lines), 1)
                
                for i, line in enumerate(lines):
                    if line.strip():
                        y1 = i * line_height
                        y2 = (i + 1) * line_height
                        ocr_results_with_text.append({
                            'text': line.strip(),
                            'bbox': [50, y1, image.width - 50, y2],
                            'confidence': 0.8,
                            'area': (image.width - 100) * line_height,
                            'source': 'surya_ocr_fallback'
                        })
            
            print(f"✅ Surya OCR found {len(ocr_results_with_text)} text regions with content")
            
            # Debug first few results
            for i, result in enumerate(ocr_results_with_text[:3]):
                print(f"   Text {i}: '{result['text'][:50]}...' conf={result['confidence']:.2f}")
            
            return ocr_results_with_text
            
        except Exception as e:
            print(f"❌ Surya OCR with text failed: {e}")
            import traceback
            traceback.print_exc()
            return []


    def analyze_layout(self, image, primary_ocr="surya", enable_florence2=True, 
                      enable_layoutlmv3=True, enable_cv2_backup=False,
                      extract_images=True, extract_text=True,
                      fusion_strategy="aggressive", florence2_image_prompt=None, florence2_text_prompt=None,
                      min_image_area=10000, min_text_area=500, 
                      surya_confidence=0.3, florence2_confidence=0.5, cv2_confidence=0.7,
                      text_grouping_mode="florence2_priority", text_merge_distance=25,
                      include_text_recognition=True, enable_debug_logging=True,
                      surya_recognition_batch=768, surya_detector_batch=54,
                      smart_grouping_enabled=True, vertical_merge_threshold=15, horizontal_merge_threshold=10):
        """Main analysis function with enhanced text grouping controls"""
        
        # Convert tensor to PIL
        pil_image = tensor_to_PIL(image)
        self.current_image = pil_image
        
        # Store grouping parameters for use in fusion
        self.text_merge_distance = text_merge_distance
        self.smart_grouping_enabled = smart_grouping_enabled
        self.vertical_merge_threshold = vertical_merge_threshold
        self.horizontal_merge_threshold = horizontal_merge_threshold
        
        if enable_debug_logging:
            print(f"🔍 Enhanced Layout Analysis starting:")
            print(f"  Image size: {pil_image.size}")
            print(f"  Primary OCR: {primary_ocr}")
            print(f"  Florence2: {enable_florence2}, LayoutLMv3: {enable_layoutlmv3}")
            print(f"  Text grouping: {text_grouping_mode}, merge_distance: {text_merge_distance}")
            print(f"  Text recognition: {include_text_recognition}")
            print(f"  Smart grouping: {smart_grouping_enabled}, v_thresh: {vertical_merge_threshold}, h_thresh: {horizontal_merge_threshold}")
        
        # Initialize components as needed
        ocr_results = []
        florence2_images = []
        florence2_text = []
        cv2_images = []
        cv2_text = []
        
        # 1. RUN SURYA OCR (with improved strategy)
        if primary_ocr == "surya":
            if include_text_recognition:
                # Full OCR with enhanced text extraction strategy
                ocr_results = self._integrate_basic_surya_ocr_with_text(pil_image, surya_confidence)
            else:
                # Detection only for better coverage (existing working code)
                ocr_results = self._integrate_basic_surya_detection(pil_image, surya_confidence)
            
            # Fallback to other OCR if Surya fails
            if not ocr_results:
                print("🔧 Surya failed, trying PaddleOCR fallback...")
                if not hasattr(self, 'fallback_ocr') or self.fallback_ocr is None:
                    self._init_fallback_ocr("paddleocr")
                
                if hasattr(self, 'fallback_ocr') and self.fallback_ocr:
                    ocr_results = self._run_fallback_ocr(pil_image, "paddleocr")
        
        # Fallback to PaddleOCR if requested
        elif primary_ocr in ["paddleocr", "easyocr"]:
            if not hasattr(self, 'fallback_ocr') or self.fallback_ocr is None:
                self._init_fallback_ocr(primary_ocr)
            
            if hasattr(self, 'fallback_ocr') and self.fallback_ocr:
                ocr_results = self._run_fallback_ocr(pil_image, primary_ocr)
        
        # 2. RUN FLORENCE2 DETECTION with optimized prompts
        if enable_florence2:
            if self.florence2_detector is None:
                self._init_florence2()
            
            if self.florence2_detector:
                florence2_images, florence2_text = self._run_florence2_detection(
                    pil_image,
                    image_prompt=florence2_image_prompt or "rectangular images in page",
                    text_prompt=florence2_text_prompt or "text, caption, paragraph, title"
                )
        
        # 3. RUN CV2 ANALYSIS only if specifically requested (minimal use)
        if enable_cv2_backup:
            cv2_images, cv2_text = self._run_cv2_analysis(pil_image)
        
        # 4. ENHANCED FUSION with LayoutLMv3
        final_images, final_text = self._smart_fusion_with_layoutlmv3(
            ocr_results, florence2_images, florence2_text, cv2_images, cv2_text,
            fusion_strategy, min_image_area, min_text_area,
            surya_confidence, florence2_confidence, cv2_confidence, text_grouping_mode,
            enable_layoutlmv3
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
        
        # 6. EXTRACT ALL TEXT CONTENT
        extracted_text = ""
        if include_text_recognition and text_boxes:
            text_content = []
            for txt_box in text_boxes:
                if txt_box.get('text', '').strip():
                    text_content.append(txt_box['text'].strip())
            
            extracted_text = "\n".join(text_content)
            print(f"📝 Extracted {len(text_content)} text blocks, total length: {len(extracted_text)} characters")
        
        # Analysis results
        analysis_results = {
            "total_detections": len(image_boxes) + len(text_boxes),
            "image_detections": len(image_boxes),
            "text_detections": len(text_boxes),
            "fusion_strategy": fusion_strategy,
            "text_grouping_mode": text_grouping_mode,
            "text_recognition_enabled": include_text_recognition,
            "extracted_text_length": len(extracted_text),
            "layoutlmv3_enabled": enable_layoutlmv3,
            "confidence_thresholds": {
                "surya": surya_confidence,
                "florence2": florence2_confidence,
                "cv2": cv2_confidence
            },
            "detection_methods_used": {
                "ocr": len(ocr_results) > 0,
                "florence2": len(florence2_images) + len(florence2_text) > 0,
                "cv2": len(cv2_images) + len(cv2_text) > 0,
                "layoutlmv3": enable_layoutlmv3
            },
            "method_statistics": {
                "ocr_results": len(ocr_results),
                "florence2_images": len(florence2_images),
                "florence2_text": len(florence2_text),
                "cv2_images": len(cv2_images),
                "cv2_text": len(cv2_text)
            }
        }
        
        return (overlay_tensor, image_boxes, text_boxes, image_crops, text_crops, extracted_text, analysis_results)

    
    def _create_enhanced_overlay(self, image: Image.Image, image_boxes: List[Dict], text_boxes: List[Dict]) -> Image.Image:
        """Create enhanced visualization showing detection sources"""
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        
        # Color coding by detection method
        colors = {
            'florence2': (255, 0, 0),        # Red
            'surya': (0, 255, 0),            # Green  
            'surya_detection': (0, 200, 0),  # Dark Green
            'surya_ocr': (0, 255, 50),       # Bright Green
            'surya_grouped': (0, 180, 100),  # Grouped Green
            'paddleocr': (0, 150, 255),      # Light Blue
            'easyocr': (100, 150, 255),      # Lighter Blue
            'cv2': (0, 0, 255),              # Blue
            'layoutlmv3': (255, 0, 255),     # Magenta
            'layoutlmv3_semantic': (200, 0, 255),  # Purple
            'multi': (255, 165, 0)           # Orange for multi-method
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
            
            # Add label with semantic info if available
            label = f"IMG{i+1}: {method_label} ({box['confidence']:.2f})"
            if box.get('semantic_label'):
                label += f" [{box['semantic_label']}]"
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
            
            # Add label with semantic info if available
            label = f"TXT{i+1}: {method_label} ({box['confidence']:.2f})"
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
    "EnhancedLayoutParser_v02": EnhancedLayoutParserNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedLayoutParser_v02": "Enhanced Layout Parser (Multi-Modal) v02",
}