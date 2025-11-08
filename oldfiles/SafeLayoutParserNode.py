import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
import folder_paths

from typing import List, Dict, Any, Tuple
from comfy.utils import common_upscale
try:
    from comfy.utils import PIL_to_tensor, tensor_to_PIL
except ImportError:
    # Fallback to manual implementation 
    import torchvision.transforms.functional as F
    
    def PIL_to_tensor(pil_image):
        return F.to_tensor(pil_image).permute(1, 2, 0)
    
    def tensor_to_PIL(tensor):
        if tensor.dim() == 4:
            tensor = tensor[0]
        return F.to_pil_image(tensor.permute(2, 0, 1))

# Safe imports with version checking
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    
    from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
    LAYOUTLMV3_AVAILABLE = True
except ImportError as e:
    LAYOUTLMV3_AVAILABLE = False
    print("LayoutLMv3 not available:", e)

# OCR imports
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

class SafeLayoutLMv3Node:
    """Safe LayoutLMv3 implementation with better error handling"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_size": (["base"], {"default": "base"}),  # Only base for safety
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0}),
                "extract_crops": ("BOOLEAN", {"default": True}),
                "language": (["en", "ch", "fr", "de"], {"default": "en"}),
                "debug_mode": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LIST", "LIST", "LIST", "LIST", "STRING")
    RETURN_NAMES = ("overlay_image", "image_boxes", "text_boxes", "image_crops", "ocr_texts", "debug_info")
    FUNCTION = "run"
    CATEGORY = "layout/safe"

    def __init__(self):
        if not LAYOUTLMV3_AVAILABLE:
            raise ImportError("LayoutLMv3 requires transformers. Check version compatibility.")
        
        self.processor = None
        self.model = None
        self.ocr_reader = None
        self.initialized = False

    def _safe_init_models(self, model_size="base", language="en", debug=True):
        """Safely initialize models with extensive error checking"""
        
        if debug:
            print("🔍 Starting safe model initialization...")
            
        # Initialize OCR
        if not EASYOCR_AVAILABLE:
            return False, "EasyOCR not available"
            
        try:
            if self.ocr_reader is None:
                if debug:
                    print("🔍 Initializing EasyOCR...")
                self.ocr_reader = easyocr.Reader([language], gpu=torch.cuda.is_available())
                if debug:
                    print("✅ EasyOCR initialized")
        except Exception as e:
            return False, f"EasyOCR initialization failed: {e}"
        
        # Initialize LayoutLMv3
        try:
            model_name = f"microsoft/layoutlmv3-{model_size}"
            if not self.initialized:
                if debug:
                    print(f"🔍 Loading LayoutLMv3 processor: {model_name}")
                
                # Try different processor initialization methods
                try:
                    # Method 1: With apply_ocr parameter
                    self.processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
                    if debug:
                        print("✅ Processor loaded with apply_ocr=False")
                except TypeError:
                    # Method 2: Without apply_ocr parameter (older transformers)
                    if debug:
                        print("⚠️ apply_ocr parameter not supported, trying without...")
                    self.processor = LayoutLMv3Processor.from_pretrained(model_name)
                    if debug:
                        print("✅ Processor loaded without apply_ocr")
                
                if debug:
                    print(f"🔍 Loading LayoutLMv3 model: {model_name}")
                self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
                
                # Move to GPU safely
                if torch.cuda.is_available():
                    if debug:
                        print("🔍 Moving model to GPU...")
                    self.model = self.model.cuda()
                    if debug:
                        print("✅ Model moved to GPU")
                
                self.model.eval()
                self.initialized = True
                if debug:
                    print("✅ LayoutLMv3 model initialized successfully")
                
        except Exception as e:
            return False, f"LayoutLMv3 initialization failed: {e}"
        
        return True, "Models initialized successfully"

    def _safe_ocr(self, image_array: np.ndarray, debug=True) -> tuple:
        """Safely run OCR with error handling"""
        try:
            if debug:
                print("🔍 Running OCR...")
            ocr_results = self.ocr_reader.readtext(image_array)
            
            results = []
            for bbox, text, confidence in ocr_results:
                # Convert bbox format safely
                try:
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
                    
                    results.append({
                        'text': text,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence
                    })
                except Exception as e:
                    if debug:
                        print(f"⚠️ Skipping invalid bbox: {e}")
                    continue
            
            if debug:
                print(f"✅ OCR completed: {len(results)} text regions detected")
            return True, results, ""
            
        except Exception as e:
            return False, [], f"OCR failed: {e}"

    def _safe_prepare_inputs(self, image: Image.Image, ocr_results: List[Dict], debug=True) -> tuple:
        """Safely prepare LayoutLMv3 inputs"""
        try:
            if debug:
                print("🔍 Preparing LayoutLMv3 inputs...")
            
            # Limit results for safety
            max_regions = 20  # Very conservative
            ocr_results = ocr_results[:max_regions]
            
            words = []
            boxes = []
            
            for result in ocr_results:
                # Split text into words safely
                text_words = result['text'].split()
                for word in text_words[:5]:  # Limit words per region
                    words.append(word)
                    boxes.append(result['bbox'])
            
            # Limit total words
            max_words = 100  # Very conservative
            if len(words) > max_words:
                if debug:
                    print(f"⚠️ Truncating from {len(words)} to {max_words} words")
                words = words[:max_words]
                boxes = boxes[:max_words]
            
            if debug:
                print(f"🔍 Processing {len(words)} words from {len(ocr_results)} regions")
            
            # Prepare inputs with extensive error checking
            try:
                encoding = self.processor(
                    image,
                    words,
                    boxes=boxes,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128  # Very conservative
                )
                
                if debug:
                    print("✅ Processor encoding successful")
                    for key, value in encoding.items():
                        if hasattr(value, 'shape'):
                            print(f"  {key}: {value.shape}")
                
            except Exception as e:
                # Try without boxes if that fails
                if debug:
                    print(f"⚠️ Encoding with boxes failed: {e}")
                    print("🔍 Trying without boxes...")
                
                encoding = self.processor(
                    image,
                    words,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128
                )
                if debug:
                    print("✅ Processor encoding successful (without boxes)")
            
            # Move to GPU safely
            if torch.cuda.is_available():
                try:
                    if debug:
                        print("🔍 Moving inputs to GPU...")
                    encoding = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in encoding.items()}
                    if debug:
                        print("✅ Inputs moved to GPU")
                except Exception as e:
                    if debug:
                        print(f"⚠️ Failed to move to GPU: {e}")
            
            return True, encoding, words, boxes, ""
            
        except Exception as e:
            return False, None, [], [], f"Input preparation failed: {e}"

    def run(self, image, model_size="base", confidence_threshold=0.5, 
            extract_crops=True, language="en", debug_mode=True):
        
        debug_log = []
        
        try:
            # Initialize models
            success, message = self._safe_init_models(model_size, language, debug_mode)
            debug_log.append(f"Model init: {message}")
            
            if not success:
                return self._create_error_return(image, f"Initialization failed: {message}", debug_log)
            
            # Convert image
            pil_image = tensor_to_PIL(image)
            image_array = np.array(pil_image)
            debug_log.append(f"Image converted: {image_array.shape}")
            
            # Run OCR
            success, ocr_results, error = self._safe_ocr(image_array, debug_mode)
            if not success:
                return self._create_error_return(image, f"OCR failed: {error}", debug_log)
            
            debug_log.append(f"OCR completed: {len(ocr_results)} regions")
            
            if not ocr_results:
                return self._create_error_return(image, "No text detected", debug_log)
            
            # Prepare inputs
            success, encoding, words, boxes, error = self._safe_prepare_inputs(pil_image, ocr_results, debug_mode)
            if not success:
                return self._create_error_return(image, f"Input prep failed: {error}", debug_log)
            
            debug_log.append(f"Inputs prepared: {len(words)} words")
            
            # Run inference
            if debug_mode:
                print("🔍 Starting model inference...")
            
            try:
                with torch.no_grad():
                    outputs = self.model(**encoding)
                
                if debug_mode:
                    print("✅ Model inference successful!")
                debug_log.append("Inference successful")
                
            except Exception as e:
                debug_log.append(f"Inference failed: {e}")
                return self._create_error_return(image, f"Inference failed: {e}", debug_log)
            
            # Create simple results for now
            text_boxes = []
            image_boxes = []
            image_crops = []
            ocr_texts = [result['text'] for result in ocr_results]
            
            # Create overlay
            overlay = pil_image.copy()
            draw = ImageDraw.Draw(overlay)
            
            for i, result in enumerate(ocr_results):
                x1, y1, x2, y2 = result['bbox']
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                draw.text((x1, y1-15), f"Text {i+1}", fill=(0, 255, 0))
            
            overlay_tensor = PIL_to_tensor(overlay)
            debug_log.append("Processing completed successfully")
            
            return (overlay_tensor, image_boxes, text_boxes, image_crops, ocr_texts, "\n".join(debug_log))
            
        except Exception as e:
            debug_log.append(f"Unexpected error: {e}")
            return self._create_error_return(image, f"Unexpected error: {e}", debug_log)

    def _create_error_return(self, image, error_msg, debug_log):
        """Create error return tuple"""
        pil_image = tensor_to_PIL(image)
        overlay_tensor = PIL_to_tensor(pil_image)
        debug_info = "\n".join(debug_log) + f"\nERROR: {error_msg}"
        return (overlay_tensor, [], [], [], [], debug_info)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "SafeLayoutLMv3Node": SafeLayoutLMv3Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SafeLayoutLMv3Node": "Safe LayoutLMv3 (Debug)",
}