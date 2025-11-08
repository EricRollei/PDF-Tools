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

# LayoutLMv3 and transformers imports
try:
    from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
    from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3ForQuestionAnswering
    LAYOUTLMV3_AVAILABLE = True
except ImportError as e:
    LAYOUTLMV3_AVAILABLE = False
    print("LayoutLMv3 not available:", e)
    print("Install with: pip install transformers torch torchvision")

# OCR imports - using EasyOCR as primary choice for simplicity
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR not available. Install with: pip install easyocr")

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("PaddleOCR not available. Install with: pip install paddleocr")

class LayoutLMv3DocumentAnalysisNode:
    """Advanced document analysis using Microsoft's LayoutLMv3"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_size": (["base", "large"], {"default": "base"}),
                "task_type": (["token_classification", "sequence_classification"], {"default": "token_classification"}),
                "ocr_engine": (["easyocr", "paddleocr"], {"default": "easyocr"}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0}),
                "extract_crops": ("BOOLEAN", {"default": True}),
                "language": (["en", "ch", "fr", "de", "ja", "ko"], {"default": "en"}),
            },
            "optional": {
                "custom_labels": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LIST", "LIST", "LIST", "LIST", "DICT")
    RETURN_NAMES = ("overlay_image", "image_boxes", "text_boxes", "image_crops", "text_tokens", "analysis_results")
    FUNCTION = "run"
    CATEGORY = "layout/advanced"

    def __init__(self):
        if not LAYOUTLMV3_AVAILABLE:
            raise ImportError("LayoutLMv3 requires transformers. Install with: pip install transformers")
        
        self.processor = None
        self.model = None
        self.ocr_reader = None
        self.current_model = None
        self.current_task = None
        
        # Default label mappings for document layout analysis
        self.layout_labels = {
            0: "O",  # Outside/Other
            1: "B-TITLE",  # Beginning of title
            2: "I-TITLE",  # Inside title
            3: "B-HEADER", # Beginning of header  
            4: "I-HEADER", # Inside header
            5: "B-TEXT",   # Beginning of text block
            6: "I-TEXT",   # Inside text block
            7: "B-LIST",   # Beginning of list
            8: "I-LIST",   # Inside list
            9: "B-TABLE",  # Beginning of table
            10: "I-TABLE", # Inside table
            11: "B-FIGURE", # Beginning of figure
            12: "I-FIGURE", # Inside figure
        }

    def _init_models(self, model_size="base", task_type="token_classification", ocr_engine="easyocr", language="en"):
        """Initialize LayoutLMv3 models and OCR engine"""
        
        # Initialize OCR engine
        if ocr_engine == "easyocr" and EASYOCR_AVAILABLE:
            if self.ocr_reader is None:
                self.ocr_reader = easyocr.Reader([language], gpu=torch.cuda.is_available())
        elif ocr_engine == "paddleocr" and PADDLEOCR_AVAILABLE:
            if self.ocr_reader is None:
                self.ocr_reader = PaddleOCR(
                    use_angle_cls=True, 
                    lang=language, 
                    use_gpu=torch.cuda.is_available(),
                    show_log=False
                )
        else:
            raise ImportError(f"OCR engine {ocr_engine} not available")
        
        # Initialize LayoutLMv3 model if needed
        model_name = f"microsoft/layoutlmv3-{model_size}"
        if self.current_model != model_name or self.current_task != task_type:
            print(f"Loading LayoutLMv3 model: {model_name} for {task_type}")
            
            # Initialize processor with apply_ocr=False since we're doing external OCR
            self.processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
            
            if task_type == "token_classification":
                self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
            elif task_type == "sequence_classification":
                self.model = LayoutLMv3ForSequenceClassification.from_pretrained(model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            self.model.eval()
            self.current_model = model_name
            self.current_task = task_type

    def _run_ocr(self, image_array: np.ndarray, ocr_engine: str) -> List[Dict]:
        """Run OCR to extract text and bounding boxes"""
        raw_results = []
        
        if ocr_engine == "easyocr":
            ocr_results = self.ocr_reader.readtext(image_array)
            for bbox, text, confidence in ocr_results:
                # Convert bbox format
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
                
                raw_results.append({
                    'text': text,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence
                })
                
        elif ocr_engine == "paddleocr":
            ocr_results = self.ocr_reader.ocr(image_array, cls=True)
            if ocr_results and ocr_results[0]:
                for line in ocr_results[0]:
                    bbox = line[0]
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    # Convert bbox format
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
                    
                    raw_results.append({
                        'text': text,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence
                    })
        
        # Group nearby text detections into text blocks
        grouped_results = self._group_text_into_blocks(raw_results)
        print(f"Grouped {len(raw_results)} text detections into {len(grouped_results)} text blocks")
        
        return grouped_results
    
    def _group_text_into_blocks(self, text_detections: List[Dict]) -> List[Dict]:
        """Group nearby text detections into logical text blocks"""
        if not text_detections:
            return []
        
        # Sort by vertical position (top to bottom)
        text_detections.sort(key=lambda x: x['bbox'][1])
        
        blocks = []
        current_block = None
        
        for detection in text_detections:
            x1, y1, x2, y2 = detection['bbox']
            
            if current_block is None:
                # Start first block
                current_block = {
                    'text': detection['text'],
                    'bbox': [x1, y1, x2, y2],
                    'confidence': detection['confidence'],
                    'detections': [detection]
                }
            else:
                # Check if this detection should be merged with current block
                curr_x1, curr_y1, curr_x2, curr_y2 = current_block['bbox']
                
                # Calculate vertical and horizontal proximity
                vertical_gap = y1 - curr_y2
                horizontal_overlap = min(x2, curr_x2) - max(x1, curr_x1)
                
                # Merge conditions:
                # 1. Small vertical gap (same paragraph)
                # 2. Some horizontal overlap or close proximity
                should_merge = (
                    vertical_gap < 30 and  # Lines close vertically
                    (horizontal_overlap > -50 or  # Some horizontal overlap
                     abs(x1 - curr_x1) < 100)  # Or similar left alignment
                )
                
                if should_merge:
                    # Merge with current block
                    current_block['text'] += ' ' + detection['text']
                    current_block['bbox'] = [
                        min(curr_x1, x1),  # leftmost
                        min(curr_y1, y1),  # topmost
                        max(curr_x2, x2),  # rightmost
                        max(curr_y2, y2)   # bottommost
                    ]
                    current_block['confidence'] = min(current_block['confidence'], detection['confidence'])
                    current_block['detections'].append(detection)
                else:
                    # Start new block
                    blocks.append(current_block)
                    current_block = {
                        'text': detection['text'],
                        'bbox': [x1, y1, x2, y2],
                        'confidence': detection['confidence'],
                        'detections': [detection]
                    }
        
        # Add the last block
        if current_block:
            blocks.append(current_block)
        
        return blocks

    def _prepare_layoutlmv3_inputs(self, image: Image.Image, ocr_results: List[Dict]) -> Dict:
        """Prepare inputs for LayoutLMv3 with aggressive validation"""
        
        # Extract text and bounding boxes
        words = []
        boxes = []
        
        # Limit OCR results to prevent memory issues
        max_regions = 50  # Limit to 50 regions to prevent crashes
        ocr_results = ocr_results[:max_regions]
        
        img_width, img_height = image.size
        print(f"🔍 Image dimensions: {img_width}x{img_height}")
        
        for result in ocr_results:
            # Split text into words and assign same bbox to each word
            text_words = result['text'].split()
            for word in text_words:
                # Skip empty words
                if not word.strip():
                    continue
                    
                words.append(word)
                
                # Aggressive bounding box validation and fixing
                x1, y1, x2, y2 = result['bbox']
                
                # Convert to integers and validate
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Ensure positive coordinates
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = max(0, x2)
                y2 = max(0, y2)
                
                # Fix invalid bounding boxes
                if x1 >= x2:
                    x2 = x1 + 10  # Minimum width
                if y1 >= y2:
                    y2 = y1 + 10  # Minimum height
                
                # Clamp to image bounds with safety margin
                x1 = min(x1, img_width - 2)
                y1 = min(y1, img_height - 2)  
                x2 = min(x2, img_width - 1)
                y2 = min(y2, img_height - 1)
                
                # Final validation
                if x1 >= x2: x2 = x1 + 1
                if y1 >= y2: y2 = y1 + 1
                
                # Ensure reasonable size (not too small)
                if (x2 - x1) < 1: x2 = x1 + 1
                if (y2 - y1) < 1: y2 = y1 + 1
                
                boxes.append([x1, y1, x2, y2])
        
        # Limit total words to LayoutLMv3's max sequence length
        max_words = 300  # More conservative limit
        if len(words) > max_words:
            print(f"Warning: Truncating from {len(words)} to {max_words} words")
            words = words[:max_words]
            boxes = boxes[:max_words]
        
        print(f"Processing {len(words)} words from {len(ocr_results)} regions")
        
        # Validate all boxes one more time
        valid_words = []
        valid_boxes = []
        
        for word, box in zip(words, boxes):
            x1, y1, x2, y2 = box
            
            # Final sanity checks
            if (x1 < x2 and y1 < y2 and 
                x1 >= 0 and y1 >= 0 and 
                x2 <= img_width and y2 <= img_height and
                len(word.strip()) > 0):
                valid_words.append(word)
                valid_boxes.append(box)
            else:
                print(f"⚠️ Skipping invalid word/box: '{word}' {box}")
        
        print(f"After validation: {len(valid_words)} valid words")
        
        if not valid_words:
            raise ValueError("No valid words/boxes after validation")
        
        try:
            # Prepare inputs using LayoutLMv3 processor
            encoding = self.processor(
                image,
                valid_words,
                boxes=valid_boxes,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256  # Even more conservative
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                encoding = {k: v.cuda() for k, v in encoding.items()}
            
            return encoding, valid_words, valid_boxes
            
        except Exception as e:
            print(f"Error in processor: {e}")
            raise e

    def _process_token_classification_results(self, predictions, words, boxes, confidence_threshold):
        """Process token classification results into structured layout information"""
        
        text_boxes = []
        image_boxes = []
        processed_tokens = []
        
        # Get predictions
        predicted_labels = torch.argmax(predictions.logits, dim=-1)
        confidences = torch.softmax(predictions.logits, dim=-1)
        
        # Process each token
        for i, (word, box, label_id, token_confidences) in enumerate(zip(words, boxes, predicted_labels[0], confidences[0])):
            label_id = label_id.item()
            confidence = token_confidences[label_id].item()
            
            if confidence < confidence_threshold:
                continue
            
            # Map label ID to label name
            label_name = self.layout_labels.get(label_id, f"LABEL_{label_id}")
            
            token_info = {
                "word": word,
                "label": label_name,
                "confidence": confidence,
                "bbox": box
            }
            processed_tokens.append(token_info)
            
            # Categorize into text or image elements
            if any(keyword in label_name.lower() for keyword in ['title', 'header', 'text', 'list']):
                text_boxes.append({
                    "type": label_name.replace('B-', '').replace('I-', '').lower(),
                    "score": confidence,
                    "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                    "text": word
                })
            elif any(keyword in label_name.lower() for keyword in ['table', 'figure']):
                image_boxes.append({
                    "type": label_name.replace('B-', '').replace('I-', '').lower(),
                    "score": confidence,
                    "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                    "text": word
                })
        
        return text_boxes, image_boxes, processed_tokens

    def _create_overlay_visualization(self, image: Image.Image, text_boxes: List, image_boxes: List):
        """Create visualization overlay"""
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        
        # Color mapping
        colors = {
            "title": (255, 0, 0),     # Red
            "header": (255, 165, 0),  # Orange  
            "text": (0, 255, 0),      # Green
            "list": (0, 255, 255),    # Cyan
            "table": (0, 0, 255),     # Blue
            "figure": (255, 255, 0),  # Yellow
        }
        
        # Draw text boxes
        for box in text_boxes:
            color = colors.get(box["type"], (128, 128, 128))
            draw.rectangle([box["x1"], box["y1"], box["x2"], box["y2"]], outline=color, width=2)
            
            # Add label
            label = f'{box["type"]}: {box["score"]:.2f}'
            draw.text((box["x1"], max(0, box["y1"]-15)), label, fill=color)
        
        # Draw image boxes  
        for box in image_boxes:
            color = colors.get(box["type"], (128, 128, 128))
            draw.rectangle([box["x1"], box["y1"], box["x2"], box["y2"]], outline=color, width=3)
            
            # Add label
            label = f'{box["type"]}: {box["score"]:.2f}'
            draw.text((box["x1"], max(0, box["y1"]-15)), label, fill=color)
        
        return overlay

    def _debug_bounding_boxes(self, words, boxes, image_size):
        """Debug function to investigate invalid bounding boxes"""
        print("🔍 Debugging bounding boxes...")
        img_width, img_height = image_size
        
        invalid_count = 0
        for i, (word, box) in enumerate(zip(words, boxes)):
            x1, y1, x2, y2 = box
            
            issues = []
            if x1 >= x2:
                issues.append(f"x1({x1}) >= x2({x2})")
            if y1 >= y2:
                issues.append(f"y1({y1}) >= y2({y2})")
            if x1 < 0 or y1 < 0:
                issues.append(f"negative coords: ({x1}, {y1})")
            if x2 > img_width or y2 > img_height:
                issues.append(f"exceeds image bounds: ({x2}, {y2}) > ({img_width}, {img_height})")
            
            if issues:
                invalid_count += 1
                print(f"  Box {i} for word '{word}': {box} - Issues: {', '.join(issues)}")
                
                if invalid_count >= 10:  # Limit output
                    print("  ... (showing first 10 invalid boxes)")
                    break
        
        print(f"🔍 Found {invalid_count} invalid boxes out of {len(boxes)} total")
        return invalid_count


    def run(self, image, model_size="base", task_type="token_classification", 
            ocr_engine="easyocr", confidence_threshold=0.5, extract_crops=True, 
            language="en", custom_labels=""):
        
        # Initialize models
        self._init_models(model_size, task_type, ocr_engine, language)
        
        # Memory warning for large model
        if model_size == "large":
            print("⚠️  Using large model - this requires significant GPU memory!")
            print("💡 If you get crashes, try using 'base' model instead")
        
        # Convert tensor to PIL and RESIZE to prevent coordinate issues
        pil_image = tensor_to_PIL(image)
        
        # RESIZE IMAGE to prevent EasyOCR coordinate issues
        original_size = pil_image.size
        if original_size[0] > 1000 or original_size[1] > 1000:
            # Resize large images to prevent coordinate issues
            aspect_ratio = original_size[0] / original_size[1]
            if aspect_ratio > 1:
                new_size = (1000, int(1000 / aspect_ratio))
            else:
                new_size = (int(1000 * aspect_ratio), 1000)
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"🔧 Resized image from {original_size} to {pil_image.size} to prevent OCR coordinate issues")
        
        image_array = np.array(pil_image)
        
        # Run OCR to get text and bounding boxes
        print("Running OCR...")
        ocr_results = self._run_ocr(image_array, ocr_engine)
        
        if not ocr_results:
            print("No text detected by OCR")
            # Return empty results
            overlay_tensor = PIL_to_tensor(pil_image)
            return (overlay_tensor, [], [], [], [], {"message": "No text detected"})
        
        print(f"OCR detected {len(ocr_results)} text regions")
        
        # Prepare LayoutLMv3 inputs
        print("Preparing LayoutLMv3 inputs...")
        try:
            encoding, words, boxes = self._prepare_layoutlmv3_inputs(pil_image, ocr_results)
        except Exception as e:
            print(f"❌ Error preparing inputs: {e}")
            return (PIL_to_tensor(pil_image), [], [], [], [], {"error": f"Input preparation failed: {e}"})
        
        # Run LayoutLMv3 inference
        print("Running LayoutLMv3 inference...")
        
        # Check GPU memory before inference
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_free = gpu_memory - memory_used
            print(f"GPU Memory: {memory_free:.1f}GB free of {gpu_memory:.1f}GB total")
        
        # Initialize outputs variable in the correct scope
        outputs = None
        
        try:
            # Clear GPU cache before inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("🔍 Debug: About to call model inference...")
            print(f"🔍 Input shapes:")
            for key, value in encoding.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape}")
            
            # Check for invalid bbox values and fix them properly
            if 'bbox' in encoding:
                bbox_tensor = encoding['bbox']
                print(f"🔍 Bbox stats:")
                print(f"  Min: {bbox_tensor.min().item()}")
                print(f"  Max: {bbox_tensor.max().item()}")
                print(f"  Contains NaN: {torch.isnan(bbox_tensor).any().item()}")
                print(f"  Contains Inf: {torch.isinf(bbox_tensor).any().item()}")
                
                # Check for invalid bounding boxes in the processed tensor
                invalid_boxes = torch.logical_or(
                    bbox_tensor[:, :, 0] >= bbox_tensor[:, :, 2],  # x1 >= x2
                    bbox_tensor[:, :, 1] >= bbox_tensor[:, :, 3]   # y1 >= y2
                )
                if invalid_boxes.any():
                    invalid_count = invalid_boxes.sum().item()
                    print(f"⚠️ Found {invalid_count} invalid bounding boxes in processed tensor!")
                    
                    # Debug the source of invalid boxes
                    print("🔍 Investigating source of invalid boxes...")
                    self._debug_bounding_boxes(words, boxes, pil_image.size)
                    
                    # Fix them in the tensor directly
                    print("🔧 Fixing invalid boxes in processed tensor...")
                    bbox_fixed = bbox_tensor.clone()
                    
                    # Fix x coordinates - ensure x2 > x1
                    bad_x = bbox_fixed[:, :, 0] >= bbox_fixed[:, :, 2]
                    bbox_fixed[:, :, 2][bad_x] = bbox_fixed[:, :, 0][bad_x] + 1
                    
                    # Fix y coordinates - ensure y2 > y1
                    bad_y = bbox_fixed[:, :, 1] >= bbox_fixed[:, :, 3]
                    bbox_fixed[:, :, 3][bad_y] = bbox_fixed[:, :, 1][bad_y] + 1
                    
                    # Update the encoding
                    encoding['bbox'] = bbox_fixed
                    print("✅ Fixed invalid bounding boxes in tensor!")
                else:
                    print("✅ All bounding boxes valid in processed tensor!")
            
            # Direct inference - we know it works up to 500 tokens
            print("🔍 Running direct inference...")
            with torch.no_grad():
                outputs = self.model(**encoding)
            print("✅ Inference successful!")
                
        except RuntimeError as e:
            error_msg = str(e).lower()
            print(f"❌ Runtime error during inference: {e}")
            
            # Check if it's a memory error
            if "out of memory" in error_msg or "cuda" in error_msg:
                print("🔧 Trying memory-saving strategies...")
                
                # Strategy 1: Remove pixel_values to save memory
                try:
                    print("🔍 Trying without pixel_values...")
                    fallback_encoding = {k: v for k, v in encoding.items() if k != 'pixel_values'}
                    with torch.no_grad():
                        outputs = self.model(**fallback_encoding)
                    print("✅ Works without pixel_values!")
                    
                except Exception as e2:
                    print(f"❌ Still fails without pixel_values: {e2}")
                    
                    # Strategy 2: Try on CPU
                    print("🔍 Trying on CPU...")
                    try:
                        cpu_encoding = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in encoding.items()}
                        cpu_model = self.model.cpu()
                        with torch.no_grad():
                            outputs = cpu_model(**cpu_encoding)
                        print("✅ Works on CPU!")
                        
                        # Move model back to GPU for next time
                        if torch.cuda.is_available():
                            self.model = self.model.cuda()
                            
                    except Exception as e3:
                        print(f"❌ Even CPU fails: {e3}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return (PIL_to_tensor(pil_image), [], [], [], [], {"error": f"All inference strategies failed: {str(e)}"})
            else:
                # Non-memory error, return immediately
                print(f"❌ Non-memory error: {e}")
                return (PIL_to_tensor(pil_image), [], [], [], [], {"error": f"Inference error: {str(e)}"})
            
            # Clear GPU cache after fallback attempts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Unexpected error during inference: {e}")
            import traceback
            traceback.print_exc()
            return (PIL_to_tensor(pil_image), [], [], [], [], {"error": f"Unexpected error: {str(e)}"})
        
        # Ensure we have outputs before proceeding
        if outputs is None:
            return (PIL_to_tensor(pil_image), [], [], [], [], {"error": "No outputs generated from model"})
        
        # Process results based on task type
        try:
            if task_type == "token_classification":
                text_boxes, image_boxes, processed_tokens = self._process_token_classification_results(
                    outputs, words, boxes, confidence_threshold
                )
                # Create analysis results for token classification
                analysis_results = {
                    "task_type": task_type,
                    "model_used": f"microsoft/layoutlmv3-{model_size}",
                    "total_text_regions": len(text_boxes),
                    "total_image_regions": len(image_boxes),
                    "total_tokens": len(processed_tokens),
                    "ocr_engine": ocr_engine,
                    "language": language
                }
            else:
                # For sequence classification, create simple results
                predicted_class = torch.argmax(outputs.logits, dim=-1).item()
                confidence = torch.softmax(outputs.logits, dim=-1).max().item()
                
                text_boxes = []
                image_boxes = []
                processed_tokens = []
                
                # Create analysis results for sequence classification
                analysis_results = {
                    "task_type": task_type,
                    "model_used": f"microsoft/layoutlmv3-{model_size}",
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "total_tokens": len(words),
                    "ocr_engine": ocr_engine,
                    "language": language
                }
                
        except Exception as e:
            print(f"❌ Error processing results: {e}")
            return (PIL_to_tensor(pil_image), [], [], [], [], {"error": f"Result processing failed: {str(e)}"})
        
        # Extract crops if requested
        image_crops = []
        if extract_crops:
            for box_list in [text_boxes, image_boxes]:
                for box in box_list:
                    try:
                        # Ensure coordinates are within image bounds
                        x1 = max(0, min(box["x1"], pil_image.width - 1))
                        y1 = max(0, min(box["y1"], pil_image.height - 1))
                        x2 = max(x1 + 1, min(box["x2"], pil_image.width))
                        y2 = max(y1 + 1, min(box["y2"], pil_image.height))
                        
                        crop = pil_image.crop((x1, y1, x2, y2))
                        image_crops.append(PIL_to_tensor(crop))
                    except Exception as e:
                        print(f"Error extracting crop: {e}")
        
        # Create visualization overlay
        try:
            overlay = self._create_overlay_visualization(pil_image, text_boxes, image_boxes)
            overlay_tensor = PIL_to_tensor(overlay)
        except Exception as e:
            print(f"Error creating overlay: {e}")
            overlay_tensor = PIL_to_tensor(pil_image)
        
        return (overlay_tensor, image_boxes, text_boxes, image_crops, processed_tokens, analysis_results)


class LayoutLMv3QuestionAnsweringNode:
    """Document Question Answering using LayoutLMv3"""
    
    @classmethod  
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "question": ("STRING", {"default": "What is the total amount?"}),
                "model_size": (["base", "large"], {"default": "base"}),
                "ocr_engine": (["easyocr", "paddleocr"], {"default": "easyocr"}),
                "language": (["en", "ch", "fr", "de", "ja", "ko"], {"default": "en"}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "DICT")
    RETURN_NAMES = ("answer", "confidence", "details")
    FUNCTION = "run"
    CATEGORY = "layout/advanced"

    def __init__(self):
        if not LAYOUTLMV3_AVAILABLE:
            raise ImportError("LayoutLMv3 requires transformers. Install with: pip install transformers")
        
        self.processor = None
        self.model = None
        self.ocr_reader = None
        self.current_model = None

    def _init_models(self, model_size="base", ocr_engine="easyocr", language="en"):
        """Initialize models"""
        # Initialize OCR
        if ocr_engine == "easyocr" and EASYOCR_AVAILABLE:
            if self.ocr_reader is None:
                self.ocr_reader = easyocr.Reader([language], gpu=torch.cuda.is_available())
        elif ocr_engine == "paddleocr" and PADDLEOCR_AVAILABLE:
            if self.ocr_reader is None:
                self.ocr_reader = PaddleOCR(use_angle_cls=True, lang=language, 
                                          use_gpu=torch.cuda.is_available(), show_log=False)
        
        # Initialize LayoutLMv3 for QA
        model_name = f"microsoft/layoutlmv3-{model_size}"
        if self.current_model != model_name:
            print(f"Loading LayoutLMv3 QA model: {model_name}")
            self.processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
            self.model = LayoutLMv3ForQuestionAnswering.from_pretrained(model_name)
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.model.eval()
            self.current_model = model_name

    def run(self, image, question, model_size="base", ocr_engine="easyocr", language="en"):
        # Initialize models
        self._init_models(model_size, ocr_engine, language)
        
        # Convert and run OCR (reuse OCR logic from main node)
        pil_image = tensor_to_PIL(image)
        image_array = np.array(pil_image)
        
        # Simple OCR extraction
        if ocr_engine == "easyocr":
            ocr_results = self.ocr_reader.readtext(image_array)
            words = []
            boxes = []
            for bbox, text, confidence in ocr_results:
                text_words = text.split()
                for word in text_words:
                    words.append(word)
                    # Convert bbox
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    boxes.append([min(x_coords), min(y_coords), max(x_coords), max(y_coords)])
        
        if not words:
            return ("No text found", 0.0, {"error": "No text detected by OCR"})
        
        # Prepare inputs for QA
        encoding = self.processor(
            pil_image, 
            question,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        
        if torch.cuda.is_available():
            encoding = {k: v.cuda() for k, v in encoding.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        # Extract answer
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)
        
        if end_idx < start_idx:
            end_idx = start_idx
        
        # Get answer tokens
        input_ids = encoding['input_ids'][0]
        answer_tokens = input_ids[start_idx:end_idx+1]
        answer = self.processor.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # Calculate confidence
        confidence = (torch.softmax(start_scores, dim=-1).max() + 
                     torch.softmax(end_scores, dim=-1).max()) / 2
        confidence = confidence.item()
        
        details = {
            "question": question,
            "start_position": start_idx.item(),
            "end_position": end_idx.item(),
            "total_words": len(words),
            "model_used": f"microsoft/layoutlmv3-{model_size}"
        }
        
        return (answer, confidence, details)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "LayoutLMv3DocumentAnalysisNode": LayoutLMv3DocumentAnalysisNode,
    "LayoutLMv3QuestionAnsweringNode": LayoutLMv3QuestionAnsweringNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayoutLMv3DocumentAnalysisNode": "LayoutLMv3 Document Analysis",
    "LayoutLMv3QuestionAnsweringNode": "LayoutLMv3 Question Answering",
}