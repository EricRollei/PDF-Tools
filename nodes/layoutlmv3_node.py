"""
Layoutlmv3 Node

Description: LayoutLMv3 model integration for document understanding and layout analysis
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
            
            self.processor = LayoutLMv3Processor.from_pretrained(model_name)
            
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
        results = []
        
        if ocr_engine == "easyocr":
            ocr_results = self.ocr_reader.readtext(image_array)
            for bbox, text, confidence in ocr_results:
                # Convert bbox format
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
                
                results.append({
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
                    
                    results.append({
                        'text': text,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence
                    })
        
        return results

    def _prepare_layoutlmv3_inputs(self, image: Image.Image, ocr_results: List[Dict]) -> Dict:
        """Prepare inputs for LayoutLMv3"""
        
        # Extract text and bounding boxes
        words = []
        boxes = []
        
        for result in ocr_results:
            # Split text into words and assign same bbox to each word
            text_words = result['text'].split()
            for word in text_words:
                words.append(word)
                boxes.append(result['bbox'])
        
        # Prepare inputs using LayoutLMv3 processor
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            encoding = {k: v.cuda() for k, v in encoding.items()}
        
        return encoding, words, boxes

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

    def run(self, image, model_size="base", task_type="token_classification", 
            ocr_engine="easyocr", confidence_threshold=0.5, extract_crops=True, 
            language="en", custom_labels=""):
        
        # Initialize models
        self._init_models(model_size, task_type, ocr_engine, language)
        
        # Convert tensor to PIL
        pil_image = tensor_to_PIL(image)
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
        encoding, words, boxes = self._prepare_layoutlmv3_inputs(pil_image, ocr_results)
        
        # Run LayoutLMv3 inference
        print("Running LayoutLMv3 inference...")
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        # Process results based on task type
        if task_type == "token_classification":
            text_boxes, image_boxes, processed_tokens = self._process_token_classification_results(
                outputs, words, boxes, confidence_threshold
            )
        else:
            # For sequence classification, create simple results
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
            confidence = torch.softmax(outputs.logits, dim=-1).max().item()
            
            text_boxes = []
            image_boxes = []
            processed_tokens = []
            
            # Create analysis results
            analysis_results = {
                "task_type": task_type,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "total_tokens": len(words)
            }
        
        # Extract crops if requested
        image_crops = []
        if extract_crops:
            for box_list in [text_boxes, image_boxes]:
                for box in box_list:
                    try:
                        crop = pil_image.crop((box["x1"], box["y1"], box["x2"], box["y2"]))
                        image_crops.append(PIL_to_tensor(crop))
                    except Exception as e:
                        print(f"Error extracting crop: {e}")
        
        # Create visualization overlay
        overlay = self._create_overlay_visualization(pil_image, text_boxes, image_boxes)
        overlay_tensor = PIL_to_tensor(overlay)
        
        # Prepare analysis results
        analysis_results = {
            "task_type": task_type,
            "model_used": f"microsoft/layoutlmv3-{model_size}",
            "total_text_regions": len(text_boxes),
            "total_image_regions": len(image_boxes),
            "total_tokens": len(processed_tokens) if task_type == "token_classification" else len(words),
            "ocr_engine": ocr_engine,
            "language": language
        }
        
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
            self.processor = LayoutLMv3Processor.from_pretrained(model_name)
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