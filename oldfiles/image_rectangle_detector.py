"""
Single Florence2 Detector Class - Replicating kijai's complete workflow

This module provides one clean class that internally does what kijai's
Florence2ModelLoader + Florence2Run do together, but as a single callable interface.
"""

import torch
import numpy as np
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict, Any
from PIL import Image

# === DaViT Patch - Apply before any transformers imports ===
def patch_davit_missing_method():
    """Non-invasive DaViT patch that doesn't interfere with global imports."""
    def _initialize_weights_dummy(self):
        """Dummy implementation for DaViT compatibility."""
        pass
    
    # Return a function that can patch objects after they're loaded
    def patch_object_if_davit(obj):
        """Patch a specific object if it's a DaViT that needs fixing."""
        try:
            if (hasattr(obj, '__class__') and 
                'DaViT' in str(type(obj)) and
                not hasattr(obj, '_initialize_weights')):
                print(f"Patching DaViT object: {type(obj)}")
                import types
                obj._initialize_weights = types.MethodType(_initialize_weights_dummy, obj)
        except Exception as e:
            # Ignore patching errors
            pass
    
    return patch_object_if_davit

# Apply the patch immediately
davit_patcher = patch_davit_missing_method()

# DO NOT import transformers at module level - defer until needed
# This prevents interference with ComfyUI initialization

# ComfyUI memory management (with fallback)
try:
    import comfy.model_management as mm
except ImportError:
    print("ComfyUI not available, using fallback memory management")
    class MockModelManagement:
        @staticmethod
        def get_torch_device():
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        @staticmethod
        def unet_offload_device():
            return torch.device("cpu")
        
        @staticmethod
        def soft_empty_cache():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    mm = MockModelManagement()


def fixed_get_imports(filename):
    """Fix for unnecessary flash_attn requirement - exactly from kijai."""
    # Import here to avoid module-level import issues
    from transformers.dynamic_module_utils import get_imports
    
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


@dataclass
class BoundingBox:
    """Simple bounding box representation."""
    x1: int
    y1: int 
    x2: int
    y2: int
    confidence: Optional[float] = None
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property 
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Returns (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def __repr__(self):
        conf_str = f", conf={self.confidence:.3f}" if self.confidence else ""
        return f"BoundingBox({self.x1}, {self.y1}, {self.x2}, {self.y2}, area={self.area}{conf_str})"


class ImageRectangleDetector:
    """
    Single class that replicates kijai's complete Florence2ModelLoader + Florence2Run workflow.
    
    Internally handles model loading with precision/attention settings AND inference,
    but provides a simple callable interface for use in any ComfyUI node.
    """
    
    def __init__(self, 
                 model_name: str = "CogFlorence-2.2-Large",
                 comfyui_base_path: str = "A:\\Comfy_Dec\\ComfyUI",
                 precision: str = "fp16",
                 attention: str = "sdpa",
                 min_box_area: int = 1000,
                 convert_to_safetensors: bool = False):
        """
        Initialize detector with kijai's exact parameters.
        
        Args:
            model_name: Model directory name in ComfyUI/models/LLM/
            comfyui_base_path: Path to ComfyUI installation  
            precision: Model precision ('fp16', 'bf16', 'fp32') - kijai's Florence2ModelLoader param
            attention: Attention mechanism ('sdpa', 'flash_attention_2', 'eager') - kijai's param
            min_box_area: Minimum bounding box area to keep
            convert_to_safetensors: Convert .bin to .safetensors - kijai's param
        """
        # Store all kijai's parameters
        self.model_name = model_name
        self.comfyui_base_path = comfyui_base_path
        self.precision = precision
        self.attention = attention
        self.min_box_area = min_box_area
        self.convert_to_safetensors = convert_to_safetensors
        
        # Model state - loaded lazily
        self.model = None
        self.processor = None
        self.dtype = None
        self.loaded_model_dict = None  # kijai's format: {model, processor, dtype}
        
        # Get model path
        self.model_path = os.path.join(comfyui_base_path, "models", "LLM", model_name)
        
        # Verify model exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
    
    def _load_model_kijai_style(self):
        """
        Load model using kijai's exact Florence2ModelLoader logic.
        This replicates the loadmodel() function from kijai's Florence2ModelLoader class.
        """
        if self.loaded_model_dict is not None:
            return self.loaded_model_dict
        
        # Import transformers only when actually loading model
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForSeq2SeqLM
            import transformers
        except ImportError:
            raise ImportError("transformers library required. Install with: pip install transformers")
            
        print(f"Loading model from {self.model_path}")
        print(f"Florence2 using {self.attention} for attention")
        print(f"Transformers version: {transformers.__version__}")
        
        # kijai's exact device and dtype setup
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.precision]
        
        # kijai's safetensors conversion logic
        if self.convert_to_safetensors:
            model_weight_path = os.path.join(self.model_path, 'pytorch_model.bin')
            if os.path.exists(model_weight_path):
                safetensors_weight_path = os.path.join(self.model_path, 'model.safetensors')
                print(f"Converting {model_weight_path} to {safetensors_weight_path}")
                if not os.path.exists(safetensors_weight_path):
                    from safetensors.torch import save_file
                    checkpoint = torch.load(model_weight_path, map_location="cpu")
                    save_file(checkpoint, safetensors_weight_path)
                    print(f"Deleting original file: {model_weight_path}")
                    os.remove(model_weight_path)
                    print(f"Original {model_weight_path} file deleted.")
        
        # Load model with proper error handling
        from unittest.mock import patch
        
        print("Loading with trust_remote_code approach...")
        try:
            # Your config.json specifies AutoModelForSeq2SeqLM, not AutoModelForCausalLM!
            with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_path,
                    attn_implementation=self.attention,
                    torch_dtype=self.dtype,
                    trust_remote_code=True
                ).to(offload_device)
                print("AutoModelForSeq2SeqLM loading successful")
                
        except Exception as e:
            print(f"AutoModelForSeq2SeqLM failed: {e}")
            print("Trying AutoModelForCausalLM as fallback...")
            
            # Fallback to AutoModelForCausalLM for older transformers compatibility
            try:
                with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        attn_implementation=self.attention,
                        torch_dtype=self.dtype,
                        trust_remote_code=True
                    ).to(offload_device)
                    print("AutoModelForCausalLM fallback successful")
                    
            except Exception as e2:
                print(f"Both loading methods failed!")
                print(f"AutoModelForSeq2SeqLM error: {e}")
                print(f"AutoModelForCausalLM error: {e2}")
                raise Exception(f"Could not load model with either method. Check config.json auto_map.")
        
        # Apply targeted DaViT patch after model is loaded
        if hasattr(self, 'model') and self.model is not None:
            try:
                # Recursively patch any DaViT components in the loaded model
                def patch_recursive(obj, visited=None):
                    if visited is None:
                        visited = set()
                    
                    if id(obj) in visited:
                        return
                    visited.add(id(obj))
                    
                    # Apply the patcher
                    davit_patcher(obj)
                    
                    # Recursively check submodules
                    if hasattr(obj, '__dict__'):
                        for attr_value in obj.__dict__.values():
                            if hasattr(attr_value, '__dict__'):
                                patch_recursive(attr_value, visited)
                    
                    # Check PyTorch module children
                    if hasattr(obj, 'children'):
                        try:
                            for child in obj.children():
                                patch_recursive(child, visited)
                        except Exception:
                            pass
                
                patch_recursive(self.model)
                print("DaViT patching complete")
                
            except Exception as e:
                print(f"DaViT patching skipped: {e}")
        
        # Load processor - kijai's approach
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        
        # Create kijai's exact return format
        self.loaded_model_dict = {
            'model': self.model, 
            'processor': self.processor,
            'dtype': self.dtype
        }
        
        print(f"Model loaded successfully on {offload_device}")
        return self.loaded_model_dict


    
    def detect_objects(self, 
                      image: Union[Image.Image, np.ndarray, torch.Tensor], 
                      text_input: str = "rectangular images in page",
                      task: str = "caption_to_phrase_grounding",
                      fill_mask: bool = True,
                      keep_model_loaded: bool = True,
                      max_new_tokens: int = 1024,
                      num_beams: int = 3,
                      do_sample: bool = True) -> List[BoundingBox]:
        """
        Detect objects using kijai's complete workflow.
        This replicates what kijai's Florence2Run does after getting the model from Florence2ModelLoader.
        
        Args:
            image: Input image
            text_input: Prompt text (kijai's parameter name)
            task: Florence2 task type (kijai's parameter)
            fill_mask: Whether to fill masks (kijai's parameter)
            keep_model_loaded: Whether to keep model in GPU memory (kijai's parameter)
            max_new_tokens: Maximum tokens to generate (kijai's parameter)
            num_beams: Number of beams for generation (kijai's parameter)
            do_sample: Whether to use sampling (kijai's parameter)
            
        Returns:
            List of BoundingBox objects for detected regions (filtered by min_box_area)
        """
        # Load model using kijai's approach
        florence2_model = self._load_model_kijai_style()
        
        # Extract components (kijai's format)
        model = florence2_model['model']
        processor = florence2_model['processor']
        dtype = florence2_model['dtype']
        
        # kijai's device management
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        # Move model to GPU for inference (kijai's approach)
        model.to(device)
        
        # Convert image to PIL format
        pil_image = self._preprocess_image(image)
        
        # Prepare task prompt - kijai's exact format
        task_prompt = f"<{task}>"
        if text_input:
            full_prompt = f"{task_prompt}{text_input}"
        else:
            full_prompt = task_prompt
        
        print(f"Running Florence2 inference with prompt: '{full_prompt}'")
        
        try:
            # Process inputs - kijai's exact approach
            inputs = processor(
                text=full_prompt,
                images=pil_image,
                return_tensors="pt"
            ).to(device)
            
            # Run generation - kijai's exact parameters from Florence2Run
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample
                )
            
            # Decode results
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            print(f"Generated text: {generated_text}")
            
            # Parse results using processor's post_process_generation
            parsed_answer = processor.post_process_generation(
                generated_text, 
                task=task, 
                image_size=(pil_image.width, pil_image.height)
            )
            
            print(f"Parsed answer: {parsed_answer}")
            
            # Extract bounding boxes
            boxes = self._extract_bounding_boxes(parsed_answer, task)
            
            # Filter by minimum area (our addition)
            filtered_boxes = self._filter_small_boxes(boxes)
            
            print(f"Detected {len(filtered_boxes)} objects after filtering")
            return filtered_boxes
            
        finally:
            # Handle model offloading - kijai's exact approach
            if not keep_model_loaded:
                print("Offloading model...")
                model.to(offload_device)
                mm.soft_empty_cache()
    
    def _extract_bounding_boxes(self, parsed_answer: Dict[str, Any], task: str) -> List[BoundingBox]:
        """Extract bounding boxes from Florence2 parsed output."""
        boxes = []
        
        if task in parsed_answer:
            result = parsed_answer[task]
            
            if 'bboxes' in result:
                bboxes = result['bboxes']
                
                for bbox in bboxes:
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        # Ensure valid coordinates
                        box = BoundingBox(
                            x1=int(max(0, x1)), 
                            y1=int(max(0, y1)), 
                            x2=int(x2), 
                            y2=int(y2)
                        )
                        
                        # Only add valid boxes
                        if box.width > 0 and box.height > 0:
                            boxes.append(box)
        
        return boxes
    
    def _filter_small_boxes(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """Filter out boxes smaller than minimum area threshold."""
        filtered = []
        
        for box in boxes:
            if box.area >= self.min_box_area:
                filtered.append(box)
            else:
                print(f"Filtering small box: {box}")
        
        return filtered
    
    def _preprocess_image(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """Convert input image to PIL Image format."""
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, np.ndarray):
            # Handle numpy arrays
            if image.dtype in [np.float32, np.float64]:
                image = (image * 255).astype(np.uint8)
            if len(image.shape) == 4:
                image = image[0]  # Remove batch dimension
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))  # CHW to HWC
            return Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            # Handle torch tensors
            if image.dim() == 4:
                image = image[0]  # Remove batch dimension
            if image.shape[0] == 3:
                image = image.permute(1, 2, 0)  # CHW to HWC
            if image.dtype in [torch.float32, torch.float16]:
                image = (image * 255).clamp(0, 255).byte()
            return Image.fromarray(image.cpu().numpy())
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the detector configuration."""
        # Import transformers only when needed
        try:
            import transformers
            transformers_version = transformers.__version__
        except ImportError:
            transformers_version = "Not installed"
            
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "precision": self.precision,
            "attention": self.attention,
            "min_box_area": self.min_box_area,
            "convert_to_safetensors": self.convert_to_safetensors,
            "loaded": self.loaded_model_dict is not None,
            "transformers_version": transformers_version
        }
    
    def update_settings(self, min_box_area: Optional[int] = None):
        """Update settings without reloading model."""
        if min_box_area is not None:
            self.min_box_area = min_box_area


# Utility functions
def visualize_detections(image: Union[Image.Image, np.ndarray], 
                        boxes: List[BoundingBox],
                        show_labels: bool = True) -> Image.Image:
    """Draw bounding boxes on image."""
    from PIL import ImageDraw, ImageFont
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    
    for i, box in enumerate(boxes):
        color = colors[i % len(colors)]
        
        # Draw rectangle
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=color, width=3)
        
        # Draw label
        if show_labels:
            label = f"Area: {box.area}"
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            draw.text((box.x1, box.y1 - 20), label, fill=color, font=font)
    
    return vis_image


def crop_images_from_boxes(image: Union[Image.Image, np.ndarray], 
                          boxes: List[BoundingBox]) -> List[Image.Image]:
    """Crop individual images from bounding boxes."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    cropped_images = []
    for box in boxes:
        cropped = image.crop((box.x1, box.y1, box.x2, box.y2))
        cropped_images.append(cropped)
    
    return cropped_images


# Example usage
if __name__ == "__main__":
    # Initialize detector with your exact kijai settings
    detector = ImageRectangleDetector(
        model_name="CogFlorence-2.2-Large",
        comfyui_base_path="A:\\Comfy_Dec\\ComfyUI",
        precision="fp16",           # kijai's Florence2ModelLoader param
        attention="sdpa",           # kijai's Florence2ModelLoader param  
        min_box_area=1000,
        convert_to_safetensors=False  # kijai's Florence2ModelLoader param
    )
    
    print("Detector configuration:")
    model_info = detector.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Example usage (replace with your image)
    image_path = "test_image.jpg"
    if os.path.exists(image_path):
        image = Image.open(image_path)
        
        # Detect objects with your exact kijai settings
        boxes = detector.detect_objects(
            image=image,
            text_input="rectangular images in page",  # kijai's parameter name
            task="caption_to_phrase_grounding",       # kijai's parameter
            fill_mask=True,                          # kijai's parameter
            keep_model_loaded=True,                  # kijai's parameter
            max_new_tokens=1024,                     # kijai's parameter (you use 2048)
            num_beams=3,                             # kijai's parameter
            do_sample=True                           # kijai's parameter
        )
        
        print(f"\nFound {len(boxes)} objects:")
        for i, box in enumerate(boxes):
            print(f"  {i+1}: {box}")
        
        # Visualize and save results
        if boxes:
            vis_image = visualize_detections(image, boxes)
            vis_image.save("detection_results.jpg")
            
            cropped_images = crop_images_from_boxes(image, boxes)
            for i, cropped in enumerate(cropped_images):
                cropped.save(f"cropped_{i+1}.jpg")
            
            print("Results saved!")
    else:
        print("Please provide a test image to run the example")