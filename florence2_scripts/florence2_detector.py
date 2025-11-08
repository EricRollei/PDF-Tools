"""
Florence2 Detector

Description: Florence2 vision model integration for object detection, image analysis, and caption generation
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
- Uses Florence-2 models (MIT License) by Microsoft: https://huggingface.co/microsoft/Florence-2-large
- See CREDITS.md for complete list of all dependencies
"""

"""
Florence2 Rectangle Detector - Clean Implementation
Based on kijai's proven ComfyUI-Florence2 implementation

Combines Florence2ModelLoader + Florence2Run into a single callable class
for detecting rectangular regions in images.
"""

import torch
import numpy as np
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict, Any
from PIL import Image, ImageDraw, ImageColor
import io
import importlib.util

# Import kijai's exact flash_attn fix
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

def fixed_get_imports(filename):
    """Fix for unnecessary flash_attn requirement - exactly from kijai."""
    try:
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        imports.remove("flash_attn")
    except:
        print(f"No flash_attn import to remove")
        pass
    return imports

# ComfyUI model management
try:
    import comfy.model_management as mm
    COMFYUI_AVAILABLE = True
except ImportError:
    print("ComfyUI not available, using fallback memory management")
    COMFYUI_AVAILABLE = False
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

# Import transformers (kijai's approach)
import transformers
from transformers import AutoModelForCausalLM, AutoProcessor
from safetensors.torch import save_file

# Handle different import contexts (ComfyUI vs test scripts)
def get_current_directory():
    """Get current directory for imports"""
    return os.path.dirname(os.path.abspath(__file__))

def import_modeling_florence2():
    """Import modeling_florence2 with multiple fallback strategies"""
    try:
        # Strategy 1: Relative import (works in ComfyUI)
        from .modeling_florence2 import Florence2ForConditionalGeneration
        return Florence2ForConditionalGeneration, "relative"
    except ImportError:
        try:
            # Strategy 2: Absolute import (works in some test contexts)
            from modeling_florence2 import Florence2ForConditionalGeneration
            return Florence2ForConditionalGeneration, "absolute"
        except ImportError:
            try:
                # Strategy 3: Direct importlib (works in test scripts)
                import importlib.util
                modeling_path = os.path.join(get_current_directory(), "modeling_florence2.py")
                if os.path.exists(modeling_path):
                    spec = importlib.util.spec_from_file_location("modeling_florence2", modeling_path)
                    modeling_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(modeling_module)
                    return modeling_module.Florence2ForConditionalGeneration, "importlib"
                else:
                    raise ImportError(f"modeling_florence2.py not found at {modeling_path}")
            except Exception as e:
                raise ImportError(f"All import strategies failed: {e}")

@dataclass
class BoundingBox:
    """Simple bounding box with label."""
    x1: int
    y1: int 
    x2: int
    y2: int
    label: str = ""
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
        return f"BoundingBox({self.x1}, {self.y1}, {self.x2}, {self.y2}, '{self.label}', area={self.area}{conf_str})"


class Florence2RectangleDetector:
    """
    Single class that combines kijai's Florence2ModelLoader + Florence2Run
    for detecting rectangular objects in images.
    
    Based on the proven working implementation from ComfyUI-Florence2.
    """
    
    def __init__(self, 
                 model_name: str = "CogFlorence-2.2-Large",
                 comfyui_base_path: str = ".",
                 min_box_area: int = 1000,
                 device: str = "auto"):
        """
        Initialize detector with kijai's proven settings.
        
        Args:
            model_name: Model directory name in ComfyUI/models/LLM/
            comfyui_base_path: Path to ComfyUI installation
            min_box_area: Minimum bounding box area to keep
        """
        self.model_name = model_name
        self.comfyui_base_path = comfyui_base_path
        self.min_box_area = min_box_area

        # Set up device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set up model attributes BEFORE any model loading
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.attention_implementation = "sdpa" if self.device == "cuda" else None
        
        # Fixed settings based on your requirements
        self.precision = "fp16"
        self.attention = "sdpa"
        self.convert_to_safetensors = False
        
        # Model state - loaded lazily
        self.florence2_model = None
        self.processor = None

        # Model path setup
        if "/" in model_name:
            # HuggingFace model
            self.model_path = model_name
        else:
            # Local model
            self.model_path = os.path.join(comfyui_base_path, "models", "LLM", model_name)
        
        # Verify model exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        print(f"Florence2RectangleDetector initialized")
        print(f"Model path: {self.model_path}")
        print(f"Settings: {self.precision}, {self.attention}, min_area={self.min_box_area}")
        print(f"Settings: {'fp16' if self.torch_dtype == torch.float16 else 'fp32'}, {self.attention_implementation or 'default'}, min_area={min_box_area}")

    def _import_local_modules(self):
        """Import local modeling and configuration modules with multiple strategies"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try to import modeling_florence2
        modeling_florence2 = None
        configuration_florence2 = None
        
        try:
            # Strategy 1: Relative import (works in ComfyUI package context)
            from .modeling_florence2 import Florence2ForConditionalGeneration
            from .configuration_florence2 import Florence2Config
            modeling_florence2 = Florence2ForConditionalGeneration
            configuration_florence2 = Florence2Config
            print("✅ Using local florence2 modules (relative import)")
            
        except ImportError:
            try:
                # Strategy 2: Absolute import (works when module is in path)
                from modeling_florence2 import Florence2ForConditionalGeneration
                from configuration_florence2 import Florence2Config
                modeling_florence2 = Florence2ForConditionalGeneration
                configuration_florence2 = Florence2Config
                print("✅ Using local florence2 modules (absolute import)")
                
            except ImportError:
                try:
                    # Strategy 3: Direct file import using importlib (works in test scripts)
                    modeling_path = os.path.join(current_dir, "modeling_florence2.py")
                    config_path = os.path.join(current_dir, "configuration_florence2.py")
                    
                    if os.path.exists(modeling_path) and os.path.exists(config_path):
                        # Import modeling_florence2
                        spec = importlib.util.spec_from_file_location("modeling_florence2", modeling_path)
                        modeling_module = importlib.util.module_from_spec(spec)
                        sys.modules["modeling_florence2"] = modeling_module  # Add to sys.modules
                        spec.loader.exec_module(modeling_module)
                        
                        # Import configuration_florence2
                        spec2 = importlib.util.spec_from_file_location("configuration_florence2", config_path)
                        config_module = importlib.util.module_from_spec(spec2)
                        sys.modules["configuration_florence2"] = config_module  # Add to sys.modules
                        spec2.loader.exec_module(config_module)
                        
                        modeling_florence2 = modeling_module.Florence2ForConditionalGeneration
                        configuration_florence2 = config_module.Florence2Config
                        print("✅ Using local florence2 modules (importlib)")
                    else:
                        raise ImportError(f"Local modules not found: {modeling_path}, {config_path}")
                        
                except Exception as e:
                    print(f"❌ All local import strategies failed: {e}")
                    return None, None
        
        return modeling_florence2, configuration_florence2


    def _load_model(self):
        """Load Florence2 model with enhanced import handling"""
        if self.florence2_model is not None:
            return self.florence2_model
        
        print(f"Loading Florence2 model from {self.model_path}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Enhanced import strategy
        Florence2ForConditionalGeneration = None
        
        try:
            # Try ComfyUI context first
            try:
                from .modeling_florence2 import Florence2ForConditionalGeneration
                print("✅ Using relative import (ComfyUI context)")
            except ImportError:
                # For test scripts: Create a proper package context
                import importlib.util
                import sys
                current_dir = os.path.dirname(os.path.abspath(__file__))
                
                # Create package structure in sys.modules to support relative imports
                package_name = "florence2_package"  # Temporary package name
                
                # Load configuration as part of the package
                config_path = os.path.join(current_dir, "configuration_florence2.py")
                if os.path.exists(config_path):
                    spec = importlib.util.spec_from_file_location(f"{package_name}.configuration_florence2", config_path)
                    config_module = importlib.util.module_from_spec(spec)
                    # Set up the module as part of a package
                    config_module.__package__ = package_name
                    sys.modules[f"{package_name}.configuration_florence2"] = config_module
                    sys.modules["configuration_florence2"] = config_module  # Also add without package
                    spec.loader.exec_module(config_module)
                
                # Load modeling as part of the package
                modeling_path = os.path.join(current_dir, "modeling_florence2.py")
                if os.path.exists(modeling_path):
                    spec2 = importlib.util.spec_from_file_location(f"{package_name}.modeling_florence2", modeling_path)
                    modeling_module = importlib.util.module_from_spec(spec2)
                    # Set up the module as part of a package  
                    modeling_module.__package__ = package_name
                    sys.modules[f"{package_name}.modeling_florence2"] = modeling_module
                    sys.modules["modeling_florence2"] = modeling_module  # Also add without package
                    spec2.loader.exec_module(modeling_module)
                    
                    Florence2ForConditionalGeneration = modeling_module.Florence2ForConditionalGeneration
                    print("✅ Using importlib with package context (test script)")
                else:
                    raise ImportError("modeling_florence2.py not found")
            
            # Use your proven loading approach
            model = Florence2ForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                attn_implementation=self.attention_implementation,
                trust_remote_code=True
            )
            print("✅ Florence2 model loaded with local implementation")
            
        except Exception as e:
            print(f"❌ Local implementation failed: {e}")
            
            # Fallback to transformers
            try:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=True
                )
                print("✅ Florence2 model loaded with transformers fallback")
            except Exception as e2:
                print(f"❌ All approaches failed: {e2}")
                raise e2
        
        model.to(self.device)
        model.eval()
        self.florence2_model = model
        return model

    def _load_processor(self):
        """Load Florence2 processor"""
        if self.processor is not None:
            return self.processor
        
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            self.processor = processor
            return processor
            
        except Exception as e:
            print(f"❌ Processor loading failed: {e}")
            raise e

    def detect_rectangles(self, 
                        image: Union[Image.Image, np.ndarray, torch.Tensor], 
                        text_input: str = "rectangular images in page",
                        return_mask: bool = True,
                        keep_model_loaded: bool = True) -> Tuple[List[BoundingBox], Optional[Image.Image]]:
        """
        Detect rectangular objects using Florence2.
        
        Args:
            image: Input image (PIL, numpy array, or torch tensor)
            text_input: Text prompt for detection
            return_mask: Whether to return a mask image
            keep_model_loaded: Whether to keep model in GPU memory
            
        Returns:
            Tuple of (bounding_boxes, mask_image)
        """
        # Load model and processor
        florence2_model = self._load_model()  # This returns the model object
        processor = self._load_processor()    # This returns the processor object
        
        # Device management
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        # Move model to GPU for inference
        florence2_model.to(device)
        
        # Convert input to PIL Image
        image_pil = self._to_pil_image(image)
        W, H = image_pil.size
        
        # Prepare prompt for detection
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        
        if text_input:
            prompt = task_prompt + " " + text_input
        else:
            prompt = task_prompt
        
        print(f"Running Florence2 inference with prompt: '{prompt}'")
        
        try:
            # Process inputs
            inputs = processor(
                text=prompt, 
                images=image_pil, 
                return_tensors="pt", 
                do_rescale=False
            ).to(self.torch_dtype).to(device)
            
            # Generate
            generated_ids = florence2_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=True,
                num_beams=3,
            )
            
            # Decode results
            results = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            print(f"Generated text: {results}")
            
            # Parse results
            parsed_answer = processor.post_process_generation(
                results, 
                task=task_prompt, 
                image_size=(W, H)
            )
            
            print(f"Parsed answer: {parsed_answer}")
            
            # Extract bounding boxes and labels
            bounding_boxes = []
            mask_image = None
            
            if task_prompt in parsed_answer:
                result = parsed_answer[task_prompt]
                
                if 'bboxes' in result and 'labels' in result:
                    bboxes = result['bboxes']
                    labels = result['labels']
                    
                    # Create mask if requested
                    if return_mask:
                        mask_image = Image.new('RGB', (W, H), 'black')
                        mask_draw = ImageDraw.Draw(mask_image)
                    
                    # Process each detection
                    for bbox, label in zip(bboxes, labels):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            
                            # Create bounding box object
                            box = BoundingBox(
                                x1=int(max(0, x1)), 
                                y1=int(max(0, y1)), 
                                x2=int(min(W, x2)), 
                                y2=int(min(H, y2)),
                                label=str(label)
                            )
                            
                            # Filter by minimum area
                            if box.width > 0 and box.height > 0 and box.area >= self.min_box_area:
                                bounding_boxes.append(box)
                                
                                # Add to mask
                                if return_mask and mask_image:
                                    mask_draw.rectangle([box.x1, box.y1, box.x2, box.y2], fill='white')
                            else:
                                print(f"Filtering small/invalid box: {box}")
            
            print(f"Detected {len(bounding_boxes)} objects after filtering")
            
            return bounding_boxes, mask_image
            
        finally:
            # Handle model offloading
            if not keep_model_loaded:
                print("Offloading model...")
                florence2_model.to(offload_device)
                mm.soft_empty_cache()

    def _to_pil_image(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """Convert various image formats to PIL Image."""
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, np.ndarray):
            # Handle numpy arrays
            if image.dtype in [np.float32, np.float64]:
                image = (image * 255).astype(np.uint8)
            if len(image.shape) == 4:
                image = image[0]  # Remove batch dimension
            if len(image.shape) == 3 and image.shape[0] in [1, 3, 4]:
                image = np.transpose(image, (1, 2, 0))  # CHW to HWC
            return Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            # Handle torch tensors - kijai's approach using torchvision
            import torchvision.transforms.functional as F
            if image.dim() == 4:
                image = image[0]  # Remove batch dimension
            return F.to_pil_image(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def crop_detected_regions(self, 
                            image: Union[Image.Image, np.ndarray, torch.Tensor], 
                            bounding_boxes: List[BoundingBox]) -> List[Image.Image]:
        """Crop individual regions from the original image."""
        image_pil = self._to_pil_image(image)
        
        cropped_images = []
        for box in bounding_boxes:
            cropped = image_pil.crop((box.x1, box.y1, box.x2, box.y2))
            cropped_images.append(cropped)
        
        return cropped_images

    def visualize_detections(self, 
                           image: Union[Image.Image, np.ndarray, torch.Tensor], 
                           bounding_boxes: List[BoundingBox],
                           show_labels: bool = True) -> Image.Image:
        """Draw bounding boxes on image - similar to kijai's visualization."""
        image_pil = self._to_pil_image(image)
        vis_image = image_pil.copy()
        draw = ImageDraw.Draw(vis_image)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'yellow', 'magenta']
        
        for i, box in enumerate(bounding_boxes):
            color = colors[i % len(colors)]
            
            # Draw rectangle
            draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=color, width=2)
            
            # Draw label
            if show_labels and box.label:
                label_text = f"{i}: {box.label} (area: {box.area})"
                # Simple text positioning
                text_x = box.x1
                text_y = max(0, box.y1 - 20)
                draw.text((text_x, text_y), label_text, fill=color)
        
        return vis_image

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the detector configuration."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "precision": self.precision,
            "attention": self.attention,
            "min_box_area": self.min_box_area,
            "convert_to_safetensors": self.convert_to_safetensors,
            "loaded": self.florence2_model is not None,
            "transformers_version": transformers.__version__,
            "comfyui_available": COMFYUI_AVAILABLE
        }

    def update_settings(self, min_box_area: Optional[int] = None):
        """Update settings without reloading model."""
        if min_box_area is not None:
            self.min_box_area = min_box_area
            print(f"Updated min_box_area to {min_box_area}")


# Convenience function for simple usage
def detect_rectangles_in_image(image_path: str, 
                             text_prompt: str = "rectangular images in page",
                             model_name: str = "CogFlorence-2.2-Large",
                             comfyui_base_path: str = ".") -> Tuple[List[BoundingBox], Image.Image]:
    """
    Simple function to detect rectangles in an image file.
    
    Args:
        image_path: Path to image file
        text_prompt: Detection prompt
        model_name: Florence2 model name
        comfyui_base_path: Path to ComfyUI installation
        
    Returns:
        Tuple of (bounding_boxes, mask_image)
    """
    detector = Florence2RectangleDetector(
        model_name=model_name,
        comfyui_base_path=comfyui_base_path
    )
    
    image = Image.open(image_path)
    boxes, mask = detector.detect_rectangles(image, text_prompt)
    
    return boxes, mask


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = Florence2RectangleDetector(
        model_name="CogFlorence-2.2-Large",
        comfyui_base_path="A:\\Comfy_Dec\\ComfyUI",  # Update this path
        min_box_area=1000
    )
    
    # Print model info
    print("Detector configuration:")
    info = detector.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Example detection (update image path)
    image_path = "test_image.jpg"
    if os.path.exists(image_path):
        print(f"\nProcessing {image_path}...")
        
        image = Image.open(image_path)
        
        # Detect rectangles
        boxes, mask = detector.detect_rectangles(
            image=image,
            text_input="rectangular images in page",
            return_mask=True,
            keep_model_loaded=True
        )
        
        print(f"\nFound {len(boxes)} rectangles:")
        for i, box in enumerate(boxes):
            print(f"  {i+1}: {box}")
        
        # Save results
        if boxes:
            # Visualize detections
            vis_image = detector.visualize_detections(image, boxes)
            vis_image.save("detection_results.jpg")
            
            # Save mask
            if mask:
                mask.save("detection_mask.jpg")
            
            # Crop individual rectangles
            cropped_images = detector.crop_detected_regions(image, boxes)
            for i, cropped in enumerate(cropped_images):
                cropped.save(f"cropped_rectangle_{i+1}.jpg")
            
            print("Results saved!")
        else:
            print("No rectangles detected.")
    else:
        print(f"Test image not found: {image_path}")
        print("Update the image_path variable to test detection.")
