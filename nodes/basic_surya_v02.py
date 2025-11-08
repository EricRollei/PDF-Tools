"""
Basic Surya V02

Description: Surya OCR integration for multilingual text recognition and layout analysis
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
- Uses Surya OCR (GNU GPL v3) by Vik Paruchuri: https://github.com/VikParuchuri/surya
- See CREDITS.md for complete list of all dependencies
"""

"""Basic Surya OCR node for ComfyUI."""

import os
import types
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from PIL import Image

# Optional Surya imports are resolved lazily so the node can load without the
# dependency. This allows ComfyUI to start even if surya-ocr is not installed.
SURYA_IMPORT_ERROR: Optional[Exception] = None

try:
    from surya.detection import DetectionPredictor
    from surya.recognition import RecognitionPredictor
    SURYA_AVAILABLE = True
except Exception as exc:  # pragma: no cover - import guard
    SURYA_IMPORT_ERROR = exc
    DetectionPredictor = None  # type: ignore[assignment]
    RecognitionPredictor = None  # type: ignore[assignment]
    SURYA_AVAILABLE = False

try:  # Foundation predictor is only available in newer Surya versions
    from surya.foundation import FoundationPredictor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    FoundationPredictor = None  # type: ignore[assignment]

try:  # Layout predictor is optional
    from surya.layout import LayoutPredictor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LayoutPredictor = None  # type: ignore[assignment]

try:
    from surya.settings import settings as SURYA_SETTINGS  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SURYA_SETTINGS = None  # type: ignore[assignment]


class SuryaOCRNode:
    """ComfyUI node that wraps Surya OCR with minimal configuration."""

    def __init__(self):
        self.device = self._auto_device()
        self.foundation_predictor = None
        self.detection_predictor = None
        self.recognition_predictor = None
        self.layout_predictor = None
        self._initialize_predictors()

    @staticmethod
    def _auto_device() -> torch.device:
        """Return the best available device without forcing a specific GPU."""
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        print("Surya OCR node: CUDA not available, using CPU.")
        return torch.device("cpu")

    @staticmethod
    def _device_string(device: torch.device) -> str:
        if device.type != "cuda":
            return "cpu"
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{index}"

    @staticmethod
    def _cuda_index(device: torch.device) -> int:
        if device.index is not None:
            return device.index
        return torch.cuda.current_device()

    def _set_device(self, device: torch.device) -> None:
        """Switch internal state to a new device and re-create predictors."""
        if device == self.device:
            return
        self.device = device
        self.foundation_predictor = None
        self.detection_predictor = None
        self.recognition_predictor = None
        self.layout_predictor = None
        if self.device.type == "cuda":
            torch.cuda.set_device(self._cuda_index(self.device))
        self._initialize_predictors()

    def _resolve_force_device(self, value: str) -> torch.device:
        """Convert UI device selection into a torch.device instance."""
        normalized = value.lower()
        if normalized.startswith("cuda"):
            if not torch.cuda.is_available():
                print("Surya OCR node: CUDA requested but unavailable, falling back to CPU.")
                return torch.device("cpu")

            index = 0
            if ":" in normalized:
                try:
                    index = int(normalized.split(":", 1)[1])
                except ValueError:
                    print(f"Surya OCR node: Unable to parse device '{value}', using cuda:0.")
                    index = 0

            if index >= torch.cuda.device_count():
                print(f"Surya OCR node: Requested device cuda:{index} is out of range, using cuda:0.")
                index = 0

            return torch.device(f"cuda:{index}")

        return torch.device("cpu")

    def _initialize_predictors(self) -> None:
        if not SURYA_AVAILABLE:
            raise ImportError(
                "surya-ocr is not installed. Install it to use the Surya OCR node"
                + (f" (import error: {SURYA_IMPORT_ERROR})" if SURYA_IMPORT_ERROR else "")
            )

        if self.device.type == "cuda":
            torch.cuda.set_device(self._cuda_index(self.device))

        device_str = self._device_string(self.device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        recognition_foundation = None
        if FoundationPredictor is not None:
            foundation_kwargs: Dict[str, Any] = {
                "device": device_str,
                "dtype": dtype,
            }

            recognition_checkpoint = None
            if SURYA_SETTINGS is not None:
                recognition_checkpoint = getattr(SURYA_SETTINGS, "RECOGNITION_MODEL_CHECKPOINT", None)

            if recognition_checkpoint:
                foundation_kwargs["checkpoint"] = recognition_checkpoint

            try:
                foundation_kwargs["attention_implementation"] = "sdpa"
                recognition_foundation = cast(Any, FoundationPredictor)(**foundation_kwargs)
            except TypeError:
                foundation_kwargs.pop("attention_implementation", None)
                recognition_foundation = cast(Any, FoundationPredictor)(**foundation_kwargs)

        self.foundation_predictor = recognition_foundation

        detection_kwargs: Dict[str, Any] = {
            "device": device_str,
            "dtype": dtype,
        }

        if SURYA_SETTINGS is not None:
            detector_checkpoint = getattr(SURYA_SETTINGS, "DETECTOR_MODEL_CHECKPOINT", None)
            if detector_checkpoint:
                detection_kwargs["checkpoint"] = detector_checkpoint

        try:
            detection_kwargs["attention_implementation"] = "sdpa"
            self.detection_predictor = cast(Any, DetectionPredictor)(**detection_kwargs)
        except TypeError:
            detection_kwargs.pop("attention_implementation", None)
            self.detection_predictor = cast(Any, DetectionPredictor)(**detection_kwargs)

        if self.foundation_predictor is not None:
            self.recognition_predictor = cast(Any, RecognitionPredictor)(
                foundation_predictor=self.foundation_predictor
            )
            self._patch_foundation_embeddings(self.foundation_predictor)
        else:
            self.recognition_predictor = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "task_mode": (["ocr_with_boxes", "ocr_without_boxes", "detection_only"], {
                    "default": "ocr_with_boxes"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
            "optional": {
                "enable_layout": ("BOOLEAN", {"default": False}),
                "batch_size_override": ("INT", {
                    "default": 0,  # 0 = use environment defaults
                    "min": 0,
                    "max": 1024,
                    "step": 1
                }),
                "clear_gpu_cache": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Clear GPU cache before processing (helps with memory errors)"
                }),
                "force_device": (["auto", "cuda:0", "cuda:1", "cpu"], {
                    "default": "auto",
                    "tooltip": "Force specific device (use cuda:0 for RTX 4090)"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "LIST")
    RETURN_NAMES = ("text_output", "json_output", "annotated_image", "detection_data")
    FUNCTION = "process_ocr"
    CATEGORY = "text/ocr"
    
    def process_ocr(self, image: torch.Tensor, task_mode: str = "ocr_with_boxes", 
                   confidence_threshold: float = 0.5, enable_layout: bool = False,
                   batch_size_override: int = 0, clear_gpu_cache: bool = False, 
                   force_device: str = "auto") -> Tuple[str, str, torch.Tensor, List[Dict]]:
        """
        Main OCR processing function
        
        Args:
            image: Input image tensor from ComfyUI
            task_mode: Type of OCR task to perform
            confidence_threshold: Minimum confidence for text detection
            enable_layout: Whether to perform layout analysis
            batch_size_override: Override default batch sizes (0 = use defaults)
            
        Returns:
            Tuple of (text_output, json_output, annotated_image, detection_data)
        """
        
        original_recognition_batch: Optional[str] = None
        original_detector_batch: Optional[str] = None

        try:
            if force_device != "auto":
                requested_device = self._resolve_force_device(force_device)
                if requested_device != self.device:
                    self._set_device(requested_device)

            if clear_gpu_cache and self.device.type == "cuda":
                torch.cuda.empty_cache()

            # Clear GPU cache before processing to prevent fragmentation
            if self.device.type == "cuda" and torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated(self.device) / 1024**3
                print(f"GPU memory at start: {initial_memory:.2f}GB")
            
            # Override batch sizes if specified
            if batch_size_override > 0:
                original_recognition_batch = os.environ.get('RECOGNITION_BATCH_SIZE')
                original_detector_batch = os.environ.get('DETECTOR_BATCH_SIZE')

                os.environ['RECOGNITION_BATCH_SIZE'] = str(batch_size_override)
                os.environ['DETECTOR_BATCH_SIZE'] = str(max(1, min(batch_size_override // 10, 64)))
            
            # Convert ComfyUI tensor to PIL Image
            print(f"Input tensor device: {image.device}, target device: {self.device}")
            pil_images = self._tensor_to_pil_batch(image)
            
            # Process based on task mode
            if task_mode == "detection_only":
                results = self._process_detection_only(pil_images, confidence_threshold)
            elif task_mode in ["ocr_with_boxes", "ocr_without_boxes"]:
                results = self._process_full_ocr(pil_images, task_mode, confidence_threshold)
            else:
                raise ValueError(f"Unknown task mode: {task_mode}")
            
            # Optional layout analysis
            if enable_layout:
                layout_results = self._process_layout(pil_images)
                results['layout'] = layout_results
            
            # Generate outputs
            text_output = self._extract_text(results)
            json_output = self._format_json_output(results)
            annotated_image = self._create_annotated_image(pil_images[0], results, image.device)
            detection_data = self._format_detection_data(results)
            
            # Clear GPU cache after processing
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                final_memory = torch.cuda.memory_allocated(self.device) / 1024**3
                print(f"GPU memory at end: {final_memory:.2f}GB (device {torch.cuda.current_device()})")
            
            return (text_output, json_output, annotated_image, detection_data)
            
        except Exception as e:
            error_msg = f"OCR processing failed: {str(e)}"
            print(f"❌ {error_msg}")
            if torch.cuda.is_available():
                print(f"Device info: Current device={torch.cuda.current_device()}")
            print(f"Target device: {self.device}")
            
            # Clear GPU cache on error to help recovery
            if self.device.type == "cuda" and torch.cuda.is_available():
                print("🧹 Clearing GPU cache after error...")
                torch.cuda.set_device(self._cuda_index(self.device))
                torch.cuda.empty_cache()
            
            # Return error state
            empty_image = torch.zeros_like(image)
            return (error_msg, f'{{"error": "{error_msg}"}}', empty_image, [])
        
        finally:
            # Restore original batch sizes
            if batch_size_override > 0:
                if original_recognition_batch is not None:
                    os.environ['RECOGNITION_BATCH_SIZE'] = original_recognition_batch
                if original_detector_batch is not None:
                    os.environ['DETECTOR_BATCH_SIZE'] = original_detector_batch
    
    def _tensor_to_pil_batch(self, tensor: torch.Tensor) -> List[Image.Image]:
        """Convert ComfyUI image tensor to PIL Images"""
        # Ensure tensor is on CPU for PIL conversion
        if tensor.device != torch.device('cpu'):
            print(f"Moving tensor from {tensor.device} to CPU for PIL conversion")
            tensor = tensor.cpu()
        
        # ComfyUI format: [batch, height, width, channels] in 0-1 range
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        pil_images = []
        for i in range(tensor.shape[0]):
            # Convert to numpy and scale to 0-255
            img_array = (tensor[i].numpy() * 255).astype(np.uint8)
            
            # Ensure proper RGB format
            if img_array.shape[2] == 3:  # RGB
                pil_image = Image.fromarray(img_array, 'RGB')
            elif img_array.shape[2] == 4:  # RGBA
                pil_image = Image.fromarray(img_array, 'RGBA').convert('RGB')
            else:
                raise ValueError(f"Unsupported number of channels: {img_array.shape[2]}")
            
            pil_images.append(self._resize_for_model(pil_image))
        
        return pil_images

    def _resize_for_model(self, image: Image.Image) -> Image.Image:
        """Downscale oversized images to avoid Surya embedding mismatches."""
        max_dim = os.environ.get("SURYA_MAX_IMAGE_DIM")
        try:
            max_dim_value = int(max_dim) if max_dim else 4096
        except ValueError:
            print(f"Invalid SURYA_MAX_IMAGE_DIM value '{max_dim}', using default 4096")
            max_dim_value = 4096

        max_dim_value = max(512, max_dim_value)
        if max(image.size) <= max_dim_value:
            return image

        scale = max_dim_value / float(max(image.size))
        new_size = (
            max(1, int(image.width * scale)),
            max(1, int(image.height * scale)),
        )

        resample_attr = getattr(Image, "Resampling", None)
        resample_method = resample_attr.LANCZOS if resample_attr else Image.LANCZOS
        resized = image.resize(new_size, resample_method)
        print(
            f"Resized image from {image.size} to {resized.size} to satisfy Surya limits"
        )
        return resized

    def _patch_foundation_embeddings(self, foundation_predictor: Any) -> None:
        """Wrap Surya's embedding call to recover from known sequence mismatches."""
        model = getattr(foundation_predictor, "model", None)
        if model is None or getattr(model, "_pdf_tools_embedding_patched", False):
            return

        original_get_embeddings = model.get_image_embeddings
        node_self = self

        def safe_get_image_embeddings(model_self, pixel_values, grid_thw, encoder_chunk_size, valid_batch_size=None, max_batch_size=None):  # type: ignore[override]
            try:
                return original_get_embeddings(pixel_values, grid_thw, encoder_chunk_size, valid_batch_size, max_batch_size)
            except AssertionError as exc:
                print(f"Surya OCR node: correcting embedding mismatch ({exc})")
                return node_self._realign_image_embeddings(
                    model_self,
                    pixel_values,
                    grid_thw,
                    encoder_chunk_size,
                    valid_batch_size,
                    max_batch_size,
                )

        model.get_image_embeddings = types.MethodType(safe_get_image_embeddings, model)
        setattr(model, "_pdf_tools_embedding_patched", True)

    def _realign_image_embeddings(
        self,
        model: Any,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        encoder_chunk_size: int,
        valid_batch_size: Optional[torch.Tensor],
        max_batch_size: Optional[int],
    ) -> torch.Tensor:
        """Recompute embeddings without asserting exact sequence equality."""
        from surya.common.xla import mark_step

        chunks = [0]
        grid_chunks = [0]
        curr_chunk_len = 0
        curr_seq_len = 0

        for i in range(len(grid_thw)):
            curr_chunk_len += (grid_thw[i][0] * grid_thw[i][1] * grid_thw[i][2]).item()
            if curr_chunk_len > encoder_chunk_size:
                chunks.append(curr_chunk_len + curr_seq_len)
                curr_seq_len += curr_chunk_len
                curr_chunk_len = 0
                grid_chunks.append(i + 1)

        if curr_chunk_len > 0:
            chunks.append(pixel_values.shape[0])
            grid_chunks.append(len(grid_thw))

        embeddings_parts: List[torch.Tensor] = []
        for i in range(len(chunks) - 1):
            start = chunks[i]
            end = chunks[i + 1]
            grid_start = grid_chunks[i]
            grid_end = grid_chunks[i + 1]

            chunk_pixels = pixel_values[start:end]
            chunk_grid_thw = grid_thw[grid_start:grid_end]
            actual_chunk_len = end - start

            chunk_pixels, chunk_grid_thw, valid_embed_len = model.maybe_static_pad_image_inputs(
                chunk_pixels,
                chunk_grid_thw,
                actual_chunk_len,
                encoder_chunk_size,
            )

            chunk_embeddings = model.vision_encoder.embed_images(
                image_batch=chunk_pixels.unsqueeze(0).to(device=model.device),
                grid_thw=chunk_grid_thw.unsqueeze(0).to(device=model.device),
            )

            embeddings_parts.append(chunk_embeddings[:valid_embed_len].squeeze(0))
            mark_step()

        if not embeddings_parts:
            raise ValueError("No image embeddings were generated. Check the input images and grid sizes.")

        embeddings = embeddings_parts[0] if len(embeddings_parts) == 1 else torch.cat(embeddings_parts, dim=0)

        encoding_2d = model.get_2d_learned_embeddings(
            grid_thw,
            device=embeddings.device,
            bbox_size=model.config.image_embed_encoding_multiplier,
        )

        if embeddings.shape[0] != encoding_2d.shape[0]:
            min_len = min(embeddings.shape[0], encoding_2d.shape[0])
            print(
                f"Surya OCR node: trimming embeddings from {embeddings.shape[0]} to {min_len} tokens (encoding {encoding_2d.shape[0]})"
            )
            embeddings = embeddings[:min_len]
            encoding_2d = encoding_2d[:min_len]

        if embeddings.shape[1] != encoding_2d.shape[1]:
            min_dim = min(embeddings.shape[1], encoding_2d.shape[1])
            print(
                f"Surya OCR node: aligning embedding width from {embeddings.shape[1]} to {min_dim} (encoding {encoding_2d.shape[1]})"
            )
            embeddings = embeddings[:, :min_dim]
            encoding_2d = encoding_2d[:, :min_dim]

        return embeddings + encoding_2d
    
    def _process_detection_only(self, images: List[Image.Image], confidence_threshold: float) -> Dict:
        """Process text detection only"""
        if self.detection_predictor is None:
            raise RuntimeError("Surya OCR node: detection predictor not initialized")

        try:
            detections = self.detection_predictor(images)
            
            # Convert results to dictionaries and filter by confidence
            filtered_detections = []
            for i, detection in enumerate(detections):                
                # Access object attributes, not dictionary keys
                detection_dict = {
                    'bboxes': [],
                    'vertical_lines': getattr(detection, 'vertical_lines', []),
                    'page': getattr(detection, 'page', 0),
                    'image_bbox': getattr(detection, 'image_bbox', None)
                }
                
                # Process bboxes
                bboxes = getattr(detection, 'bboxes', [])
                
                for j, bbox in enumerate(bboxes):
                    # Convert bbox object to dict
                    bbox_dict = {
                        'bbox': getattr(bbox, 'bbox', None),
                        'polygon': getattr(bbox, 'polygon', None),
                        'confidence': getattr(bbox, 'confidence', 1.0)
                    }
                    
                    if bbox_dict['confidence'] >= confidence_threshold:
                        detection_dict['bboxes'].append(bbox_dict)
                
                filtered_detections.append(detection_dict)
            
            print(f"✅ Detection complete: {sum(len(d['bboxes']) for d in filtered_detections)} boxes found")
            return {'detections': filtered_detections}
            
        except Exception as e:
            print(f"Error in _process_detection_only: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _process_full_ocr(self, images: List[Image.Image], task_mode: str, confidence_threshold: float) -> Dict:
        """Process full OCR (detection + recognition)"""
        if self.recognition_predictor is None or self.detection_predictor is None:
            raise RuntimeError("Surya OCR node: predictors not initialized")

        try:
            # Run OCR with detection predictor
            ocr_results = self.recognition_predictor(images, det_predictor=self.detection_predictor)
            
            # Debug: Compare with standalone detection
            standalone_detections = self.detection_predictor(images)
            standalone_count = sum(len(getattr(d, 'bboxes', [])) for d in standalone_detections)
            
            # Convert results to dictionaries and filter by confidence
            filtered_results = []
            total_text_lines = 0
            total_lines_before_filter = 0
            
            for i, result in enumerate(ocr_results):
                # Access object attributes, not dictionary keys
                result_dict = {
                    'text_lines': [],
                    'page': getattr(result, 'page', 0),
                    'image_bbox': getattr(result, 'image_bbox', None)
                }
                
                # Process text lines
                text_lines = getattr(result, 'text_lines', [])
                total_lines_before_filter += len(text_lines)
                
                for j, line in enumerate(text_lines):
                    # Convert line object to dict - handle all nested objects
                    line_dict = {
                        'text': getattr(line, 'text', ''),
                        'confidence': getattr(line, 'confidence', 1.0),
                        'polygon': getattr(line, 'polygon', None),
                        'bbox': getattr(line, 'bbox', None),
                    }
                    
                    # Convert chars to serializable format
                    chars = getattr(line, 'chars', [])
                    line_dict['chars'] = []
                    for char in chars:
                        char_dict = {
                            'text': getattr(char, 'text', ''),
                            'bbox': getattr(char, 'bbox', None),
                            'polygon': getattr(char, 'polygon', None),
                            'confidence': getattr(char, 'confidence', 1.0),
                            'bbox_valid': getattr(char, 'bbox_valid', True)
                        }
                        line_dict['chars'].append(char_dict)
                    
                    # Convert words to serializable format
                    words = getattr(line, 'words', [])
                    line_dict['words'] = []
                    for word in words:
                        word_dict = {
                            'text': getattr(word, 'text', ''),
                            'bbox': getattr(word, 'bbox', None),
                            'polygon': getattr(word, 'polygon', None),
                            'confidence': getattr(word, 'confidence', 1.0),
                            'bbox_valid': getattr(word, 'bbox_valid', True)
                        }
                        line_dict['words'].append(word_dict)
                    
                    if line_dict['confidence'] >= confidence_threshold:
                        result_dict['text_lines'].append(line_dict)
                        total_text_lines += 1
                
                filtered_results.append(result_dict)
            
            # Debug output - compare detection vs OCR
            print(f"🔍 Detection comparison:")
            print(f"  Standalone detection found: {standalone_count} boxes")
            print(f"  OCR found: {total_lines_before_filter} lines")
            print(f"  OCR after confidence filter ({confidence_threshold}): {total_text_lines} lines")
            
            if standalone_count > total_text_lines:
                print(f"⚠️  OCR missed {standalone_count - total_text_lines} detections!")
                print(f"   Consider lowering confidence_threshold (current: {confidence_threshold})")
            
            print(f"✅ OCR complete: {total_text_lines} text lines extracted")
            return {'ocr_results': filtered_results, 'task_mode': task_mode}
            
        except Exception as e:
            print(f"Error in _process_full_ocr: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _process_layout(self, images: List[Image.Image]) -> Dict:
        """Process layout analysis"""
        if not self.layout_predictor:
            if LayoutPredictor is None:
                print("Surya OCR node: Layout predictor not available, skipping layout analysis.")
                return {}

            predictor_kwargs = {}
            if self.foundation_predictor is not None:
                predictor_kwargs['foundation_predictor'] = self.foundation_predictor

            try:
                self.layout_predictor = cast(Any, LayoutPredictor)(**predictor_kwargs)
            except TypeError:
                self.layout_predictor = cast(Any, LayoutPredictor)()
        
        layout_results = self.layout_predictor(images)
        return layout_results
    
    def _extract_text(self, results: Dict) -> str:
        """Extract plain text from results"""
        text_lines = []
        
        if 'ocr_results' in results:
            for page_result in results['ocr_results']:
                for line in page_result.get('text_lines', []):
                    text_lines.append(line.get('text', ''))
        
        return '\n'.join(text_lines)
    
    def _format_json_output(self, results: Dict) -> str:
        """Format results as JSON string with safe serialization"""
        import json
        
        def make_json_serializable(obj):
            """Recursively convert objects to JSON-serializable format"""
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return list(obj)  # Convert tuples to lists
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif hasattr(obj, '__dict__'):
                # Convert custom objects to dictionaries
                return make_json_serializable(obj.__dict__)
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            else:
                # Fallback: convert to string
                return str(obj)
        
        try:
            # Make results completely serializable
            safe_results = make_json_serializable(results)
            return json.dumps(safe_results, indent=2, ensure_ascii=False)
        except Exception as e:
            error_msg = f"Failed to serialize results: {str(e)}"
            print(f"JSON serialization error: {error_msg}")
            return f'{{"error": "{error_msg}", "partial_results": {{"text_count": {len(results.get("ocr_results", []))}}}}}'
    
    def _create_annotated_image(self, original_image: Image.Image, results: Dict, target_device: torch.device = None) -> torch.Tensor:
        """Create annotated image with bounding boxes"""
        try:
            from PIL import ImageDraw, ImageFont
            
            # Create copy for annotation
            annotated = original_image.copy()
            draw = ImageDraw.Draw(annotated)
            
            # Try to load a font, fallback to default
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            # Draw bounding boxes
            if 'ocr_results' in results:
                for page_result in results['ocr_results']:
                    for line in page_result.get('text_lines', []):
                        bbox = line.get('bbox')
                        if bbox and len(bbox) >= 4:
                            # Draw rectangle
                            draw.rectangle(bbox, outline='red', width=2)
                            
                            # Draw text if available
                            text = line.get('text', '')[:50]  # Truncate long text
                            if text:
                                # Position text above the box
                                text_y = max(5, bbox[1] - 15)
                                draw.text((bbox[0], text_y), text, fill='red', font=font)
            
            elif 'detections' in results:
                for page_result in results['detections']:
                    for bbox_info in page_result.get('bboxes', []):
                        bbox = bbox_info.get('bbox')
                        if bbox and len(bbox) >= 4:
                            draw.rectangle(bbox, outline='blue', width=2)
                            
                            # Draw confidence score
                            confidence = bbox_info.get('confidence', 1.0)
                            conf_text = f"{confidence:.2f}"
                            text_y = max(5, bbox[1] - 15)
                            draw.text((bbox[0], text_y), conf_text, fill='blue', font=font)
            
            # Convert back to tensor
            img_array = np.array(annotated).astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            # Move to target device if specified
            if target_device is not None and target_device != torch.device('cpu'):
                tensor = tensor.to(target_device)
            
            return tensor
            
        except Exception as e:
            print(f"Failed to create annotated image: {e}")
            # Return original image as tensor
            img_array = np.array(original_image).astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_array).unsqueeze(0)

            # Move to target device if specified
            if target_device is not None and target_device != torch.device('cpu'):
                tensor = tensor.to(target_device)

            return tensor
    
    def _format_detection_data(self, results: Dict) -> List[Dict]:
        """Format detection data for downstream nodes"""
        detection_data = []
        
        if 'ocr_results' in results:
            for page_result in results['ocr_results']:
                for line in page_result.get('text_lines', []):
                    detection_data.append({
                        'text': line.get('text', ''),
                        'bbox': line.get('bbox'),
                        'confidence': line.get('confidence', 1.0),
                        'polygon': line.get('polygon'),
                    })
        
        return detection_data

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SuryaOCR_v02": SuryaOCRNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SuryaOCR_v02": "Surya OCR v02 (RTX 4090 Optimized)"
}

