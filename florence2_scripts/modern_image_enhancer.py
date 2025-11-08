"""
Modern Image Enhancer

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
Modern Image Enhancer v2.0 - GPU Accelerated with Advanced Algorithms
High-quality enhancement using modern algorithms with GPU acceleration

Features:
- GPU-accelerated wavelet denoising using CuPy
- GPU-based selective sharpening with frequency domain processing
- Fast LAB color enhancement on GPU
- Advanced tone mapping with GPU acceleration
- Florence2-guided selective enhancement (GPU-accelerated)
- Maintains all quality-focused algorithms from v1.0

Author: Eric Hiss & Claude Sonnet 4
Version: 2.0 - GPU Accelerated Advanced
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from typing import Optional, List, Dict, Any, Tuple
import cv2
import time
import os

# GPU acceleration imports
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    import cupyx.scipy.signal as cp_signal
    CUPY_AVAILABLE = True
    print("✅ CuPy available for GPU acceleration")
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠️ CuPy not available - using CPU fallback")

try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import kornia
    import kornia.filters as KF
    import kornia.enhance as KE
    import kornia.color as KC
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False

# Advanced algorithm libraries
try:
    import pywt  # PyWavelets for wavelet denoising
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("PyWavelets not available - using fallback denoising")

try:
    from skimage import restoration, exposure, color
    from skimage.filters import gaussian
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("scikit-image not available - using fallback methods")

# GPU-accelerated PyWavelets alternative
try:
    import cupy_wavelets as cp_pywt  # GPU wavelets if available
    GPU_WAVELETS_AVAILABLE = True
except ImportError:
    GPU_WAVELETS_AVAILABLE = False

# Advanced Sharpening from Eric_Image_Processing_Nodes
try:
    from pathlib import Path
    import sys
    
    # Try to import from Eric_Image_Processing_Nodes if available
    # Get path to custom_nodes directory (parent of parent of this file)
    current_file = Path(__file__)
    custom_nodes_path = current_file.parent.parent.parent
    image_proc_path = custom_nodes_path / "Eric_Image_Processing_Nodes"
    
    if image_proc_path.exists():
        # Add to path if not already there
        if str(image_proc_path) not in sys.path:
            sys.path.insert(0, str(image_proc_path))
        
        # Import the advanced sharpening processor
        from scripts.advanced_sharpening import AdvancedSharpeningProcessor
        ADVANCED_SHARPENING_AVAILABLE = True
        print("✅ Advanced Sharpening Processor available - using research-grade algorithms")
    else:
        ADVANCED_SHARPENING_AVAILABLE = False
        print("⚠️ Eric_Image_Processing_Nodes not found - using basic sharpening fallback")
except Exception as e:
    ADVANCED_SHARPENING_AVAILABLE = False
    print(f"⚠️ Advanced Sharpening not available: {e}")
    print("   Using basic sharpening fallback")


class ModernImageEnhancer:
    """
    Modern image enhancement with GPU acceleration while preserving advanced algorithms
    """
    
    def __init__(self, debug_mode: bool = False, use_gpu: bool = True):
        self.debug_mode = debug_mode
        self.use_gpu = use_gpu
        
        # Initialize advanced sharpening processor if available
        if ADVANCED_SHARPENING_AVAILABLE:
            try:
                self.advanced_sharpener = AdvancedSharpeningProcessor()
                if debug_mode:
                    print("🎯 Advanced Sharpening Processor initialized")
                    print("   Available methods: smart, hiraloam, edge_directional, multiscale, guided")
            except Exception as e:
                self.advanced_sharpener = None
                if debug_mode:
                    print(f"⚠️ Failed to initialize advanced sharpener: {e}")
        else:
            self.advanced_sharpener = None
            if debug_mode:
                print("💻 Using basic sharpening (advanced sharpener not available)")
        
        # Setup GPU device
        if CUPY_AVAILABLE and use_gpu:
            try:
                # Initialize CuPy and check GPU
                cp.cuda.Device(0).use()
                self.gpu_available = True
                if debug_mode:
                    mempool = cp.get_default_memory_pool()
                    gpu_memory = cp.cuda.Device().mem_info
                    print(f"🚀 GPU acceleration enabled with CuPy")
                    print(f"   💾 GPU Memory: {gpu_memory[1] / 1e9:.1f}GB total, {gpu_memory[0] / 1e9:.1f}GB free")
            except Exception as e:
                self.gpu_available = False
                if debug_mode:
                    print(f"⚠️ GPU initialization failed: {e}")
        else:
            self.gpu_available = False
            if debug_mode:
                print("💻 Using CPU processing")
        
        # Enhanced settings (same as v1.0)
        self.settings = {
            # LAB color enhancement
            "lab_color_enhancement": True,
            "lab_a_contrast": 0.12,
            "lab_b_contrast": 0.12,
            "lab_lightness_preserve": True,
            
            # Advanced sharpening settings
            "subject_sharpen_boost": 0.25,
            "base_sharpen_strength": 0.6,
            "sharpen_radius": 1.2,
            "sharpen_threshold": 0.02,
            "frequency_domain_sharpening": True,  # NEW: Use frequency domain
            
            # Advanced denoising settings  
            "denoise_strength": 0.3,
            "denoise_preserve_edges": True,
            "wavelet_levels": 3,  # Wavelet decomposition levels
            "wavelet_type": 'db4',  # Daubechies 4 wavelet
            
            # Advanced tone mapping
            "tone_mapping_strength": 0.4,
            "shadow_recovery": 0.2,
            "highlight_recovery": 0.1,
            "local_adaptation": True,  # Use local tone mapping
            
            # GPU-specific optimizations
            "gpu_batch_size": 1,
            "gpu_memory_fraction": 0.8,
            "use_gpu_wavelets": True,
            "use_gpu_fft": True,
            
            # Quality preservation
            "jpeg_artifact_removal": True,
            "preserve_color_profiles": True,
            "output_quality": "best"
        }
        
        # Same enhancement profiles as v1.0
        self.profiles = {
            "Digital Magazine": {
                "denoise": 0.1,
                "sharpen": 0.5, 
                "tone_map": 0.2,
                "color_pop": 0.6,
                "artifact_removal": True
            },
            "Scanned Photo": {
                "denoise": 0.5,
                "sharpen": 0.7,
                "tone_map": 0.6,
                "color_pop": 0.8,
                "artifact_removal": True
            },
            "Vintage/Compressed": {
                "denoise": 0.8,
                "sharpen": 0.8,
                "tone_map": 0.8,
                "color_pop": 1.0,
                "artifact_removal": True
            },
            "Minimal": {
                "denoise": 0.0,
                "sharpen": 0.2,
                "tone_map": 0.0,
                "color_pop": 0.3,
                "artifact_removal": False
            }
        }
    
    def enhance_image_with_params(self,
                                 image: Image.Image,
                                 denoise: float = 0.0,
                                 sharpen: float = 0.5,
                                 tone_map: float = 0.0,
                                 color_pop: float = 0.3,
                                 artifact_removal: bool = False,
                                 subject_boxes: Optional[List] = None,
                                 sharpening_method: str = "gpu_basic") -> Image.Image:
        """
        Main enhancement method with direct parameter control (NEW API)
        
        Args:
            image: Input PIL Image
            denoise: Noise reduction strength (0.0-1.0)
            sharpen: Sharpening strength (0.0-1.0)
            tone_map: Tone mapping strength (0.0-1.0)
            color_pop: Color enhancement strength (0.0-1.0)
            artifact_removal: Remove JPEG artifacts (bool)
            subject_boxes: Florence2 bounding boxes for selective enhancement
            sharpening_method: Sharpening algorithm
                - "gpu_basic": Fast GPU sharpening (recommended)
                - "smart_adaptive": CPU adaptive sharpening
                - "hiraloam_gentle": High Radius Low Amount
                - "edge_directional": Edge-aware directional
                - "multiscale_quality": Multi-scale Laplacian (slow)
                - "guided_filter": Guided filter edge-preserving
        
        Returns:
            Enhanced PIL Image
        """
        
        # Build custom profile settings from parameters
        profile_settings = {
            "denoise": denoise,
            "sharpen": sharpen,
            "tone_map": tone_map,
            "color_pop": color_pop,
            "artifact_removal": artifact_removal
        }
        
        start_time = time.time()
        
        if self.debug_mode:
            print(f"🎨 Enhancing image {image.size} with custom parameters")
            print(f"    Device: {'GPU (CuPy)' if self.gpu_available else 'CPU'}")
            print(f"    Denoise: {denoise:.2f}, Sharpen: {sharpen:.2f}, Tone: {tone_map:.2f}, Color: {color_pop:.2f}")
            print(f"    Artifacts: {'Remove' if artifact_removal else 'Keep'}, Method: {sharpening_method}")
        
        # Map sharpening method names
        method_map = {
            "gpu_basic": "basic",
            "smart_adaptive": "smart",
            "hiraloam_gentle": "hiraloam",
            "edge_directional": "edge_directional",
            "multiscale_quality": "multiscale",
            "guided_filter": "guided"
        }
        internal_method = method_map.get(sharpening_method, "basic")
        
        # Preserve original color profile
        original_profile = None
        if hasattr(image, 'info') and 'icc_profile' in image.info:
            original_profile = image.info['icc_profile']
        
        # Convert to working format
        if image.mode != 'RGB':
            if self.debug_mode:
                print(f"    Converting from {image.mode} to RGB for processing")
            image = image.convert('RGB')
        
        # Choose processing path
        strength = 1.0  # Parameters are already absolute values
        
        if self.gpu_available:
            if self.debug_mode:
                print("🚀 Using GPU-accelerated pipeline (fast)")
            try:
                enhanced = self._gpu_advanced_pipeline(image, profile_settings, strength, subject_boxes)
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️ GPU pipeline failed: {e}, falling back to CPU")
                enhanced = self._cpu_advanced_pipeline(image, profile_settings, strength, subject_boxes, internal_method)
        else:
            if self.debug_mode:
                print("💻 Using CPU pipeline")
            enhanced = self._cpu_advanced_pipeline(image, profile_settings, strength, subject_boxes, internal_method)
        
        # Restore color profile
        if original_profile:
            enhanced.info['icc_profile'] = original_profile
        
        processing_time = time.time() - start_time
        
        if self.debug_mode:
            print(f"✅ Enhancement complete in {processing_time:.2f}s")
        
        return enhanced
    
    def enhance_image(self, 
                     image: Image.Image, 
                     profile: str = "Digital Magazine",
                     strength: float = 1.0,
                     subject_boxes: Optional[List] = None,
                     sharpening_method: str = "auto") -> Image.Image:
        """
        LEGACY: Profile-based enhancement method (kept for backwards compatibility)
        
        Args:
            image: Input PIL Image
            profile: Enhancement profile name
            strength: Overall strength multiplier (0.0-2.0)
            subject_boxes: Florence2 bounding boxes for selective enhancement
            sharpening_method: Sharpening algorithm to use
                - "auto": Auto-detect best method based on image characteristics
                - "smart": Smart adaptive sharpening with overshoot detection
                - "hiraloam": High Radius Low Amount (natural, gentle)
                - "edge_directional": Edge-aware directional sharpening
                - "multiscale": Multi-scale Laplacian pyramid
                - "guided": Guided filter edge-preserving
                - "basic": Basic unsharp mask (fallback)
        """
        
        # Handle "None" profile - return image unchanged (fastest option)
        if profile == "None":
            if self.debug_mode:
                print(f"⚡ Enhancement profile: None - returning image unchanged (fastest)")
            return image
        
        if profile not in self.profiles:
            profile = "Digital Magazine"
            
        profile_settings = self.profiles[profile]
        
        start_time = time.time()
        
        if self.debug_mode:
            print(f"🎨 Enhancing image {image.size} with profile '{profile}' (strength: {strength:.1f})")
            print(f"    Device: {'GPU (CuPy)' if self.gpu_available else 'CPU'}")
            print(f"    Sharpening: {sharpening_method}")
            print(f"    Input mode: {image.mode}")
        
        # Preserve original color profile
        original_profile = None
        if hasattr(image, 'info') and 'icc_profile' in image.info:
            original_profile = image.info['icc_profile']
        
        # Convert to working format
        if image.mode != 'RGB':
            if self.debug_mode:
                print(f"    Converting from {image.mode} to RGB for processing")
            image = image.convert('RGB')
        
        # Choose processing path
        if self.gpu_available:
            if self.debug_mode:
                print(f"    🚀 Using GPU-accelerated pipeline (fast)")
            try:
                enhanced = self._gpu_advanced_pipeline(image, profile_settings, strength, subject_boxes)
            except Exception as e:
                if self.debug_mode:
                    print(f"    ⚠️ GPU pipeline failed: {e}")
                    print(f"    💻 Falling back to CPU pipeline (slower)")
                enhanced = self._cpu_advanced_pipeline(image, profile_settings, strength, subject_boxes, sharpening_method)
        else:
            if self.debug_mode:
                print(f"    💻 Using CPU pipeline (no GPU available)")
            enhanced = self._cpu_advanced_pipeline(image, profile_settings, strength, subject_boxes, sharpening_method)
        
        # Restore color profile
        if original_profile and self.settings["preserve_color_profiles"]:
            if not hasattr(enhanced, 'info'):
                enhanced.info = {}
            enhanced.info['icc_profile'] = original_profile
        
        processing_time = time.time() - start_time
        
        if self.debug_mode:
            print(f"✅ Enhancement complete in {processing_time:.2f}s")
        
        return enhanced
    
    def _gpu_advanced_pipeline(self, image: Image.Image, profile_settings: Dict, 
                              strength: float, subject_boxes: Optional[List]) -> Image.Image:
        """GPU-accelerated pipeline using advanced algorithms"""
        
        try:
            import time
            
            # Convert to CuPy array
            step_start = time.time()
            img_array = cp.array(np.array(image), dtype=cp.float32) / 255.0
            if self.debug_mode:
                print(f"    ⏱️ GPU array conversion: {time.time()-step_start:.3f}s")
                print(f"    GPU array shape: {img_array.shape}, device: {img_array.device}")
            
            # Step 1: GPU-accelerated JPEG artifact removal
            if profile_settings.get("artifact_removal", False):
                step_start = time.time()
                img_array = self._gpu_remove_artifacts(img_array, profile_settings["denoise"] * strength)
                if self.debug_mode:
                    print(f"    ⏱️ Artifact removal: {time.time()-step_start:.3f}s")
            
            # Step 2: GPU-accelerated wavelet denoising
            if profile_settings["denoise"] > 0:
                step_start = time.time()
                img_array = self._gpu_wavelet_denoise(img_array, profile_settings["denoise"] * strength)
                if self.debug_mode:
                    print(f"    ⏱️ Wavelet denoising: {time.time()-step_start:.3f}s")
            
            # Step 3: GPU-accelerated advanced tone mapping
            if profile_settings["tone_map"] > 0:
                step_start = time.time()
                img_array = self._gpu_advanced_tone_mapping(img_array, profile_settings["tone_map"] * strength)
                if self.debug_mode:
                    print(f"    ⏱️ Tone mapping: {time.time()-step_start:.3f}s")
            
            # Step 4: GPU-accelerated selective sharpening
            if profile_settings["sharpen"] > 0:
                step_start = time.time()
                img_array = self._gpu_selective_sharpen(img_array, profile_settings["sharpen"] * strength, subject_boxes, image.size)
                if self.debug_mode:
                    print(f"    ⏱️ Selective sharpening: {time.time()-step_start:.3f}s")
            
            # Step 5: GPU-accelerated LAB color enhancement
            if profile_settings["color_pop"] > 0:
                step_start = time.time()
                img_array = self._gpu_lab_color_enhancement(img_array, profile_settings["color_pop"] * strength)
                if self.debug_mode:
                    print(f"    ⏱️ LAB color enhancement: {time.time()-step_start:.3f}s")
            
            # Step 6: Final exposure adjustment
            step_start = time.time()
            img_array = self._gpu_exposure_adjustment(img_array, strength * 0.1)
            if self.debug_mode:
                print(f"    ⏱️ Exposure adjustment: {time.time()-step_start:.3f}s")
            
            # Convert back to PIL
            step_start = time.time()
            img_array = cp.clip(img_array * 255.0, 0, 255).astype(cp.uint8)
            enhanced_np = cp.asnumpy(img_array)
            enhanced = Image.fromarray(enhanced_np, mode='RGB')
            if self.debug_mode:
                print(f"    ⏱️ GPU→CPU conversion: {time.time()-step_start:.3f}s")
            
            # Clean up GPU memory
            del img_array
            if hasattr(cp, 'get_default_memory_pool'):
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
            
            return enhanced
            
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ GPU processing failed: {e}, falling back to CPU")
            
            # Clean up and fallback
            if hasattr(cp, 'get_default_memory_pool'):
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
            
            return self._cpu_advanced_pipeline(image, profile_settings, strength, subject_boxes)
    
    def _gpu_wavelet_denoise(self, img_array: cp.ndarray, strength: float) -> cp.ndarray:
        """GPU-accelerated wavelet denoising using PyWavelets + CuPy"""
        if strength <= 0:
            return img_array
        
        try:
            if self.debug_mode:
                print(f"    🌊 GPU wavelet denoising (strength: {strength:.2f})")
            
            # Process each channel separately
            denoised_channels = []
            
            for channel in range(3):  # RGB
                channel_data = img_array[:, :, channel]
                
                if PYWT_AVAILABLE:
                    # Convert to CPU for PyWavelets, then back to GPU
                    channel_cpu = cp.asnumpy(channel_data)
                    
                    # Wavelet decomposition
                    coeffs = pywt.wavedec2(channel_cpu, self.settings["wavelet_type"], 
                                        level=self.settings["wavelet_levels"])
                    
                    # Estimate noise and apply thresholding
                    sigma = np.std(coeffs[-1]) * strength * 0.1
                    
                    # Apply soft thresholding
                    coeffs_thresh = list(coeffs)
                    for i in range(1, len(coeffs)):
                        if isinstance(coeffs[i], tuple):
                            coeffs_thresh[i] = tuple([
                                pywt.threshold(detail, sigma, mode='soft') 
                                for detail in coeffs[i]
                            ])
                        else:
                            coeffs_thresh[i] = pywt.threshold(coeffs[i], sigma, mode='soft')
                    
                    # Reconstruct and convert back to GPU
                    denoised_cpu = pywt.waverec2(coeffs_thresh, self.settings["wavelet_type"])
                    denoised_channel = cp.array(denoised_cpu)
                    
                else:
                    # Fallback: GPU bilateral filter approximation
                    kernel_size = int(5 + strength * 6)
                    sigma_spatial = kernel_size / 3.0
                    
                    # Create Gaussian kernel
                    x = cp.arange(kernel_size) - kernel_size // 2
                    gaussian_kernel = cp.exp(-(x**2) / (2 * sigma_spatial**2))
                    gaussian_kernel = gaussian_kernel / cp.sum(gaussian_kernel)
                    
                    # Apply edge-preserving denoising
                    denoised_channel = cp_ndimage.convolve1d(channel_data, gaussian_kernel, axis=0)
                    denoised_channel = cp_ndimage.convolve1d(denoised_channel, gaussian_kernel, axis=1)
                
                denoised_channels.append(denoised_channel)
            
            # Combine channels
            result = cp.stack(denoised_channels, axis=2)
            
            if self.debug_mode:
                print(f"    ✅ GPU wavelet denoising complete")
            
            return result
            
        except Exception as e:
            if self.debug_mode:
                print(f"    ⚠️ GPU wavelet denoising failed: {e}")
            return img_array
    
    def _gpu_selective_sharpen(self, img_array: cp.ndarray, strength: float, 
                              subject_boxes: Optional[List], image_size: Tuple[int, int]) -> cp.ndarray:
        """GPU-accelerated selective sharpening with frequency domain processing"""
        if strength <= 0:
            return img_array
        
        try:
            if self.debug_mode:
                print(f"    🔪 GPU selective sharpening (strength: {strength:.2f})")
            
            height, width = img_array.shape[:2]
            
            # Create subject mask on GPU
            if subject_boxes:
                subject_mask = self._gpu_create_subject_mask((width, height), subject_boxes)
            else:
                subject_mask = cp.ones((height, width), dtype=cp.float32)
            
            # Parameters
            background_strength = strength * self.settings["base_sharpen_strength"]
            subject_strength = background_strength * (1 + self.settings["subject_sharpen_boost"])
            radius = self.settings["sharpen_radius"]
            
            if self.settings["frequency_domain_sharpening"] and self.settings["use_gpu_fft"]:
                # Advanced frequency domain sharpening
                result = self._gpu_frequency_domain_sharpen(img_array, subject_mask, 
                                                          background_strength, subject_strength, radius)
            else:
                # Spatial domain sharpening (unsharp mask)
                result = self._gpu_spatial_sharpen(img_array, subject_mask, 
                                                 background_strength, subject_strength, radius)
            
            if self.debug_mode:
                print(f"    ✅ GPU selective sharpening complete")
            
            return result
            
        except Exception as e:
            if self.debug_mode:
                print(f"    ⚠️ GPU selective sharpening failed: {e}")
            return img_array
    
    def _gpu_frequency_domain_sharpen(self, img_array: cp.ndarray, subject_mask: cp.ndarray,
                                     bg_strength: float, subj_strength: float, radius: float) -> cp.ndarray:
        """Advanced frequency domain sharpening using GPU FFT"""
        
        enhanced_channels = []
        
        for channel in range(3):  # RGB
            channel_data = img_array[:, :, channel]
            
            # FFT to frequency domain
            fft_channel = cp.fft.fft2(channel_data)
            fft_shifted = cp.fft.fftshift(fft_channel)
            
            # Create high-pass filter (for sharpening)
            height, width = channel_data.shape
            center_y, center_x = height // 2, width // 2
            
            # Create coordinate grids
            y, x = cp.ogrid[:height, :width]
            distance = cp.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # High-pass filter (inverted Gaussian)
            cutoff = min(height, width) / (4 * radius)  # Convert radius to frequency domain
            high_pass_filter = 1 - cp.exp(-(distance**2) / (2 * cutoff**2))
            
            # Apply selective enhancement based on subject mask
            # Create varying strength map
            strength_map = bg_strength + (subj_strength - bg_strength) * subject_mask
            
            # Apply filter with varying strength (VECTORIZED - critical for performance!)
            enhancement_factor = 1 + strength_map * high_pass_filter
            fft_shifted = fft_shifted * enhancement_factor
            
            # Convert back to spatial domain
            fft_unshifted = cp.fft.ifftshift(fft_shifted)
            enhanced_channel = cp.real(cp.fft.ifft2(fft_unshifted))
            
            enhanced_channels.append(enhanced_channel)
        
        return cp.stack(enhanced_channels, axis=2)

    def _gpu_local_tone_mapping(self, lightness: cp.ndarray, strength: float) -> cp.ndarray:
        """GPU-accelerated local tone mapping"""
        try:
            # Simple local contrast enhancement using Gaussian pyramid
            # Create multi-scale representation
            scales = []
            current = lightness
            
            for i in range(3):  # 3 scale levels
                # Gaussian blur for current scale
                sigma = 2.0 ** i
                kernel_size = int(6 * sigma) | 1
                
                # Create 1D Gaussian kernel
                x = cp.arange(kernel_size) - kernel_size // 2
                gaussian_kernel = cp.exp(-(x**2) / (2 * sigma**2))
                gaussian_kernel = gaussian_kernel / cp.sum(gaussian_kernel)
                
                # Apply separable convolution
                blurred = cp_ndimage.convolve1d(current, gaussian_kernel, axis=0)
                blurred = cp_ndimage.convolve1d(blurred, gaussian_kernel, axis=1)
                
                scales.append(current - blurred)  # Detail at this scale
                current = blurred
            
            # Enhance details at each scale
            enhanced = current  # Base (lowest frequency)
            for i, detail in enumerate(scales):
                enhancement_factor = strength * (0.3 + 0.2 * i)  # More enhancement for finer details
                enhanced += detail * (1 + enhancement_factor)
            
            return enhanced
            
        except Exception as e:
            if self.debug_mode:
                print(f"    GPU local tone mapping failed: {e}")
            return lightness

    def _gpu_global_tone_mapping(self, lightness: cp.ndarray, strength: float) -> cp.ndarray:
        """GPU-accelerated global tone mapping"""
        try:
            # Simple S-curve tone mapping
            # Normalize to 0-1 range
            L_norm = lightness / 100.0
            
            # Apply S-curve for contrast enhancement
            # This mimics Photoshop's curve adjustment
            curve_strength = strength * 0.5
            enhanced = L_norm + curve_strength * cp.sin(2 * cp.pi * L_norm) / (2 * cp.pi)
            
            # Clamp and convert back to LAB L range
            enhanced = cp.clip(enhanced, 0, 1) * 100.0
            
            return enhanced
            
        except Exception as e:
            if self.debug_mode:
                print(f"    GPU global tone mapping failed: {e}")
            return lightness

    def _gpu_remove_artifacts(self, img_array: cp.ndarray, strength: float) -> cp.ndarray:
        """GPU-accelerated artifact removal using bilateral filter approximation"""
        if strength <= 0:
            return img_array
        
        try:
            if self.debug_mode:
                print(f"    🧹 GPU artifact removal (strength: {strength:.2f})")
            
            # Process each channel separately
            filtered_channels = []
            
            for channel in range(3):  # RGB
                channel_data = img_array[:, :, channel]
                
                # Approximate bilateral filter using cascaded Gaussian filters
                # This is faster than true bilateral filtering
                kernel_size = int(5 + strength * 6)
                sigma_spatial = kernel_size / 3.0
                
                # Create Gaussian kernel
                x = cp.arange(kernel_size) - kernel_size // 2
                gaussian_kernel = cp.exp(-(x**2) / (2 * sigma_spatial**2))
                gaussian_kernel = gaussian_kernel / cp.sum(gaussian_kernel)
                
                # Apply edge-preserving filter (approximation)
                # Multiple passes with small kernels approximate bilateral filter
                filtered = channel_data
                for _ in range(2):  # 2 passes for edge preservation
                    filtered = cp_ndimage.convolve1d(filtered, gaussian_kernel, axis=0)
                    filtered = cp_ndimage.convolve1d(filtered, gaussian_kernel, axis=1)
                
                # Blend with original based on edge strength
                edge_strength = cp.abs(cp_ndimage.sobel(channel_data))
                edge_mask = 1.0 - cp.clip(edge_strength * 2.0, 0, 1)  # Preserve edges
                
                filtered = channel_data * (1 - edge_mask * strength) + filtered * (edge_mask * strength)
                filtered_channels.append(filtered)
            
            result = cp.stack(filtered_channels, axis=2)
            
            if self.debug_mode:
                print(f"    ✅ GPU artifact removal complete")
            
            return result
            
        except Exception as e:
            if self.debug_mode:
                print(f"    ⚠️ GPU artifact removal failed: {e}")
            return img_array

    def _gpu_exposure_adjustment(self, img_array: cp.ndarray, strength: float) -> cp.ndarray:
        """GPU exposure adjustment"""
        if strength <= 0:
            return img_array
        
        try:
            # Very subtle brightness adjustment
            brightness_factor = 1.0 + strength * 0.05
            adjusted = img_array * brightness_factor
            
            # Clamp to valid range
            adjusted = cp.clip(adjusted, 0, 1)
            
            if self.debug_mode:
                print(f"    💡 GPU exposure adjustment applied ({brightness_factor:.3f})")
            
            return adjusted
            
        except Exception as e:
            if self.debug_mode:
                print(f"    ⚠️ GPU exposure adjustment failed: {e}")
            return img_array
            
    def _gpu_spatial_sharpen(self, img_array: cp.ndarray, subject_mask: cp.ndarray,
                            bg_strength: float, subj_strength: float, radius: float) -> cp.ndarray:
        """Spatial domain unsharp mask sharpening on GPU"""
        
        # Create Gaussian kernel for blurring
        kernel_size = int(radius * 6) | 1  # Ensure odd size
        sigma = radius
        
        # Create 2D Gaussian kernel
        x = cp.arange(kernel_size) - kernel_size // 2
        y = x[:, cp.newaxis]
        gaussian_kernel = cp.exp(-(x**2 + y**2) / (2 * sigma**2))
        gaussian_kernel = gaussian_kernel / cp.sum(gaussian_kernel)
        
        enhanced_channels = []
        
        for channel in range(3):  # RGB
            channel_data = img_array[:, :, channel]
            
            # Apply Gaussian blur using convolution
            blurred = cp_signal.convolve2d(channel_data, gaussian_kernel, mode='same', boundary='reflect')
            
            # High-pass filter
            high_pass = channel_data - blurred
            
            # Apply threshold to avoid noise amplification
            threshold = self.settings["sharpen_threshold"]
            high_pass_masked = cp.where(cp.abs(high_pass) > threshold, high_pass, 0)
            
            # Apply selective sharpening - FIX: Remove the extra dimension
            strength_map = bg_strength + (subj_strength - bg_strength) * subject_mask
            enhanced_channel = channel_data + high_pass_masked * strength_map  # Remove [:, :, cp.newaxis]
            
            enhanced_channels.append(enhanced_channel)
        
        return cp.stack(enhanced_channels, axis=2)
    
    def _gpu_create_subject_mask(self, image_size: Tuple[int, int], subject_boxes: List) -> cp.ndarray:
        """Create subject mask on GPU with soft falloff"""
        width, height = image_size
        mask = cp.zeros((height, width), dtype=cp.float32)
        
        for box in subject_boxes:
            # Handle different box formats
            if hasattr(box, 'x1'):
                x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
            elif isinstance(box, (list, tuple)) and len(box) >= 4:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            else:
                continue
            
            # Clamp to image bounds
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))
            
            # Create coordinate grids
            y_coords, x_coords = cp.ogrid[:height, :width]
            
            # Distance to box edges
            dist_x = cp.minimum(cp.abs(x_coords - x1), cp.abs(x_coords - x2))
            dist_y = cp.minimum(cp.abs(y_coords - y1), cp.abs(y_coords - y2))
            
            # Inside box
            inside_mask = (x_coords >= x1) & (x_coords < x2) & (y_coords >= y1) & (y_coords < y2)
            
            # Soft falloff outside box (20 pixel radius)
            falloff_distance = cp.minimum(dist_x, dist_y)
            falloff_mask = cp.maximum(0, 1.0 - falloff_distance / 20.0)
            
            # Combine masks
            combined_mask = cp.where(inside_mask, 1.0, falloff_mask)
            mask = cp.maximum(mask, combined_mask)
        
        return mask
    
    def _gpu_advanced_tone_mapping(self, img_array: cp.ndarray, strength: float) -> cp.ndarray:
        """GPU-accelerated advanced tone mapping using LAB space"""
        if strength <= 0:
            return img_array
        
        try:
            if self.debug_mode:
                print(f"    🎨 GPU advanced tone mapping (strength: {strength:.2f})")
            
            # Convert RGB to LAB
            lab_array = self._gpu_rgb_to_lab(img_array)
            
            # Extract lightness channel
            lightness = lab_array[:, :, 0]
            
            if self.settings["local_adaptation"]:
                # Local adaptive enhancement
                enhanced_lightness = self._gpu_local_tone_mapping(lightness, strength)
            else:
                # Global tone mapping
                enhanced_lightness = self._gpu_global_tone_mapping(lightness, strength)
            
            # Reconstruct LAB and convert back to RGB
            lab_array[:, :, 0] = enhanced_lightness
            rgb_array = self._gpu_lab_to_rgb(lab_array)
            
            if self.debug_mode:
                print(f"    ✅ GPU tone mapping complete")
            
            return rgb_array
            
        except Exception as e:
            if self.debug_mode:
                print(f"    ⚠️ GPU tone mapping failed: {e}")
            return img_array
    
    def _gpu_lab_color_enhancement(self, img_array: cp.ndarray, strength: float) -> cp.ndarray:
        """GPU-accelerated LAB color enhancement"""
        if strength <= 0:
            return img_array
        
        try:
            if self.debug_mode:
                print(f"    🌈 GPU LAB color enhancement (strength: {strength:.2f})")
            
            # Convert to LAB
            lab_array = self._gpu_rgb_to_lab(img_array)
            
            # Extract channels
            L, A, B = lab_array[:, :, 0], lab_array[:, :, 1], lab_array[:, :, 2]
            
            # Apply contrast to A and B channels
            a_contrast = self.settings["lab_a_contrast"] * strength
            b_contrast = self.settings["lab_b_contrast"] * strength
            
            # Enhance A channel (Green-Red)
            A_mean = cp.mean(A)
            A_enhanced = A_mean + (A - A_mean) * (1 + a_contrast)
            
            # Enhance B channel (Blue-Yellow)
            B_mean = cp.mean(B)
            B_enhanced = B_mean + (B - B_mean) * (1 + b_contrast)
            
            # Reconstruct LAB
            lab_enhanced = cp.stack([L, A_enhanced, B_enhanced], axis=2)
            
            # Convert back to RGB
            rgb_enhanced = self._gpu_lab_to_rgb(lab_enhanced)
            
            if self.debug_mode:
                print(f"    ✅ GPU LAB enhancement complete")
            
            return rgb_enhanced
            
        except Exception as e:
            if self.debug_mode:
                print(f"    ⚠️ GPU LAB enhancement failed: {e}")
            return img_array
    
    # GPU helper methods for color space conversion
    def _gpu_rgb_to_lab(self, rgb_array: cp.ndarray) -> cp.ndarray:
        """Convert RGB to LAB color space on GPU"""
        # Simplified LAB conversion for GPU processing
        # For full accuracy, this could be expanded with proper color space matrices
        
        # Normalize RGB
        rgb_norm = rgb_array / 1.0
        
        # Convert to XYZ (simplified)
        # This is a simplified conversion - for production use, implement full sRGB->XYZ->LAB
        xyz = cp.zeros_like(rgb_norm)
        xyz[:, :, 0] = 0.412453 * rgb_norm[:, :, 0] + 0.357580 * rgb_norm[:, :, 1] + 0.180423 * rgb_norm[:, :, 2]  # X
        xyz[:, :, 1] = 0.212671 * rgb_norm[:, :, 0] + 0.715160 * rgb_norm[:, :, 1] + 0.072169 * rgb_norm[:, :, 2]  # Y
        xyz[:, :, 2] = 0.019334 * rgb_norm[:, :, 0] + 0.119193 * rgb_norm[:, :, 1] + 0.950227 * rgb_norm[:, :, 2]  # Z
        
        # Convert XYZ to LAB (simplified)
        lab = cp.zeros_like(xyz)
        lab[:, :, 0] = 116 * cp.cbrt(xyz[:, :, 1]) - 16  # L
        lab[:, :, 1] = 500 * (cp.cbrt(xyz[:, :, 0]) - cp.cbrt(xyz[:, :, 1]))  # A
        lab[:, :, 2] = 200 * (cp.cbrt(xyz[:, :, 1]) - cp.cbrt(xyz[:, :, 2]))  # B
        
        return lab
    
    def _gpu_lab_to_rgb(self, lab_array: cp.ndarray) -> cp.ndarray:
        """Convert LAB to RGB color space on GPU"""
        # Simplified LAB to RGB conversion
        # Convert LAB to XYZ first
        L, A, B = lab_array[:, :, 0], lab_array[:, :, 1], lab_array[:, :, 2]
        
        # LAB to XYZ
        fy = (L + 16) / 116
        fx = A / 500 + fy
        fz = fy - B / 200
        
        xyz = cp.zeros_like(lab_array)
        xyz[:, :, 0] = fx ** 3  # X
        xyz[:, :, 1] = fy ** 3  # Y  
        xyz[:, :, 2] = fz ** 3  # Z
        
        # XYZ to RGB (simplified)
        rgb = cp.zeros_like(xyz)
        rgb[:, :, 0] = 3.240479 * xyz[:, :, 0] - 1.537150 * xyz[:, :, 1] - 0.498535 * xyz[:, :, 2]  # R
        rgb[:, :, 1] = -0.969256 * xyz[:, :, 0] + 1.875992 * xyz[:, :, 1] + 0.041556 * xyz[:, :, 2]  # G
        rgb[:, :, 2] = 0.055648 * xyz[:, :, 0] - 0.204043 * xyz[:, :, 1] + 1.057311 * xyz[:, :, 2]  # B
        
        return cp.clip(rgb, 0, 1)
    
    # Keep all the advanced CPU methods as fallbacks
    def _cpu_advanced_pipeline(self, image: Image.Image, profile_settings: Dict, 
                              strength: float, subject_boxes: Optional[List],
                              sharpening_method: str = "auto") -> Image.Image:
        """CPU fallback using all the original advanced algorithms"""
        
        enhanced = image.copy()
        
        if self.debug_mode:
            print(f"    💻 Using CPU advanced processing pipeline")
        
        # Step 1: JPEG artifact removal
        if profile_settings.get("artifact_removal", False):
            enhanced = self._remove_jpeg_artifacts(enhanced, profile_settings["denoise"] * strength)
        
        # Step 2: Wavelet denoising
        if profile_settings["denoise"] > 0:
            enhanced = self._wavelet_denoise(enhanced, profile_settings["denoise"] * strength)
        
        # Step 3: Advanced tone mapping
        if profile_settings["tone_map"] > 0:
            enhanced = self._tone_mapping(enhanced, profile_settings["tone_map"] * strength)
        
        # Step 4: Advanced Sharpening
        if profile_settings["sharpen"] > 0:
            enhanced = self._apply_sharpening(enhanced, profile_settings["sharpen"] * strength, subject_boxes, method=sharpening_method)
        
        # Step 5: LAB color enhancement
        if profile_settings["color_pop"] > 0:
            enhanced = self._lab_color_enhancement(enhanced, profile_settings["color_pop"] * strength)
        
        # Step 6: Final exposure adjustment
        enhanced = self._final_exposure_adjustment(enhanced, strength * 0.1)
        
        return enhanced
    
    # Include all original CPU methods unchanged
    def _remove_jpeg_artifacts(self, image: Image.Image, strength: float) -> Image.Image:
        """Remove JPEG compression artifacts using modern denoising"""
        if strength <= 0:
            return image
        
        try:
            # Convert to numpy for OpenCV processing
            img_array = np.array(image)
            
            # Use Non-Local Means denoising (excellent for JPEG artifacts)
            if hasattr(cv2, 'fastNlMeansDenoising'):
                h = int(3 + strength * 7)  # Denoising strength
                template_window_size = 7
                search_window_size = 21
                
                denoised = cv2.fastNlMeansDenoising(
                    img_array, None, h, template_window_size, search_window_size
                )
                
                if self.debug_mode:
                    print(f"    Applied JPEG artifact removal (strength: {h})")
                
                return Image.fromarray(denoised)
            else:
                # Fallback: gentle Gaussian blur
                radius = strength * 0.5
                return image.filter(ImageFilter.GaussianBlur(radius=radius))
                
        except Exception as e:
            if self.debug_mode:
                print(f"    JPEG artifact removal failed: {e}")
            return image
    
    def _wavelet_denoise(self, image: Image.Image, strength: float) -> Image.Image:
        """Wavelet-based denoising (superior to Gaussian blur)"""
        if strength <= 0:
            return image
        
        try:
            if PYWT_AVAILABLE:
                # Convert to numpy
                img_array = np.array(image, dtype=np.float32) / 255.0
                
                # Process each channel separately for RGB
                denoised_channels = []
                
                for channel in range(3):
                    channel_data = img_array[:, :, channel]
                    
                    # Wavelet decomposition - fix parameter name
                    coeffs = pywt.wavedec2(channel_data, 'db4', level=3)
                    
                    # Estimate noise standard deviation
                    sigma = np.std(coeffs[-1]) * strength * 0.1
                    
                    # Apply soft thresholding to remove noise
                    coeffs_thresh = list(coeffs)
                    for i in range(1, len(coeffs)):
                        if isinstance(coeffs[i], tuple):
                            coeffs_thresh[i] = tuple([
                                pywt.threshold(detail, sigma, mode='soft') 
                                for detail in coeffs[i]
                            ])
                        else:
                            coeffs_thresh[i] = pywt.threshold(coeffs[i], sigma, mode='soft')
                    
                    # Reconstruct
                    denoised_channel = pywt.waverec2(coeffs_thresh, 'db4')
                    denoised_channels.append(denoised_channel)
                
                # Combine channels
                denoised_array = np.stack(denoised_channels, axis=2)
                denoised_array = np.clip(denoised_array * 255, 0, 255).astype(np.uint8)
                
                if self.debug_mode:
                    print(f"    Applied wavelet denoising (strength: {strength:.2f})")
                
                return Image.fromarray(denoised_array)
            
            else:
                # Fallback: Non-local means if available, else Gaussian
                img_array = np.array(image)
                
                if hasattr(cv2, 'fastNlMeansDenoising'):
                    h = int(strength * 10)
                    denoised = cv2.fastNlMeansDenoising(img_array, None, h, 7, 21)
                    return Image.fromarray(denoised)
                else:
                    # Simple Gaussian fallback
                    radius = strength * 0.8
                    return image.filter(ImageFilter.GaussianBlur(radius=radius))
                
        except Exception as e:
            if self.debug_mode:
                print(f"    Wavelet denoising failed: {e}")
            return image
    
    def _tone_mapping(self, image: Image.Image, strength: float) -> Image.Image:
        """Modern tone mapping (like Photoshop's Shadow/Highlight tool)"""
        if strength <= 0:
            return image
        
        try:
            if SKIMAGE_AVAILABLE:
                # Convert to numpy array
                img_array = np.array(image, dtype=np.float32) / 255.0
                
                if self.debug_mode:
                    print(f"    Tone mapping input: {img_array.shape}, range: {img_array.min():.3f}-{img_array.max():.3f}")
                
                # Apply adaptive histogram equalization with better parameters
                img_lab = color.rgb2lab(img_array)
                
                # Work only on lightness channel
                lightness = img_lab[:, :, 0]
                
                # Apply adaptive equalization (like smart tone mapping)
                kernel_size = max(lightness.shape) // 8  # Adaptive kernel size
                clip_limit = 0.01 + strength * 0.02  # Gentle clipping
                
                # Use local contrast enhancement
                enhanced_lightness = exposure.equalize_adapthist(
                    lightness / 100.0,  # Normalize LAB L range
                    kernel_size=kernel_size,
                    clip_limit=clip_limit
                ) * 100.0
                
                # Blend with original based on strength
                img_lab[:, :, 0] = (1 - strength) * lightness + strength * enhanced_lightness
                
                # Convert back to RGB
                enhanced_array = color.lab2rgb(img_lab)
                enhanced_array = np.clip(enhanced_array * 255, 0, 255).astype(np.uint8)
                
                result = Image.fromarray(enhanced_array, mode='RGB')
                
                if self.debug_mode:
                    print(f"    Tone mapping output: {result.mode}, size: {result.size}")
                    print(f"    Applied tone mapping (strength: {strength:.2f})")
                
                return result
            
            else:
                # Fallback: simple shadow/highlight adjustment
                enhancer = ImageEnhance.Contrast(image)
                contrast_factor = 1.0 + strength * 0.2
                result = enhancer.enhance(contrast_factor)
                
                if self.debug_mode:
                    print(f"    Applied contrast enhancement (factor: {contrast_factor:.2f})")
                
                return result
                
        except Exception as e:
            if self.debug_mode:
                print(f"    Tone mapping failed: {e}")
            return image
    
    def _selective_sharpen(self, image: Image.Image, strength: float, subject_boxes: Optional[List] = None) -> Image.Image:
        """Frequency-based sharpening with selective enhancement on subjects"""
        if strength <= 0:
            return image
        
        try:
            if self.debug_mode:
                print(f"    Sharpening input: {image.mode}, size: {image.size}")
            
            # Convert to numpy for processing
            img_array = np.array(image, dtype=np.float32)
            
            # Create sharpening mask (more sharpening on subjects)
            if subject_boxes:
                sharpen_mask = self._create_subject_mask(image.size, subject_boxes)
                # Background gets base sharpening, subjects get boosted sharpening
                background_strength = strength * self.settings["base_sharpen_strength"]
                subject_strength = background_strength * (1 + self.settings["subject_sharpen_boost"])
            else:
                # Uniform sharpening if no subject detection
                sharpen_mask = np.ones((image.height, image.width), dtype=np.float32)
                background_strength = subject_strength = strength * self.settings["base_sharpen_strength"]
            
            # High-pass sharpening (Photoshop-style)
            radius = self.settings["sharpen_radius"]
            
            # Create Gaussian blurred version for high-pass filter
            blurred = cv2.GaussianBlur(img_array, (0, 0), radius)
            
            # High-pass = original - blurred
            high_pass = img_array - blurred
            
            # Apply threshold to avoid sharpening noise
            threshold = self.settings["sharpen_threshold"] * 255
            high_pass_masked = np.where(np.abs(high_pass) > threshold, high_pass, 0)
            
            # Apply selective sharpening channel by channel to preserve color
            enhanced_array = img_array.copy()
            
            # Process each color channel separately
            for c in range(img_array.shape[2]):  # RGB channels
                for y in range(img_array.shape[0]):
                    for x in range(img_array.shape[1]):
                        mask_value = sharpen_mask[y, x]
                        local_strength = background_strength + (subject_strength - background_strength) * mask_value
                        enhanced_array[y, x, c] += high_pass_masked[y, x, c] * local_strength
            
            # Clip to valid range
            enhanced_array = np.clip(enhanced_array, 0, 255).astype(np.uint8)
            
            result = Image.fromarray(enhanced_array, mode='RGB')
            
            if self.debug_mode:
                print(f"    Sharpening output: {result.mode}, size: {result.size}")
                print(f"    Applied selective sharpening (bg: {background_strength:.2f}, subj: {subject_strength:.2f})")
            
            return result
            
        except Exception as e:
            if self.debug_mode:
                print(f"    Selective sharpening failed: {e}")
            
            # Fallback: simple unsharp mask
            try:
                radius = strength * 2.0
                amount = strength * 1.5
                
                # Simple unsharp mask implementation using PIL
                from PIL import ImageFilter
                blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
                
                # Create high-pass by subtracting blurred from original
                orig_array = np.array(image, dtype=np.float32)
                blur_array = np.array(blurred, dtype=np.float32)
                
                # Apply unsharp mask formula
                sharpened_array = orig_array + amount * (orig_array - blur_array)
                sharpened_array = np.clip(sharpened_array, 0, 255).astype(np.uint8)
                
                result = Image.fromarray(sharpened_array, mode='RGB')
                
                if self.debug_mode:
                    print(f"    Applied fallback sharpening: {result.mode}")
                
                return result
                
            except Exception as e2:
                if self.debug_mode:
                    print(f"    Fallback sharpening also failed: {e2}")
                return image
    
    def _create_subject_mask(self, image_size: Tuple[int, int], subject_boxes: List) -> np.ndarray:
        """Create a mask highlighting subject areas for selective enhancement"""
        width, height = image_size
        mask = np.zeros((height, width), dtype=np.float32)
        
        for box in subject_boxes:
            # Handle different box formats
            if hasattr(box, 'x1'):
                x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
            elif isinstance(box, (list, tuple)) and len(box) >= 4:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            else:
                continue
            
            # Clamp to image bounds
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))
            
            # Vectorized mask creation (much faster than nested loops)
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            
            # Inside box mask
            inside_mask = (x_coords >= x1) & (x_coords < x2) & (y_coords >= y1) & (y_coords < y2)
            
            # Distance-based falloff mask
            dist_x = np.minimum(np.abs(x_coords - x1), np.abs(x_coords - x2))
            dist_y = np.minimum(np.abs(y_coords - y1), np.abs(y_coords - y2))
            falloff_distance = np.minimum(dist_x, dist_y)
            
            # Soft falloff over 20 pixels
            falloff_mask = np.maximum(0, 1.0 - falloff_distance / 20.0)
            
            # Combine masks
            combined_mask = np.where(inside_mask, 1.0, falloff_mask)
            mask = np.maximum(mask, combined_mask)
        
        return mask
    
    def _apply_sharpening(self, image: Image.Image, strength: float, 
                         subject_boxes: Optional[List] = None,
                         method: str = "auto") -> Image.Image:
        """
        Apply sharpening using advanced algorithms from Eric_Image_Processing_Nodes
        or fallback to basic sharpening
        
        Args:
            image: Input PIL Image
            strength: Sharpening strength (0.0-2.0)
            subject_boxes: Optional subject bounding boxes for selective sharpening
            method: Sharpening method to use
                - "auto": Auto-detect best method based on image characteristics
                - "smart": Smart adaptive sharpening with overshoot detection
                - "hiraloam": High Radius Low Amount (natural, gentle)
                - "edge_directional": Edge-aware directional sharpening
                - "multiscale": Multi-scale Laplacian pyramid
                - "guided": Guided filter edge-preserving
                - "basic": Basic unsharp mask (fallback)
        
        Returns:
            Sharpened PIL Image
        """
        
        if strength <= 0:
            return image
        
        # PERFORMANCE: Advanced sharpening methods are CPU-only and VERY slow (300-400s per image!)
        # When GPU is available, ALWAYS use fast GPU-accelerated basic sharpening
        # The advanced methods are only useful when GPU is not available
        if self.gpu_available and method == "auto":
            if self.debug_mode:
                print(f"    ⚡ GPU available: Using GPU-accelerated sharpening (fast)")
            method = "basic"  # This triggers GPU pipeline usage
        elif method == "auto":
            # Only use CPU-based advanced methods when GPU not available
            method = self._detect_best_sharpening_method(image)
            if self.debug_mode:
                print(f"    Auto-detected CPU sharpening method: {method} (slow)")
        
        # Try advanced sharpening if available and method is not "basic"
        if self.advanced_sharpener and method != "basic":
            try:
                # Convert PIL to numpy array
                img_array = np.array(image)
                
                # Map method to processor function
                method_map = {
                    "smart": self.advanced_sharpener.smart_sharpening,
                    "hiraloam": self.advanced_sharpener.hiraloam_sharpening,
                    "edge_directional": self.advanced_sharpener.edge_directional_sharpening,
                    "multiscale": self.advanced_sharpener.multiscale_laplacian_sharpening,
                    "guided": self.advanced_sharpener.guided_filter_sharpening
                }
                
                sharpen_func = method_map.get(method)
                if sharpen_func:
                    if self.debug_mode:
                        print(f"    Applying {method} sharpening (strength: {strength:.2f})")
                    
                    # Apply advanced sharpening
                    # Note: amount parameter typically 0.0-2.0, matches our strength
                    sharpened_array = sharpen_func(img_array, amount=strength)
                    
                    # Convert back to PIL
                    result = Image.fromarray(sharpened_array.astype(np.uint8))
                    
                    if self.debug_mode:
                        print(f"    Advanced sharpening complete: {method}")
                    
                    return result
                else:
                    if self.debug_mode:
                        print(f"    Unknown method '{method}', falling back to basic sharpening")
                    
            except Exception as e:
                if self.debug_mode:
                    print(f"    Advanced sharpening failed: {e}")
                    print(f"    Falling back to basic sharpening")
        
        # Fallback to basic selective sharpening
        if self.debug_mode:
            print(f"    Using basic sharpening (strength: {strength:.2f})")
        return self._selective_sharpen(image, strength, subject_boxes)
    
    def _detect_best_sharpening_method(self, image: Image.Image) -> str:
        """
        Intelligently choose sharpening method based on image characteristics
        
        Analyzes:
        - Image sharpness (Laplacian variance)
        - Edge density
        - Image size
        - Color distribution
        
        Returns:
            Best sharpening method name
        """
        
        try:
            # Convert to grayscale for analysis
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # 1. Check if already sharp (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 2. Check edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 3. Check image size
            width, height = image.size
            total_pixels = width * height
            
            # 4. Check for JPEG artifacts (high frequency noise)
            # Look at high-frequency components
            dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
            
            # Sample high frequency regions (corners of spectrum)
            h, w = magnitude.shape
            corner_size = min(50, h//10, w//10)
            corners = [
                magnitude[:corner_size, :corner_size],
                magnitude[:corner_size, -corner_size:],
                magnitude[-corner_size:, :corner_size],
                magnitude[-corner_size:, -corner_size:]
            ]
            high_freq_energy = np.mean([np.mean(corner) for corner in corners])
            
            # Decision tree based on characteristics
            if self.debug_mode:
                print(f"    Image analysis:")
                print(f"      Sharpness (Laplacian): {laplacian_var:.1f}")
                print(f"      Edge density: {edge_density:.3f}")
                print(f"      Total pixels: {total_pixels:,}")
                print(f"      High-freq energy: {high_freq_energy:.1f}")
            
            # Already very sharp - use gentle method to avoid over-sharpening
            if laplacian_var > 150:
                return "guided"  # Gentle, edge-preserving
            
            # High edge density (detailed image) - preserve edges
            elif edge_density > 0.15:
                return "edge_directional"  # Edge-aware
            
            # Low edge density (simple/scanned image) - aggressive sharpening
            elif edge_density < 0.05:
                return "hiraloam"  # Natural high-radius enhancement
            
            # High frequency noise (compressed/JPEG artifacts) - careful sharpening
            elif high_freq_energy > 5000:
                return "guided"  # Smooths while sharpening
            
            # Small image (<1MP) - use fast method
            elif total_pixels < 1_000_000:
                return "smart"  # Fast adaptive
            
            # Large high-quality image - use best quality
            elif total_pixels > 4_000_000 and laplacian_var > 50:
                return "multiscale"  # Highest quality multi-resolution
            
            # Default: smart adaptive
            else:
                return "smart"
                
        except Exception as e:
            if self.debug_mode:
                print(f"    Auto-detection failed: {e}")
                print(f"    Defaulting to 'smart' method")
            return "smart"
    
    def _lab_color_enhancement(self, image: Image.Image, strength: float) -> Image.Image:
        """LAB color enhancement (A/B channel contrast like Photoshop)"""
        if strength <= 0 or not self.settings["lab_color_enhancement"]:
            return image
        
        try:
            if SKIMAGE_AVAILABLE:
                # Convert to LAB color space
                img_array = np.array(image, dtype=np.float32) / 255.0
                
                if self.debug_mode:
                    print(f"    LAB enhancement input: {img_array.shape}, mode check: {image.mode}")
                
                lab_array = color.rgb2lab(img_array)
                
                # Extract channels
                L, A, B = lab_array[:, :, 0], lab_array[:, :, 1], lab_array[:, :, 2]
                
                # Apply contrast curves to A and B channels (not lightness)
                a_contrast = self.settings["lab_a_contrast"] * strength
                b_contrast = self.settings["lab_b_contrast"] * strength
                
                # Enhance A channel (Green-Red)
                A_mean = np.mean(A)
                A_enhanced = A_mean + (A - A_mean) * (1 + a_contrast)
                
                # Enhance B channel (Blue-Yellow)  
                B_mean = np.mean(B)
                B_enhanced = B_mean + (B - B_mean) * (1 + b_contrast)
                
                # Reconstruct LAB image
                lab_enhanced = np.stack([L, A_enhanced, B_enhanced], axis=2)
                
                # Convert back to RGB
                rgb_enhanced = color.lab2rgb(lab_enhanced)
                rgb_enhanced = np.clip(rgb_enhanced * 255, 0, 255).astype(np.uint8)
                
                result = Image.fromarray(rgb_enhanced, mode='RGB')
                
                if self.debug_mode:
                    print(f"    LAB enhancement output: {result.mode}, size: {result.size}")
                    print(f"    Applied LAB color enhancement (A: {a_contrast:.2f}, B: {b_contrast:.2f})")
                
                return result
            
            else:
                # Fallback: simple saturation boost
                enhancer = ImageEnhance.Color(image)
                saturation_factor = 1.0 + strength * 0.15
                result = enhancer.enhance(saturation_factor)
                
                if self.debug_mode:
                    print(f"    Applied saturation boost (factor: {saturation_factor:.2f})")
                
                return result
                
        except Exception as e:
            if self.debug_mode:
                print(f"    LAB color enhancement failed: {e}")
            return image
    
    def _final_exposure_adjustment(self, image: Image.Image, strength: float) -> Image.Image:
        """Final subtle exposure adjustment"""
        if strength <= 0:
            return image
        
        try:
            # Very subtle brightness adjustment
            enhancer = ImageEnhance.Brightness(image)
            brightness_factor = 1.0 + strength * 0.05  # Very subtle
            enhanced = enhancer.enhance(brightness_factor)
            
            if self.debug_mode:
                print(f"    Applied final exposure adjustment ({brightness_factor:.3f})")
            
            return enhanced
            
        except Exception as e:
            if self.debug_mode:
                print(f"    Final exposure adjustment failed: {e}")
            return image
    
    def update_settings(self, **kwargs):
        """Update internal settings easily"""
        for key, value in kwargs.items():
            if key in self.settings:
                self.settings[key] = value
                if self.debug_mode:
                    print(f"Updated setting {key} = {value}")
            else:
                if self.debug_mode:
                    print(f"Warning: Unknown setting {key}")
    
    def get_available_profiles(self) -> List[str]:
        """Get list of available enhancement profiles"""
        return list(self.profiles.keys())
    
    def create_custom_profile(self, name: str, settings: Dict[str, float]):
        """Create a custom enhancement profile"""
        required_keys = ["denoise", "sharpen", "tone_map", "color_pop", "artifact_removal"]
        
        if all(key in settings for key in required_keys):
            self.profiles[name] = settings
            if self.debug_mode:
                print(f"Created custom profile: {name}")
        else:
            raise ValueError(f"Profile must contain keys: {required_keys}")

    def benchmark_performance(self, test_image: Image.Image, iterations: int = 3) -> Dict[str, Any]:
        """Benchmark GPU vs CPU performance"""
        
        results = {
            "image_size": test_image.size,
            "iterations": iterations,
            "gpu_available": self.gpu_available
        }
        
        profile_settings = self.profiles["Digital Magazine"]
        
        # GPU benchmark
        if self.gpu_available:
            gpu_times = []
            for i in range(iterations):
                start_time = time.time()
                try:
                    _ = self._gpu_advanced_pipeline(test_image, profile_settings, 1.0, None)
                    gpu_times.append(time.time() - start_time)
                except Exception as e:
                    if self.debug_mode:
                        print(f"GPU benchmark iteration {i+1} failed: {e}")
                    break
            
            if gpu_times:
                results["gpu_avg_time"] = sum(gpu_times) / len(gpu_times)
                results["gpu_min_time"] = min(gpu_times)
                results["gpu_max_time"] = max(gpu_times)
        
        # CPU benchmark
        cpu_times = []
        for i in range(iterations):
            start_time = time.time()
            try:
                _ = self._cpu_advanced_pipeline(test_image, profile_settings, 1.0, None)
                cpu_times.append(time.time() - start_time)
            except Exception as e:
                if self.debug_mode:
                    print(f"CPU benchmark iteration {i+1} failed: {e}")
                break
        
        if cpu_times:
            results["cpu_avg_time"] = sum(cpu_times) / len(cpu_times)
            results["cpu_min_time"] = min(cpu_times)
            results["cpu_max_time"] = max(cpu_times)
            
            # Calculate speedup
            if "gpu_avg_time" in results and results["gpu_avg_time"] > 0:
                results["speedup"] = results["cpu_avg_time"] / results["gpu_avg_time"]
        
        return results

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information"""
        info = {
            "system_available": True,
            "gpu_available": self.gpu_available
        }
        
        # System memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["system_memory"] = {
                "total": mem.total,
                "available": mem.available,
                "percent": mem.percent
            }
        except ImportError:
            info["system_memory"] = "psutil not available"
        
        # GPU memory
        if self.gpu_available:
            try:
                gpu_mem = cp.cuda.Device().mem_info
                mempool = cp.get_default_memory_pool()
                info["gpu_memory"] = {
                    "total": gpu_mem[1],
                    "free": gpu_mem[0],
                    "used": gpu_mem[1] - gpu_mem[0],
                    "percent": ((gpu_mem[1] - gpu_mem[0]) / gpu_mem[1]) * 100,
                    "pool_used": mempool.used_bytes(),
                    "pool_total": mempool.total_bytes()
                }
            except Exception as e:
                info["gpu_memory"] = f"Error: {e}"
        
        return info

    def enhance_image_safe(self, 
                        image: Image.Image, 
                        profile: str = "Digital Magazine",
                        strength: float = 1.0,
                        subject_boxes: Optional[List] = None,
                        max_retries: int = 2) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Safe enhancement with detailed error reporting and retries
        
        Returns:
            Tuple of (enhanced_image, processing_info)
        """
        
        processing_info = {
            "success": False,
            "method_used": "none",
            "processing_time": 0.0,
            "errors": [],
            "retries": 0
        }
        
        start_time = time.time()
        
        for attempt in range(max_retries + 1):
            try:
                enhanced = self.enhance_image(image, profile, strength, subject_boxes)
                
                processing_info.update({
                    "success": True,
                    "method_used": "gpu" if self.gpu_available else "cpu",
                    "processing_time": time.time() - start_time,
                    "retries": attempt,
                    "final_size": enhanced.size,
                    "final_mode": enhanced.mode
                })
                
                return enhanced, processing_info
                
            except Exception as e:
                error_msg = f"Attempt {attempt + 1}: {str(e)}"
                processing_info["errors"].append(error_msg)
                
                if self.debug_mode:
                    print(f"⚠️ Enhancement failed on attempt {attempt + 1}: {e}")
                
                if attempt < max_retries:
                    # Clean up GPU memory before retry
                    if self.gpu_available:
                        try:
                            cp.get_default_memory_pool().free_all_blocks()
                        except:
                            pass
                    time.sleep(0.1)  # Brief pause before retry
                else:
                    # Final fallback: return original image
                    processing_info.update({
                        "method_used": "fallback_original",
                        "processing_time": time.time() - start_time
                    })
                    
                    if self.debug_mode:
                        print("❌ All enhancement attempts failed, returning original image")
                    
                    return image, processing_info
        
        return image, processing_info

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration and dependencies"""
        
        config_status = {
            "gpu_acceleration": {
                "available": self.gpu_available,
                "cupy": CUPY_AVAILABLE,
                "gpu_memory": None
            },
            "algorithms": {
                "wavelets": PYWT_AVAILABLE,
                "skimage": SKIMAGE_AVAILABLE,
                "opencv": True  # cv2 is imported
            },
            "profiles": {
                "count": len(self.profiles),
                "valid": True
            },
            "settings": {
                "valid": True,
                "issues": []
            }
        }
        
        # Check GPU memory if available
        if self.gpu_available:
            try:
                gpu_mem = cp.cuda.Device().mem_info
                config_status["gpu_acceleration"]["gpu_memory"] = {
                    "total_gb": gpu_mem[1] / 1e9,
                    "free_gb": gpu_mem[0] / 1e9
                }
            except:
                config_status["gpu_acceleration"]["gpu_memory"] = "Unable to query"
        
        # Validate profiles
        required_profile_keys = ["denoise", "sharpen", "tone_map", "color_pop", "artifact_removal"]
        for name, profile in self.profiles.items():
            if not all(key in profile for key in required_profile_keys):
                config_status["profiles"]["valid"] = False
                config_status["profiles"][f"invalid_{name}"] = "Missing required keys"
        
        # Validate settings
        if self.settings["wavelet_levels"] < 1 or self.settings["wavelet_levels"] > 5:
            config_status["settings"]["issues"].append("wavelet_levels should be 1-5")
        
        if self.settings["sharpen_radius"] <= 0 or self.settings["sharpen_radius"] > 5:
            config_status["settings"]["issues"].append("sharpen_radius should be 0-5")
        
        if config_status["settings"]["issues"]:
            config_status["settings"]["valid"] = False
        
        return config_status

# Example usage
if __name__ == "__main__":
    # Create enhancer
    enhancer = ModernImageEnhancer(debug_mode=True)
    
    # Test with sample image
    test_image_path = "test.jpg"
    if os.path.exists(test_image_path):
        image = Image.open(test_image_path)
        
        # Test different profiles
        profiles_to_test = ["Minimal", "Digital Magazine", "Scanned Photo"]
        
        for profile in profiles_to_test:
            enhanced = enhancer.enhance_image(image, profile=profile, strength=1.0)
            enhanced.save(f"enhanced_{profile.lower().replace(' ', '_')}.jpg", quality=95)
            print(f"Saved enhanced_{profile.lower().replace(' ', '_')}.jpg")
    
    # Show available profiles
    print(f"Available profiles: {enhancer.get_available_profiles()}")
    
    # Example of adjusting LAB settings
    enhancer.update_settings(
        lab_a_contrast=0.15,  # Slightly more color pop
        lab_b_contrast=0.10   # Less yellow-blue enhancement
    )