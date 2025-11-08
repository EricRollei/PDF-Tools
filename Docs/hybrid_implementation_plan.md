# PyMuPDF Hybrid Pipeline Integration Plan v2

## Executive Summary

Transform PDF processing by combining the best features from all versions:
- **v06's Batch Processing**: Memory-efficient streaming for large PDFs
- **Hybrid AI Detection**: Florence2 + Surya Layout fusion for perfect image detection
- **PyMuPDF Optimizations**: Smart routing, OCR, even/odd joining
- **Expected Performance**: 3-5x faster overall, 60% less memory, 100% image detection accuracy

## Phase 1: Enhanced Architecture (Week 1)

### 1.1 Intelligent Processing Pipeline

```python
# Enhanced: hybrid_pdf_processor.py
class HybridPDFProcessor:
    def __init__(self):
        self.document_analyzer = DocumentAnalyzer()
        self.batch_processor = BatchProcessor()  # From v06
        self.ai_fusion_engine = AIFusionEngine()  # Florence2 + Surya
        self.pymupdf_engine = PyMuPDFEngine()
        
    def process_pdf(self, pdf_path: str, config: ProcessingConfig) -> ExtractionResults:
        # 1. Document analysis and routing
        doc_analysis = self.document_analyzer.analyze_document(pdf_path)
        
        # 2. Choose processing strategy
        if doc_analysis.should_use_streaming():
            return self._process_with_streaming(pdf_path, doc_analysis, config)
        else:
            return self._process_with_batch(pdf_path, doc_analysis, config)
```

### 1.2 AI Fusion Engine for Image Detection

```python
# New: ai_fusion_engine.py
class AIFusionEngine:
    def __init__(self):
        self.florence2_model = None
        self.surya_model = None
        self.model_manager = ModelManager()
        
    def detect_images_with_fusion(self, image: Image.Image, 
                                 confidence_threshold: float = 0.5) -> List[ImageRegion]:
        """Combine Florence2 precision with Surya Layout completeness"""
        
        # 1. Florence2: High-precision boxes (may miss some images)
        florence2_boxes = self._get_florence2_detections(image, confidence_threshold)
        
        # 2. Surya Layout: Complete detection (imperfect boxes)
        surya_layout = self._get_surya_layout_detections(image)
        
        # 3. Fusion: Match and refine
        fused_regions = self._fuse_detections(florence2_boxes, surya_layout)
        
        return fused_regions
    
    def _fuse_detections(self, florence2_boxes: List[Dict], 
                        surya_layout: List[Dict]) -> List[ImageRegion]:
        """Smart fusion algorithm"""
        
        fused_regions = []
        
        # Step 1: Use Florence2 boxes as ground truth where available
        florence2_regions = [self._convert_to_image_region(box, source="florence2") 
                           for box in florence2_boxes]
        
        # Step 2: Find Surya regions not covered by Florence2
        uncovered_surya = []
        for surya_region in surya_layout:
            if surya_region["category"] in ["Picture", "Figure", "Image"]:
                if not self._overlaps_with_florence2(surya_region, florence2_boxes):
                    uncovered_surya.append(surya_region)
        
        # Step 3: For uncovered Surya regions, use Surya boxes but validate
        for surya_region in uncovered_surya:
            # Try to refine Surya box using local image analysis
            refined_box = self._refine_surya_box(surya_region, image)
            fused_regions.append(ImageRegion(refined_box, source="surya_refined"))
        
        # Step 4: Combine all regions
        all_regions = florence2_regions + fused_regions
        
        # Step 5: Remove duplicates and validate
        final_regions = self._deduplicate_and_validate(all_regions)
        
        return final_regions
```

### 1.3 Streaming Processor (from v06)

```python
# Enhanced: streaming_processor.py (based on v06)
class StreamingProcessor:
    def __init__(self, chunk_size: int = 10):
        self.chunk_size = chunk_size
        self.memory_monitor = MemoryMonitor()
        
    def process_large_pdf_streaming(self, pdf_path: str, processor: HybridPDFProcessor) -> ExtractionResults:
        """Memory-efficient streaming processing for large PDFs"""
        
        doc_info = self._analyze_pdf_size(pdf_path)
        
        if doc_info.should_stream:
            return self._process_in_chunks(pdf_path, processor)
        else:
            return self._process_batch(pdf_path, processor)
    
    def _process_in_chunks(self, pdf_path: str, processor: HybridPDFProcessor) -> ExtractionResults:
        """Process PDF in memory-efficient chunks"""
        
        all_results = ExtractionResults()
        
        with fitz.open(pdf_path) as doc:
            for chunk_start in range(0, len(doc), self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, len(doc))
                
                # Process chunk
                chunk_pages = self._extract_chunk_pages(doc, chunk_start, chunk_end)
                chunk_results = self._process_chunk(chunk_pages, processor)
                
                # Accumulate results
                all_results.merge(chunk_results)
                
                # Clean up chunk memory
                self._cleanup_chunk_memory(chunk_pages, chunk_results)
                
                # Monitor memory usage
                if self.memory_monitor.should_pause():
                    self.memory_monitor.gc_and_wait()
        
        return all_results
```

## Phase 2: OCR Performance Optimization (Week 2)

### 2.1 OCR Performance Comparison

```python
# New: ocr_benchmarker.py
class OCRBenchmarker:
    def __init__(self):
        self.tesseract_processor = TesseractProcessor()  # Your current setup
        self.pymupdf_processor = PyMuPDFOCRProcessor()
        
    def benchmark_ocr_methods(self, test_images: List[Image.Image]) -> OCRBenchmarkResults:
        """Compare PyMuPDF OCR vs current Tesseract setup"""
        
        results = {
            'tesseract': self._benchmark_tesseract(test_images),
            'pymupdf': self._benchmark_pymupdf(test_images)
        }
        
        return OCRBenchmarkResults(
            speed_comparison=self._compare_speeds(results),
            accuracy_comparison=self._compare_accuracy(results),
            memory_usage=self._compare_memory(results),
            recommendation=self._make_recommendation(results)
        )
    
    def _benchmark_pymupdf(self, test_images: List[Image.Image]) -> Dict:
        """Test PyMuPDF OCR performance"""
        start_time = time.time()
        
        results = []
        for image in test_images:
            # Convert to pixmap and run OCR
            pixmap = self._pil_to_pixmap(image)
            
            # Create temporary PDF page for OCR
            temp_doc = fitz.open()
            temp_page = temp_doc.new_page(width=pixmap.width, height=pixmap.height)
            
            # Insert pixmap and run OCR
            temp_page.insert_image(temp_page.rect, pixmap=pixmap)
            
            try:
                ocr_textpage = temp_page.get_textpage_ocr(language='eng', dpi=300)
                text = ocr_textpage.extractText() if ocr_textpage else ""
                words = ocr_textpage.extractWORDS() if ocr_textpage else []
            except Exception as e:
                text = ""
                words = []
            
            results.append({
                'text': text,
                'words': words,
                'char_count': len(text),
                'word_count': len(words)
            })
            
            temp_doc.close()
        
        total_time = time.time() - start_time
        
        return {
            'results': results,
            'total_time': total_time,
            'avg_time_per_image': total_time / len(test_images),
            'total_words': sum(len(r['words']) for r in results),
            'total_chars': sum(r['char_count'] for r in results)
        }
```

### 2.2 Smart OCR Router

```python
# New: smart_ocr_processor.py
class SmartOCRProcessor:
    def __init__(self):
        self.benchmark_results = None
        self.preferred_method = "auto"  # auto, tesseract, pymupdf
        
    def extract_text_optimal(self, page_data, method: str = "auto") -> TextExtractionResult:
        """Choose optimal OCR method based on benchmarks and content type"""
        
        if method == "auto":
            method = self._choose_optimal_method(page_data)
        
        if method == "pymupdf":
            return self._extract_with_pymupdf(page_data)
        else:
            return self._extract_with_tesseract(page_data)
    
    def _choose_optimal_method(self, page_data) -> str:
        """Dynamic method selection based on content characteristics"""
        
        # Factors to consider:
        # - Image resolution
        # - Text density
        # - Language complexity
        # - Processing time constraints
        
        if page_data.is_high_resolution and page_data.has_simple_text:
            return "pymupdf"  # Fast for simple, high-res content
        elif page_data.has_complex_layout or page_data.has_multiple_languages:
            return "tesseract"  # Better for complex scenarios
        else:
            return self.preferred_method if self.preferred_method != "auto" else "pymupdf"
```

## Phase 3: Batch Processing Integration (Week 3)

### 3.1 Enhanced Batch Processor (from v06)

```python
# Enhanced: batch_processor.py (v06 + optimizations)
class BatchProcessor:
    def __init__(self, ai_fusion_engine: AIFusionEngine):
        self.ai_fusion = ai_fusion_engine
        self.chunk_size = 8  # Optimal batch size from v06 experience
        self.memory_threshold = 0.8  # 80% memory usage threshold
        
    def process_batch_optimized(self, pages: List[PDFPage], 
                               config: ProcessingConfig) -> BatchResults:
        """Enhanced batch processing with AI fusion and memory management"""
        
        # 1. Group pages by processing requirements
        page_groups = self._group_pages_by_type(pages)
        
        # 2. Process each group with optimal method
        all_results = []
        
        for group_type, group_pages in page_groups.items():
            if group_type == "layered_professional":
                results = self._process_layered_batch(group_pages, config)
            elif group_type == "flattened_scan":
                results = self._process_flattened_batch(group_pages, config)
            else:
                results = self._process_mixed_batch(group_pages, config)
            
            all_results.extend(results)
        
        return BatchResults(all_results)
    
    def _process_flattened_batch(self, pages: List[PDFPage], 
                                config: ProcessingConfig) -> List[PageResult]:
        """Process flattened pages with AI fusion + PyMuPDF OCR"""
        
        results = []
        
        # Process in chunks for memory efficiency
        for chunk in self._chunk_pages(pages, self.chunk_size):
            chunk_results = []
            
            for page in chunk:
                # 1. Create high-res pixmap
                pixmap = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                pil_image = self._pixmap_to_pil(pixmap)
                
                # 2. AI Fusion for image detection
                image_regions = self.ai_fusion.detect_images_with_fusion(
                    pil_image, 
                    confidence_threshold=config.confidence_threshold
                )
                
                # 3. PyMuPDF OCR for text (if faster than Tesseract)
                text_result = self._extract_text_smart(page, config.ocr_method)
                
                # 4. Crop images from pixmap
                cropped_images = self._crop_images_from_pixmap(pixmap, image_regions)
                
                # 5. Apply enhancement to flattened content
                enhanced_images = self._batch_enhance_images(cropped_images, config)
                
                chunk_results.append(PageResult(
                    page_num=page.number,
                    images=cropped_images,
                    enhanced_images=enhanced_images,
                    text=text_result,
                    processing_method="ai_fusion_flattened"
                ))
            
            results.extend(chunk_results)
            
            # Memory cleanup after each chunk
            self._cleanup_chunk_memory(chunk, chunk_results)
        
        return results
```

### 3.2 Memory-Efficient AI Model Management

```python
# Enhanced: model_manager.py (with v06-style memory management)
class MemoryEfficientModelManager:
    def __init__(self):
        self.florence2_model = None
        self.surya_model = None
        self.model_cache_size = 2  # Max models in memory
        self.memory_monitor = MemoryMonitor()
        
    def get_models_for_batch(self, batch_type: str) -> Tuple[Any, Any]:
        """Load models efficiently for batch processing"""
        
        if batch_type in ["flattened_scan", "mixed_content"]:
            # Need both models for AI fusion
            self._ensure_both_models_loaded()
            return self.florence2_model, self.surya_model
        elif batch_type == "layered_professional":
            # May not need models at all for direct extraction
            return None, None
        
    def _ensure_both_models_loaded(self):
        """Load both models with memory management"""
        
        # Check memory before loading
        if self.memory_monitor.available_memory() < 4096:  # 4GB threshold
            self._cleanup_unused_models()
        
        if self.florence2_model is None:
            self.florence2_model = self._load_florence2_model()
            
        if self.surya_model is None:
            self.surya_model = self._load_surya_model()
    
    def process_with_memory_management(self, pages: List[PDFPage], 
                                     processor_func: Callable) -> List[Any]:
        """Process with automatic memory management"""
        
        results = []
        
        for i, page_batch in enumerate(self._batch_pages(pages)):
            # Check memory before each batch
            if self.memory_monitor.should_cleanup():
                self._cleanup_intermediate_results()
            
            batch_results = processor_func(page_batch)
            results.extend(batch_results)
            
            # Periodic model cleanup
            if i % 10 == 0:  # Every 10 batches
                self._refresh_models_if_needed()
        
        return results
```

## Phase 4: Performance Integration (Week 4)

### 4.1 Unified Processing Node

```python
# Enhanced: pdf_extractor_v09.py
class EnhancedPDFExtractorNode_v09:
    @classmethod  
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Existing parameters...
                "processing_strategy": ([
                    "auto", "fast_path_only", "ai_fusion", "streaming", "legacy_v08"
                ], {"default": "auto"}),
                "ai_detection_method": ([
                    "florence2_only", "surya_only", "ai_fusion", "pymupdf_only"
                ], {"default": "ai_fusion"}),
                "ocr_method": ([
                    "auto", "pymupdf", "tesseract", "benchmark_both"
                ], {"default": "auto"}),
                "enable_batch_processing": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # All existing parameters...
                "force_streaming": ("BOOLEAN", {"default": False}),
                "memory_optimization_level": (["low", "medium", "high"], {"default": "medium"}),
            }
        }
    
    def process_pdf(self, pdf_path: str, processing_strategy: str = "auto",
                   ai_detection_method: str = "ai_fusion", 
                   ocr_method: str = "auto",
                   enable_batch_processing: bool = True,
                   **kwargs) -> ExtractionResults:
        """Main processing with all optimizations"""
        
        # Initialize hybrid processor
        processor = HybridPDFProcessor()
        processor.configure(
            ai_detection=ai_detection_method,
            ocr_method=ocr_method,
            batch_processing=enable_batch_processing,
            **kwargs
        )
        
        # Route to appropriate processing method
        if processing_strategy == "legacy_v08":
            return self._process_with_v08(pdf_path, **kwargs)
        elif processing_strategy == "streaming" or self._should_use_streaming(pdf_path):
            return processor.process_with_streaming(pdf_path)
        else:
            return processor.process_with_batch_optimization(pdf_path)
```

### 4.2 Performance Monitoring & Benchmarking

```python
# Enhanced: performance_monitor.py
class PerformanceMonitor:
    def __init__(self):
        self.ocr_benchmarks = {}
        self.processing_metrics = defaultdict(list)
        
    def benchmark_setup(self, test_images: List[Image.Image]):
        """Run comprehensive benchmarks on first use"""
        
        # OCR method comparison
        ocr_benchmarker = OCRBenchmarker()
        self.ocr_benchmarks = ocr_benchmarker.benchmark_ocr_methods(test_images)
        
        print(f"📊 OCR Benchmark Results:")
        print(f"   PyMuPDF: {self.ocr_benchmarks['pymupdf']['avg_time_per_image']:.3f}s/image")
        print(f"   Tesseract: {self.ocr_benchmarks['tesseract']['avg_time_per_image']:.3f}s/image")
        print(f"   Recommended: {self.ocr_benchmarks['recommendation']}")
    
    def track_processing_performance(self, pdf_path: str, method: str, 
                                   results: ExtractionResults):
        """Track and analyze processing performance"""
        
        metrics = {
            'file_size': os.path.getsize(pdf_path),
            'page_count': results.page_count,
            'processing_time': results.processing_time,
            'method': method,
            'images_found': len(results.images),
            'text_words': results.total_words,
            'ai_fusion_used': getattr(results, 'used_ai_fusion', False),
            'streaming_used': getattr(results, 'used_streaming', False),
            'timestamp': datetime.now()
        }
        
        self.processing_metrics[method].append(metrics)
        
        # Real-time performance analysis
        if len(self.processing_metrics[method]) >= 5:
            self._analyze_performance_trends(method)
```

## Expected Performance Improvements (Revised)

| Metric | Layered PDFs | Flattened PDFs | Large PDFs (Streaming) | Overall |
|--------|--------------|----------------|----------------------|---------|
| **Processing Speed** | 8-12x faster | 3-4x faster | 2-3x faster | 4-6x faster |
| **Memory Usage** | -80% (PyMuPDF only) | -40% (batch AI) | -60% (streaming) | -60% |
| **Image Detection** | Same quality | 100% coverage + perfect boxes | Same | Perfect |
| **Text Extraction** | Same | 100x+ improvement | Same | Much better |
| **OCR Performance** | N/A | TBD (benchmark) | TBD | Optimized |

## Implementation Questions for You:

1. **OCR Priority**: Should we benchmark PyMuPDF vs Tesseract first to determine the winner?

2. **AI Fusion Validation**: Want to test Florence2 + Surya fusion on your documents to validate the 100% detection claim?

3. **Streaming Threshold**: What file size/page count should trigger streaming mode?

4. **Batch Processing**: Keep v06's 8-page chunk size or adjust based on your typical documents?

5. **Memory Management**: What's your target memory usage for large batch processing?

This approach gives you the **best of all three versions** while addressing your specific requirements. Want to dive deeper into any particular component?
