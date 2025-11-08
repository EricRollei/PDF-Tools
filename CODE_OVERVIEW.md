# PDF Tools - Code Structure Summary

## Overview
This is a ComfyUI custom node package containing **downloader nodes** and **PDF/image processing nodes**. Despite the "PDF_tools" name, it's evolved into a comprehensive media download and processing toolkit.

## Directory Structure

```
PDF_tools/
├── __init__.py                    # Main entry point - loads all nodes
├── requirements.txt               # Python dependencies
├── INSTALLATION_GUIDE.md          # Setup instructions (NEW)
│
├── nodes/                         # ComfyUI node implementations
│   ├── gallery_dl_downloader.py   # Download from 100+ websites (Instagram, Reddit, etc.)
│   ├── yt_dlp_downloader.py       # Video/audio downloader (YouTube, TikTok, etc.)
│   ├── pdf_extractor_v08.py       # Latest PDF extraction node (MAIN VERSION)
│   ├── pdf_extractor_v07.py       # Previous version
│   ├── enhanced_layout_parser_v06.py  # Layout analysis node (MAIN VERSION)
│   ├── eric_rectangle_detector_node.py  # Rectangle detection node
│   └── [older versions]           # v01-v06 versions for reference
│
├── florence2_scripts/             # Florence2 AI vision model integration
│   ├── florence2_detector.py      # Main detector (based on kijai's implementation)
│   ├── modeling_florence2.py      # Florence2 model definition
│   ├── modern_image_enhancer.py   # Image enhancement utilities
│   └── [test/demo scripts]        # Various test and demo files
│
├── sam2_scripts/                  # SAM2 (Segment Anything 2) integration
│   ├── sam2_florence_segmentation.py  # Combined SAM2+Florence2
│   ├── sam2_integration.py        # SAM2 wrapper
│   └── [other scripts]
│
├── configs/                       # Configuration files
│   ├── auth_config.json           # Authentication for various sites
│   ├── instagram_cookies.json     # Cookie storage
│   ├── gallery-dl*.conf           # Gallery-dl configurations
│   └── yt-dlp*.conf              # Yt-dlp configurations
│
├── Docs/                          # Documentation and guides
│   ├── gallery_dl_node_complete_guide.md    # Gallery-dl setup guide
│   ├── yt_dlp_node_complete_guide.md        # Yt-dlp setup guide
│   ├── SETUP_COMPLETE.md          # Authentication setup
│   ├── test_*.py                  # Test scripts for various features
│   └── [various guides]           # Implementation guides
│
├── oldfiles/                      # Deprecated/backup files
├── test-*/                        # Test output directories
└── [other directories]            # History, cache, etc.
```

## Main Components

### 1. Gallery-dl Downloader (`nodes/gallery_dl_downloader.py`)
**Purpose:** Download media from 100+ websites  
**Key Features:**
- URL list or URL file input
- Browser cookie support (Chrome, Firefox, etc.)
- Exported cookie file support
- Download archive (SQLite) to avoid duplicates
- Automatic file organization (images/videos/audio/other)
- Video filtering
- Metadata extraction
- Rate limiting and retry logic
- Instagram-specific optimizations

**Main Class:** `GalleryDLDownloader`
**Key Methods:**
- `_build_command()` - Constructs gallery-dl command
- `_convert_cookie_file()` - Converts cookie formats
- `_organize_downloaded_files()` - Sorts files by type
- `download()` - Main download execution

**External Dependencies:**
- `gallery-dl` command-line tool
- `browser-cookie3` for browser cookie extraction

### 2. Yt-dlp Downloader (`nodes/yt_dlp_downloader.py`)
**Purpose:** Download videos and audio from 1000+ sites  
**Key Features:**
- URL list or batch file input
- Format selection (quality presets)
- Audio extraction (MP3, FLAC, etc.)
- Subtitle download and embedding
- Playlist support
- Rate limiting
- Cookie support
- File organization

**Main Class:** `YtDlpDownloader`
**Key Methods:**
- `_build_command()` - Constructs yt-dlp command
- `_organize_downloaded_files()` - Sorts files by type
- `download()` - Main download execution

**External Dependencies:**
- `yt-dlp` command-line tool
- `ffmpeg` for audio/video processing

### 3. PDF Extractor v08 (`nodes/pdf_extractor_v08.py`)
**Purpose:** Extract images and text from PDFs  
**Key Features:**
- Page extraction as images
- Quality assessment
- Spread detection (book scanning)
- OCR integration
- Metadata preservation
- Multiple output formats

**Main Classes:**
- `PDFExtractorV08Node` - Main ComfyUI node
- `PDFProcessor` - Core PDF processing logic

**Dependencies:**
- `PyMuPDF` (fitz) for PDF reading
- `PyPDF2` as fallback
- Florence2 for image analysis

### 4. Florence2 Detector (`florence2_scripts/florence2_detector.py`)
**Purpose:** AI-powered vision model for image analysis  
**Key Features:**
- Rectangle detection
- Object detection
- Caption generation
- Visual question answering

**Main Classes:**
- `Florence2RectangleDetector` - Main detector
- `BoundingBox` - Bounding box data structure

**Based on:** kijai's ComfyUI-Florence2 implementation

**Dependencies:**
- `transformers` library
- `safetensors`
- Florence2 models from Hugging Face

### 5. Enhanced Layout Parser v06 (`nodes/enhanced_layout_parser_v06.py`)
**Purpose:** Document layout analysis  
**Key Features:**
- Text block detection
- Figure/table detection
- Multi-engine OCR support
- Computer vision-based analysis

**Dependencies:**
- `opencv-python`
- Optional: Surya OCR, LayoutLMv3

## Architecture Patterns

### Node Loading System
The `__init__.py` dynamically loads all nodes from the `nodes/` directory:
1. Scans `nodes/` for `.py` files
2. Imports each module
3. Extracts `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`
4. Registers with ComfyUI

### Downloader Pattern
Both gallery-dl and yt-dlp nodes follow the same pattern:
1. **Command Building** - Construct CLI command with all options
2. **Subprocess Execution** - Run external tool via subprocess
3. **Output Parsing** - Parse stdout/stderr for status
4. **File Organization** - Move downloaded files into organized folders
5. **Status Reporting** - Return summary with file counts and errors

### Error Handling
All nodes use extensive try-except blocks with:
- Graceful fallbacks (e.g., PyMuPDF → PyPDF2)
- Detailed debug output
- Status reporting to user
- Optional strict error mode

### Memory Management
AI model nodes (Florence2, SAM2) include:
- GPU memory checking
- Model offloading to CPU
- Batch size management
- CUDA cache clearing

## Key Design Decisions

### 1. External Command-Line Tools
**Why:** Gallery-dl and yt-dlp are mature, well-maintained tools with extensive site support. Wrapping them provides instant access to 1000+ sites without reimplementing complex protocols.

**Trade-offs:**
- ✅ Comprehensive site support
- ✅ Active maintenance
- ❌ Subprocess overhead
- ❌ Error parsing complexity

### 2. File Organization Within Profile Directories
**Why:** Preserves gallery-dl's original directory structure while organizing by file type.

**Implementation:** Files are moved into subdirectories (images/, videos/, audio/, other/) within each profile/site directory.

### 3. Cookie Authentication
**Why:** Many sites (Instagram, Reddit) require authentication for full access.

**Implementation:**
- Browser cookie extraction via `browser-cookie3`
- Exported cookie file support (JSON/Netscape format)
- Automatic cookie format conversion

**Challenges:**
- Chrome requires admin privileges on Windows
- Cookie expiration handling
- Site-specific cookie formats

### 4. Modular AI Model Integration
**Why:** Allow optional AI features without forcing all dependencies.

**Implementation:**
- Try-except imports for optional models
- Graceful degradation when models unavailable
- Lazy loading of heavy models

## Configuration System

### Config Files (`configs/`)
- **auth_config.json** - Site credentials and API keys
- **instagram_cookies.json** - Cookie storage
- **gallery-dl.conf** - Gallery-dl options
- **yt-dlp.conf** - Yt-dlp options

### Runtime Configuration
Nodes accept configuration via:
1. Node parameters (primary)
2. Config files (secondary)
3. Environment variables (some cases)

Priority: Node params > Config file > Defaults

## Testing Strategy

### Test Scripts (`Docs/test_*.py`)
- **test_instagram_optimal_setup.py** - Instagram download test
- **test_file_sorting.py** - File organization test
- **test_gallery_dl_node.py** - Gallery-dl integration test
- **test_yt_dlp_node.py** - Yt-dlp integration test
- And many more...

### Test Pattern
```python
# 1. Setup test environment
# 2. Create node instance
# 3. Run with test data
# 4. Verify output
# 5. Cleanup
```

## Dependencies Summary

### Core (Required)
- `Pillow` - Image processing
- `numpy` - Array operations
- `PyMuPDF` - PDF reading
- `opencv-python` - Computer vision

### AI/ML (Optional)
- `torch` - Deep learning (from ComfyUI)
- `transformers` - Hugging Face models
- `safetensors` - Model format
- `timm` - Vision models

### Downloaders (Optional)
- `gallery-dl` - Web media downloader
- `yt-dlp` - Video downloader
- `browser-cookie3` - Cookie extraction

### Utilities
- `requests` - HTTP requests
- `tqdm` - Progress bars
- `colorama` - Terminal colors

## Performance Considerations

### Memory Usage
- **Florence2**: ~2-4GB VRAM
- **SAM2**: ~4-8GB VRAM
- **PDF Processing**: Depends on PDF size
- **Downloads**: Minimal (streaming)

### Optimization Strategies
1. **Model Offloading** - Move models to CPU when not in use
2. **Batch Processing** - Process multiple items together
3. **Caching** - Cache model outputs
4. **Streaming Downloads** - Don't load entire files in memory

## Known Issues & Limitations

### Reddit API
- May hang with old credentials
- 2FA complications
- **Solution:** Use browser cookies instead

### Chrome Cookie Access
- Requires admin privileges on Windows
- **Solution:** Use Firefox or export cookies

### Instagram Rate Limiting
- Too many requests can trigger rate limits
- **Solution:** Use `--sleep` parameter (implemented)

### Large PDF Files
- Memory intensive for large PDFs
- **Solution:** Process page by page

## Future Improvement Areas

1. **Async Downloads** - Parallel downloads for better performance
2. **Download Queue** - Queue system for batch downloads
3. **Better Error Recovery** - Automatic retry with backoff
4. **Model Caching** - Better model management
5. **UI Improvements** - Progress bars in ComfyUI
6. **Database Integration** - Track all downloads in database
7. **Webhook Support** - Notify on completion

## How to Extend

### Adding a New Downloader Node
1. Create new file in `nodes/`
2. Implement node class with required methods
3. Add `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`
4. Test with various URLs
5. Document in `Docs/`

### Adding a New PDF Feature
1. Extend `PDFProcessor` class
2. Add new parameters to node
3. Update `_build_command()` or processing logic
4. Test with various PDFs
5. Update documentation

### Adding a New AI Model
1. Create new script in `florence2_scripts/` or `sam2_scripts/`
2. Implement model loading and inference
3. Add memory management
4. Create node wrapper in `nodes/`
5. Test on GPU and CPU
6. Document model requirements

## Code Quality Notes

### Good Practices
- ✅ Extensive error handling
- ✅ Detailed documentation
- ✅ Debug output for troubleshooting
- ✅ Modular design
- ✅ Graceful degradation

### Areas for Improvement
- ⚠️ Some code duplication between versions
- ⚠️ Test coverage could be better
- ⚠️ Some large functions could be split
- ⚠️ More type hints would help
- ⚠️ Some hardcoded paths

## Security Considerations

### Cookie Handling
- Cookies stored in JSON files
- **Risk:** Plaintext cookie storage
- **Mitigation:** Document proper file permissions

### Subprocess Execution
- User input passed to subprocess
- **Risk:** Command injection
- **Mitigation:** Input validation and sanitization (implemented)

### External Commands
- Relies on external binaries
- **Risk:** Malicious binary execution
- **Mitigation:** Check binary availability, use known paths

## Version History

- **v08** - Current PDF extractor with enhanced features
- **v07** - Added spread detection
- **v06** - Enhanced layout parser
- Earlier versions in oldfiles/

## Contributing Guidelines

If extending this codebase:
1. Follow existing patterns
2. Add comprehensive error handling
3. Include debug output
4. Test with real data
5. Document in Docs/
6. Update requirements.txt if adding dependencies
7. Keep backward compatibility when possible

## Summary

This is a **well-structured, feature-rich** ComfyUI custom node package that combines:
- Mature external tools (gallery-dl, yt-dlp)
- Modern AI models (Florence2, SAM2)
- Robust error handling
- Comprehensive documentation

The codebase is **production-ready** for the download nodes and **functional but evolving** for the AI model integration. The modular design makes it easy to use only the features you need.
