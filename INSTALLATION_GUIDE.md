# PDF Tools Custom Node - Installation Guide

## Overview

This is a comprehensive ComfyUI custom node package that provides powerful tools for:
- **PDF Extraction & Processing** - Extract images and text from PDFs
- **Media Downloading** - Download images/videos from Instagram, Reddit, Twitter, YouTube, etc.
- **AI-Powered Image Analysis** - Florence2 vision models for rectangle detection
- **Layout Analysis** - Detect document layouts and structures
- **Image Enhancement** - Modern image enhancement for better quality

## What's Included

### 1. **Gallery-dl Downloader Node**
Download media from 100+ websites including:
- Instagram (posts, stories, reels)
- Reddit (posts, galleries)
- Twitter/X (images, videos)
- Imgur, DeviantArt, Artstation, and more

Features:
- Browser cookie authentication
- File organization by type (images/videos/audio)
- Download archive to avoid duplicates
- Metadata extraction

### 2. **Yt-dlp Downloader Node**
Download videos and audio from:
- YouTube (videos, playlists, channels)
- TikTok, Twitch, Instagram videos
- 1000+ video platforms

Features:
- Format selection (quality presets)
- Audio extraction (MP3, FLAC, etc.)
- Subtitle download and embedding
- Playlist support

### 3. **PDF Extractor Nodes**
Multiple versions for extracting content from PDFs:
- Extract images with quality assessment
- OCR text recognition
- Spread detection (book scanning)
- Metadata preservation
- Multiple output formats

### 4. **Florence2 Rectangle Detector**
AI-powered image analysis using Microsoft's Florence2 model:
- Detect rectangular regions in images
- Caption generation
- Object detection
- Visual question answering

### 5. **Layout Parser Nodes**
Document layout analysis:
- Detect text blocks, figures, tables
- Enhanced OCR with multiple engines
- Computer vision-based layout detection

## Installation Steps

### Step 1: Navigate to Your Python Environment

```powershell
cd A:\Comfy25\ComfyUI_windows_portable
```

### Step 2: Install Core Requirements

Use the embedded Python to install packages:

```powershell
.\python_embeded\python.exe -m pip install -r custom_nodes\PDF_tools\requirements.txt
```

### Step 3: Install External Tools (Optional but Recommended)

#### 3a. Install gallery-dl (for web downloads)
```powershell
.\python_embeded\python.exe -m pip install gallery-dl
```

#### 3b. Install yt-dlp (for video downloads)
```powershell
.\python_embeded\python.exe -m pip install yt-dlp
```

#### 3c. Install FFmpeg (Required for yt-dlp audio extraction)
1. Download FFmpeg from: https://www.gyan.dev/ffmpeg/builds/
2. Extract to `C:\ffmpeg\` (or any location)
3. Add `C:\ffmpeg\bin` to your system PATH
4. Verify: `ffmpeg -version`

### Step 4: Verify Installation

Test that packages are installed:

```powershell
# Test gallery-dl
.\python_embeded\python.exe -m gallery_dl --version

# Test yt-dlp
.\python_embeded\python.exe -m yt_dlp --version

# Test PyMuPDF
.\python_embeded\python.exe -c "import fitz; print(f'PyMuPDF {fitz.__version__}')"

# Test transformers
.\python_embeded\python.exe -c "import transformers; print(f'Transformers {transformers.__version__}')"
```

### Step 5: Configure Authentication (Optional)

#### For Instagram/Reddit downloads:

1. **Export cookies from your browser:**
   - Use browser extension: "Get cookies.txt LOCALLY" (Chrome/Firefox)
   - Save as `configs/instagram_cookies.json` (Netscape format is fine)

2. **Or use browser cookies directly:**
   - Set `use_browser_cookies: True` in the node
   - Chrome requires admin privileges on Windows
   - Firefox works without admin

#### For Reddit API (if using config file):
- See `Docs/reddit_app_creation_guide.py` for setup instructions
- Note: Reddit API requires OAuth and may have rate limits

### Step 6: Start ComfyUI

```powershell
# Start ComfyUI normally
.\run_nvidia_gpu.bat

# Or with admin privileges (for Chrome cookie access)
# Right-click run_nvidia_gpu.bat → "Run as administrator"
```

## Quick Test

### Test Gallery-dl Node:

1. Add "Gallery-dl Downloader" node to workflow
2. Set URL: `https://www.instagram.com/janaioannaa/` (or any public profile)
3. Set output_dir: `./test-output`
4. Run workflow
5. Check `test-output/instagram/janaioannaa/images/` for downloaded files

### Test Yt-dlp Node:

1. Add "Yt-dlp Downloader" node to workflow
2. Set URL: `https://www.youtube.com/watch?v=dQw4w9WgXcQ` (example)
3. Set output_dir: `./yt-output`
4. Run workflow
5. Check output directory for downloaded video

### Test PDF Extractor:

1. Add "PDF Extractor v08" node to workflow
2. Load a PDF file
3. Set output directory
4. Run to extract pages as images

## Common Issues & Solutions

### Issue: "gallery-dl: command not found"
**Solution:** Install with pip: `python_embeded\python.exe -m pip install gallery-dl`

### Issue: "yt-dlp: command not found"
**Solution:** Install with pip: `python_embeded\python.exe -m pip install yt-dlp`

### Issue: "FFmpeg not found" (yt-dlp)
**Solution:** 
1. Download FFmpeg: https://www.gyan.dev/ffmpeg/builds/
2. Extract and add to PATH
3. Or place `ffmpeg.exe` in ComfyUI root directory

### Issue: "PyMuPDF not found"
**Solution:** `python_embeded\python.exe -m pip install PyMuPDF`

### Issue: Chrome cookies not accessible
**Solutions:**
1. Run ComfyUI as administrator (Windows security restriction)
2. Or use Firefox instead (doesn't require admin)
3. Or export cookies to file and use `cookie_file` parameter

### Issue: Instagram/Reddit downloads fail
**Solutions:**
1. Export cookies from logged-in browser session
2. Place in `configs/instagram_cookies.json`
3. Set `cookie_file` parameter in node
4. Make sure you're logged into the site in your browser

### Issue: "CUDA out of memory" (Florence2/AI models)
**Solutions:**
1. Close other GPU applications
2. Reduce batch size in node settings
3. Use smaller model variants
4. Enable model offloading in ComfyUI settings

### Issue: Transformers version conflicts
**Solution:** 
```powershell
.\python_embeded\python.exe -m pip install transformers>=4.35.0 --upgrade
```

## Package Size Warnings

Some packages are large and optional:

- **Surya OCR**: ~1GB models (advanced OCR)
- **SAM2**: ~1-2GB models (segmentation)
- **Florence2 models**: ~500MB-2GB (vision models, auto-downloaded)
- **PaddleOCR**: ~500MB models (Chinese/English OCR)
- **EasyOCR**: ~1GB models (multi-language OCR)

These are **commented out** in requirements.txt - only install if needed.

## Minimal Installation

If you only want specific features:

### Just Gallery-dl (web downloads):
```powershell
.\python_embeded\python.exe -m pip install gallery-dl browser-cookie3 requests
```

### Just Yt-dlp (video downloads):
```powershell
.\python_embeded\python.exe -m pip install yt-dlp
```

### Just PDF extraction:
```powershell
.\python_embeded\python.exe -m pip install PyMuPDF Pillow numpy
```

### Just Florence2 (AI vision):
```powershell
.\python_embeded\python.exe -m pip install transformers safetensors accelerate timm
```

## Next Steps

1. **Review the documentation:**
   - `Docs/gallery_dl_node_complete_guide.md` - Gallery-dl setup
   - `Docs/yt_dlp_node_complete_guide.md` - Yt-dlp setup
   - `Docs/SETUP_COMPLETE.md` - Authentication setup

2. **Test with example workflows:**
   - Start with simple single-URL downloads
   - Test authentication with your accounts
   - Try batch downloads from files

3. **Configure for your needs:**
   - Set up cookie files for authenticated sites
   - Create custom config files for specific sites
   - Organize download directories

## Getting Help

- Check the `Docs/` folder for detailed guides
- Review test scripts in `Docs/test_*.py` for examples
- Check ComfyUI console for debug output (nodes provide detailed status)

## System Requirements

- **OS**: Windows 10/11 (primary), Linux (should work)
- **GPU**: NVIDIA GPU with CUDA (for AI models, optional for downloaders)
- **RAM**: 8GB minimum, 16GB+ recommended for AI models
- **Storage**: 5-10GB for packages + models
- **Python**: 3.10+ (comes with ComfyUI portable)

## What's Working

✅ Gallery-dl downloads (Instagram, Reddit, Twitter, etc.)
✅ Yt-dlp downloads (YouTube, TikTok, etc.)
✅ PDF extraction with PyMuPDF
✅ Florence2 rectangle detection
✅ Browser cookie authentication
✅ File organization by type
✅ Download archives (no duplicates)
✅ Metadata extraction
✅ Debug output and error handling

## Known Limitations

⚠️ Reddit API may hang with old credentials (use browser cookies instead)
⚠️ Chrome cookies require admin privileges on Windows
⚠️ Some sites require valid login cookies
⚠️ Large models (Florence2, SAM2) require GPU memory
⚠️ Transformers versions may need updates

## Support & Updates

This is a custom node package. For issues:
1. Check the documentation in `Docs/`
2. Review test scripts for working examples
3. Check ComfyUI console for detailed error messages
4. Ensure all requirements are installed correctly

Happy downloading and processing! 🚀
