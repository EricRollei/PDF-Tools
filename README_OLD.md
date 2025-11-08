# PDF Tools - ComfyUI Custom Node Package

A comprehensive ComfyUI custom node collection for PDF processing with AI-powered analysis.

## 📢 Important Notice: Package Split

The **download functionality has moved** to a separate package:

- **PDF Tools** (this package): PDF extraction, OCR, AI vision processing
- **[Download Tools](../download-tools/)** (new package): gallery-dl and yt-dlp downloaders

If you need the download nodes (yt-dlp, gallery-dl), please install the **download-tools** package separately:

```powershell
cd ComfyUI/custom_nodes/download-tools
.\install.ps1
```

Both packages work independently and can be installed together or separately.

---

## 🎉 Quick Start

### Just Moved to New Install?

1. **Check Installation:**
   ```powershell
   cd A:\Comfy25\ComfyUI_windows_portable
   powershell -ExecutionPolicy Bypass -File .\ComfyUI\custom_nodes\PDF_tools\check_install.ps1
   ```

2. **Install Dependencies:**
   - The check script will offer to install everything
   - Or manually run: `.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\PDF_tools\requirements.txt`

3. **Start ComfyUI:**
   - Run your normal startup script
   - Nodes will appear in the "Add Node" menu

## 📦 What's Included

### PDF Extraction Nodes
- **PDF Extractor v08** - Advanced PDF image extraction with quality assessment and spread detection
- **PDF Extractor v09** - Latest version with enhanced image processing
- **PDF Extractor v05-v07** - Previous versions for compatibility
- **Simple PDF Extractor** - Basic PDF extraction without advanced features

### OCR & Layout Analysis Nodes
- **Surya OCR Layout Node** - State-of-the-art multilingual OCR with layout detection
- **Surya Layout OCR Hybrid** - Combined layout analysis and text extraction
- **Basic Surya** / **Basic Surya v02** - Simplified Surya OCR interfaces
- **Enhanced Layout Parser v06** - Advanced document layout understanding
- **LayoutLMv3 Node** - Microsoft LayoutLMv3 for document AI

### AI Vision & Detection Nodes
- **Florence2 Rectangle Detector** - Microsoft Florence-2 for object detection and vision analysis
- **Florence2 Cropper Node** - Crop images based on Florence-2 detections
- **Eric Rectangle Detector** - Custom rectangle detection implementation

### Remote Processing Nodes
- **PaddleOCR VL Remote** - Remote PaddleOCR processing for Chinese/multilingual documents

### Downloader Nodes (Moved)
> **Note:** Download functionality has moved to the [download-tools](../download-tools/) package.
> Install it separately if you need gallery-dl or yt-dlp downloaders.

## 🚀 Features

✅ **PDF Extraction** - Extract images from PDFs with quality scoring and duplicate detection  
✅ **AI-Powered OCR** - Surya, PaddleOCR, LayoutLMv3 for accurate text extraction  
✅ **Layout Analysis** - Understand document structure (headers, paragraphs, tables, figures)  
✅ **Vision Models** - Florence-2 for object detection and image understanding  
✅ **Spread Detection** - Automatically detect two-page spreads in scanned documents  
✅ **Quality Assessment** - Evaluate image sharpness, contrast, and overall quality  
✅ **Batch Processing** - Process multiple PDFs or images efficiently  
✅ **GPU Acceleration** - Leverage CUDA for faster AI model inference  

## 📚 Documentation

- **[INSTALLATION_SUCCESS.md](INSTALLATION_SUCCESS.md)** - Installation complete? Read this!
- **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** - Detailed setup instructions
- **[CODE_OVERVIEW.md](CODE_OVERVIEW.md)** - Understanding the codebase
- **[requirements.txt](requirements.txt)** - All Python dependencies

### Detailed Guides (in Docs/)
- `gallery_dl_node_complete_guide.md` - Gallery-dl usage guide
- `yt_dlp_node_complete_guide.md` - Yt-dlp usage guide  
- `SETUP_COMPLETE.md` - Authentication configuration
- Test scripts for all features

## 🔧 Requirements

### Core Dependencies (Auto-installed)
- PyMuPDF (fitz) - PDF processing
- Pillow - Image processing
- numpy - Array operations
- opencv-python - Computer vision
- transformers - AI models

### Downloader Tools (Auto-installed)
- gallery-dl - Web media downloader
- yt-dlp - Video/audio downloader
- browser-cookie3 - Cookie extraction

### Optional
- FFmpeg - For yt-dlp audio extraction (download separately)
- Tesseract - For enhanced OCR (download separately)

## � Configuration Files

**Important:** This repository uses **template config files** to protect your personal credentials.

### First Time Setup
1. Copy template files: `configs/*.example` → remove `.example` extension
2. Edit with your credentials (usernames, passwords, API keys)
3. Your personal configs are automatically excluded from Git

See **[CONFIG_SETUP.md](CONFIG_SETUP.md)** for detailed instructions.

Your personal configuration files (without `.example`) will **never** be committed to GitHub! 🔒

## �💡 Quick Examples

### Download from Instagram
```
Node: Gallery-dl Downloader
URL: https://www.instagram.com/username/
Output: ./instagram-downloads
Enable: organize_files, use_download_archive
```

### Download from YouTube
```
Node: Yt-dlp Downloader
URL: https://www.youtube.com/watch?v=VIDEO_ID
Format: best (or audio-only for music)
Output: ./youtube-downloads
```

### Extract PDF Pages
```
Node: PDF Extractor v08
Input: Your PDF file
Output: ./pdf-output
Enable: quality_assessment, spread_detection
```

## 🔐 Authentication

Many sites (Instagram, Reddit) require authentication:

**Option 1: Browser Cookies**
- Log into site in browser
- Enable `use_browser_cookies` in node
- Select browser (Firefox, Chrome, etc.)

**Option 2: Export Cookies**
- Install browser extension "Get cookies.txt LOCALLY"
- Export cookies while logged in
- Save to `configs/instagram_cookies.json`
- Set `cookie_file` parameter

## 📁 Project Structure

```
PDF_tools/
├── nodes/              # ComfyUI node implementations
├── florence2_scripts/  # Florence2 AI vision models
├── sam2_scripts/       # SAM2 segmentation models  
├── configs/            # Authentication & config files
├── Docs/               # Comprehensive documentation
└── [test directories]  # Test output locations
```

## ⚙️ System Requirements

- Windows 10/11 (primary), Linux (should work)
- Python 3.10+ (included with ComfyUI)
- NVIDIA GPU with CUDA (recommended for AI models)
- 8GB+ RAM (16GB+ for AI models)
- 5-10GB storage for packages + models

## 🐛 Troubleshooting

### "gallery-dl not found"
✅ **Already installed!** Use: `.\python_embeded\python.exe -m gallery_dl`

### "Chrome cookies not accessible"  
💡 **Solution:** Run ComfyUI as admin, or use Firefox, or export cookies

### "Instagram/Reddit downloads fail"
💡 **Solution:** Export cookies from logged-in browser session

### "CUDA out of memory"
💡 **Solution:** Close other GPU apps, reduce batch size, use smaller models

See INSTALLATION_GUIDE.md for detailed troubleshooting.

## 🎯 Best Practices

1. **Use Download Archives** - Enable to avoid re-downloading
2. **Enable File Organization** - Keep downloads sorted by type
3. **Export Cookies for Private Content** - More reliable than browser cookies
4. **Start with Public URLs** - Test setup before authenticated downloads
5. **Monitor GPU Usage** - When using AI models

## 📝 Version Info

- **Gallery-dl:** 1.30.9 ✅
- **Yt-dlp:** 2025.9.5 ✅
- **PyMuPDF:** 1.26.4 ✅
- **Transformers:** 4.55.0 ✅
- **Torch:** 2.7.1+cu128 ✅

## 🤝 Contributing

This is a working codebase! If you extend it:
- Follow existing patterns
- Add comprehensive error handling
- Include debug output
- Update documentation
- Test with real data

## 📄 License

Copyright (c) 2025 Eric Hiss. All rights reserved.

This software is dual-licensed:

- **Non-Commercial Use:** Licensed under [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](http://creativecommons.org/licenses/by-nc/4.0/)
- **Commercial Use:** Requires a separate commercial license. Contact: eric@historic.camera or eric@rollei.us

See [LICENSE.md](LICENSE.md) for full licensing terms.

### Third-Party Dependencies

This project relies on numerous open-source libraries, each with its own license. See [CREDITS.md](CREDITS.md) for a comprehensive list of all dependencies and their licenses.

**Important License Notes:**
- Some dependencies (gallery-dl, PyMuPDF) use GPL/AGPL licenses
- Commercial use requires compliance with all dependency licenses
- See CREDITS.md for detailed licensing information

## 👥 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style and standards
- Testing requirements
- Pull request process
- Development setup

## 📧 Contact

- **Author:** Eric Hiss
- **GitHub:** [EricRollei](https://github.com/EricRollei)
- **Email:** eric@historic.camera, eric@rollei.us

## 🙏 Acknowledgments

Special thanks to:
- The **ComfyUI** community for the amazing platform
- **Hugging Face** for hosting models and providing ML tools
- **Microsoft Research** for Florence-2 vision models
- **Meta AI** for SAM2 (Segment Anything Model 2)
- All open-source developers whose libraries make this project possible

See [CREDITS.md](CREDITS.md) for detailed acknowledgments.

## 🎉 Getting Started

1. ✅ **Check installation:** Run `check_install.ps1`
2. ✅ **Read:** [INSTALLATION_SUCCESS.md](INSTALLATION_SUCCESS.md)
3. ✅ **Start ComfyUI:** Nodes will appear automatically
4. ✅ **Test:** Try a simple download
5. ✅ **Configure:** Set up authentication if needed

---

**Ready to go!** All dependencies are installed. Read INSTALLATION_SUCCESS.md for next steps.

## 📚 Additional Documentation

- [LICENSE.md](LICENSE.md) - Full license terms
- [CREDITS.md](CREDITS.md) - Third-party libraries and acknowledgments
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
