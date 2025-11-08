# Gallery-dl ComfyUI Node - Complete Setup Guide

## Overview
This is a comprehensive ComfyUI node for downloading images and media from various websites using gallery-dl. The node includes support for authentication, file organization, and robust error handling.

## ✅ Completed Features

### Core Features
- **URL Support**: Download from URL lists or URL files
- **Authentication**: Browser cookies and exported cookie files
- **File Organization**: Automatically sort files into subfolders by type
- **Download Archive**: Avoid re-downloading the same files
- **Video Filtering**: Option to skip video files
- **Metadata Extraction**: Save download information to JSON
- **Debug Output**: Comprehensive status and error reporting

### File Organization

Downloaded files are automatically sorted into subfolders by type, **WITHIN each profile directory**:

**Example for Instagram profile:**
```
gallery-dl-output/
├── instagram/
│   └── janaioannaa/
│       ├── images/     ← Photos and images (jpg, png, webp, etc.)
│       ├── videos/     ← Video files (mp4, webm, mkv, etc.)
│       ├── audio/      ← Audio files (mp3, flac, wav, etc.)
│       └── other/      ← Documents and other files (pdf, txt, etc.)
└── gallery-dl-metadata.json
```

**Key Features:**
- Files are organized WITHIN each profile/site directory
- Preserves gallery-dl's original directory structure
- Each profile gets its own organized subdirectories
- No files are moved to the root level

## 🎯 Optimal Setup by Platform

### Instagram (Best Practice)
```
✅ config_path: LEAVE EMPTY (not needed)
✅ cookie_file: './configs/instagram_cookies.json'
✅ use_browser_cookies: False
✅ organize_files: True
✅ use_download_archive: True
```

**Why this works:**
- Instagram only needs cookies for authentication
- No config file required
- Exported cookies are more reliable than browser cookies
- Files get organized automatically

### Reddit (Best Practice)
```
✅ config_path: './configs/gallery-dl-no-reddit.conf'
✅ cookie_file: LEAVE EMPTY
✅ use_browser_cookies: True
✅ organize_files: True
✅ use_download_archive: True
```

**Why this works:**
- Reddit API credentials are problematic due to 2FA
- Browser cookies work better for Reddit
- Config file can disable Reddit if needed
- Files get organized automatically

### General Sites (Twitter, Imgur, etc.)
```
✅ config_path: LEAVE EMPTY (unless site-specific config needed)
✅ cookie_file: LEAVE EMPTY
✅ use_browser_cookies: True
✅ organize_files: True
✅ use_download_archive: True
```

## 📁 File Structure
```
PDF_tools/
├── nodes/
│   └── gallery_dl_downloader.py          # Main node implementation
├── configs/
│   ├── instagram_cookies.json            # Your exported Instagram cookies
│   ├── gallery-dl-no-reddit.conf         # Config that disables Reddit
│   ├── gallery-dl.conf                   # Full config with Reddit credentials
│   └── gallery-dl-browser-cookies.conf   # Browser cookies config
└── Docs/
    ├── test_file_sorting.py               # Test file organization
    └── test_instagram_optimal_setup.py    # Test optimal setup
```

## 🔧 Node Parameters

### Required Parameters
- **url_list**: URLs to download (one per line)
- **output_dir**: Where to save downloaded files

### Optional Parameters
- **url_file**: Text file containing URLs
- **config_path**: Path to gallery-dl config file
- **cookie_file**: Path to exported cookie JSON file
- **use_browser_cookies**: Extract cookies from browser
- **browser_name**: Which browser to use (firefox, chrome, edge, etc.)
- **use_download_archive**: Avoid re-downloading files
- **archive_file**: Path to download archive database
- **skip_videos**: Only download images, skip videos
- **extract_metadata**: Save download metadata to JSON
- **organize_files**: Sort files into subfolders by type

## 🍪 Cookie Setup

### Option 1: Exported Cookie File (Recommended for Instagram)
1. Install a browser extension like "Cookie Editor" or "EditThisCookie"
2. Visit Instagram and log in
3. Export cookies as JSON
4. Save to `./configs/instagram_cookies.json`
5. Use in node: `cookie_file: './configs/instagram_cookies.json'`

### Option 2: Browser Cookies (Recommended for Reddit)
1. Log into the site in your browser
2. Set `use_browser_cookies: True`
3. Choose your browser (Firefox works without admin privileges)

## 🚀 Usage Examples

### Example 1: Instagram Posts
```
url_list: https://www.instagram.com/p/ABC123/
output_dir: ./instagram-downloads
cookie_file: ./configs/instagram_cookies.json
organize_files: True
```

### Example 2: Reddit Posts
```
url_list: https://www.reddit.com/r/pics/comments/abc123/
output_dir: ./reddit-downloads
use_browser_cookies: True
organize_files: True
```

### Example 3: Multiple Sites
```
url_list: 
https://imgur.com/gallery/ABC123
https://twitter.com/user/status/123456
https://example.com/image.jpg
output_dir: ./mixed-downloads
use_browser_cookies: True
organize_files: True
```

## 🔍 Debug Information

The node provides comprehensive debug output including:
- Cookie file conversion status
- Authentication attempts
- File organization results
- Download progress
- Error messages and warnings

## 🧪 Testing

Run the test scripts to verify functionality:
```bash
# Test file organization
python Docs/test_file_sorting.py

# Test Instagram setup
python Docs/test_instagram_optimal_setup.py
```

## 🛠️ Troubleshooting

### Common Issues

1. **"gallery-dl not found"**
   - Install gallery-dl: `pip install gallery-dl`

2. **Authentication fails**
   - For Instagram: Use exported cookie file
   - For Reddit: Use browser cookies, avoid API credentials

3. **Files not organizing**
   - Ensure `organize_files: True`
   - Check debug output for move operations

4. **Downloads are slow**
   - Node includes automatic rate limiting (1 second between requests)
   - This is intentional to be respectful to servers

5. **Browser cookies not working**
   - Try Firefox (works without admin privileges)
   - For Chrome/Edge, run ComfyUI as administrator

### Debug Steps
1. Check the debug output in the node's summary
2. Verify file paths are correct
3. Test authentication with a single URL first
4. Check if gallery-dl works from command line

## 📚 Additional Resources

- [gallery-dl documentation](https://github.com/mikf/gallery-dl)
- [Supported sites list](https://github.com/mikf/gallery-dl/blob/master/docs/supportedsites.md)
- [gallery-dl configuration](https://github.com/mikf/gallery-dl/blob/master/docs/configuration.rst)

## 🔄 Version History

- **v1.0.0**: Initial release with file organization and authentication support
- All major features implemented and tested
- Ready for production use

## 🎉 Summary

The Gallery-dl ComfyUI node is now complete with:
- ✅ Robust authentication (browser cookies + exported cookies)
- ✅ File organization by type (images/, videos/, audio/, other/)
- ✅ Comprehensive error handling and debug output
- ✅ Optimal setup guides for Instagram and Reddit
- ✅ Extensive testing and validation

The node is ready for production use and supports all major requirements!
