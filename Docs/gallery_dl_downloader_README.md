# Gallery-dl Downloader Node for ComfyUI

## Overview
This ComfyUI node allows you to download images and media from various websites using the powerful gallery-dl tool with integrated authentication and comprehensive debugging.

## 🆕 New Features (v1.1)
- ✅ **Integrated Auth Config**: Use your existing authentication configuration
- ✅ **Enhanced Debugging**: Detailed status reporting with debug information
- ✅ **Browser Cookie Testing**: Real-time cookie access validation
- ✅ **Local Config Support**: Auth config now stored in PDF_tools/configs/
- ✅ **Permission Fix**: No more cross-directory permission issues

## Authentication Integration

### Your Auth Config Support
The node now uses your existing `auth_config.json` file with automatic conversion to gallery-dl format.

**Location**: `PDF_tools/configs/auth_config.json` (copied from your Metadata_system)

**Default Path**: `./configs/auth_config.json` (relative to PDF_tools directory)

### Debugging Output
The status report now includes comprehensive debugging information:

```
🔧 Debug Information:
🔍 Loading auth config from: /path/to/auth_config.json
✅ Auth config loaded successfully with 15 sites  
🔑 Found auth config for domain: reddit.com
🍪 Extracting cookies from chrome browser
✅ Chrome: Found 1247 cookies
📄 Using generated authentication config: /path/to/temp_config.json
✅ Generated config for reddit
⚡ Rate limiting: 1 second between requests, 3 retries on failure
```

## Supported Sites
Gallery-dl supports hundreds of websites including:
- **Image hosting**: Imgur, Flickr, 500px, etc.
- **Social media**: Twitter, Instagram, Pinterest, etc.
- **Art platforms**: DeviantArt, ArtStation, Pixiv, etc.
- **Boorus**: Danbooru, Gelbooru, e621, etc.
- **And many more!**

## Installation

### 1. Install gallery-dl
First, you need to install gallery-dl:

```bash
pip install gallery-dl
```

### 2. Verify Installation
Check that gallery-dl is properly installed:

```bash
gallery-dl --version
```

### 3. Optional: Install yt-dlp for video support
For better video handling:

```bash
pip install yt-dlp
```

## Usage

### Basic Usage
1. Add the "Gallery-dl Downloader" node to your ComfyUI workflow
2. Enter URLs in the `url_list` field (one per line)
3. Set your desired `output_dir`
4. Run the workflow

### Advanced Features

#### Browser Cookies
Enable `use_browser_cookies` to use your browser's login cookies for authenticated downloads:
- Supports: Firefox, Chrome, Chromium, Edge, Safari, Opera
- Useful for downloading from accounts you're logged into

#### Download Archive
Keep `use_download_archive` enabled to avoid re-downloading the same files:
- Creates a SQLite database to track downloaded content
- Speeds up subsequent runs

#### Configuration Files
Use `config_path` to specify a gallery-dl configuration file for:
- Custom download directories per site
- Filename templates
- Site-specific settings
- API keys and authentication

#### Video Filtering
Enable `skip_videos` to download only images and skip video files.

## Example URLs
```
# Single image
https://imgur.com/abcd123

# Album/Gallery
https://imgur.com/a/xyz789

# User profile
https://www.deviantart.com/username

# Specific post
https://twitter.com/username/status/1234567890
```

## Output
The node returns:
- `output_dir`: Path to downloaded files
- `summary`: Human-readable download summary
- `download_count`: Number of files downloaded
- `success`: Boolean indicating if download succeeded

## Configuration File Example
Create a `gallery-dl.conf` file for advanced settings:

```json
{
    "extractor": {
        "base-directory": "./downloads",
        "archive": "./archive.sqlite3",
        "sleep": 1.0,
        "retries": 3
    },
    "output": {
        "filename": "{category}_{subcategory}_{id}.{extension}",
        "directory": ["{category}", "{subcategory}"]
    }
}
```

## Troubleshooting

### Gallery-dl not found
If you get "gallery-dl not found" error:
1. Make sure gallery-dl is installed: `pip install gallery-dl`
2. Check your PATH environment variable
3. Try installing in the same Python environment as ComfyUI

### Permission errors
If you get permission errors:
1. Make sure the output directory is writable
2. Run ComfyUI with appropriate permissions
3. Check file system permissions

### Download failures
If downloads fail:
1. Check if the URL is supported by gallery-dl
2. Some sites may require authentication (use browser cookies)
3. Check site-specific rate limits
4. Verify internet connection

## 🔧 Troubleshooting Common Issues

### Permission Errors with Auth Config
**Problem**: "Permission denied" when accessing config files

**Solutions**:
1. ✅ **Use Local Config**: Config now stored in `PDF_tools/configs/auth_config.json`
2. ✅ **Default Path**: Leave auth_config_path as `./configs/auth_config.json`
3. ✅ **Check Permissions**: Ensure ComfyUI has read access to the config directory
4. ✅ **Full Path**: Use absolute path if needed: `C:\path\to\PDF_tools\configs\auth_config.json`

### Browser Cookie Issues
**Problem**: Browser cookies not being extracted

**Debugging Steps**:
1. **Check Debug Output**: Look for cookie information in the status report:
   ```
   🍪 Extracting cookies from chrome browser
   ✅ Chrome: Found 1247 cookies
   ```

2. **Common Issues**:
   - ❌ `browser_cookie3 library not installed` → Install: `pip install browser-cookie3`
   - ❌ `Chrome cookie access failed` → Close Chrome completely and try again
   - ❌ `Found 0 cookies` → Check if you're logged into the browser
   - ❌ `Permission denied` → Run as administrator or check browser profile permissions

3. **Browser-Specific Fixes**:
   - **Chrome**: Close all Chrome windows, ensure no background processes
   - **Firefox**: Check if Firefox is using a custom profile location
   - **Edge**: Similar to Chrome, ensure it's completely closed
   - **Safari**: macOS only, may need keychain access permissions

### Debug Information Not Showing
**Problem**: No debugging output in status report

**Solutions**:
1. ✅ **Enable Debug**: The new version automatically includes debug information
2. ✅ **Check Summary**: Debug info appears in the "summary" output field
3. ✅ **Look for Section**: Debug info appears under "🔧 Debug Information:"

### Gallery-dl Installation Issues
**Problem**: "gallery-dl not found" error

**Solutions**:
1. **Install**: `pip install gallery-dl`
2. **Upgrade**: `pip install --upgrade gallery-dl`
3. **Virtual Environment**: Install in the same Python environment as ComfyUI
4. **PATH Issue**: Add Python Scripts directory to PATH

### Authentication Not Working
**Problem**: Downloads fail despite having auth config

**Debug Steps**:
1. **Check Debug Output** for auth-related messages:
   ```
   ✅ Auth config loaded successfully with 15 sites
   🔑 Found auth config for domain: reddit.com
   ✅ Generated config for reddit
   ```

2. **Common Issues**:
   - Domain mismatch (reddit.com vs www.reddit.com)
   - Expired cookies or API keys
   - Incorrect auth type for the site
   - Missing required auth fields

3. **Solutions**:
   - Update cookies by logging in again
   - Check API key expiration dates
   - Verify domain names match exactly
   - Enable browser cookies as backup authentication

## Links
- [Gallery-dl Documentation](https://github.com/mikf/gallery-dl)
- [Supported Sites List](https://github.com/mikf/gallery-dl/blob/master/docs/supportedsites.md)
- [Configuration Guide](https://github.com/mikf/gallery-dl/blob/master/docs/configuration.rst)
