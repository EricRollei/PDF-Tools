# Yt-dlp Downloader Node - Complete Guide

## Overview

The Yt-dlp Downloader Node is a powerful ComfyUI node that allows you to download audio and video content from hundreds of websites using the yt-dlp library. It provides a user-friendly interface with advanced features like format selection, audio extraction, subtitle downloads, and automatic file organization.

## Features

### 🎬 Video Downloads
- Download videos from YouTube, Vimeo, TikTok, Twitter, and hundreds more sites
- Format selection (best quality, specific resolution, audio-only, etc.)
- Playlist support with range selection
- Concurrent fragment downloads for speed
- Rate limiting to be respectful to servers

### 🎵 Audio Extraction
- Extract audio from videos in various formats (MP3, AAC, FLAC, etc.)
- Configurable audio quality (32-320 kbps)
- Perfect for music, podcasts, and lectures
- Automatic format conversion with ffmpeg

### 📝 Subtitle Support
- Download subtitles in multiple languages
- Embed subtitles directly in video files
- Support for auto-generated captions
- Organize subtitle files separately

### 🍪 Authentication
- Browser cookie extraction (Firefox, Chrome, Edge, Safari)
- Custom cookie file support
- Configuration file support
- Automatic authentication detection

### 📂 File Organization
- Automatic organization by file type (videos/, audio/, subtitles/, other/)
- Channel/uploader-based directory structure
- Duplicate handling with smart renaming
- Custom output templates

### 🔧 Advanced Features
- Download archive to prevent re-downloads
- Metadata extraction and embedding
- Thumbnail downloads
- Post-processing options
- Custom command-line options
- Comprehensive error handling and debugging

## Installation Requirements

### Required
- **yt-dlp**: `pip install yt-dlp`
- **Python 3.9+**: Modern Python version

### Optional but Recommended
- **ffmpeg**: Required for audio extraction, format merging, and subtitle embedding
  - Windows: Download from https://ffmpeg.org/download.html
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg` or equivalent
- **browser-cookie3**: For browser cookie extraction
  - `pip install browser-cookie3`

## Quick Start

### Basic Video Download
```python
# In ComfyUI workflow
yt_dlp_node = YtDlpDownloader()
result = yt_dlp_node.execute(
    url_list="https://www.youtube.com/watch?v=example",
    output_dir="./downloads"
)
```

### Audio Extraction
```python
result = yt_dlp_node.execute(
    url_list="https://www.youtube.com/watch?v=example",
    output_dir="./music",
    extract_audio=True,
    audio_format="mp3",
    audio_quality="192"
)
```

## Parameter Reference

### Required Parameters

#### `url_list`
- **Type**: Multiline String
- **Description**: URLs to download, one per line
- **Example**:
  ```
  https://www.youtube.com/watch?v=example1
  https://soundcloud.com/user/track
  # Comments starting with # are ignored
  ```

#### `output_dir`
- **Type**: String
- **Default**: `./yt-dlp-output`
- **Description**: Directory where files will be saved
- **Example**: `./downloads` or `C:/MyVideos`

### Optional Parameters

#### File Input/Output

##### `batch_file`
- **Type**: String
- **Default**: Empty
- **Description**: Path to text file containing URLs (one per line)
- **Example**: `./urls.txt`

##### `config_path`
- **Type**: String
- **Default**: Empty
- **Description**: Path to yt-dlp configuration file
- **Example**: `./configs/yt-dlp.conf`

##### `use_download_archive`
- **Type**: Boolean
- **Default**: True
- **Description**: Use archive file to skip already downloaded content

##### `archive_file`
- **Type**: String
- **Default**: `./yt-dlp-archive.txt`
- **Description**: Path to download archive file

#### Authentication

##### `cookie_file`
- **Type**: String
- **Default**: Empty
- **Description**: Path to Netscape cookies file
- **Example**: `./cookies.txt`

##### `use_browser_cookies`
- **Type**: Boolean
- **Default**: True
- **Description**: Extract cookies from browser (ignored if cookie file provided)

##### `browser_name`
- **Type**: Dropdown
- **Options**: firefox, chrome, chromium, edge, safari, opera
- **Default**: firefox
- **Description**: Browser to extract cookies from

#### Format Selection

##### `format_selector`
- **Type**: Dropdown
- **Options**: best, worst, best[height<=720], best[height<=480], bestvideo+bestaudio, bestvideo, bestaudio, mp4, webm
- **Default**: best
- **Description**: Video format to download

**Common Format Examples**:
- `best`: Highest quality available
- `worst`: Lowest quality (for testing)
- `bestaudio`: Audio only
- `best[height<=720]`: Max 720p resolution
- `bestvideo+bestaudio`: Best video + best audio merged

#### Audio Extraction

##### `extract_audio`
- **Type**: Boolean
- **Default**: False
- **Description**: Extract audio from videos (requires ffmpeg)

##### `audio_format`
- **Type**: Dropdown
- **Options**: mp3, aac, flac, m4a, opus, vorbis, wav
- **Default**: mp3
- **Description**: Audio format when extracting

##### `audio_quality`
- **Type**: Dropdown
- **Options**: 32, 64, 128, 192, 256, 320
- **Default**: 192
- **Description**: Audio bitrate in kbps

#### Subtitles

##### `download_subtitles`
- **Type**: Boolean
- **Default**: False
- **Description**: Download subtitle files

##### `subtitle_langs`
- **Type**: String
- **Default**: en
- **Description**: Subtitle languages (comma-separated)
- **Examples**: 
  - `en`: English only
  - `en,es,fr`: Multiple languages
  - `all`: All available languages

##### `embed_subtitles`
- **Type**: Boolean
- **Default**: False
- **Description**: Embed subtitles in video files (requires ffmpeg)

#### File Organization

##### `organize_files`
- **Type**: Boolean
- **Default**: True
- **Description**: Sort files into subfolders by type

##### `write_info_json`
- **Type**: Boolean
- **Default**: True
- **Description**: Save video metadata to JSON files

#### Performance

##### `rate_limit`
- **Type**: String
- **Default**: Empty
- **Description**: Maximum download speed
- **Examples**: `1M` (1MB/s), `500K` (500KB/s)

##### `concurrent_fragments`
- **Type**: Dropdown
- **Options**: 1, 2, 4, 8
- **Default**: 1
- **Description**: Number of fragments to download simultaneously

#### Playlist Options

##### `playlist_start`
- **Type**: String
- **Default**: Empty
- **Description**: Start index for playlists
- **Example**: `1` (start from first video)

##### `playlist_end`
- **Type**: String
- **Default**: Empty
- **Description**: End index for playlists
- **Example**: `10` (stop at tenth video)

#### Advanced

##### `extra_options`
- **Type**: String
- **Default**: Empty
- **Description**: Additional yt-dlp command-line options
- **Examples**: 
  - `--write-thumbnail`: Download thumbnails
  - `--embed-metadata`: Embed metadata in files
  - `--sleep-interval 2`: Sleep 2 seconds between downloads

## Output Structure

With file organization enabled, downloads are structured as:

```
output_dir/
├── ChannelName1/
│   ├── videos/
│   │   ├── video1.mp4
│   │   └── video2.webm
│   ├── audio/
│   │   ├── song1.mp3
│   │   └── song2.flac
│   ├── subtitles/
│   │   ├── video1.en.srt
│   │   └── video1.es.srt
│   └── other/
│       ├── video1.info.json
│       └── thumbnail.jpg
└── ChannelName2/
    └── videos/
        └── video3.mp4
```

## Configuration Files

Pre-made configuration files are available in the `configs/` directory:

### `yt-dlp.conf` - General Purpose
- Good quality settings (max 1080p)
- Basic metadata and thumbnails
- Rate limiting enabled
- Suitable for most use cases

### `yt-dlp-audio.conf` - Audio Optimized
- Audio extraction enabled
- High quality MP3 (320K)
- Optimized for music and podcasts
- Playlist handling enabled

### `yt-dlp-hq.conf` - High Quality Archival
- Best available quality
- All metadata and subtitles
- Comprehensive archiving
- Slower but thorough

## Usage Examples

### Example 1: Basic YouTube Download
```python
result = node.execute(
    url_list="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_dir="./youtube-downloads",
    format_selector="best[height<=720]",
    organize_files=True
)
```

### Example 2: Podcast Audio Extraction
```python
result = node.execute(
    url_list="""
    https://www.youtube.com/watch?v=podcast1
    https://www.youtube.com/watch?v=podcast2
    """,
    output_dir="./podcasts",
    extract_audio=True,
    audio_format="mp3",
    audio_quality="128",
    organize_files=True
)
```

### Example 3: Playlist with Range
```python
result = node.execute(
    url_list="https://www.youtube.com/playlist?list=PLExample",
    output_dir="./playlist-downloads",
    playlist_start="1",
    playlist_end="10",
    format_selector="best[height<=480]",
    rate_limit="500K"
)
```

### Example 4: Multilingual Subtitles
```python
result = node.execute(
    url_list="https://www.youtube.com/watch?v=example",
    output_dir="./videos-with-subs",
    download_subtitles=True,
    subtitle_langs="en,es,fr,de",
    embed_subtitles=True,
    organize_files=True
)
```

### Example 5: Batch File Processing
```python
# Create urls.txt file:
# https://www.youtube.com/watch?v=example1
# https://vimeo.com/example2
# https://soundcloud.com/user/track

result = node.execute(
    url_list="",  # Empty when using batch file
    batch_file="./urls.txt",
    output_dir="./batch-downloads",
    use_download_archive=True
)
```

### Example 6: High Quality with Config
```python
result = node.execute(
    url_list="https://www.youtube.com/watch?v=example",
    output_dir="./hq-downloads",
    config_path="./configs/yt-dlp-hq.conf",
    use_browser_cookies=True,
    browser_name="firefox"
)
```

## Supported Sites

Yt-dlp supports hundreds of websites. Some popular ones include:

### Video Platforms
- YouTube, YouTube Music
- Vimeo, Dailymotion
- TikTok, Instagram
- Twitter/X, Facebook
- Twitch, Kick

### Audio Platforms
- SoundCloud, Bandcamp
- Mixcloud, AudioBoom
- Spotify (limited)

### Educational
- Khan Academy, Coursera
- edX, Udemy
- TED Talks

### News & Media
- BBC iPlayer, CNN
- NPR, PBS
- Various news outlets

See the full list at: https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md

## Troubleshooting

### Common Issues

#### "yt-dlp not found"
**Solution**: Install yt-dlp
```bash
pip install yt-dlp
```

#### "Audio extraction failed"
**Solution**: Install ffmpeg
- Windows: Download from https://ffmpeg.org/
- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

#### "Cookie access failed"
**Solution**: 
- Use Firefox (doesn't require admin rights)
- Or export cookies manually to a file
- Or disable cookie usage for public content

#### "Download failed with 403/404"
**Possible causes**:
- Video is private/deleted
- Geographic restrictions
- Authentication required
- Rate limiting

**Solutions**:
- Check if URL is accessible in browser
- Use browser cookies for authentication
- Enable rate limiting
- Try different format selector

#### "Files not organized"
**Solution**:
- Ensure `organize_files=True`
- Check if files were actually downloaded
- Verify write permissions in output directory

### Debug Information

The node provides detailed debug output including:
- Detected target sites
- Authentication status
- Command used
- File organization progress
- Error details

Check the summary output for debug information when troubleshooting.

## Performance Tips

### Speed Optimization
1. **Concurrent Fragments**: Set to 4 or 8 for faster downloads
2. **Format Selection**: Use specific formats instead of "best"
3. **Rate Limiting**: Don't set too low unless necessary
4. **Archive File**: Enable to skip re-downloads

### Quality Optimization
1. **Format Selector**: Use `bestvideo+bestaudio` for best quality
2. **Audio Quality**: Set to 320K for best audio
3. **Subtitles**: Download all languages with `all`
4. **Metadata**: Enable JSON and embedding

### Reliability
1. **Download Archive**: Always enable for large batches
2. **Error Handling**: Built-in retry and ignore-errors
3. **Browser Cookies**: Use for authenticated content
4. **Rate Limiting**: Be respectful to servers

## Advanced Usage

### Custom Format Selectors
```python
# Download only 720p MP4
format_selector="best[height=720][ext=mp4]"

# Audio only, prefer MP3
format_selector="bestaudio[ext=mp3]/bestaudio"

# Best video under 100MB
format_selector="best[filesize<100M]"
```

### Complex Extra Options
```python
extra_options="""
--write-thumbnail
--embed-metadata
--sleep-interval 2
--max-sleep-interval 5
--add-metadata
"""
```

### Custom Output Templates (via config file)
```
# In yt-dlp.conf
--output "%(uploader)s/%(upload_date)s - %(title)s [%(id)s].%(ext)s"
```

## API Reference

### Return Values

The node returns a tuple with:
1. **output_dir** (str): Path to output directory
2. **summary** (str): Detailed summary with debug info
3. **download_count** (int): Number of files downloaded
4. **success** (bool): Whether download succeeded

### Error Handling

The node includes comprehensive error handling:
- Timeout protection (10 minutes)
- Automatic retries (3 attempts)
- Graceful failure with partial results
- Detailed error reporting

## Best Practices

### For Regular Use
1. Enable download archive
2. Use file organization
3. Set reasonable rate limits
4. Use browser cookies for authenticated content

### For Archival
1. Use high-quality config
2. Download all subtitles
3. Save all metadata
4. Use descriptive output templates

### For Audio Content
1. Use audio extraction
2. Set appropriate quality
3. Organize by type
4. Consider format compatibility

### For Batch Processing
1. Use batch files for large lists
2. Enable archive to resume interrupted downloads
3. Use playlist ranges for testing
4. Monitor disk space

## Version History

- **v1.0.0** (January 2025): Initial release
  - Full yt-dlp integration
  - File organization
  - Audio extraction
  - Subtitle support
  - Browser cookie integration
  - Comprehensive error handling

## Support

For issues and questions:
1. Check the debug output in the summary
2. Verify yt-dlp and ffmpeg installation
3. Test with simple URLs first
4. Check yt-dlp documentation for format-specific issues

## License

This node is part of the PDF_tools ComfyUI extension and follows the same licensing terms.

---

*Happy downloading! 🎉*
