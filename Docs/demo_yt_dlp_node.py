#!/usr/bin/env python3
"""
Demo script for the Yt-dlp Downloader Node
Shows practical usage examples for different scenarios

Author: Eric Hiss
Date: January 2025
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the current directory to Python path to import our node
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

try:
    from nodes.yt_dlp_downloader import YtDlpDownloader, YtDlpNode, check_yt_dlp_installation
    print("✅ Successfully imported Yt-dlp node components")
except ImportError as e:
    print(f"❌ Failed to import yt-dlp node: {e}")
    sys.exit(1)


def demo_check_installation():
    """Demo: Check if yt-dlp is properly installed"""
    print("\n🔍 Demo: Checking yt-dlp installation")
    print("-" * 40)
    
    is_installed, message = check_yt_dlp_installation()
    
    if is_installed:
        print(f"✅ yt-dlp is installed: {message.strip()}")
    else:
        print(f"❌ yt-dlp not found: {message}")
        print("📦 Install with: pip install yt-dlp")
    
    return is_installed


def demo_basic_download():
    """Demo: Basic video download"""
    print("\n🎬 Demo: Basic Video Download")
    print("-" * 40)
    
    # Example URLs (feel free to replace with your own)
    demo_urls = """
# YouTube examples
https://www.youtube.com/watch?v=jNQXAC9IVRw

# Or try these (comment/uncomment as needed):
# https://vimeo.com/148751763
# https://soundcloud.com/example/track
"""
    
    print("Example usage for basic video download:")
    print(f"URLs:\n{demo_urls}")
    
    print("\nNode configuration:")
    print("- Output: ./yt-dlp-downloads")
    print("- Format: best quality available")
    print("- Archive: enabled (avoid re-downloads)")
    print("- File organization: enabled")
    print("- Metadata: JSON files saved")
    
    # Create a YtDlpNode and show how to use it
    node = YtDlpNode()
    
    print("\nComfyUI Node call example:")
    print("""
result = node.execute(
    url_list=demo_urls,
    output_dir="./yt-dlp-downloads",
    format_selector="best",
    use_download_archive=True,
    organize_files=True,
    write_info_json=True
)
""")
    
    # Don't actually run the download in demo mode
    print("💡 To run this, uncomment the execution code in the script")


def demo_audio_extraction():
    """Demo: Audio-only downloads"""
    print("\n🎵 Demo: Audio Extraction")
    print("-" * 40)
    
    print("Example usage for audio extraction:")
    print("- Extract MP3 audio from videos")
    print("- Quality: 192 kbps")
    print("- Organize into audio/ folder")
    print("- Good for music, podcasts, lectures")
    
    print("\nNode configuration:")
    print("""
result = node.execute(
    url_list="https://www.youtube.com/watch?v=example",
    output_dir="./audio-downloads",
    extract_audio=True,
    audio_format="mp3",
    audio_quality="192",
    organize_files=True,
    write_info_json=True
)
""")
    
    print("📁 Output structure:")
    print("audio-downloads/")
    print("├── ChannelName/")
    print("│   ├── audio/")
    print("│   │   ├── song1.mp3")
    print("│   │   └── song2.mp3")
    print("│   └── other/")
    print("│       ├── song1.info.json")
    print("│       └── song2.info.json")


def demo_playlist_download():
    """Demo: Playlist downloads with range selection"""
    print("\n📋 Demo: Playlist Download")
    print("-" * 40)
    
    print("Example usage for playlist downloads:")
    print("- Download specific range from playlist")
    print("- Rate limiting to be respectful")
    print("- Concurrent fragments for speed")
    
    print("\nNode configuration:")
    print("""
result = node.execute(
    url_list="https://www.youtube.com/playlist?list=example",
    output_dir="./playlist-downloads",
    playlist_start="1",
    playlist_end="10",
    rate_limit="1M",
    concurrent_fragments="4",
    organize_files=True,
    use_download_archive=True
)
""")
    
    print("🔧 Advanced options:")
    print("- playlist_start: Start from video 1")
    print("- playlist_end: Stop at video 10") 
    print("- rate_limit: Max 1MB/s download speed")
    print("- concurrent_fragments: Download 4 fragments in parallel")


def demo_subtitle_download():
    """Demo: Subtitle downloads"""
    print("\n📝 Demo: Subtitle Download")
    print("-" * 40)
    
    print("Example usage for subtitle downloads:")
    print("- Download subtitles in multiple languages")
    print("- Embed subtitles in video files")
    print("- Organize into subtitles/ folder")
    
    print("\nNode configuration:")
    print("""
result = node.execute(
    url_list="https://www.youtube.com/watch?v=example",
    output_dir="./video-with-subs",
    download_subtitles=True,
    subtitle_langs="en,es,fr",
    embed_subtitles=True,
    organize_files=True
)
""")
    
    print("📁 Output structure:")
    print("video-with-subs/")
    print("├── ChannelName/")
    print("│   ├── videos/")
    print("│   │   └── video.mp4 (with embedded subs)")
    print("│   ├── subtitles/")
    print("│   │   ├── video.en.srt")
    print("│   │   ├── video.es.srt")
    print("│   │   └── video.fr.srt")


def demo_advanced_options():
    """Demo: Advanced configuration options"""
    print("\n⚙️ Demo: Advanced Options")
    print("-" * 40)
    
    print("Example usage with advanced options:")
    print("- Custom config file")
    print("- Browser cookies for authentication")
    print("- Extra command-line options")
    print("- Format selection")
    
    print("\nNode configuration:")
    print("""
result = node.execute(
    url_list="https://www.youtube.com/watch?v=example",
    output_dir="./advanced-download",
    config_path="./configs/yt-dlp-hq.conf",
    use_browser_cookies=True,
    browser_name="firefox",
    format_selector="best[height<=720]",
    extra_options="--write-thumbnail --embed-metadata",
    organize_files=True
)
""")
    
    print("🔧 Advanced features:")
    print("- config_path: Use predefined configuration")
    print("- browser_cookies: Extract cookies from Firefox")
    print("- format_selector: Limit to 720p max")
    print("- extra_options: Additional yt-dlp flags")


def demo_batch_file():
    """Demo: Batch file usage"""
    print("\n📄 Demo: Batch File Download")
    print("-" * 40)
    
    print("Example batch file content (urls.txt):")
    print("""
# Music videos
https://www.youtube.com/watch?v=example1
https://www.youtube.com/watch?v=example2

# Podcast episodes  
https://soundcloud.com/podcast/episode1
https://soundcloud.com/podcast/episode2

# Documentary
https://vimeo.com/documentary

# Comments starting with # are ignored
""")
    
    print("\nNode configuration:")
    print("""
result = node.execute(
    url_list="",  # Empty when using batch file
    batch_file="./urls.txt",
    output_dir="./batch-downloads",
    extract_audio=True,  # Good for mixed content
    organize_files=True,
    use_download_archive=True
)
""")


def demo_error_handling():
    """Demo: Error handling and debugging"""
    print("\n🐛 Demo: Error Handling")
    print("-" * 40)
    
    print("The node provides detailed debug information:")
    print("- Detected target sites")
    print("- Authentication status")
    print("- File organization progress")
    print("- Command used")
    print("- Error details")
    
    print("\nExample debug output:")
    print("""
🔧 Debug Information:
📁 Output directory: ./downloads
🎯 Detected target sites: youtube
✅ Firefox: Found 156 cookies
📦 Using download archive: ./yt-dlp-archive.txt
🎬 Format selector: best
📄 Writing metadata JSON files
⚡ Rate limit: 1M
🔧 Added standard options: no-warnings, ignore-errors, 3 retries
📊 Files before download: 0
📊 Files after download: 3
📊 New files this run: 3
📂 Starting file organization by type...
📄 Queued video.mp4 for organization in ChannelName
📁 Moved video.mp4 to ChannelName/videos/ folder
📂 Organization completed successfully. Final file count: 3
""")


def demo_config_files():
    """Demo: Configuration files"""
    print("\n📄 Demo: Configuration Files")
    print("-" * 40)
    
    configs_dir = Path(__file__).parent.parent / "configs"
    
    print("Available configuration files:")
    
    config_files = [
        ("yt-dlp.conf", "General purpose downloads"),
        ("yt-dlp-audio.conf", "Audio extraction optimized"),
        ("yt-dlp-hq.conf", "High quality archival"),
    ]
    
    for config_file, description in config_files:
        config_path = configs_dir / config_file
        print(f"- {config_file}: {description}")
        if config_path.exists():
            print(f"  ✅ Available at: {config_path}")
        else:
            print(f"  ❌ Not found at: {config_path}")
    
    print("\nUsage example:")
    print("""
result = node.execute(
    url_list="https://www.youtube.com/watch?v=example",
    output_dir="./downloads",
    config_path="./configs/yt-dlp-hq.conf"  # Use high quality config
)
""")


def main():
    """Run all demos"""
    print("🚀 Yt-dlp Downloader Node - Usage Demos")
    print("=" * 50)
    
    # Check installation first
    if not demo_check_installation():
        print("\n⚠️ Note: Some demos may not work without yt-dlp installed")
    
    demos = [
        demo_basic_download,
        demo_audio_extraction,
        demo_playlist_download,
        demo_subtitle_download,
        demo_batch_file,
        demo_advanced_options,
        demo_config_files,
        demo_error_handling,
    ]
    
    for demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"❌ Demo error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed!")
    print("\n💡 Tips:")
    print("- Install yt-dlp: pip install yt-dlp")
    print("- Install ffmpeg for audio extraction and subtitle embedding")
    print("- Use browser cookies for authenticated downloads")
    print("- Enable file organization for better structure")
    print("- Use download archive to avoid re-downloads")
    print("- Check the configs/ folder for example configurations")


if __name__ == "__main__":
    main()
