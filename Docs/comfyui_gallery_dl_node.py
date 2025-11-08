"""
Gallery-dl Downloader Node for ComfyUI
Downloads images and media from various websites using gallery-dl

Features:
- Download from URLs or URL files
- Browser cookie support
- Download archive to avoid duplicates
- Video filtering options
- Metadata extraction
- Configurable output directory

Author: Eric Hiss
Version: 1.0.0
Date: July 2025
"""

import os
import subprocess
import tempfile
import json
import shutil

from pathlib import Path
from typing import List, Tuple, Dict, Any

class GalleryDLDownloader:
    def __init__(
        self,
        url_list: List[str] = None,
        url_file: str = None,
        output_dir: str = "./gallery-dl-output",
        config_path: str = None,
        use_browser_cookies: bool = False,
        browser_name: str = "firefox",
        use_download_archive: bool = True,
        archive_file: str = "./gallery-dl-archive.sqlite3",
        skip_videos: bool = False,
        extract_metadata: bool = True
    ):
        self.url_list = url_list or []
        self.url_file = url_file
        self.output_dir = output_dir
        self.config_path = config_path
        self.use_browser_cookies = use_browser_cookies
        self.browser_name = browser_name
        self.use_download_archive = use_download_archive
        self.archive_file = archive_file
        self.skip_videos = skip_videos
        self.extract_metadata = extract_metadata

    def _check_gallery_dl_installed(self) -> bool:
        """Check if gallery-dl is installed and accessible."""
        try:
            result = subprocess.run(
                ["gallery-dl", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _build_command(self, urls: List[str]) -> List[str]:
        command = ["gallery-dl"]

        command += ["-d", self.output_dir]

        if self.config_path:
            command += ["--config", self.config_path]

        if self.use_browser_cookies:
            command += ["--cookies-from-browser", self.browser_name]

        if self.use_download_archive:
            command += ["--download-archive", self.archive_file]

        if self.skip_videos:
            command += ["--filter", "extension not in ('mp4', 'webm', 'mkv', 'avi', 'mov', 'wmv', 'flv', 'm4v')"]

        # Add rate limiting to be respectful to servers
        command += ["--sleep", "1"]
        
        # Add retry logic
        command += ["--retries", "3"]

        command += urls
        return command

    def run(self) -> Dict[str, Any]:
        """Execute the gallery-dl download process."""
        # Check if gallery-dl is installed
        if not self._check_gallery_dl_installed():
            raise RuntimeError(
                "gallery-dl is not installed or not accessible. "
                "Please install it using: pip install gallery-dl"
            )
        
        # Prepare URLs
        if self.url_file:
            if not os.path.exists(self.url_file):
                raise FileNotFoundError(f"URL file not found: {self.url_file}")
            with open(self.url_file, "r", encoding="utf-8") as f:
                urls = [line.strip() for line in f if line.strip()]
        else:
            urls = [url.strip() for url in self.url_list if url.strip()]

        if not urls:
            raise ValueError("No URLs provided for download.")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Build and execute command
        command = self._build_command(urls)
        
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=self.output_dir  # Set working directory
            )
            stdout, stderr = process.communicate(timeout=300)  # 5 minute timeout
        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError("Download process timed out after 5 minutes")

        # Count downloaded files
        downloaded_files = []
        if os.path.exists(self.output_dir):
            for root, dirs, files in os.walk(self.output_dir):
                for file in files:
                    if not file.endswith('.json'):  # Skip metadata files
                        downloaded_files.append(os.path.join(root, file))

        result = {
            "command": " ".join(command),
            "stdout": stdout,
            "stderr": stderr,
            "returncode": process.returncode,
            "output_dir": self.output_dir,
            "downloaded_files": downloaded_files,
            "download_count": len(downloaded_files),
            "success": process.returncode == 0
        }

        # Save metadata if requested
        if self.extract_metadata:
            metadata_path = Path(self.output_dir) / "gallery-dl-metadata.json"
            try:
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save metadata: {e}")

        return result


class GalleryDLNode:
    """
    ComfyUI Node for downloading images and media using gallery-dl.
    
    Supports downloading from various websites including:
    - Image hosting sites (imgur, flickr, etc.)
    - Social media platforms (Twitter, Instagram, etc.)
    - Art platforms (DeviantArt, ArtStation, etc.)
    - And many more supported by gallery-dl
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "url_list": ("STRING", {
                    "multiline": True, 
                    "default": "# Enter URLs here, one per line\n# Example:\n# https://imgur.com/gallery/example\n# https://example.com/image.jpg"
                }),
                "output_dir": ("STRING", {
                    "default": "./gallery-dl-output",
                    "tooltip": "Directory where downloaded files will be saved"
                })
            },
            "optional": {
                "url_file": ("STRING", {
                    "default": "",
                    "tooltip": "Path to a text file containing URLs (one per line)"
                }),
                "config_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to gallery-dl configuration file"
                }),
                "use_browser_cookies": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use browser cookies for authentication"
                }),
                "browser_name": (["firefox", "chrome", "chromium", "edge", "safari", "opera"], {
                    "default": "firefox",
                    "tooltip": "Browser to extract cookies from"
                }),
                "use_download_archive": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use archive file to skip already downloaded content"
                }),
                "archive_file": ("STRING", {
                    "default": "./gallery-dl-archive.sqlite3",
                    "tooltip": "Path to the download archive database"
                }),
                "skip_videos": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip video files, download only images"
                }),
                "extract_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save download metadata to JSON file"
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("output_dir", "summary", "download_count", "success")
    FUNCTION = "execute"
    CATEGORY = "Downloaders"
    
    DESCRIPTION = "Download images and media from various websites using gallery-dl"

    def execute(
        self,
        url_list: str,
        output_dir: str,
        url_file: str = None,
        config_path: str = None,
        use_browser_cookies: bool = False,
        browser_name: str = "firefox",
        use_download_archive: bool = True,
        archive_file: str = "./gallery-dl-archive.sqlite3",
        skip_videos: bool = False,
        extract_metadata: bool = True
    ) -> Tuple[str, str, int, bool]:
        """
        Execute the gallery-dl download process.
        
        Returns:
            tuple: (output_dir, summary, download_count, success)
        """
        try:
            # Clean up input parameters
            if url_file and url_file.strip() == "":
                url_file = None
            if config_path and config_path.strip() == "":
                config_path = None
            if archive_file and archive_file.strip() == "":
                archive_file = "./gallery-dl-archive.sqlite3"
            
            # Convert relative paths to absolute paths
            output_dir = os.path.abspath(output_dir)
            if archive_file:
                archive_file = os.path.abspath(archive_file)
            if config_path:
                config_path = os.path.abspath(config_path)
            if url_file:
                url_file = os.path.abspath(url_file)

            # Create downloader instance
            downloader = GalleryDLDownloader(
                url_list=url_list.splitlines() if url_list.strip() else [],
                url_file=url_file,
                output_dir=output_dir,
                config_path=config_path,
                use_browser_cookies=use_browser_cookies,
                browser_name=browser_name,
                use_download_archive=use_download_archive,
                archive_file=archive_file,
                skip_videos=skip_videos,
                extract_metadata=extract_metadata
            )

            # Execute download
            result = downloader.run()
            
            # Create summary
            if result['success']:
                summary = (
                    f"✅ Download completed successfully\n"
                    f"📁 Output directory: {result['output_dir']}\n"
                    f"📊 Files downloaded: {result['download_count']}\n"
                    f"💾 Command used: {result['command'][:100]}{'...' if len(result['command']) > 100 else ''}"
                )
                if result['stderr']:
                    summary += f"\n⚠️ Warnings: {result['stderr'][:200]}{'...' if len(result['stderr']) > 200 else ''}"
            else:
                summary = (
                    f"❌ Download failed (exit code: {result['returncode']})\n"
                    f"📁 Output directory: {result['output_dir']}\n"
                    f"📊 Files downloaded: {result['download_count']}\n"
                    f"🔍 Error: {result['stderr'][:300]}{'...' if len(result['stderr']) > 300 else ''}"
                )
            
            return (
                result['output_dir'], 
                summary, 
                result['download_count'], 
                result['success']
            )
            
        except Exception as e:
            error_summary = f"❌ Error: {str(e)}"
            return (
                output_dir if 'output_dir' in locals() else "./gallery-dl-output", 
                error_summary, 
                0, 
                False
            )


# Additional utility functions for the node
def check_gallery_dl_installation():
    """Helper function to check if gallery-dl is properly installed."""
    try:
        result = subprocess.run(
            ["gallery-dl", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        return result.returncode == 0, result.stdout if result.returncode == 0 else result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "gallery-dl not found in PATH"


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "GalleryDLDownloader": GalleryDLNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GalleryDLDownloader": "Gallery-dl Downloader",
}

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "GalleryDLNode", "GalleryDLDownloader"]
