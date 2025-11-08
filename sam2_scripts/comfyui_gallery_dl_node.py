"""
Comfyui Gallery Dl Node

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
- Uses gallery-dl (GNU GPL v2) by Mike Fährmann: https://github.com/mikf/gallery-dl
- See CREDITS.md for complete list of all dependencies
"""

import os
import subprocess
import tempfile
import json

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
            command += ["--filter", "type != 'video'"]

        command += urls
        return command

    def run(self) -> Dict[str, Any]:
        if self.url_file:
            with open(self.url_file, "r", encoding="utf-8") as f:
                urls = [line.strip() for line in f if line.strip()]
        else:
            urls = [url.strip() for url in self.url_list if url.strip()]

        if not urls:
            raise ValueError("No URLs provided for download.")

        os.makedirs(self.output_dir, exist_ok=True)

        command = self._build_command(urls)

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()

        result = {
            "command": " ".join(command),
            "stdout": stdout,
            "stderr": stderr,
            "returncode": process.returncode,
            "output_dir": self.output_dir
        }

        if self.extract_metadata:
            metadata_path = Path(self.output_dir) / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

        return result


class GalleryDLNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Tuple[str, Dict[str, Any]]]:
        return {
            "required": {
                "url_list": ("STRING", {"multiline": True}),
                "output_dir": ("STRING", {"default": "./gallery-dl-output"})
            },
            "optional": {
                "url_file": ("STRING", {}),
                "config_path": ("STRING", {}),
                "use_browser_cookies": ("BOOLEAN", {"default": False}),
                "browser_name": ("STRING", {"default": "firefox"}),
                "use_download_archive": ("BOOLEAN", {"default": True}),
                "archive_file": ("STRING", {"default": "./gallery-dl-archive.sqlite3"}),
                "skip_videos": ("BOOLEAN", {"default": False}),
                "extract_metadata": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_dir", "summary")
    FUNCTION = "execute"
    CATEGORY = "Downloaders"

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
    ) -> Tuple[str, str]:

        downloader = GalleryDLDownloader(
            url_list=url_list.splitlines(),
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

        result = downloader.run()
        summary = f"Downloaded to: {result['output_dir']}\nReturn code: {result['returncode']}\nErrors: {result['stderr'][:300]}"
        return result['output_dir'], summary
