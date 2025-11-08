"""
Paddleocr Vl Client

Description: PaddleOCR integration for advanced OCR with multiple language support
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
- Uses PaddleOCR (Apache 2.0) by PaddlePaddle: https://github.com/PaddlePaddle/PaddleOCR
- See CREDITS.md for complete list of all dependencies
"""

"""Lightweight client for the PaddleOCR-VL REST service."""

import argparse
import base64
import json
import pathlib
from typing import Any, Dict, List

import requests


def _encode_image(image_path: pathlib.Path) -> str:
    with image_path.open("rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def _load_payload(image_path: pathlib.Path, args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "file": _encode_image(image_path),
        "fileType": 1,
        "useLayoutDetection": args.use_layout_detection,
        "useDocUnwarping": args.use_doc_unwarping,
        "useDocOrientationClassify": args.use_doc_orientation_classify,
        "useChartRecognition": args.use_chart_recognition,
        "formatBlockContent": args.format_block_content,
        "layoutThreshold": args.layout_threshold,
        "prettifyMarkdown": args.prettify_markdown,
        "showFormulaNumber": args.show_formula_number,
    }
    return payload


def _save_outputs(output_dir: pathlib.Path, stem: str, result: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_parts: List[str] = []
    text_parts: List[str] = []

    for idx, entry in enumerate(result.get("layoutParsingResults", [])):
        pruned = entry.get("prunedResult", {})
        for block in pruned.get("parsing_res_list", []):
            content = block.get("block_content", "").strip()
            if content:
                text_parts.append(content)

        markdown = entry.get("markdown", {})
        md_text = markdown.get("text", "").strip()
        if md_text:
            page_name = f"{stem}_page_{idx:03d}.md"
            (output_dir / page_name).write_text(md_text, encoding="utf-8")
            markdown_parts.append(md_text)

    if text_parts:
        (output_dir / f"{stem}.txt").write_text("\n".join(text_parts), encoding="utf-8")
    (output_dir / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Call a PaddleOCR-VL REST endpoint")
    parser.add_argument("images", nargs="+", type=pathlib.Path, help="Image paths to process")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/layout-parsing", help="REST endpoint URL")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout in seconds")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("paddleocr_results"), help="Output directory")
    parser.add_argument("--use-layout-detection", dest="use_layout_detection", action="store_true")
    parser.add_argument("--no-layout-detection", dest="use_layout_detection", action="store_false")
    parser.set_defaults(use_layout_detection=True)
    parser.add_argument("--use-doc-unwarping", action="store_true")
    parser.add_argument("--use-doc-orientation-classify", action="store_true")
    parser.add_argument("--use-chart-recognition", action="store_true")
    parser.add_argument("--format-block-content", action="store_true")
    parser.add_argument("--layout-threshold", type=float, default=0.5)
    parser.add_argument("--prettify-markdown", dest="prettify_markdown", action="store_true")
    parser.add_argument("--no-prettify-markdown", dest="prettify_markdown", action="store_false")
    parser.set_defaults(prettify_markdown=True)
    parser.add_argument("--show-formula-number", action="store_true")
    args = parser.parse_args()

    session = requests.Session()

    for path in args.images:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        payload = _load_payload(path, args)
        response = session.post(args.endpoint, json=payload, timeout=args.timeout)
        response.raise_for_status()
        body = response.json()
        result = body.get("result", body)
        stem = path.stem
        _save_outputs(args.output, stem, result)
        print(f"Processed {path} -> {args.output}")


if __name__ == "__main__":
    main()
