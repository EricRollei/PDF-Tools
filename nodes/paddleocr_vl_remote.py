"""
Paddleocr Vl Remote

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

import base64
import json
import io
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
import torch
from PIL import Image, ImageDraw


class PaddleOCRVLRemoteNode:
    """Send ComfyUI images to a PaddleOCR-VL REST endpoint."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "endpoint": ("STRING", {"default": "http://127.0.0.1:8080/layout-parsing"}),
            },
            "optional": {
                "timeout": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 600.0, "step": 1.0}),
                "use_layout_detection": ("BOOLEAN", {"default": True}),
                "use_doc_unwarping": ("BOOLEAN", {"default": False}),
                "use_doc_orientation_classify": ("BOOLEAN", {"default": False}),
                "use_chart_recognition": ("BOOLEAN", {"default": False}),
                "format_block_content": ("BOOLEAN", {"default": False}),
                "layout_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "prettify_markdown": ("BOOLEAN", {"default": True}),
                "show_formula_number": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "LIST", "IMAGE")
    RETURN_NAMES = ("plain_text", "markdown_text", "json_output", "layout_blocks", "annotated_image")
    FUNCTION = "invoke"
    CATEGORY = "text/ocr"

    def __init__(self):
        self.session = requests.Session()

    def invoke(
        self,
        image: torch.Tensor,
        endpoint: str,
        timeout: float = 60.0,
        use_layout_detection: bool = True,
        use_doc_unwarping: bool = False,
        use_doc_orientation_classify: bool = False,
        use_chart_recognition: bool = False,
        format_block_content: bool = False,
        layout_threshold: float = 0.5,
        prettify_markdown: bool = True,
        show_formula_number: bool = False,
    ) -> Tuple[str, str, str, List[Dict[str, Any]], torch.Tensor]:
        try:
            pil_batch = self._tensor_to_pil_batch(image)
            combined_text: List[str] = []
            markdown_pages: List[str] = []
            layout_blocks: List[Dict[str, Any]] = []
            json_results: List[Dict[str, Any]] = []
            annotated_tensors: List[torch.Tensor] = []

            for page_index, pil_image in enumerate(pil_batch):
                payload = self._build_payload(
                    pil_image,
                    use_layout_detection=use_layout_detection,
                    use_doc_unwarping=use_doc_unwarping,
                    use_doc_orientation_classify=use_doc_orientation_classify,
                    use_chart_recognition=use_chart_recognition,
                    format_block_content=format_block_content,
                    layout_threshold=layout_threshold,
                    prettify_markdown=prettify_markdown,
                    show_formula_number=show_formula_number,
                )
                response = self.session.post(endpoint, json=payload, timeout=timeout)
                response.raise_for_status()
                data = response.json()

                if "result" not in data:
                    raise ValueError("Unexpected response shape: missing 'result'")

                json_results.append(data.get("result", {}))
                page_text, page_markdown, page_blocks = self._parse_result(data["result"], page_index)
                if page_text:
                    combined_text.append(page_text)
                if page_markdown:
                    markdown_pages.append(page_markdown)
                layout_blocks.extend(page_blocks)

                annotated = self._draw_overlays(pil_image, page_blocks, page_index)
                annotated_array = np.asarray(annotated, dtype=np.float32) / 255.0
                annotated_tensor = torch.from_numpy(annotated_array)
                annotated_tensors.append(annotated_tensor.unsqueeze(0))

            plain_text = "\n\n".join(part for part in combined_text if part)
            markdown_text = "\n\n".join(part for part in markdown_pages if part)
            json_output = json.dumps(json_results, ensure_ascii=False, indent=2)
            if annotated_tensors:
                annotated_batch = torch.cat(annotated_tensors, dim=0)
            else:
                annotated_batch = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return plain_text, markdown_text, json_output, layout_blocks, annotated_batch

        except Exception as exc:
            error_msg = f"PaddleOCR-VL request failed: {exc}"
            fallback_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return error_msg, "", json.dumps({"error": error_msg}, ensure_ascii=False), [], fallback_image

    def _tensor_to_pil_batch(self, tensor: torch.Tensor) -> List[Image.Image]:
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        pil_images: List[Image.Image] = []
        for idx in range(tensor.shape[0]):
            array = tensor[idx].cpu().numpy()
            array = (array * 255.0).clip(0, 255).astype(np.uint8)
            if array.shape[2] == 4:
                image = Image.fromarray(array[:, :, :3], "RGB")
            else:
                image = Image.fromarray(array, "RGB")
            pil_images.append(image)
        return pil_images

    def _build_payload(
        self,
        image: Image.Image,
        *,
        use_layout_detection: bool,
        use_doc_unwarping: bool,
        use_doc_orientation_classify: bool,
        use_chart_recognition: bool,
        format_block_content: bool,
        layout_threshold: float,
        prettify_markdown: bool,
        show_formula_number: bool,
    ) -> Dict[str, Any]:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        payload: Dict[str, Any] = {
            "file": encoded,
            "fileType": 1,
            "useLayoutDetection": use_layout_detection,
            "useDocUnwarping": use_doc_unwarping,
            "useDocOrientationClassify": use_doc_orientation_classify,
            "useChartRecognition": use_chart_recognition,
            "formatBlockContent": format_block_content,
            "layoutThreshold": layout_threshold,
            "prettifyMarkdown": prettify_markdown,
            "showFormulaNumber": show_formula_number,
        }
        return payload

    def _parse_result(
        self,
        result: Dict[str, Any],
        page_index: int,
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        text_parts: List[str] = []
        blocks: List[Dict[str, Any]] = []
        markdown_pages: List[str] = []

        for entry in result.get("layoutParsingResults", []):
            pruned = entry.get("prunedResult", {})
            for block in pruned.get("parsing_res_list", []):
                content = block.get("block_content", "").strip()
                label = block.get("block_label", "")
                score = block.get("block_confidence", block.get("block_score"))
                bbox = block.get("block_bbox")
                order = block.get("block_order")
                if content:
                    text_parts.append(content)
                blocks.append(
                    {
                        "page_index": page_index,
                        "label": label,
                        "content": content,
                        "bbox": bbox,
                        "score": score,
                        "order": order,
                    }
                )

            markdown = entry.get("markdown", {})
            markdown_text = markdown.get("text", "").strip()
            if markdown_text:
                markdown_pages.append(markdown_text)

        plain_text = "\n".join(text_parts)
        markdown_text = "\n\n".join(markdown_pages)
        return plain_text, markdown_text, blocks

    def _draw_overlays(
        self,
        pil_image: Image.Image,
        page_blocks: List[Dict[str, Any]],
        page_index: int,
    ) -> Image.Image:
        annotated = pil_image.convert("RGB").copy()
        draw = ImageDraw.Draw(annotated)
        for block_number, block in enumerate(page_blocks, start=1):
            bbox = block.get("bbox") or block.get("block_bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            label = block.get("label") or block.get("block_label") or "block"
            content = block.get("content") or block.get("block_content") or ""
            snippet = content.strip().replace("\n", " ")
            if len(snippet) > 40:
                snippet = snippet[:37] + "..."
            caption = f"P{page_index+1}-{block_number}: {label}"
            if snippet:
                caption = f"{caption} | {snippet}"

            draw.rectangle([x1, y1, x2, y2], outline="#34d399", width=3)
            text_x, text_y = x1 + 4, y1 + 4
            try:
                text_width = draw.textlength(caption)
            except Exception:
                text_width = 8 * len(caption)
            try:
                draw.rectangle(
                    [text_x - 4, text_y - 2, text_x - 4 + text_width + 8, text_y + 16],
                    fill="#111827",
                    outline="#34d399",
                )
            except Exception:
                pass
            draw.text((text_x, text_y), caption, fill="#f9fafb")
        return annotated


NODE_CLASS_MAPPINGS = {
    "PaddleOCRVLRemoteNode": PaddleOCRVLRemoteNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PaddleOCRVLRemoteNode": "PaddleOCR-VL Remote",
}
