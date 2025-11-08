"""
Paddleocr Vl Rest Server

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

"""Standalone FastAPI server that exposes PaddleOCR-VL over HTTP.

Run inside the dedicated PaddleOCR-VL environment:

    python tools/paddleocr_vl_rest_server.py --host 0.0.0.0 --port 8080 --device cpu

"""

import argparse
import base64
import io
import logging
import tempfile
import traceback
import uuid
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
def _json_safe(value: Any) -> Any:
    if isinstance(value, Image.Image):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value

try:  # Provide a narrow workaround until Paddle ships a Windows wheel with Paddle safetensors support
    import safetensors

    _original_safe_open = safetensors.safe_open

    try:
        import torch
    except Exception:  # noqa: BLE001 - torch might not be installed
        torch = None

    try:
        import paddle
    except Exception as paddle_exc:  # noqa: BLE001 - propagate more helpful message later
        paddle = None

    def _torch_tensor_to_paddle(tensor: "torch.Tensor") -> "paddle.Tensor":  # type: ignore[name-defined]
        if torch is None or paddle is None:
            raise RuntimeError(
                "PyTorch + Paddle are both required to load PaddleOCR-VL weights on Windows CPU. "
                "Ensure paddlepaddle==3.2.0 and torch (CPU wheel) are installed in this environment."
            )
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        array = tensor.detach().cpu().numpy()
        return paddle.to_tensor(array)

    class _TorchSliceWrapper:
        def __init__(self, tensor: "torch.Tensor") -> None:  # type: ignore[name-defined]
            self._tensor = tensor

        def __getitem__(self, item: Any):  # type: ignore[override]
            result = self._tensor.__getitem__(item)
            return _torch_tensor_to_paddle(result)

        def numpy(self) -> "paddle.Tensor":  # type: ignore[override]
            return _torch_tensor_to_paddle(self._tensor)

        def as_numpy(self) -> "paddle.Tensor":
            return self.numpy()

        @property
        def dtype(self) -> Any:
            return self.numpy().dtype

        def __getattr__(self, name: str) -> Any:
            return getattr(self._tensor, name)

    class _TorchTensorDictWrapper:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def keys(self) -> Any:
            return self._inner.keys()

        def get_tensor(self, key: str):  # type: ignore[override]
            tensor = self._inner.get_tensor(key)
            return _torch_tensor_to_paddle(tensor)

        def get_slice(self, key: str) -> _TorchSliceWrapper:
            tensor = self._inner.get_slice(key)
            return _TorchSliceWrapper(tensor)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    class _SafeOpenWrapper:
        def __init__(self, handle: Any) -> None:
            self._handle = handle

        def __enter__(self) -> _TorchTensorDictWrapper:
            entered = self._handle.__enter__()
            return _TorchTensorDictWrapper(entered)

        def __exit__(self, exc_type, exc, tb) -> Any:  # noqa: ANN001 - signature matches context manager protocol
            return self._handle.__exit__(exc_type, exc, tb)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._handle, name)

    def _safe_open_with_paddle_support(*args: Any, framework: Optional[str] = None, **kwargs: Any):
        if framework == "paddle":
            if torch is None:
                raise RuntimeError(
                    "PyTorch is required to load PaddleOCR-VL weights on Windows CPU. Install with "
                    "`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`."
                )
            framework = "pt"
        handle = _original_safe_open(*args, framework=framework, **kwargs)
        if framework == "pt":
            return _SafeOpenWrapper(handle)
        return handle

    safetensors.safe_open = _safe_open_with_paddle_support
except Exception:  # noqa: BLE001 - best effort patch, fall back to default behaviour if import fails
    safetensors = None

from paddleocr import PaddleOCRVL
from PIL import Image
import uvicorn


class LayoutParsingRequest(BaseModel):
    file: str
    fileType: Optional[int] = 1  # 1 = image, 0 = PDF
    useLayoutDetection: Optional[bool] = None
    useDocUnwarping: Optional[bool] = None
    useDocOrientationClassify: Optional[bool] = None
    useChartRecognition: Optional[bool] = None
    formatBlockContent: Optional[bool] = None
    layoutThreshold: Optional[float] = None
    layoutNms: Optional[bool] = None
    layoutUnclipRatio: Optional[float] = None
    layoutMergeBboxesMode: Optional[str] = None
    promptLabel: Optional[str] = None
    prettifyMarkdown: Optional[bool] = None
    showFormulaNumber: Optional[bool] = None
    repetitionPenalty: Optional[float] = None
    temperature: Optional[float] = None
    topP: Optional[float] = None
    minPixels: Optional[int] = None
    maxPixels: Optional[int] = None


def _load_input_bytes(payload: LayoutParsingRequest) -> bytes:
    try:
        return base64.b64decode(payload.file, validate=True)
    except Exception as exc:  # noqa: BLE001 - we want any base64 error
        raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {exc}") from exc


def _predict_single(
    pipeline: PaddleOCRVL,
    data: LayoutParsingRequest,
) -> Dict[str, Any]:
    decoded = _load_input_bytes(data)

    if data.fileType == 0:
        suffix = ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(decoded)
            tmp.flush()
            outputs = pipeline.predict(
                input=tmp.name,
                use_layout_detection=data.useLayoutDetection,
                use_doc_unwarping=data.useDocUnwarping,
                use_doc_orientation_classify=data.useDocOrientationClassify,
                use_chart_recognition=data.useChartRecognition,
                layout_threshold=data.layoutThreshold,
                layout_nms=data.layoutNms,
                layout_unclip_ratio=data.layoutUnclipRatio,
                layout_merge_bboxes_mode=data.layoutMergeBboxesMode,
                format_block_content=data.formatBlockContent,
                repetition_penalty=data.repetitionPenalty,
                temperature=data.temperature,
                top_p=data.topP,
                min_pixels=data.minPixels,
                max_pixels=data.maxPixels,
            )
    else:
        image = Image.open(io.BytesIO(decoded)).convert("RGB")
        np_image = np.array(image)
        outputs = pipeline.predict(
            input=np_image,
            use_layout_detection=data.useLayoutDetection,
            use_doc_unwarping=data.useDocUnwarping,
            use_doc_orientation_classify=data.useDocOrientationClassify,
            use_chart_recognition=data.useChartRecognition,
            layout_threshold=data.layoutThreshold,
            layout_nms=data.layoutNms,
            layout_unclip_ratio=data.layoutUnclipRatio,
            layout_merge_bboxes_mode=data.layoutMergeBboxesMode,
            format_block_content=data.formatBlockContent,
            repetition_penalty=data.repetitionPenalty,
            temperature=data.temperature,
            top_p=data.topP,
            min_pixels=data.minPixels,
            max_pixels=data.maxPixels,
        )

    layout_results: List[Dict[str, Any]] = []

    for result in outputs:
        result_json = getattr(result, "json", None)
        if callable(result_json):  # guard older API returning method
            result_json = result_json()
        if isinstance(result_json, dict) and "res" in result_json:
            pruned = result_json["res"]
        else:
            pruned = result_json or {}
        pruned = _json_safe(pruned)

        markdown_info = getattr(result, "markdown", {}) or {}
        markdown_text = markdown_info.get("markdown_texts")
        if isinstance(markdown_text, list):
            markdown_text = "\n".join(markdown_text)
        elif markdown_text is None:
            markdown_text = ""
        markdown_images = _json_safe(markdown_info.get("markdown_images", {}))
        continuation = markdown_info.get("page_continuation_flags", (False, False))

        layout_results.append(
            {
                "prunedResult": pruned,
                "markdown": {
                    "text": markdown_text,
                    "images": markdown_images,
                    "isStart": bool(continuation[0]) if isinstance(continuation, (list, tuple)) and continuation else False,
                    "isEnd": bool(continuation[1]) if isinstance(continuation, (list, tuple)) and continuation else False,
                },
                "outputImages": None,  # Visualization not supported in this lightweight server
                "inputImage": None,
            }
        )

    return {
        "layoutParsingResults": layout_results,
        "dataInfo": {
            "fileType": data.fileType,
            "count": len(layout_results),
        },
    }


def build_app(pipeline: PaddleOCRVL) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    predict_fn = partial(_predict_single, pipeline)

    @app.post("/layout-parsing")
    async def layout_parsing_endpoint(payload: LayoutParsingRequest) -> Dict[str, Any]:
        try:
            result = predict_fn(payload)
            return {
                "logId": str(uuid.uuid4()),
                "errorCode": 0,
                "errorMsg": "Success",
                "result": result,
            }
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - return clean error to client
            logging.error("PaddleOCR-VL inference failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/health")
    async def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve PaddleOCR-VL over HTTP (FastAPI)")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--device", default="cpu", help="Device string passed to PaddleOCRVL (e.g., cpu, gpu:0)")
    parser.add_argument("--vl-rec-backend", default=None, help="Optional VLM backend (e.g., vllm-server)")
    parser.add_argument("--vl-rec-server-url", default=None, help="Optional VLM server URL")
    parser.add_argument("--concurrency", type=int, default=1, help="Max concurrent requests (FastAPI workers)")
    parser.add_argument("--disable-layout", action="store_true", help="Disable layout detection by default")
    parser.add_argument("--enable-orientation", action="store_true", help="Enable orientation classification by default")
    parser.add_argument("--enable-unwarping", action="store_true", help="Enable document unwarping by default")
    parser.add_argument("--enable-chart", action="store_true", help="Enable chart recognition by default")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = PaddleOCRVL(
        device=args.device,
        vl_rec_backend=args.vl_rec_backend,
        vl_rec_server_url=args.vl_rec_server_url,
        use_layout_detection=not args.disable_layout,
        use_doc_orientation_classify=args.enable_orientation,
        use_doc_unwarping=args.enable_unwarping,
        use_chart_recognition=args.enable_chart,
    )

    app = build_app(pipeline)
    uvicorn.run(app, host=args.host, port=args.port, workers=max(1, args.concurrency))


if __name__ == "__main__":
    main()
