# Credits and Third-Party Licenses

This project relies on numerous excellent open-source libraries and tools. We are grateful to all the developers and maintainers who make these projects possible.

## Core Dependencies

### Image Processing Libraries

- **[Pillow (PIL Fork)](https://pillow.readthedocs.io/)**
  - License: HPND (Historical Permission Notice and Disclaimer)
  - Copyright: Alex Clark and Contributors
  - Used for: Core image manipulation and processing

- **[NumPy](https://numpy.org/)**
  - License: BSD 3-Clause
  - Copyright: NumPy Developers
  - Used for: Array operations and numerical computing

- **[OpenCV (opencv-python)](https://opencv.org/)**
  - License: Apache 2.0
  - Copyright: OpenCV team
  - Used for: Computer vision operations and image analysis

### PDF Processing Libraries

- **[PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)**
  - License: GNU AGPL v3 / Commercial
  - Copyright: Artifex Software, Inc.
  - Used for: PDF parsing, rendering, and content extraction

- **[PyPDF2](https://pypdf2.readthedocs.io/)**
  - License: BSD 3-Clause
  - Copyright: Mathieu Fenniak and Contributors
  - Used for: Alternative PDF manipulation

### AI/ML Frameworks

- **[PyTorch](https://pytorch.org/)**
  - License: BSD 3-Clause
  - Copyright: Facebook, Inc. and its affiliates
  - Used for: Deep learning operations and tensor computations

- **[Transformers](https://huggingface.co/docs/transformers/)**
  - License: Apache 2.0
  - Copyright: Hugging Face, Inc.
  - Used for: AI model loading and inference

- **[timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)**
  - License: Apache 2.0
  - Copyright: Ross Wightman
  - Used for: Vision model architectures

- **[SafeTensors](https://github.com/huggingface/safetensors)**
  - License: Apache 2.0
  - Copyright: Hugging Face, Inc.
  - Used for: Secure model weight storage and loading

- **[Accelerate](https://github.com/huggingface/accelerate)**
  - License: Apache 2.0
  - Copyright: Hugging Face, Inc.
  - Used for: Distributed training and inference

- **[SentencePiece](https://github.com/google/sentencepiece)**
  - License: Apache 2.0
  - Copyright: Google Inc.
  - Used for: Text tokenization in transformer models

### Media Downloader Tools

- **[gallery-dl](https://github.com/mikf/gallery-dl)**
  - License: GNU GPL v2
  - Copyright: Mike Fährmann
  - Used for: Downloading images and media from 100+ websites
  - Website: https://github.com/mikf/gallery-dl

- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)**
  - License: Unlicense (Public Domain)
  - Copyright: yt-dlp contributors
  - Used for: Downloading videos and audio from 1000+ platforms
  - Website: https://github.com/yt-dlp/yt-dlp

- **[browser-cookie3](https://github.com/borisbabic/browser_cookie3)**
  - License: GNU GPL v3
  - Copyright: Boris Babic
  - Used for: Extracting cookies from web browsers for authentication

### Utility Libraries

- **[Requests](https://requests.readthedocs.io/)**
  - License: Apache 2.0
  - Copyright: Kenneth Reitz and Contributors
  - Used for: HTTP requests and API interactions

- **[urllib3](https://urllib3.readthedocs.io/)**
  - License: MIT
  - Copyright: Andrey Petrov and Contributors
  - Used for: HTTP client functionality

- **[tqdm](https://github.com/tqdm/tqdm)**
  - License: MIT / MPL 2.0
  - Copyright: Casper da Costa-Luis and Contributors
  - Used for: Progress bars and status indicators

- **[colorama](https://github.com/tartley/colorama)**
  - License: BSD 3-Clause
  - Copyright: Jonathan Hartley
  - Used for: Cross-platform colored terminal output

- **[jsonschema](https://python-jsonschema.readthedocs.io/)**
  - License: MIT
  - Copyright: Julian Berman
  - Used for: JSON schema validation

## AI Models

This project may download and use the following pretrained models:

### Florence2 Vision Models

- **[Florence-2](https://huggingface.co/microsoft/Florence-2-large)**
  - License: MIT
  - Copyright: Microsoft Corporation
  - Used for: Vision-language tasks, object detection, and image captioning
  - Citation: If you use Florence-2 in research, please cite the original Microsoft paper

### SAM2 (Segment Anything Model 2)

- **[SAM2](https://github.com/facebookresearch/segment-anything-2)**
  - License: Apache 2.0
  - Copyright: Meta Platforms, Inc.
  - Used for: Image segmentation and object masking
  - Citation: If you use SAM2 in research, please cite the Meta AI paper

## Optional Dependencies

### OCR Libraries (Not included by default)

- **[Tesseract OCR](https://github.com/tesseract-ocr/tesseract)**
  - License: Apache 2.0
  - Copyright: Google Inc.
  - Used for: Optical character recognition

- **[pytesseract](https://github.com/madmaze/pytesseract)**
  - License: Apache 2.0
  - Copyright: Matthias A. Lee
  - Used for: Python wrapper for Tesseract

- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)**
  - License: Apache 2.0
  - Copyright: PaddlePaddle Authors
  - Used for: Advanced OCR with multiple language support

- **[EasyOCR](https://github.com/JaidedAI/EasyOCR)**
  - License: Apache 2.0
  - Copyright: Jaided AI
  - Used for: Easy-to-use OCR with 80+ language support

- **[Surya OCR](https://github.com/VikParuchuri/surya)**
  - License: GNU GPL v3
  - Copyright: Vik Paruchuri
  - Used for: Modern multilingual OCR and layout analysis

### External Tools

- **[FFmpeg](https://ffmpeg.org/)**
  - License: GNU GPL v2+ / LGPL v2.1+
  - Copyright: FFmpeg team
  - Used for: Video/audio conversion and processing (required by yt-dlp for audio extraction)

- **[Poppler](https://poppler.freedesktop.org/)**
  - License: GNU GPL v2+
  - Copyright: Poppler developers
  - Used for: PDF rendering (optional, for pdf2image)

## ComfyUI Integration

This package is designed as a custom node collection for:

- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)**
  - License: GNU GPL v3
  - Copyright: ComfyAnonymous and Contributors
  - The powerful and modular stable diffusion GUI

## License Compatibility

This project is released under a dual license (Non-Commercial: CC BY-NC 4.0 / Commercial: Contact for license). Please note:

- **GPL-licensed dependencies** (gallery-dl, browser-cookie3, Surya OCR, PyMuPDF AGPL): If you distribute this software, you must comply with GPL/AGPL requirements, which may require releasing your source code.
- **Commercial use**: If you wish to use this commercially, you must obtain both our commercial license AND ensure compliance with all third-party dependency licenses.
- **PyMuPDF**: Note that PyMuPDF (fitz) uses AGPL v3 for open-source use. Commercial licenses are available from Artifex Software.

## Attribution

If you use this project in your work, please provide attribution:

```
PDF Tools for ComfyUI
Author: Eric Hiss (GitHub: EricRollei)
License: Dual License (CC BY-NC 4.0 / Commercial)
Repository: [Your GitHub Repository URL]
```

## Acknowledgments

Special thanks to:

- The **ComfyUI** community for creating an amazing extensible platform
- **Hugging Face** for hosting models and providing excellent ML tools
- **Microsoft Research** for the Florence-2 vision models
- **Meta AI** for the Segment Anything Model (SAM2)
- All the open-source developers who maintain the libraries this project depends on

## Reporting Issues

If you believe any license information is incorrect or incomplete, please:

1. Open an issue on our GitHub repository
2. Contact: eric@historic.camera or eric@rollei.us

We take license compliance seriously and will address any concerns promptly.

---

**Last Updated:** January 2025
