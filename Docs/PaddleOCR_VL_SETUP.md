# PaddleOCR-VL Standalone Environment Setup

This guide walks through creating an isolated runtime for PaddleOCR-VL so you can keep the ComfyUI environment on CUDA 12.8 while hosting OCR services separately.

## 1. Create the Environment

1. Open PowerShell in a folder where you keep tooling environments (e.g. `A:\venvs`).
2. Create and activate a fresh virtual environment with Python 3.10+:

   ```pwsh
   python -m venv paddleocr-vl-env
   .\paddleocr-vl-env\Scripts\Activate.ps1
   ```

3. Upgrade pip and wheel:

   ```pwsh
   python -m pip install --upgrade pip wheel
   ```

## 2. Install PaddleOCR-VL Dependencies

> PaddleOCR-VL currently ships CUDA 12.6 wheels. Keep this environment isolated so it does not downgrade CUDA for ComfyUI.

```pwsh
# Core runtime (adjust the CUDA mirror if you target a different driver)
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install "paddleocr[doc-parser]"
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl

# Optional accelerators (install only on compatible GPUs)
# python -m pip install flash-attn==2.8.3  # Required for NVIDIA 50-series cards
```

> **Windows note:** Paddle’s GPU wheels target Linux. Use WSL2 + CUDA toolkit, or run inside an NVIDIA CUDA-enabled Docker container if you need GPU acceleration. On bare Windows, fall back to CPU (`paddlepaddle==3.2.0`) or WSL2.
>
> If you stay on native Windows with the CPU build, replace the safetensors command above with:
>
> ```pwsh
> python -m pip install safetensors
> ```

## 3. Start the Inference Service

You have two supported hosting modes; pick one that fits your workflow.

### Option A: PaddleOCR GenAI Server (vLLM/SGLang backend)

1. (Optional) install backend deps if you want vLLM/SGLang acceleration:

   ```pwsh
   paddleocr install_genai_server_deps vllm
   # or: paddleocr install_genai_server_deps sglang
   ```

2. Launch the service (replace `vllm` with `sglang` if needed):

   ```pwsh
   paddleocr genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --host 0.0.0.0 --port 8118
   ```

3. Leave this terminal running; the ComfyUI node will call `http://127.0.0.1:8118/v1` for inference.

### Option B: Lightweight FastAPI Server (recommended for Windows CPU)

The official PaddleX server expects a GPU-capable Paddle build. On native Windows with the CPU wheel, use the bundled FastAPI script instead.

1. Install the small web stack in your PaddleOCR environment:

   ```pwsh
   python -m pip install fastapi uvicorn[standard]
   python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

2. Launch the server script:

   ```pwsh
   python ..\tools\paddleocr_vl_rest_server.py --host 0.0.0.0 --port 8080 --device cpu
   ```

   Add `--disable-layout`, `--enable-orientation`, etc., if you want to change the default modules the server loads.

   > The script now patches `safetensors.safe_open` so Paddle’s Windows CPU build can read the bundled `.safetensors` weights by routing through PyTorch. Keep the CPU wheel of PyTorch installed (see step 1). If you still see errors mentioning `framework paddle` or `bfloat16`, reinstall `safetensors` inside this venv (`python -m pip install --force-reinstall safetensors`) and retry.

3. Keep this terminal running. The ComfyUI node will call `http://127.0.0.1:8080/layout-parsing` with base64-encoded images.

## 4. Verify the Service

Run a quick smoke test from the same environment:

```pwsh
python - <<'PY'
import base64, json, requests, pathlib

url = "http://127.0.0.1:8080/layout-parsing"  # or http://127.0.0.1:8118/v1 for genai server
image_path = pathlib.Path("./demo.jpg")

with image_path.open("rb") as f:
    payload = {"file": base64.b64encode(f.read()).decode("ascii"), "fileType": 1}

response = requests.post(url, json=payload)
response.raise_for_status()
print(json.dumps(response.json(), indent=2)[:800])
PY
```

Expect a JSON payload containing `layoutParsingResults` with bounding boxes and Markdown text.

### CLI Helper (Optional)

Once the service responds, you can batch images from Windows or WSL using the helper script included in this repository:

```pwsh
python ..\tools\paddleocr_vl_client.py \
   --endpoint http://127.0.0.1:8080/layout-parsing \
   --output ..\test-output\paddleocr \
   ..\test-output\sample_page.png
```

The client saves `{stem}.json`, `{stem}.txt`, and one Markdown file per page under the chosen output directory.

## 5. Wire It Into ComfyUI

1. Keep the PaddleOCR-VL service running in its own terminal session.
2. Drop the **PaddleOCR-VL Remote** node into your workflow (GUI or `comfycli`). It ships with four outputs:
   - Plain text joined across all blocks.
   - Full Markdown output (one blob per page).
   - Raw JSON (mirrors the REST payload for debugging).
   - A Python list of layout blocks containing `page_index`, `label`, `bbox`, and `content` for downstream crop logic.
3. Set the `endpoint` input to `http://127.0.0.1:8080/layout-parsing` (PaddleX REST) or `http://127.0.0.1:8118/v1` (GenAI server). Tweak the boolean inputs to enable orientation, unwarping, chart parsing, or Markdown formatting at call time.
4. Feed the node’s layout list into your existing crop/translation nodes, or serialize the Markdown/text outputs straight to disk via the standard file-writer nodes.
5. For headless batches, reference the node in a saved workflow and trigger it with `python -m comfycli --workflow path/to/workflow.json --output out_dir`—the endpoint and toggles can be overridden with `--set NodeID.endpoint="http://..."` arguments if you need to switch servers on the fly.

## 6. Maintenance Tips

- If performance is slow, set `use_queues=True` on the server and increase `vl_rec_max_concurrency` (GenAI backend) or run multiple worker processes.
- Backup the environment: copy the entire `paddleocr-vl-env` folder or export requirements with `python -m pip freeze > requirements.txt`.
- Update PaddleOCR cautiously; new releases may target different CUDA toolchains. Test in a clone of the environment first.
