"""
Serve Qwen/Qwen3.5-35B-A3B with vLLM on Modal as an OpenAI-compatible endpoint.

Deploy:
    modal deploy vllm_qwen_serve.py

After deploy, Modal prints a URL like:
    https://<workspace>--vllm-qwen-serve-serve.modal.run

Point any OpenAI-compatible client at:
    base_url = "<that URL>/v1"
    api_key  = <the VLLM_API_KEY you stored in the Modal secret>
    model    = "Qwen/Qwen3.5-35B-A3B"

Reference: https://modal.com/docs/examples/vllm_inference
"""

import json

import modal

# ─── Model ───────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
# Pin a revision once you've validated one. Leave None to track main.
MODEL_REVISION: str | None = None

# ─── GPU / concurrency ───────────────────────────────────────────────────────

GPU_CONFIG = "A100-80GB:2"
TENSOR_PARALLEL_SIZE = 2
MAX_CONCURRENT_REQUESTS = 64

# ─── Boot trade-off ──────────────────────────────────────────────────────────
# False = slower cold start, faster steady-state generation (recommended for prod).
# True  = faster cold start, slower per-token (useful while iterating).
FAST_BOOT = False

# ─── Image ───────────────────────────────────────────────────────────────────

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.19.0")
    # Upgrade transformers in a second step: vllm 0.19.0 pins transformers<5,
    # but qwen3_5_moe ships in transformers 5.x. Splitting steps sidesteps the
    # resolver — the second install upgrades without re-checking vllm's pin.
    .uv_pip_install("transformers==5.5.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# ─── Caches (persist weights + JIT artifacts across cold starts) ─────────────

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# ─── Secrets ─────────────────────────────────────────────────────────────────
#   modal secret create huggingface-token HF_TOKEN=hf_...
#   modal secret create vllm-api-key      VLLM_API_KEY=sk-...   (any string)

hf_secret = modal.Secret.from_name("huggingface-token")
api_key_secret = modal.Secret.from_name("vllm-api-key")

# ─── App ─────────────────────────────────────────────────────────────────────

app = modal.App("vllm-qwen-serve")

MINUTES = 60
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=GPU_CONFIG,
    scaledown_window=15 * MINUTES,
    timeout=20 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[hf_secret, api_key_secret],
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_REQUESTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def serve():
    import os
    import subprocess

    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--uvicorn-log-level=info",
        "--api-key", os.environ["VLLM_API_KEY"],
        "--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE),
        "--async-scheduling",
        "--enforce-eager" if FAST_BOOT else "--no-enforce-eager",
        # text-only: the model is a VLM but we skip multimedia to save VRAM
        "--limit-mm-per-prompt",
        f"'{json.dumps({'image': 0, 'video': 0, 'audio': 0})}'",
        # tool calls (OpenAI tools API) — Qwen3 family uses Hermes-style
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
        # NOTE: no --reasoning-parser on purpose. With Qwen3.5 thinking on
        # (the model default), the client wants raw <think>...</think> tags
        # in `content` so its own stripper can run. Adding a reasoning parser
        # would route the trace to `reasoning_content` and the stripper would
        # silently no-op, making reasoning look "shallow" to the caller.
    ]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]

    print(" ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)


# ─── Smoke tests ─────────────────────────────────────────────────────────────


@app.function(
    image=modal.Image.debian_slim().uv_pip_install("aiohttp"),
    secrets=[api_key_secret],
    timeout=25 * MINUTES,
)
async def smoke_test(content: str | None = None):
    """Run inside Modal so the api-key secret is already mounted.

    Invoke with:  modal run vllm_qwen_serve.py::smoke_test
    """
    import os

    import aiohttp

    url = serve.get_web_url()
    api_key = os.environ["VLLM_API_KEY"]
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": content or "Reply with one word: ping?"}
        ],
        "max_tokens": 32,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    timeout = aiohttp.ClientTimeout(total=20 * MINUTES)
    async with aiohttp.ClientSession(base_url=url, timeout=timeout) as session:
        async with session.get("/health") as r:
            print(f"/health -> {r.status}")
        async with session.post(
            "/v1/chat/completions", json=payload, headers=headers
        ) as r:
            print(f"/v1/chat/completions -> {r.status}")
            print(await r.text())


@app.local_entrypoint()
def main(content: str | None = None):
    smoke_test.remote(content)
