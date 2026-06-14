"""
Serve Qwen/Qwen3.5-35B-A3B with vLLM on Modal as an OpenAI-compatible endpoint.

Deploy:
    modal deploy vllm_qwen_serve.py

After deploy, Modal prints a URL like:
    https://<workspace>--vllm-qwen-serve-serve.modal.run

Point any OpenAI-compatible client at:
    base_url = "<that URL>/v1"
    api_key  = <the VLLM_API_KEY you stored in the Modal secret>
    model    = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"

Reference: https://modal.com/docs/examples/vllm_inference
"""

import json

import modal

# ─── Model ───────────────────────────────────────────────────────────────────

# Default switched 2026-06-15 to the agentic coder (FP8) for a single-GPU run.
# Qwen3-Coder-30B-A3B-Instruct: purpose-built for agentic tool-calling, NON-thinking
# (no endless-<think>), 256K context, A3B (~3B active) = fast. Emits the qwen3_coder
# tool-call format natively (matches TOOL_CALL_PARSER). FP8 weights ~30GB -> fits ONE
# A100-80GB with plenty of KV-cache room for long (150-turn) sessions. ~50% SWE-bench.
MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
#   bf16 fallback (if FP8 Marlin kernels misbehave on Ampere/A100): use
#     MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"  WITH GPU_CONFIG="A100-80GB:2", TP=2
#   Stronger alternative to evaluate (often beats Qwen3-30B on coding/tool-use):
#     GLM-4.7-Flash (30B MoE). Needs a different parser: set TOOL_CALL_PARSER="glm47"
#     and MODEL_NAME to the exact GLM-4.7-Flash repo id (verify on HF, e.g. zai-org/...).
# Pin a revision once validated one. Leave None to track main.
MODEL_REVISION: str | None = None

# Tool-call parser MUST match the format the model emits, or vLLM returns the
# tool call as raw text in `content` and OpenAI-style clients (OpenClaw) never
# see structured `tool_calls` -> the agent narrates "let me execute" forever.
# Verified 2026-06-13: Qwen3.5-35B-A3B emits <tool_call><function=..><parameter=..>
# (the qwen3_coder format). The previous value "hermes" expects JSON-in-<tool_call>
# and silently fails to parse it (cf. vLLM issue #31871). "qwen3_xml" is the newer
# variant if your vLLM build supports it.
TOOL_CALL_PARSER = "qwen3_coder"

# ─── GPU / concurrency ───────────────────────────────────────────────────────

# 1x A100-80GB is enough for a 30B-A3B model in FP8 (weights ~30GB; the rest of
# the 80GB is KV cache). The previous 35B model needed for me to have 
# 2 GPUs because it's 70GB in bf16 AND a multimodal VLM; the 30B FP8 coder does not. 
# 1 GPU ~= half the cost.
GPU_CONFIG = "A100-80GB:1"
TENSOR_PARALLEL_SIZE = 1
# vLLM serves many requests CONCURRENTLY (continuous batching) up to this cap, so
# the agent's bursty per-turn traffic (tool-loop steps + subagents) is batched, not
# queued. Self-hosting removes the shared-tier ~40-RPM throttle entirely issue that 
# i had :) 

# the 64 concurrent request cap is a soft limit to prevent OOMs from unbounded concurrency; 
# it's not a rate limit. The actual max QPS depends on the prompt length and model speed, 
# but with a 30B A3B in FP8, you can expect around 10-20 concurrent requests before hitting GPU saturation. 
# Adjust as needed based on your workload and latency requirements.
# now limited only by GPU throughput. & now can safely hit the endpoint many-at-once :)
MAX_CONCURRENT_REQUESTS = 64
# Bound context so the KV cache fits comfortably on one GPU. 131072 is ample for a
# coding session; raise toward 262144 only if you truly need it.
MAX_MODEL_LEN = 131072

# Keep N containers warm to kill cold-start 408s during an active session.
# 0 = scale-to-zero (cheap, but first call after idle times out while the GPU
# boots — observed as HTTP 408 "Missing request"). Set to 1 ONLY while doing a
# long live run (e.g. huge coding session), then back to 0 — 2xA100-80GB
# running idle is expensive.
MIN_CONTAINERS = 0

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
    min_containers=MIN_CONTAINERS,
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
        "--max-model-len", str(MAX_MODEL_LEN),
        "--async-scheduling",
        "--enforce-eager" if FAST_BOOT else "--no-enforce-eager",
        # text-only: the model is a VLM but we skip multimedia to save VRAM
        "--limit-mm-per-prompt",
        f"'{json.dumps({'image': 0, 'video': 0, 'audio': 0})}'",
        # tool calls (OpenAI tools API). Parser MUST match the emitted format;
        # "hermes" (the old value) could NOT parse Qwen3.5's output and returned
        # it as raw text -> OpenClaw saw no tool_calls -> infinite "let me
        # execute" loop. See TOOL_CALL_PARSER note at top.
        "--enable-auto-tool-choice",
        "--tool-call-parser", TOOL_CALL_PARSER,
        # NOTE: no --reasoning-parser on purpose. With Qwen3.5 thinking on
        # (the model default), the client wants raw <think>...</think> tags
        # in `content` so its own stripper can run. Adding a reasoning parser
        # would route the trace to `reasoning_content` and the stripper would
        # silently no-op, making reasoning look "shallow" to the caller.
        # (Moot for Qwen3-Coder-*-Instruct, which is non-thinking.)
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
