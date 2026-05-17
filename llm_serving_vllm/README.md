# Modal vLLM Serving — Qwen3.5-35B-A3B (OpenAI-Compatible)

This project deploys **`Qwen/Qwen3.5-35B-A3B`** on Modal using **vLLM**, exposed as an **OpenAI-compatible HTTP server**. Any client that speaks the OpenAI API (the `openai` SDK, LiteLLM, OpenRouter-style wrappers, etc.) can point at it without code changes — just swap the `base_url` and `api_key`.

Designed as a **drop-in replacement for OpenRouter** for projects that already use an `OpenAI`-style client.

---

## 🧠 What You Get

* `POST /v1/chat/completions` — chat completions
* `POST /v1/completions` — text completions
* `GET  /v1/models` — list served models
* `GET  /health` — liveness probe
* `GET  /docs` — Swagger UI

All protected by a bearer-token API key you control.

---

## 🏗 Architecture

| Component | Choice | Why |
|---|---|---|
| Serving engine | **vLLM 0.19.0** | Production-grade, batches requests, speaks OpenAI protocol natively |
| Model | **Qwen/Qwen3.5-35B-A3B** | MoE — 35B total params, 3B active per token |
| GPU | **A100-80GB × 2** with `--tensor-parallel-size 2` | ~70 GB of bf16 weights split across two cards |
| Weight cache | `huggingface-cache` Modal Volume | Avoids re-downloading ~70 GB on every cold start |
| JIT cache | `vllm-cache` Modal Volume | Skips Torch compile / CUDA graph capture after first boot |
| Concurrency | `@modal.concurrent(max_inputs=64)` | vLLM batches; one replica handles many concurrent requests |
| Auth | `--api-key $VLLM_API_KEY` (Modal Secret) | Bearer-token auth on every endpoint |
| Tool calls | `--enable-auto-tool-choice --tool-call-parser hermes` | OpenAI `tools` + `tool_choice="auto"` supported; Qwen3 family uses Hermes-style |
| Reasoning passthrough | *No* `--reasoning-parser` | Raw `<think>…</think>` flows through `content` so the client's existing stripper can run |

For the full rationale behind each of these choices, see [`SYSTEM_DESIGN.md`](./SYSTEM_DESIGN.md).

---

## 🔑 One-Time Setup

### 1. Install + auth Modal

```bash
pip install modal
modal token set --token-id <token-id> --token-secret as-<token-secret>
```

### 2. Create the two Modal Secrets

```bash
# HuggingFace token (needed if you ever serve a gated repo; harmless otherwise)
modal secret create huggingface-token HF_TOKEN=hf_xxx_your_token

# Pick ANY string — this becomes the API key clients must send
modal secret create vllm-api-key VLLM_API_KEY=sk-anything-you-like
```

Reference: <https://modal.com/docs/guide/secrets>

---

## 🚀 Deploy

```bash
modal deploy vllm_qwen_serve.py
```

Modal will:

1. Build the CUDA + vLLM image (slow first time, cached after)
2. Provision 2 × A100-80GB
3. Download ~70 GB of weights into the `huggingface-cache` Volume (slow first cold start; fast after)
4. Boot vLLM, capture CUDA graphs, and start the OpenAI server on port 8000
5. Print a public HTTPS URL:

```
https://joshua-abok--vllm-qwen-serve-serve.modal.run
```

> ⏱ **Cold start expectations.** First deploy: 10–20 min (build + weight download). Subsequent cold starts: ~60–120 s (volumes are warm).

---

## 🧪 Smoke Test

The smoke test runs **inside Modal** so the `vllm-api-key` secret is already mounted — you don't need to export anything locally.

```bash
modal run vllm_qwen_serve.py
```

You should see `/health -> 200` and a JSON chat completion. To test with a custom prompt:

```bash
modal run vllm_qwen_serve.py --content "Say hi in one word."
```

Or hit it with `curl` from your laptop (replace `$KEY` with your actual `VLLM_API_KEY`):

```bash
curl https://joshua-abok--vllm-qwen-serve-serve.modal.run/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-35B-A3B",
    "messages": [{"role": "user", "content": "Say hi in one word."}],
    "max_tokens": 16
  }'
```

---

## 🔌 Wiring Into Your Existing Project (No Code Changes)

Your existing scenario runner does:

```python
llm = make_client(config.OPENROUTER_API_KEY, config.OPENROUTER_BASE_URL)
```

To switch from OpenRouter to this Modal deployment, change **only the three values** in that project's `config.py`:

```python
# Before (OpenRouter)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY  = "<openrouter key>"
DEFAULT_MODEL       = "qwen/qwen3.5-35b-a3b"

# After (this Modal deployment)
OPENROUTER_BASE_URL = "https://joshua-abok--vllm-qwen-serve-serve.modal.run/v1"
OPENROUTER_API_KEY  = "<the VLLM_API_KEY string from the Modal secret>"
DEFAULT_MODEL       = "Qwen/Qwen3.5-35B-A3B"
```

That's it. `make_client(...)`, `run_agent(...)`, and the rest of the scenario loop run unchanged — they talk to vLLM's `/v1/chat/completions` exactly like they talked to OpenRouter's.

> ℹ The `model=` field in your client calls must match `--served-model-name`, which we set to `Qwen/Qwen3.5-35B-A3B` in the script. Change one and the other in lockstep.

---

## 🧠 Tool Calling & Thinking — Contract With the Client

The server is configured so your existing agent code keeps working without changes:

* **Tool calls.** Pass standard OpenAI `tools=[…]` plus `tool_choice="auto"`. vLLM's Hermes parser surfaces structured calls under `message.tool_calls[]` just like OpenAI.
* **Thinking traces stay in `content`.** Qwen3.5 emits `<think>…</think>` blocks by default. The server *does not* split them into a separate `reasoning_content` field — they arrive raw in `message.content`, where your client-side stripper can clean them out and feed the rest to the agent loop.
* **To disable thinking** (shorter responses, no trace), pass per-request:
  ```python
  extra_body={"chat_template_kwargs": {"enable_thinking": False}}
  ```
  Or prefix the user prompt with `/no_think`. This is a chat-template flag, not a server flag — leaving it off means the model thinks (recommended for deep reasoning).

---

## ⚙️ Tuning Knobs (top of `vllm_qwen_serve.py`)

| Variable | Default | Effect |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen3.5-35B-A3B` | Any HF repo vLLM supports |
| `MODEL_REVISION` | `None` | Pin a commit to avoid surprises when upstream updates |
| `GPU_CONFIG` | `"A100-80GB:2"` | Swap for `"H100:2"` or `"H200:1"` |
| `TENSOR_PARALLEL_SIZE` | `2` | Must equal the GPU count |
| `MAX_CONCURRENT_REQUESTS` | `64` | How many requests one replica batches |
| `FAST_BOOT` | `False` | `True` = quicker cold start, slower steady-state |

To scale horizontally (multiple replicas under load), Modal auto-scales the function; nothing to configure in the script.

---

## 🛠 Troubleshooting

* **`"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser` (HTTP 400)**
  Both flags are in the script — but a *warm container* started before the flags were added will keep serving the old config. Run `modal container list`, then `modal container stop <id>` to force the next request to boot fresh.
* **OOM during boot** — Drop to a smaller context (`--max-model-len 32768` in the `cmd` list) or move to `H200:1` / `H100:4`.
* **401 Unauthorized** — Client `api_key` doesn't match the Modal secret value. Re-check both sides.
* **Long first request after idle** — Modal scaled to zero. Either bump `scaledown_window`, set `min_containers=1` on `@app.function`, or accept the cold start (~2–4 min once volumes are warm).
* **Agent looks like it stopped reasoning ("shallow")** — Your client is probably passing `enable_thinking: False` or `/no_think`. Remove it; raw `<think>` tags will flow through `content` for your stripper to handle.
* **`finish_reason="length"`** — Thinking traces eat tokens. Raise `max_tokens` on the client side, or disable thinking for that specific call.

---

## 📚 References

* Modal vLLM example: <https://modal.com/docs/examples/vllm_inference>
* Modal high-performance LLM guide: <https://modal.com/docs/guide/high-performance-llm-inference>
* vLLM OpenAI server docs: <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>
* Model card: <https://huggingface.co/Qwen/Qwen3.5-35B-A3B>
