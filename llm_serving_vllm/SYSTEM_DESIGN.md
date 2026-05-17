# SYSTEM_DESIGN — `llm_serving_vllm`

This document captures the **why** behind the deployment in `vllm_qwen_serve.py`. The README explains *how* to deploy and use it; this file explains the design decisions, the alternatives we considered, and the operational lessons learned from validating the deployment end-to-end against a live agent.

---

## 1. Problem & Goal

We have a scenario-runner project that calls a hosted LLM via an OpenAI-compatible interface:

```python
llm = make_client(config.OPENROUTER_API_KEY, config.OPENROUTER_BASE_URL)
```

Until now, that hit OpenRouter's hosted `Qwen3.5-35B-A3B`. We want to:

1. **Host the same model ourselves** so we control latency, throughput, cost, and (eventually) fine-tuned LoRA adapters.
2. **Expose it as an OpenAI-compatible endpoint** so the scenario runner is a config-only change — no client refactor.
3. **Use Modal** for serverless GPU infrastructure, consistent with sibling subfolders in this repo (`llm_inferencing/`, `llm-web-endpoint/`, `run_jupyter_server/`).

The non-goal: building anything fancy on top of vLLM. The endpoint is a **transport substitute**, not a feature layer.

---

## 2. Architecture (One Picture)

```
┌──────────────────────────────────────────┐         ┌────────────────────────────────────────┐
│  scenario_runner (separate project)      │         │  Modal — app: vllm-qwen-serve          │
│                                          │         │  ┌──────────────────────────────────┐  │
│  make_client(                            │  HTTPS  │  │ @modal.web_server(port=8000)     │  │
│    api_key   = OPENROUTER_API_KEY,       │ ──────► │  │   subprocess.Popen("vllm serve …")│  │
│    base_url  = OPENROUTER_BASE_URL,      │  bearer │  │     │                            │  │
│  )                                       │   auth  │  │     ▼                            │  │
│       │                                  │         │  │  vLLM 0.19.0 (OpenAI server)     │  │
│       ▼                                  │         │  │     │                            │  │
│  POST /v1/chat/completions               │         │  │     ▼                            │  │
│   { model:"Qwen/Qwen3.5-35B-A3B",        │         │  │  Qwen3.5-35B-A3B (TP=2)          │  │
│     messages:[…], tools:[…],             │         │  │     on 2 × A100-80GB             │  │
│     tool_choice:"auto" }                 │         │  └──────────────────────────────────┘  │
│                                          │         │      ▲                  ▲              │
└──────────────────────────────────────────┘         │      │                  │              │
                                                     │  huggingface-cache  vllm-cache         │
                                                     │     (Volume)         (Volume)          │
                                                     └────────────────────────────────────────┘
```

Two Modal Volumes survive across container lifetimes so cold-start cost amortizes:

* `huggingface-cache` — weights (~70 GB of bf16). Downloaded once; subsequent cold starts read from disk.
* `vllm-cache` — Torch compile artifacts and CUDA graphs. Saves ~30–60 s of recompile per boot.

Two Modal Secrets are mounted as env vars:

* `huggingface-token` → `HF_TOKEN` (currently unused since the repo is public — kept for future gated/private models).
* `vllm-api-key` → `VLLM_API_KEY` (any string; vLLM enforces it as a bearer token on every endpoint).

---

## 3. Key Design Decisions

### 3.1 Why vLLM (and not Transformers `pipeline`, TGI, SGLang, or llama.cpp)

| Engine | Considered | Reason |
|---|---|---|
| **vLLM** | ✅ chosen | Continuous batching, paged attention, native OpenAI server, native Hermes/Qwen tool parsers, mature MoE support |
| Transformers `pipeline` | ❌ | What the sibling `llm_inferencing/` and `llm-web-endpoint/` use — fine for single-request demos, but no batching = throughput collapses under concurrent load |
| TGI (HuggingFace) | ❌ | Comparable to vLLM but more opinionated about routing; Modal example community uses vLLM |
| SGLang | ❌ | Excellent throughput but the OpenAI-compat layer is younger; tool-calling parsers less battle-tested for our model |
| llama.cpp | ❌ | CPU/Apple-silicon focus; we have datacenter GPUs and need MoE |

The deciding factor was the **OpenAI server is built-in** — `vllm serve` gives us `/v1/chat/completions` for free, which is exactly what the scenario runner already speaks.

### 3.2 Why this model: `Qwen/Qwen3.5-35B-A3B`

* The scenario runner was already pointed at it via OpenRouter — minimizing surprises.
* MoE shape (35B total / 3B active per token) means compute-per-token is small, even though we pay weight-storage cost for the full 35B.
* Thinking mode is on by default, which the agent loop relies on for deep reasoning.

The model is a **vision-language model** (architecture `Qwen3_5MoeForConditionalGeneration`, model_type `qwen3_5_moe`), but we don't need the vision side. See §3.6 for how we disable it.

### 3.3 Why `A100-80GB:2` with `--tensor-parallel-size 2` (and not H200:1 or H100:2)

| Shape | Fits? | Per-hour cost | Why we didn't pick it |
|---|---|---|---|
| `H200:1` (141 GB) | ✅ comfortable | highest | Cleanest single-GPU, but most expensive |
| `H100:2` (2 × 80 GB) | ✅ tight | medium | Newer silicon, slightly better throughput, but cost premium not justified for current workload |
| **`A100-80GB:2`** | ✅ workable | **lowest of the three** | Matches GPUs the team already uses (`run_jupyter_server/jupyter_sandbox.py`) — least-surprise choice |
| `A100-40GB:N` | ❌ | n/a | 35B in bf16 won't fit in 80 GB total |
| `L40S:N` | ❌ | n/a | No NVLink / poor TP scaling for MoE |

bf16 weights for 35B params are roughly 70 GB. With KV cache, activations, and CUDA graph buffers, single-80GB-A100 doesn't fit, so TP=2 is the floor.

### 3.4 Why split `vllm` and `transformers` into two `uv_pip_install` steps

```python
.uv_pip_install("vllm==0.19.0")
.uv_pip_install("transformers==5.5.0")
```

`vllm==0.19.0` pins `transformers>=4.56,<5`. But Qwen3.5's `model_type=qwen3_5_moe` config only exists in `transformers>=5.x`. Doing both in one resolver pass fails:

```
× No solution found: vllm==0.19.0 depends on transformers<5, but you require transformers==5.5.0
```

Splitting them sidesteps the resolver: vLLM installs first with whatever transformers it wants, then the second step upgrades transformers to 5.5.0 without re-checking vLLM's pin. Discovered the hard way on our first deploy attempt.

### 3.5 Why `--api-key $VLLM_API_KEY` instead of no auth or a Modal proxy

vLLM has native bearer-token auth via `--api-key`. Modal's web URLs are public, so without it anyone with the URL could run the model on our dime. The token is whatever string we put in the `vllm-api-key` Modal Secret — vLLM checks `Authorization: Bearer <token>` on every endpoint (including `/v1/models`, but **not** `/health`).

The scenario runner already sends an `Authorization` header (because it was talking to OpenRouter), so this is zero client-side work — just point `OPENROUTER_API_KEY` at the vLLM key.

### 3.6 Why `--limit-mm-per-prompt '{"image":0,"video":0,"audio":0}'`

Qwen3.5-35B-A3B is multimodal — its config includes `image_token_id` and the model class is `Qwen3_5MoeForConditionalGeneration`. Loading the full vision tower would:

* Allocate ~several GB of additional VRAM for the vision encoder
* Slow down boot
* Be a complete waste for our text-only scenario runner

This flag tells vLLM "you're text-only — don't allocate vision/audio paths." Confirmed in the boot logs:

```
All limits of multimodal modalities supported by the model are set to 0, running in text-only mode.
```

### 3.7 Why `--enable-auto-tool-choice --tool-call-parser hermes`

The scenario runner uses OpenAI's `tools=[…]` + `tool_choice="auto"` to let the model decide when to call agent tools. vLLM only honors `tool_choice="auto"` if explicitly enabled — without the flag you get HTTP 400:

```
"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser
```

Qwen3 family models emit tool calls in **Hermes format** (`<tool_call>{...}</tool_call>`). vLLM 0.19.0 ships `vllm/tool_parsers/hermes_tool_parser.py` for exactly this. We considered `qwen3_xml` and `qwen3_coder` — those are for the *Coder* variant and for the experimental XML format respectively. `hermes` is the right pick for the chat variant we're serving.

### 3.8 Why we *removed* `--reasoning-parser qwen3` after first adding it

Initial intuition: route `<think>…</think>` blocks to a separate `reasoning_content` field so `message.content` is "clean." We did this on the second redeploy.

Result: the scenario runner reported **"shallow reasoning"** — the agent was making first-impression decisions instead of tracing deeply.

Root cause: the runner has its own `<think>` stripper that expects raw thinking in `content`. With the reasoning parser on:
* The trace was being silently routed to `reasoning_content` (which the runner doesn't read).
* From the runner's perspective, `content` looked like the model wasn't thinking at all.
* The runner's stripper became a no-op.

Fix: removed the reasoning parser. Raw `<think>…</think>` now flows through `content` where the runner's stripper handles it. **The contract is now: client owns thinking-trace parsing.**

This is the kind of decision worth documenting because it looks wrong at first glance ("why aren't you using the dedicated parser?"). The answer is: the parser's design assumes the client wants the trace *separated*; our client wants it *inline-then-stripped*.

### 3.9 Why `@modal.concurrent(max_inputs=64)` and not a smaller number

vLLM's headline feature is **continuous batching** — multiple requests can share one forward pass. The limit is KV-cache memory, not request count. 64 is generous for 2 × 80 GB; if we hit OOM under load, we lower it. Modal also auto-scales replicas, so the function-level concurrency just controls per-replica batch depth.

### 3.10 Why `FAST_BOOT = False`

vLLM's CUDA graph capture and Torch compile add ~30 s to cold start but materially improve steady-state throughput (no per-step Python overhead). For our usage pattern (long-lived containers serving scenarios for hours), the cold-start cost is amortized hundreds of times over. `True` is for development iteration where you keep redeploying and want quick feedback.

---

## 4. Cold-Start Mechanics (and Why They Matter)

Three distinct cold-start regimes — important to understand because they shape ops:

| Regime | Trigger | Wall-clock | What happens |
|---|---|---|---|
| **First-ever deploy** | `modal deploy` on empty volumes | ~15–20 min | Image build + 70 GB weight download + CUDA graph capture |
| **Warm-volumes cold start** | First request after scale-to-zero | ~2–4 min | Weights from volume + CUDA graphs from cache + engine init (~5 min for the engine alone in practice — see `core.py` log `init engine took 308.71 seconds`) |
| **Warm-container request** | Any request while replica is alive | tens of ms to a few seconds | Just inference |

Modal's `scaledown_window=15 * MINUTES` keeps a warm replica for 15 minutes after the last request. Under steady scenario-runner load, replicas should stay warm continuously. The cold-start regimes matter most after long idle periods or fresh deploys.

### 4.1 The redeploy-doesn't-evict-warm-containers gotcha

**Trip wire:** `modal deploy` swaps the function definition for *new* container starts. **Existing warm containers keep serving the old config until they idle out.**

We hit this twice in this session:
1. Added tool-calling flags → redeployed → client still got HTTP 400 → discovered the previous container was still alive with old args.
2. Removed reasoning parser → redeployed → would have hit the same trap.

**Workaround:** after every redeploy that changes the vLLM `cmd`, run

```bash
modal container list
modal container stop <each container id>
```

Now the next request boots a fresh container with the new flags.

This is documented in the README's Troubleshooting section as well.

---

## 5. Validation Trail

The deployment was validated end-to-end before being handed to the scenario runner:

1. **Static checks:** confirmed `Qwen3_5MoeForConditionalGeneration` exists in `vllm/model_executor/models/qwen3_5.py` (v0.19.0) as a real class, not a registry stub; confirmed `transformers==5.5.0` ships the `qwen3_5_moe` module.
2. **First deploy attempt:** failed on the dependency-resolver conflict (§3.4); fixed by splitting installs.
3. **Second deploy + smoke test:** model loaded, engine init in 308 s, CUDA graphs captured, `GET /health` → 200, `POST /v1/chat/completions` → 200 with valid OpenAI ChatCompletion JSON.
4. **Tool-call regression:** caller reported HTTP 400; root-caused to missing `--enable-auto-tool-choice` and stale container.
5. **Reasoning regression:** caller reported shallow reasoning; root-caused to `--reasoning-parser qwen3` routing traces away from the client's expected location.
6. **Current state:** all four flag categories settled (auth, multimodal-off, tool-calling on, reasoning-passthrough). Documented in `vllm_qwen_serve.py`.

---

## 6. Known Trade-offs & Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| vLLM bumps a minor version → breaks something | medium over time | Pinned to `vllm==0.19.0`; revisit only when upgrading deliberately |
| Qwen pushes a new revision that changes tokenizer / chat template | low–medium | Set `MODEL_REVISION` to a known-good commit hash once we identify one |
| Thinking traces blow past `max_tokens` → truncated responses | observed | Caller raises `max_tokens` or sends `enable_thinking: False` for that request |
| A 2 × A100-80GB replica left warm overnight runs up cost | high if forgotten | `scaledown_window=15*MINUTES`; no `min_containers=1` set |
| Hermes parser fails on a Qwen tool-call edge case | low | Caller already validates tool-call payloads; vLLM parser is well-tested |
| Modal Volume corruption / accidental delete of `huggingface-cache` | low | Re-download takes 10–15 min one time; not a permanent data loss |

---

## 7. What This Deployment Intentionally Doesn't Do

* **No LoRA adapter support.** vLLM can serve base + LoRA via `--enable-lora --lora-modules`. We left this out because no adapter exists yet. When one does (presumably from training pipelines in sibling folders), add the flags here and re-deploy.
* **No prompt caching server-side.** vLLM has prefix caching available behind flags; we haven't measured whether our scenario prompts benefit enough to justify enabling it.
* **No autoscaling tuning.** Defaults are fine for current load. If we see queue depth grow under burst traffic, raise `max_inputs`, drop `scaledown_window`, or set `min_containers > 0`.
* **No streaming-specific tuning.** vLLM streams by default if the client requests it; we haven't measured first-token-latency vs. throughput trade-offs.
* **No observability beyond Modal's built-in dashboard.** vLLM exposes Prometheus metrics at `/metrics`; not scraped today.

These are all "add when needed" rather than "missing" — flagged so future-you knows what to grab off the shelf.

---

## 8. References

* Modal vLLM example (the starting point): <https://modal.com/docs/examples/vllm_inference>
* Modal high-performance LLM guide: <https://modal.com/docs/guide/high-performance-llm-inference>
* vLLM OpenAI server: <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>
* vLLM tool parsers source: <https://github.com/vllm-project/vllm/tree/v0.19.0/vllm/tool_parsers>
* Hermes tool-calling spec (NousResearch): <https://github.com/NousResearch/Hermes-Function-Calling>
* Qwen3.5 model card: <https://huggingface.co/Qwen/Qwen3.5-35B-A3B>
