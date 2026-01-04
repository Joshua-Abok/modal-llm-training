# Modal LLM Web Endpoint (GPU Inference)

This project demonstrates how to deploy a **GPU-accelerated Large Language Model (LLM) as an HTTP web endpoint** using **Modal Labs** and **Hugging Face Transformers**.

The endpoint runs on an **NVIDIA H100 GPU**, accepts a prompt via HTTP `POST`, and returns a generated response from the `Qwen/Qwen3-1.7B-FP8` model.

---

## 🚀 What This Project Does

* Builds a custom Modal image with:

  * `transformers`
  * `torch`
  * `fastapi`
  * `uvicorn`
* Provisions an **H100 GPU on demand**
* Exposes an **HTTP POST web endpoint**
* Accepts a text prompt as input
* Returns a structured JSON response
* Automatically scales with traffic (managed by Modal)

This is designed as a **production-ready inference pattern**, not a local demo.

---

## 🧠 How It Works

### Core Components

```python
app = modal.App("llm-web-endpoint")
image = modal.Image.debian_slim().uv_pip_install(
    "transformers[torch]", "fastapi", "uvicorn"
)
```

* Creates a Modal app
* Builds a minimal Debian-based container image
* Installs required inference and web dependencies

---

```python
@app.function(gpu="h100", image=image)
@modal.web_endpoint(method="POST")
def chat_endpoint(prompt: str | None = None):
```

* Declares a Modal function
* Requests an **H100 GPU**
* Exposes the function as an **HTTP POST endpoint**

---

```python
chatbot = pipeline(
    model="Qwen/Qwen3-1.7B-FP8",
    device_map="cuda",
    max_new_tokens=1024,
)
```

* Loads a Hugging Face chat model
* Automatically places it on the GPU
* Runs inference using the Transformers pipeline API

---

## 🔑 Modal Authentication (Required)

Before deploying, authenticate with Modal.

### Install Modal CLI

```bash
pip install modal
```

### Set Token ID and Secret

```bash
modal token set --token-id <token-id> --token-secret as-<token-secret>
```

Expected output:

```
Token verified successfully!
Token written to ~/.modal.toml
```

📚 Reference:
[https://modal.com/docs/reference/modal.config](https://modal.com/docs/reference/modal.config)

---

## ▶️ Deploying the Web Endpoint

From the project directory:

```bash
modal deploy inference.py
```

Modal will:

1. Build the container image
2. Cache CUDA + PyTorch dependencies
3. Provision GPU infrastructure
4. Deploy a managed HTTPS endpoint

You’ll receive a URL like:

```
https://<username>--llm-web-endpoint-chat-endpoint.modal.run
```

---

## 🔌 Calling the Endpoint

### Example using `curl`

```bash
curl -X POST \
  https://<endpoint-url> \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain what Modal Labs is in simple terms."
  }'
```

### Example Response

```json
{
  "prompt": "Explain what Modal Labs is in simple terms.",
  "response": "Modal Labs is a cloud platform that allows developers to run Python code on powerful infrastructure like GPUs without managing servers..."
}
```

---

## ⚠️ Important Notes

* **Cold starts** may take 10–30 seconds (GPU + model loading)
* GPUs are **only billed while handling requests**
* First request is slow; subsequent requests are fast
* This version loads the model **per request**

---

