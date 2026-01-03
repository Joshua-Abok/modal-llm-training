# Modal LLM Inference Example (GPU)

This script demonstrates how to run **GPU-accelerated LLM inference on Modal** using the `transformers` library and a Hugging Face model (`Qwen/Qwen3-1.7B-FP8`). The example deploys a Modal function that runs on an **H100 GPU**, constructs a prompt, and generates a response using a chat-style pipeline.

---

## 📌 What This Project Does

* Builds a **custom Modal image** with `transformers[torch]`
* Runs an **LLM inference job on an H100 GPU**
* Reads the contents of the Python file itself
* Sends that code as a prompt to the model
* Asks the model to explain what the code does
* Returns the model’s generated response

This is a **pure inference example** (not training) designed to show how Modal handles:

* Image building
* GPU provisioning
* Remote execution
* Streaming logs and outputs

---

## 🧠 How the Code Works

### Key Components

```python
app = modal.App("example-inference")
image = modal.Image.debian_slim().uv_pip_install("transformers[torch]")
```

* Creates a Modal app named `example-inference`
* Builds a lightweight Debian image and installs `transformers` + PyTorch

```python
@app.function(gpu="h100", image=image)
def chat(prompt: str | None = None) -> list[dict]:
```

* Declares a Modal function
* Requests an **H100 GPU**
* Uses the custom image

```python
if prompt is None:
    prompt = f"/no_think Read this code.\n\n{Path(__file__).read_text()}\nIn one paragraph, what does the code do?"
```

* If no prompt is provided, the function:

  * Reads its own source file
  * Asks the model to summarize what the code does

```python
chatbot = pipeline(
    model="Qwen/Qwen3-1.7B-FP8",
    device_map="cuda",
    max_new_tokens=1024
)
```

* Loads a Hugging Face chat model
* Automatically places it on the GPU
* Runs inference and prints the result

---

## 🔑 Modal Authentication (Required)

Before running this project, you must authenticate with Modal.

### 1. Install Modal CLI

```bash
pip install modal
```

### 2. Set Your Token ID & Secret

```bash
modal token set --token-id <token-id> --token-secret as-<token-secret>
```

You should see:

```
Token verified successfully!
Token written to ~/.modal.toml
```

📚 Token setup reference:
[https://modal.com/docs/reference/modal.config](https://modal.com/docs/reference/modal.config)

---

## ▶️ Running the Example

From the project directory:

```bash
modal run inference.py
```

Modal will:

1. Build the container image
2. Download CUDA, PyTorch, and Transformers
3. Provision an H100 GPU
4. Execute the `chat` function remotely
5. Stream logs and model output back to your terminal

You’ll also get a **run URL** like:

```
https://modal.com/apps/<username>/main/<run-id>
```

---

## 📤 Example Output

The model receives the source code as input and returns a summary such as:

> “The code defines a Modal function that runs a chatbot using the transformers library with the Qwen model. It reads the contents of the current Python file, constructs a prompt asking what the code does, and uses the chatbot to generate a response.”

This confirms:

* The model loaded successfully
* GPU inference worked
* The prompt and response loop is functioning

---

## 🧩 Notes & Tips

* Image build time is **long on first run** due to CUDA + PyTorch downloads
* Subsequent runs are much faster (cached image)
* You can replace the model with any Hugging Face model that supports GPU inference
* This pattern can be extended into:

  * API endpoints
  * Batch inference jobs
  * Streaming chat services

---


