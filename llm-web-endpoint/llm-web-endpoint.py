from pathlib import Path
import modal


app = modal.App("llm-web-endpoint")
image = modal.Image.debian_slim().uv_pip_install("transformers[torch]", "fastapi", "uvicorn")

@app.function(gpu="h100", image=image)
@modal.web_endpoint(method="POST")
def chat_endpoint(prompt: str | None = None):
    from transformers import pipeline 

    if prompt is None:
        prompt = (
            "Read the following code and explain what it does: \n\n"
            + f"{Path(__file__).read_text()}"
        )

    chatbot = pipeline(
        model="Qwen/Qwen3-1.7B-FP8", 
        device_map="cuda", 
        max_new_tokens=1024,
    )

    result = chatbot(
        [{"role": "user", "content": prompt}]
        )
    
    response_text = result[0]["generated_text"][-1]["content"]

    return {
        "prompt": prompt,
        "response": response_text,
    }