"""
Jupyter Lab on Modal using the Tunnel approach.
Based on: https://github.com/modal-labs/modal-examples/blob/main/11_notebooks/jupyter_inside_modal.py

Run with:  modal run jupyter_tunnel.py

This uses modal.forward() to create a secure tunnel to Jupyter.
The URL will be printed in your terminal.
"""

import modal

app = modal.App("jupyter-tunnel-server")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands(
        "apt-get update",
        "apt-get install -y git",
    )
    .pip_install(
        "jupyterlab",
        "jupyter>=1.0",
        "ipywidgets>=8.0",
        "pandas",
        "numpy",
        "websockets>=12.0",
        "requests>=2.28",
        "pyarrow",
        "plotly",
        "scipy>=1.11",
        "scikit-learn",
        "hdbscan",
    )
)

JUPYTER_PORT = 8888
JUPYTER_TOKEN = "my-secure-token-change-me"  # Change this!

REPO_URL_TEMPLATE = "https://{token}@github.com/Joshua-Abok/market_microstructure_manipulation.git"
PROJECT_DIR = "/root/project"


@app.function(
    image=image,
    timeout=60 * 60 * 24,  # 24 hours
    secrets=[modal.Secret.from_name("github-token")],
    max_containers=1,
)
def run_jupyter(timeout: int = 86400):
    import os
    import subprocess
    import time

    # Clone the repo
    github_token = os.environ["GITHUB_TOKEN"]
    clone_url = REPO_URL_TEMPLATE.format(token=github_token)

    print("📦 Cloning repository...")
    result = subprocess.run(
        ["git", "clone", clone_url, PROJECT_DIR],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"❌ Git clone failed: {result.stderr}")
        raise RuntimeError(f"Clone failed: {result.stderr}")
    print("✅ Repository cloned successfully.")

    # Start Jupyter with a tunnel
    with modal.forward(JUPYTER_PORT) as tunnel:
        jupyter_process = subprocess.Popen(
            [
                "jupyter", "lab",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={JUPYTER_PORT}",
                "--ServerApp.allow_origin=*",
                "--ServerApp.allow_remote_access=True",
            ],
            env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN},
            cwd=PROJECT_DIR,
        )

        print(f"\n🔗 Jupyter Lab is available at:\n")
        print(f"   {tunnel.url}/?token={JUPYTER_TOKEN}\n")
        print("Press Ctrl+C in your terminal to stop.\n")

        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                time.sleep(5)
            print(f"⏰ Reached {timeout}s timeout. Shutting down.")
        except KeyboardInterrupt:
            print("🛑 Shutting down...")
        finally:
            jupyter_process.kill()


@app.local_entrypoint()
def main(timeout: int = 86400):
    run_jupyter.remote(timeout=timeout)
