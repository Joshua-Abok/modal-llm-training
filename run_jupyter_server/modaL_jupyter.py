import modal

app = modal.App("jupyter-cpu-server")

image = (
    modal.Image.debian_slim()
    .run_commands(
        "apt-get update",
        "apt-get install -y git"
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
        "hdbscan"
    )
)


@app.function(
    image=image,
    timeout=60 * 60 * 24,  # 24 hours
    secrets=[modal.Secret.from_name("github-token")]
)
@modal.web_server(port=8888)
def run_jupyter():
    import subprocess
    import os

    # Clone private repo with error handling
    token = os.environ["GITHUB_TOKEN"]
    result = subprocess.run(
        [
            "git", "clone",
            f"https://{token}@github.com/Joshua-Abok/market_microstructure_manipulation.git",
            "/root/project"
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Git clone failed: {result.stderr}")
        raise RuntimeError(f"Failed to clone repo: {result.stderr}")

    print("Repo cloned successfully.")
    os.chdir("/root/project")

    # Start Jupyter Lab with correct flags for Modal's proxy
    subprocess.Popen(
        [
            "jupyter", "lab",
            "--ip=0.0.0.0",
            "--port=8888",
            "--no-browser",
            "--allow-root",
            "--ServerApp.token=",
            "--ServerApp.password=",
            "--ServerApp.allow_origin=*",
            "--ServerApp.allow_remote_access=True",
            "--ServerApp.base_url=/",
        ]
    )