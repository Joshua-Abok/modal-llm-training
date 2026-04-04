import modal

app = modal.App("jupyter-gpu")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "jupyterlab",
        "pandas",
        "numpy",
        "pyarrow",
        "plotly",
        "scikit-learn",
        "hdbscan"
    )
)

# @app.function(secrets=[modal.Secret.from_name("github-token")])
@app.function(
    image=image,
    # gpu="any",  # or "A10G", "T4", etc.
    timeout=60 * 60 * 24,  # 24 hours
    # secrets=[modal.Secret.from_name("github-token")]
)

@modal.web_server(port=8888)
def run_jupyter():
    import subprocess
    import os

    # Clone your repo if not already there
    # if not os.path.exists("/root/project"):
    #     subprocess.run([
    #         "git", "clone",
    #         "https://github.com/Joshua-Abok/market_microstructure_manipulation.git",
    #         "/root/project"
    #     ])

    # os.chdir("/root/project")

    # token = os.environ["GITHUB_TOKEN"]
    # subprocess.run([
    #     "git", "clone",
    #     f"https://{token}@github.com/Joshua-Abok/market_microstructure_manipulation.git",
    #     "/root/project"
    # ])

    os.chdir("/root/market_microstructure_manipulation")

    subprocess.run([
        "jupyter", "lab",
        "--ip=0.0.0.0",
        "--port=8888",
        "--no-browser",
        "--allow-root",
        "--NotebookApp.token=''",  # disables token (optional)
        "--NotebookApp.password=''"
    ])