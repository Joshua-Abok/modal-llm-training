"""
Jupyter Lab on Modal using the Sandbox approach.
Based on: https://modal.com/docs/examples/jupyter_sandbox

IMPORTANT: Run with plain Python, NOT 'modal run':

    python jupyter_sandbox.py

This will print a URL you can open in your browser.
When done, terminate via Modal dashboard or Ctrl+C.
"""

import json
import secrets
import time
import urllib.request

import modal

# ─── Configuration ───────────────────────────────────────────────────────────

JUPYTER_PORT = 8888
SANDBOX_TIMEOUT = 60 * 60 * 6  # 6 hours (adjust as needed)
REPO_URL = "https://$GITHUB_TOKEN@github.com/Joshua-Abok/market_microstructure_manipulation.git"
PROJECT_DIR = "/root/project"

# ─── Image (defined at module level, no app needed) ──────────────────────────

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

# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    # Create the app INSIDE main(), not at module level.
    # This avoids the "App is already running" error.
    app = modal.App.lookup("jupyter-sandbox-server", create_if_missing=True)

    # Generate a random token for Jupyter auth
    token = secrets.token_urlsafe(16)
    token_secret = modal.Secret.from_dict({"JUPYTER_TOKEN": token})

    print("🚀 Creating sandbox...")

    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            "bash",
            "-c",
            f"""
            set -e
            echo "📦 Cloning repository..."
            git clone {REPO_URL} {PROJECT_DIR}
            echo "✅ Clone complete."
            cd {PROJECT_DIR}
            echo "🚀 Starting Jupyter Lab..."
            exec jupyter lab \
                --no-browser \
                --allow-root \
                --ip=0.0.0.0 \
                --port={JUPYTER_PORT} \
                --ServerApp.token="$JUPYTER_TOKEN" \
                --ServerApp.allow_origin='*' \
                --ServerApp.allow_remote_access=True
            """,
            encrypted_ports=[JUPYTER_PORT],
            secrets=[
                token_secret,
                modal.Secret.from_name("github-token"),
            ],
            timeout=SANDBOX_TIMEOUT,
            image=image,
            app=app,
            gpu=None,  # add a GPU if you need it!
        )

    print(f"📦 Sandbox ID: {sandbox.object_id}")

    # Get the tunnel URL
    tunnel = sandbox.tunnels()[JUPYTER_PORT]
    url = f"{tunnel.url}/?token={token}"

    # Wait for Jupyter to be ready
    print("⏳ Waiting for Jupyter to start...")
    if _wait_for_jupyter(tunnel.url, token, timeout=90):
        print(f"\n✅ Jupyter Lab is ready!")
        print(f"🔗 Open this URL in your browser:\n")
        print(f"   {url}\n")
    else:
        print("⚠️  Timed out waiting, but Jupyter might still be starting.")
        print(f"🔗 Try this URL:\n\n   {url}\n")

    print(f"📌 To terminate: modal sandbox terminate {sandbox.object_id}")
    print("   Or use the Modal dashboard: https://modal.com/sandboxes")
    print("\n🛑 Press Ctrl+C to exit this script (sandbox keeps running).")

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n👋 Exiting. Sandbox is still running in the background.")
        print(f"   Terminate with: modal sandbox terminate {sandbox.object_id}")


def _wait_for_jupyter(base_url, token, timeout=90):
    """Poll Jupyter's status endpoint until it responds."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(
                f"{base_url}/api/status?token={token}",
                headers={"User-Agent": "Modal-Jupyter-Launcher"},
            )
            resp = urllib.request.urlopen(req, timeout=5)
            if resp.getcode() == 200:
                data = json.loads(resp.read().decode())
                if data.get("started", False):
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


if __name__ == "__main__":
    main()