import os
import sys
import subprocess
import torch

ROOT_DIR = os.path.dirname(__file__)
DUST3R_REPO_DIR = os.path.join(ROOT_DIR, 'dust3r')
_DUST3R_PKG_MARKER = os.path.join(DUST3R_REPO_DIR, 'dust3r', 'model.py')


def _ensure_dust3r_source() -> None:
    if os.path.exists(_DUST3R_PKG_MARKER):
        return

    print("Local DUSt3R source missing. Cloning repository...")
    subprocess.check_call([
        "git",
        "clone",
        "-b",
        "dev",
        "--recursive",
        "https://github.com/camenduru/dust3r",
        DUST3R_REPO_DIR,
    ])


_ensure_dust3r_source()

if DUST3R_REPO_DIR not in sys.path:
    sys.path.insert(0, DUST3R_REPO_DIR)

from model import initialize
from gradio_ui import build_ui
from config import WEIGHTS_PATH, OUTPUT_DIR, SERVER_NAME, SERVER_PORT, SHARE, SHOW_ERROR


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading DUSt3R model...")
    model = initialize(WEIGHTS_PATH, device)
    
    print("Starting Multi-View 3D Reconstruction (MV3DR)...")
    app = build_ui(OUTPUT_DIR, model, device)
    app.launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        share=SHARE,
        show_error=SHOW_ERROR
    )


if __name__ == "__main__":
    main()
