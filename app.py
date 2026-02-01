import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'dust3r'))

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
    app.launch()


if __name__ == "__main__":
    main()
