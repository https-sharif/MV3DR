import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DUST3R_DIR = os.path.join(ROOT_DIR, "dust3r")
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
OUTPUT_DIR = os.path.join(ROOT_DIR, "results")
WEIGHTS_PATH = os.path.join(DUST3R_DIR, "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")

BATCH_SIZE = 1
IMAGE_SIZE = 512

DEFAULT_ITERATIONS = 300
DEFAULT_CONF_THRESHOLD = 0.001

CONTRAST_ENHANCEMENT = 1.2
SHARPNESS_ENHANCEMENT = 1.5

SERVER_NAME = "0.0.0.0"
SERVER_PORT = 7860
SHARE = True
SHOW_ERROR = True
