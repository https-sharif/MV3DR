# Multi-View 3D Reconstruction (MV3DR)

Convert multiple 2D images into 3D models using DUSt3R (Dense and Unconstrained Stereo 3D Reconstruction). Simple web interface powered by Gradio.

## Quick Start

```bash
git clone https://github.com/https-sharif/MV3DR.git
cd MV3DR
bash setup.sh
python main.py
```

Open `http://0.0.0.0:7860` in your browser, upload 2+ images, and generate your 3D model.

## Features

- Multi-view 3D reconstruction from 2+ images
- Export as point cloud or mesh (GLB format)
- Real-time depth maps and confidence visualization
- Background filtering and point cloud cleaning
- Adjustable quality settings (100-1000 iterations)
- Interactive 3D viewer with fullscreen mode

## How It Works

1. **Preprocess**: Enhances image contrast and sharpness
2. **Inference**: DUSt3R model predicts depth and generates 3D points
3. **Alignment**: Merges multiple views into unified 3D space (300 iterations default)
4. **Post-process**: Cleans point cloud and filters background
5. **Export**: Generates GLB file for viewing/download

## Setup

**Requirements**: Python 3.10+, 16GB RAM, ~3GB storage

Run the setup script:

```bash
bash setup.sh
```

This installs PyTorch, Gradio, Trimesh, and downloads the DUSt3R model (~2.1GB).

Then start the app:

```bash
python main.py
```

### Manual Installation

If the setup script fails:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install torch torchvision gradio trimesh scipy roma einops
git clone -b dev --recursive https://github.com/camenduru/dust3r
cd dust3r && pip install -e . && cd ..

mkdir -p dust3r/checkpoints
wget https://huggingface.co/camenduru/dust3r/resolve/main/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  -P dust3r/checkpoints/
```

## Usage

1. Upload 2+ images of your object from different angles
2. (Optional) Adjust settings:
   - Iterations: 100-1000 (default: 300)
   - Point cloud or mesh export
   - Background filtering
3. Click "Run Inference"
4. Download GLB file or view in browser

**Tips**: Use 4-8 images with good overlap between views. Matte objects work better than shiny ones.

## Configuration

Edit `config.py` to customize:
- Image size (default: 512px)
- Iterations (default: 300)
- Server port (default: 7860)

## License

CC BY-NC-SA 4.0 - Non-commercial use only

## Credits

Built with [DUSt3R](https://github.com/naver/dust3r), [Gradio](https://gradio.app/), [PyTorch](https://pytorch.org/), and [Trimesh](https://github.com/mikedh/trimesh).
