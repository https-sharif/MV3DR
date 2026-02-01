# Multi-View 3D Reconstruction (MV3DR)

A web application for multi-view 3D object reconstruction using DUSt3R (Dense and Unconstrained Stereo 3D Reconstruction).

## Overview

MV3DR is a production-ready system that generates 3D models from multiple 2D images through dense stereo reconstruction. The application features a dark-themed web interface built with Gradio, providing real-time visualization of depth maps, confidence heatmaps, and interactive 3D outputs.

## Features

- Multi-view stereo reconstruction from 2+ images
- Dual export modes: point cloud or textured mesh
- Real-time depth map and confidence visualization
- Advanced post-processing with background filtering
- Point cloud cleaning and alignment optimization
- Interactive 3D viewer with fullscreen support
- Modular architecture with separated core modules

## Technical Stack

- **Model**: DUSt3R (Dense and Unconstrained Stereo 3D Reconstruction)
- **Framework**: PyTorch with mixed-precision inference
- **Interface**: Gradio with custom dark theme
- **3D Processing**: Trimesh for mesh/point cloud generation
- **Visualization**: Matplotlib for heatmaps

## Architecture

```
MV3DR/
├── main.py              # Application entry point
├── config.py            # Centralized configuration
├── model.py             # DUSt3R model initialization
├── pipeline.py          # Main reconstruction pipeline
├── gradio_ui.py         # Web interface
├── core/
│   ├── inference.py     # Model inference with AMP
│   ├── preprocessing.py # Image enhancement
│   ├── postprocessing.py# 3D output generation
│   └── visualization.py # Artifact generation
└── assets/
    ├── style.css        # Dark theme styling
    └── fullscreen.js    # 3D viewer controls
```

## Pipeline

1. **Preprocessing**: Contrast and sharpness enhancement
2. **Pair Generation**: Create symmetrized image pairs
3. **Inference**: DUSt3R stereo reconstruction with mixed precision
4. **Alignment**: Multi-view point cloud optimization
5. **Post-processing**: Depth cleaning and background filtering
6. **Export**: GLB file generation (mesh or point cloud)
7. **Visualization**: Depth maps and confidence heatmaps

## Performance

- RTX 4090: ~15-20 seconds (3 images)
- RTX 3080: ~25-30 seconds (3 images)
- T4 GPU: ~40-50 seconds (3 images)
- CPU: ~5-10 minutes (3 images)

## Requirements

- Python 3.10+
- CUDA-capable GPU (16GB+ VRAM recommended)
- 16GB+ RAM

## License

Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
Built on DUSt3R by Naver Corporation.

## Credits

- DUSt3R: Naver Corporation
- Gradio: Hugging Face
- Trimesh: Michael Dawson-Haggerty
