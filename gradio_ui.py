import gradio as gr
import functools
import os

from pipeline import run_pipeline
from config import ASSETS_DIR, IMAGE_SIZE


def load_assets():
    with open(os.path.join(ASSETS_DIR, 'style.css'), 'r') as f:
        css = f.read()
    
    with open(os.path.join(ASSETS_DIR, 'fullscreen.js'), 'r') as f:
        fullscreen_js = f.read()
    
    return css, fullscreen_js


def build_ui(out_dir: str, model, device: str) -> gr.Blocks:
    css, fullscreen_js = load_assets()
    pipeline = functools.partial(run_pipeline, out_dir, model, device, IMAGE_SIZE)

    with gr.Blocks(title="Multi-View 3D Reconstruction (MV3DR)", css=css, theme=gr.themes.Base(), fill_width=True) as app:
        gr.Markdown("# Multi-View 3D Reconstruction (MV3DR)")

        with gr.Row():
            with gr.Column(scale=1):
                input_files = gr.File(file_count="multiple", label="Images")
                run_btn = gr.Button("Run Inference")
                with gr.Accordion("Settings", open=False):
                    n_iterations = gr.Slider(100, 1000, 300, label="Alignment Iterations")
                    render_mode = gr.Checkbox(True, label="Render as Point Cloud")
                    post_proc = gr.Checkbox(True, label="Filter Background Points")
                    clean_depth = gr.Checkbox(True, label="Clean Point Cloud")

            with gr.Column(scale=2):
                output_model = gr.Model3D(label="3D Output", height=600, elem_id="model-container")
                full_screen_btn = gr.Button("Toggle Full Screen ⛶", size="sm")

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## RGB | DEPTH | CONFIDENCE")
                artifact_gallery = gr.Gallery(columns=3, height="auto", label="Logs")

        full_screen_btn.click(None, None, None, js=fullscreen_js)

        saved_state = gr.State()
        run_btn.click(fn=pipeline,
                      inputs=[input_files, n_iterations, render_mode, post_proc, clean_depth],
                      outputs=[saved_state, output_model, artifact_gallery],
                      show_progress=True)

    return app
