import gradio as gr
import functools
from pathlib import Path
from pipeline import run_pipeline

ASSETS = Path(__file__).parent / "assets"
CSS = (ASSETS / "style.css").read_text()
FULLSCREEN_JS = (ASSETS / "fullscreen.js").read_text()


def build_ui(outdir, model, device):

    pipeline = functools.partial(run_pipeline, outdir, model, device, 512)

    with gr.Blocks(
        title="3D Object Reconstruction",
        css=CSS,
        theme=gr.themes.Base(),
        fill_width=True,
    ) as app:

        gr.Markdown("# 3D Object Reconstruction")

        with gr.Row():

            with gr.Column(scale=1):
                files = gr.File(file_count="multiple", label="Images")
                run_btn = gr.Button("Run Inference", variant="primary")

                with gr.Accordion("Settings", open=False):
                    iters = gr.Slider(100, 1000, 300, step=50, label="Alignment Iteration")
                    as_pc = gr.Checkbox(True, label="Render as Point Cloud")
                    refine = gr.Checkbox(True, label="Filter Background Points")
                    clean = gr.Checkbox(True, label="Clean-up depthmaps")

            with gr.Column(scale=2):
                model3d = gr.Model3D(
                    label="3D Output",
                    height=600,
                    elem_id="model-container",
                )
                fs_btn = gr.Button("Toggle Full Screen ⛶", size="sm")

        gr.Markdown("---")
        gallery = gr.Gallery(columns=3, label="RGB | DEPTH | CONFIDENCE")

        fs_btn.click(None, None, None, js=FULLSCREEN_JS)

        state = gr.State()
        run_btn.click(
            fn=pipeline,
            inputs=[files, iters, as_pc, refine, clean],
            outputs=[state, model3d, gallery],
            show_progress="minimal",
        )

    return app
