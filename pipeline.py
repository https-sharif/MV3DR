import torch
import copy
from typing import List, Tuple, Optional

from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from core.preprocessing import preprocess_images
from core.inference import run_inference
from core.postprocessing import filter_background_points, create_3d_output
from core.visualization import generate_artifacts
from config import IMAGE_SIZE, DEFAULT_ITERATIONS


def run_pipeline(outdir: str, model: torch.nn.Module, device: str, img_size: int,
                filelist: List[str], niter: int, as_pc: bool, refinement: bool, 
                clean_depth: bool = True) -> Tuple[Optional[object], Optional[str], List]:
   
    if not filelist or len(filelist) == 0:
        return None, None, []
    
    processed_list = preprocess_images(filelist)
    imgs = load_images(processed_list, size=img_size)
    
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    
    output = run_inference(pairs, model, device)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode)

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        scene.compute_global_alignment(init='mst', niter=niter, schedule='linear', lr=0.01)

    if clean_depth:
        scene = scene.clean_pointcloud()
    
    if refinement:
        scene = filter_background_points(scene)

    glb_path = create_3d_output(
        outdir, scene.imgs, scene.get_pts3d(),
        to_numpy(scene.get_masks()), scene.get_focals().cpu(),
        scene.get_im_poses().cpu(), as_pointcloud=as_pc
    )

    artifacts = generate_artifacts(scene)

    return scene, glb_path, artifacts
