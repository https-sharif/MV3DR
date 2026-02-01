import torch, copy, numpy as np, trimesh, matplotlib.pyplot as plt
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy, to_cpu, collate_with_cat as collate
from dust3r.viz import pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils import preprocess

BATCH_SIZE = 1

def interleave(img1, img2):
    out = {}
    for k, v1 in img1.items():
        v2 = img2[k]
        if isinstance(v1, torch.Tensor):
            out[k] = torch.stack((v1, v2), dim=1).flatten(0, 1)
        else:
            out[k] = [x for p in zip(v1, v2) for x in p]
    return out

def inference(pairs, model, device):
    results = []
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = collate(pairs[i:i+BATCH_SIZE])
        for view in batch:
            for k in ["img", "pts3d", "valid_mask", "camera_pose", "camera_intrinsics"]:
                if k in view:
                    view[k] = view[k].to(device)
        v1, v2 = batch
        v1, v2 = interleave(v1, v2), interleave(v2, v1)

        with torch.cuda.amp.autocast():
            p1, p2 = model(v1, v2)
        results.append(to_cpu(dict(view1=v1, view2=v2, pred1=p1, pred2=p2)))

    return collate(results, lists=True)

def create_glb(outdir, scene):
    meshes = [
        pts3d_to_trimesh(scene.imgs[i], scene.get_pts3d()[i], scene.get_masks()[i])
        for i in range(len(scene.imgs))
    ]
    mesh = trimesh.Trimesh(**cat_meshes(meshes))
    mesh.apply_translation(-mesh.centroid)

    scene_out = trimesh.Scene(mesh)
    out = f"{outdir}/object.glb"
    scene_out.export(out)
    return out
