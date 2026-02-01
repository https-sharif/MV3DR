import os
import numpy as np
import trimesh
from typing import List, Tuple

from dust3r.utils.device import to_numpy
from dust3r.viz import pts3d_to_trimesh, cat_meshes
from config import DEFAULT_CONF_THRESHOLD


def filter_background_points(scene, conf_threshold: float = DEFAULT_CONF_THRESHOLD):
    masks = scene.get_masks()
    confs = [c for c in scene.im_conf]
    
    for i, (mask, conf) in enumerate(zip(masks, confs)):
        conf_mask = conf > conf_threshold
        masks[i] = mask & conf_mask
    
    return scene


def create_3d_output(outdir: str, imgs: List, pts3d: List, mask: List, 
                     focals: List, cams2world: List, as_pointcloud: bool = False) -> str:

    pts3d, imgs, focals, cams2world = map(to_numpy, [pts3d, imgs, focals, cams2world])
    scene = trimesh.Scene()

    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([imgs[i][mask[i]] for i in range(len(imgs))])
        geometry = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
    else:
        meshes = [pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]) for i in range(len(imgs))]
        geometry = trimesh.Trimesh(**cat_meshes(meshes))

    centroid = geometry.centroid
    geometry.apply_translation(-centroid)
    scene.add_geometry(geometry)

    flip_correction = np.eye(4)
    flip_correction[1, 1] = -1
    flip_correction[2, 2] = -1
    scene.apply_transform(flip_correction)

    outfile = os.path.join(outdir, 'object.glb')
    scene.export(outfile)
    
    return outfile
