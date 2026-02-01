import matplotlib.pyplot as plt
from typing import List, Tuple

from dust3r.utils.image import rgb
from dust3r.utils.device import to_numpy


def generate_artifacts(scene) -> List[Tuple]:
    artifacts = []
    cmap = plt.get_cmap('jet')
    
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])

    for i in range(len(scene.imgs)):
        artifacts.append((scene.imgs[i], f"View {i+1}: RGB"))

        d_norm = depths[i] / (depths[i].max() + 1e-8)
        artifacts.append((rgb(d_norm), f"View {i+1}: Depth Map"))

        c_norm = cmap(confs[i] / (confs[i].max() + 1e-8))
        artifacts.append((rgb(c_norm), f"View {i+1}: Confidence Heatmap"))

    return artifacts
