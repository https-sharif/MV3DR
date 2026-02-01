import torch
from typing import List, Dict, Any

from dust3r.utils.device import to_cpu, collate_with_cat as collate
from config import BATCH_SIZE


def run_inference(pairs: List, model: torch.nn.Module, device: str, batch_size: int = BATCH_SIZE) -> Dict[str, Any]:
    result = []
    
    for i in range(0, len(pairs), batch_size):
        batch = collate(pairs[i:i+batch_size])
        
        for view in batch:
            for k in ["img", "pts3d", "valid_mask", "camera_pose", "camera_intrinsics"]:
                if k in view:
                    view[k] = view[k].to(device)
        
        v1, v2 = batch
        
        with torch.cuda.amp.autocast():
            p1, p2 = model(v1, v2)
        
        result.append(to_cpu(dict(view1=v1, view2=v2, pred1=p1, pred2=p2)))
    
    return collate(result, lists=True)
