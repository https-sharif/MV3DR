import os
import tempfile
from PIL import Image, ImageEnhance
from typing import List

from config import CONTRAST_ENHANCEMENT, SHARPNESS_ENHANCEMENT


def preprocess_images(image_paths: List[str]) -> List[str]:
    cleaned_paths = []
    
    for i, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB")
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(CONTRAST_ENHANCEMENT)
        
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(SHARPNESS_ENHANCEMENT)
        
        save_path = os.path.join(tempfile.gettempdir(), f"input_refined_{i}.png")
        img.save(save_path)
        cleaned_paths.append(save_path)
    
    return cleaned_paths
