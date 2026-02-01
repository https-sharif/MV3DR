import os, tempfile
from PIL import Image, ImageEnhance

def preprocess(image_paths):
    cleaned = []
    for i, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB")
        img = ImageEnhance.Contrast(img).enhance(1.2)
        img = ImageEnhance.Sharpness(img).enhance(1.5)

        out = os.path.join(tempfile.gettempdir(), f"input_refined_{i}.png")
        img.save(out)
        cleaned.append(out)
    return cleaned
