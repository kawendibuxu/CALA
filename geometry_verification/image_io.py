from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_image_paths(image_dir, seq_len):
    root = Path(image_dir)
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(paths) < seq_len:
        raise ValueError(f"Need at least {seq_len} images in {root}, found {len(paths)}.")
    return paths[:seq_len]


def load_rgb_sequence(paths, image_size):
    transform = T.Compose(
        [
            T.Resize(tuple(image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ]
    )

    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(transform(image.convert("RGB")))
    return torch.stack(frames, dim=0)


def make_grid_query_points(image_size, grid_rows=12, grid_cols=16, margin_ratio=0.08):
    height, width = image_size
    x_margin = width * margin_ratio
    y_margin = height * margin_ratio

    xs = torch.linspace(x_margin, width - x_margin - 1, grid_cols)
    ys = torch.linspace(y_margin, height - y_margin - 1, grid_rows)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
