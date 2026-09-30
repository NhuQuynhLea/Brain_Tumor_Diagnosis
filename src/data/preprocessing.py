"""Image preprocessing pipeline: resize -> normalize -> tensor.

Uses albumentations so the same composition API is shared with augmentation.py.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_eval_transform(cfg) -> A.Compose:
    """Deterministic transform for validation/test: resize + ImageNet normalize."""
    h, w = cfg.preprocessing.resize
    return A.Compose(
        [
            A.Resize(height=h, width=w),
            A.Normalize(mean=cfg.preprocessing.normalize.mean,
                        std=cfg.preprocessing.normalize.std,
                        max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def denormalize(tensor, mean, std):
    """Undo normalization for visualization. tensor: (C,H,W) torch.Tensor -> (H,W,C) numpy."""
    import numpy as np
    import torch

    img = tensor.detach().cpu().clone()
    for c in range(img.shape[0]):
        img[c] = img[c] * std[c] + mean[c]
    img = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)
