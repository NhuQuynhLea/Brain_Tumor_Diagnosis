"""Training-time data augmentation pipeline (albumentations)."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transform(cfg) -> A.Compose:
    """Random augmentation for training: rotate/flip/zoom/brightness/elastic + resize + normalize."""
    aug = cfg.augmentation
    h, w = cfg.preprocessing.resize
    return A.Compose(
        [
            A.Rotate(limit=aug.rotation_limit, p=0.5),
            A.HorizontalFlip(p=aug.horizontal_flip_p),
            A.Affine(scale=(1 - aug.scale_limit, 1 + aug.scale_limit), p=0.4),
            A.RandomBrightnessContrast(
                brightness_limit=aug.brightness_limit,
                contrast_limit=aug.contrast_limit,
                p=aug.brightness_contrast_p,
            ),
            A.ElasticTransform(
                alpha=aug.elastic_alpha,
                sigma=aug.elastic_sigma,
                p=aug.elastic_p,
            ),
            A.Resize(height=h, width=w),
            A.Normalize(mean=cfg.preprocessing.normalize.mean,
                        std=cfg.preprocessing.normalize.std,
                        max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
