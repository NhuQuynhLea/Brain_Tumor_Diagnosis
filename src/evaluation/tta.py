"""Test-time augmentation (TTA) inference.

Modes:
  full       - every sample gets K views; prob = mean over views
  selective  - only low-confidence samples (max prob < tau) get the extra views
  subject    - gate at group level: every slice of a low-confidence subject
               (mean slice confidence < tau) gets the extra views

All TTA views are deterministic transforms drawn from the same family/limits as
build_train_transform, so they stay inside the training augmentation envelope.
"""

import time

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2


def build_tta_views(cfg) -> list[tuple[str, A.Compose]]:
    """Deterministic TTA views. View 0 is the plain eval transform (identity)."""
    h, w = cfg.preprocessing.resize
    aug = cfg.get("augmentation") or {}
    rot = float(aug.get("rotation_limit", 15))
    scale_hi = 1.0 + float(aug.get("scale_limit", 0.10))
    scale_lo = 1.0 - float(aug.get("scale_limit", 0.10))
    tail = [
        A.Resize(height=h, width=w),
        A.Normalize(mean=cfg.preprocessing.normalize.mean,
                    std=cfg.preprocessing.normalize.std,
                    max_pixel_value=255.0),
        ToTensorV2(),
    ]
    return [
        ("id",     A.Compose(tail)),
        ("hflip",  A.Compose([A.HorizontalFlip(p=1.0)] + tail)),
        ("rot+",   A.Compose([A.Rotate(limit=(rot, rot), p=1.0,
                                     border_mode=cv2.BORDER_REFLECT_101)] + tail)),
        ("rot-",   A.Compose([A.Rotate(limit=(-rot, -rot), p=1.0,
                                     border_mode=cv2.BORDER_REFLECT_101)] + tail)),
        ("scale+", A.Compose([A.Affine(scale=(scale_hi, scale_hi), p=1.0)] + tail)),
        ("scale-", A.Compose([A.Affine(scale=(scale_lo, scale_lo), p=1.0)] + tail)),
    ]


def load_raw_images(df: pd.DataFrame) -> list[np.ndarray]:
    """Read manifest filepaths -> list of (H, W, 3) uint8 RGB (shapes may vary)."""
    imgs = []
    for fp in df["filepath"]:
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Could not read image: {fp}")
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return imgs


@torch.no_grad()
def forward_views(model: torch.nn.Module, imgs_rgb: np.ndarray,
                  views: list[tuple[str, A.Compose]], device,
                  batch_size: int = 32) -> np.ndarray:
    """Run every view on every image. Returns probs (N, K, C)."""
    model.eval()
    n = len(imgs_rgb)
    probs = None
    for vi, (_, tf) in enumerate(views):
        for i in range(0, n, batch_size):
            x = torch.stack([tf(image=im)["image"] for im in imgs_rgb[i:i + batch_size]])
            p = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()
            if probs is None:
                probs = np.zeros((n, len(views), p.shape[1]))
            probs[i:i + batch_size, vi] = p
    return probs


def gate_and_aggregate(probs_views: np.ndarray, df: pd.DataFrame,
                       tau: float, mode: str = "selective"):
    """Merge view probs under a confidence gate.

    probs_views: (N, K, C). Returns (final_probs (N, C), trigger mask (N, bool)).
    For 'full', every row merges; tau is ignored.
    """
    p0 = probs_views[:, 0, :]
    k = probs_views.shape[1]
    merged = probs_views.mean(axis=1)
    conf = p0.max(1)
    if mode == "selective":
        trigger = conf < tau
    elif mode == "subject":
        subj_conf = pd.Series(conf).groupby(df["subject"].to_numpy()).transform("mean")
        trigger = (subj_conf < tau).to_numpy()
    else:  # full
        trigger = np.ones(len(p0), dtype=bool)
    return np.where(trigger[:, None], merged, p0), trigger


def run_tta_eval(model, df: pd.DataFrame, cfg, device, mode: str = "selective",
                 tau: float = 0.8, n_views: int = 5, batch_size: int = 32) -> dict:
    """Gated TTA over a manifest DataFrame; measures real wall-clock timing.

    Pass 1 runs the identity view on all rows. Triggered rows get views 1..K-1
    and their final prob is the mean over all K views.
    """
    views = build_tta_views(cfg)[:max(1, n_views)]
    n = len(df)

    t0 = time.perf_counter()
    imgs = load_raw_images(df)
    t_load = time.perf_counter()
    probs0 = forward_views(model, imgs, views[:1], device, batch_size)[:, 0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    probs, trigger = probs0.copy(), np.zeros(n, dtype=bool)
    tta_s = 0.0
    if mode in ("full", "selective", "subject") and len(views) > 1:
        conf = probs0.max(1)
        if mode == "full":
            idx = np.arange(n)
        elif mode == "selective":
            idx = np.where(conf < tau)[0]
        else:
            subj_conf = pd.Series(conf).groupby(df["subject"].to_numpy()).transform("mean")
            idx = np.where((subj_conf < tau).to_numpy())[0]
        trigger[idx] = True
        t2 = time.perf_counter()
        if len(idx):
            vprobs = forward_views(model, [imgs[i] for i in idx], views[1:],
                                   device, batch_size)
            if device.type == "cuda":
                torch.cuda.synchronize()
            probs[idx] = (probs0[idx] + vprobs.sum(axis=1)) / len(views)
        tta_s = time.perf_counter() - t2

    total_s = time.perf_counter() - t0
    return {
        "probs": probs,
        "probs0": probs0,
        "conf": probs0.max(1),
        "trigger": trigger,
        "views": [v[0] for v in views],
        "timing": {
            "mode": mode,
            "tau": tau,
            "n_views": len(views),
            "n_samples": n,
            "n_triggered": int(trigger.sum()),
            "trigger_rate": float(trigger.mean()),
            "load_s": round(t_load - t0, 3),
            "pass1_s": round(t1 - t_load, 3),
            "tta_s": round(tta_s, 3),
            "total_s": round(total_s, 3),
            "effective_ms_per_slice": round(total_s / n * 1000, 2),
        },
    }


def compute_view_probs(model, df: pd.DataFrame, cfg, device,
                       n_views: int = 5, batch_size: int = 32):
    """Run ALL TTA views on ALL rows (for tau sweeps / full-TTA ceiling).
    Returns (probs (N, K, C), df, elapsed_s)."""
    views = build_tta_views(cfg)[:max(1, n_views)]
    t0 = time.perf_counter()
    imgs = load_raw_images(df)
    probs = forward_views(model, imgs, views, device, batch_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return probs, df, time.perf_counter() - t0
