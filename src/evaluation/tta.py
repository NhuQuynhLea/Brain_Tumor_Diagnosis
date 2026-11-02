"""Test-time augmentation (TTA) inference.

Modes:
  full    - every sample gets K views; prob = mean over views
  mcgtta  - Multi-Scale Consistency-Guided TTA: low-confidence samples
            (max prob < tau) are probed with the scale views and escalate to
            full multi-view aggregation only when the probes disagree.
            Mirrors the MPAC block's multi-dilation paths: scale-sensitive
            inputs are exactly where multi-path fusion is unstable.

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
from tqdm import tqdm


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
    for fp in tqdm(df["filepath"], desc="Loading images", unit="img"):
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Could not read image: {fp}")
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return imgs


@torch.no_grad()
def forward_views(model: torch.nn.Module, imgs_rgb: list[np.ndarray],
                  views: list[tuple[str, A.Compose]], device,
                  batch_size: int = 32, desc: str = "Inference") -> np.ndarray:
    """Run every view on every image. Returns probs (N, K, C)."""
    model.eval()
    n = len(imgs_rgb)
    probs = None
    pbar = tqdm(total=len(views) * -(-n // batch_size), desc=desc, unit="batch")
    for vi, (vname, tf) in enumerate(views):
        pbar.set_postfix(view=vname)
        for i in range(0, n, batch_size):
            x = torch.stack([tf(image=im)["image"] for im in imgs_rgb[i:i + batch_size]])
            p = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()
            if probs is None:
                probs = np.zeros((n, len(views), p.shape[1]))
            probs[i:i + batch_size, vi] = p
            pbar.update(1)
    pbar.close()
    return probs


STRATEGY_NAMES = ("mcgtta",)


def union_trigger(p0: np.ndarray, taus: list[float]) -> np.ndarray:
    """Rows entering the confidence gate at any tau in the sweep, so a single
    views pass on this subset serves every tau simulation."""
    return np.where(p0.max(1) < max(taus))[0]


def simulate_mcgtta(probs_T: np.ndarray, idx_T: np.ndarray, p0: np.ndarray,
                    tau: float, view_names: list[str]):
    """MCG-TTA (Multi-Scale Consistency-Guided TTA): replayed on cached views.

    Rows with max prob < tau first receive the scale probe views (mirroring
    the MPAC block's multi-dilation paths). If the probes agree with the base
    prediction and each other, the final prob is the mean of identity+probes;
    on disagreement the row escalates to all K views. Non-gated rows keep p0.

    probs_T: (T, K, C) probs for rows idx_T (view 0 = identity).
    Returns (probs (N, C), trigger mask (N,), n_extra_forwards, n_escalated).
    """
    n, _ = p0.shape
    k = probs_T.shape[1]
    conf = p0.max(1)
    pred = p0.argmax(1)
    in_T = np.zeros(n, bool)
    in_T[idx_T] = True
    t_of_row = -np.ones(n, dtype=int)
    t_of_row[idx_T] = np.arange(len(idx_T))

    rows = np.where((conf < tau) & in_T)[0]
    tt = t_of_row[rows]

    probs = p0.copy()
    probes = [i for i, v in enumerate(view_names) if v.startswith("scale")]
    votes = np.stack([probs_T[tt, v].argmax(1) for v in [0] + probes], axis=1)
    esc = (votes != pred[rows][:, None]).any(1) | (votes[:, 1:] != votes[:, :1]).any(1)
    used = [0] + probes
    probs[rows[~esc]] = probs_T[tt[~esc]][:, used].mean(axis=1)
    probs[rows[esc]] = probs_T[tt[esc]].mean(axis=1)
    extra = len(rows) * len(probes) + int(esc.sum()) * (k - 1 - len(probes))
    return probs, (conf < tau) & in_T, int(extra), int(esc.sum())


def run_tta_eval(model, df: pd.DataFrame, cfg, device, mode: str = "none",
                 n_views: int = 5, batch_size: int = 32) -> dict:
    """Plain or full-TTA evaluation over a manifest DataFrame; measures real
    wall-clock timing. mode: "none" (identity view only) or "full" (all views)."""
    views = build_tta_views(cfg)[:max(1, n_views)]
    n = len(df)

    t0 = time.perf_counter()
    imgs = load_raw_images(df)
    t_load = time.perf_counter()
    probs0 = forward_views(model, imgs, views[:1], device, batch_size,
                           desc="Pass 1 (plain)")[:, 0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    probs, trigger = probs0.copy(), np.zeros(n, dtype=bool)
    tta_s = 0.0
    if mode == "full" and len(views) > 1:
        idx = np.arange(n)
        trigger[idx] = True
        t2 = time.perf_counter()
        if len(idx):
            vprobs = forward_views(model, [imgs[i] for i in idx], views[1:],
                                   device, batch_size,
                                   desc=f"TTA pass ({len(idx)} triggered)")
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
    probs = forward_views(model, imgs, views, device, batch_size,
                          desc="All views")
    if device.type == "cuda":
        torch.cuda.synchronize()
    return probs, df, time.perf_counter() - t0
