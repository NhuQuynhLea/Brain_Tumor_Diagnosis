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


STRATEGY_NAMES = ("conf", "subject", "weakclass", "disagree", "scalempac")


def union_trigger(p0: np.ndarray, df: pd.DataFrame, taus: list[float],
                  weak_idx: int | None = None) -> np.ndarray:
    """Rows that ANY selective strategy could enter at ANY tau in the sweep,
    so a single views pass on this subset serves every simulation."""
    conf = p0.max(1)
    pred = p0.argmax(1)
    m = conf < max(taus)
    if weak_idx is not None:
        m |= pred == weak_idx
    if "subject" in df.columns:
        sc = pd.Series(conf).groupby(df["subject"].to_numpy()).transform("mean")
        m |= (sc < max(taus)).to_numpy()
    return np.where(m)[0]


def simulate_strategy(probs_T: np.ndarray, idx_T: np.ndarray, p0: np.ndarray,
                      df: pd.DataFrame, tau: float, strategy: str,
                      view_names: list[str], weak_idx: int | None = None):
    """Replay one strategy's gate/escalation on cached per-row view probs.

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

    if strategy == "subject" and "subject" in df.columns:
        subj_conf = pd.Series(conf).groupby(df["subject"].to_numpy()).transform("mean")
        entry = (subj_conf < tau).to_numpy()
    elif strategy == "weakclass":
        entry = (conf < tau) | (pred == weak_idx)
    else:  # conf, disagree, scalempac share the confidence entry gate
        entry = conf < tau
    entry &= in_T
    rows = np.where(entry)[0]
    tt = t_of_row[rows]

    probs = p0.copy()
    n_escalated = 0
    if strategy in ("conf", "subject", "weakclass"):
        probs[rows] = probs_T[tt].mean(axis=1)
        extra = len(rows) * (k - 1)
    elif strategy == "disagree":
        probe = view_names.index("hflip")
        esc = probs_T[tt, probe].argmax(1) != pred[rows]
        probs[rows[~esc]] = probs_T[tt[~esc]][:, [0, probe]].mean(axis=1)
        probs[rows[esc]] = probs_T[tt[esc]].mean(axis=1)
        extra = len(rows) + esc.sum() * (k - 2)
        n_escalated = int(esc.sum())
    elif strategy == "scalempac":
        probes = [i for i, v in enumerate(view_names) if v.startswith("scale")]
        votes = np.stack([probs_T[tt, v].argmax(1) for v in [0] + probes], axis=1)
        esc = (votes != pred[rows][:, None]).any(1) | (votes[:, 1:] != votes[:, :1]).any(1)
        used = [0] + probes
        probs[rows[~esc]] = probs_T[tt[~esc]][:, used].mean(axis=1)
        probs[rows[esc]] = probs_T[tt[esc]].mean(axis=1)
        extra = len(rows) * len(probes) + esc.sum() * (k - 1 - len(probes))
        n_escalated = int(esc.sum())
    else:
        raise ValueError(f"unknown strategy '{strategy}'")
    return probs, entry, int(extra), n_escalated


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
    probs0 = forward_views(model, imgs, views[:1], device, batch_size,
                           desc="Pass 1 (plain)")[:, 0]
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
    probs = forward_views(model, imgs, views, device, batch_size,
                          desc="All views")
    if device.type == "cuda":
        torch.cuda.synchronize()
    return probs, df, time.perf_counter() - t0
