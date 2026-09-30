"""Model complexity: parameter count, FLOPs, model size, inference latency."""

import time
from pathlib import Path

import torch
import torch.nn as nn


def measure_flops(model: nn.Module, input_shape=(1, 3, 224, 224), device="cpu") -> float:
    """FLOPs per forward pass using torch's native flop counter."""
    from torch.utils.flop_counter import FlopCounterMode

    model = model.to(device).eval()
    x = torch.zeros(input_shape, device=device)
    with FlopCounterMode(display=False) as counter:
        model(x)
    return float(counter.get_total_flops())


def model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)


def measure_inference_time(model: nn.Module, input_shape=(1, 3, 224, 224),
                           device="cpu", n_warmup=5, n_runs=30) -> float:
    """Median per-image inference latency in milliseconds."""
    model = model.to(device).eval()
    x = torch.randn(input_shape, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return float(sorted(times)[len(times) // 2] * 1000)


def profile_model(model: nn.Module, device="cpu", image_size=224,
                  checkpoint_path: Path | None = None) -> dict:
    from src.models.model_factory import count_parameters

    counts = count_parameters(model)
    result = {
        "params_total": counts["total"],
        "params_trainable": counts["trainable"],
        "size_mb": round(model_size_mb(model), 2),
        "gflops": round(measure_flops(model, (1, 3, image_size, image_size), device) / 1e9, 3),
        "inference_ms": round(measure_inference_time(model, (1, 3, image_size, image_size), device), 2),
    }
    if checkpoint_path and Path(checkpoint_path).exists():
        result["checkpoint_mb"] = round(Path(checkpoint_path).stat().st_size / (1024 ** 2), 2)
    return result
