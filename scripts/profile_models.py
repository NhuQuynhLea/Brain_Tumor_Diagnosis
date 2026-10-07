"""Profile params / FLOPs / MACs / latency / FPS for registered models (task 9.11).

Writes results/metrics/complexity_all.csv (paper Table-2 style: Params, MACs, FPS)
and is merged into the accuracy-efficiency figure by paper_figures.py.

    python scripts/profile_models.py                      # all registered models
    python scripts/profile_models.py --models lsnet mpac_resnet resnet50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.evaluation.complexity import measure_flops, measure_inference_time, model_size_mb
from src.models.model_factory import available_models, build_model, count_parameters
from src.utils.config import load_config
from src.utils.tracking import get_logger

logger = get_logger("profile_models", Path("results/logs/profile_models.log"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=None, help="Subset (default: all)")
    args = p.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = args.models or available_models()
    logger.info("Profiling %s on %s", names, device.type)

    rows = {}
    for name in names:
        model = build_model(cfg, name=name)
        counts = count_parameters(model)
        flops = measure_flops(model, (1, 3, cfg.data.image_size, cfg.data.image_size), device)
        ms = measure_inference_time(model, (1, 3, cfg.data.image_size, cfg.data.image_size), device)
        rows[name] = {
            "params_total": counts["total"],
            "size_mb": round(model_size_mb(model), 2),
            "gflops": round(flops / 1e9, 3),
            "gmacs": round(flops / 2e9, 3),
            "inference_ms": ms,
            "fps": round(1000.0 / ms, 1),
        }
        logger.info("%-18s params=%d gmacs=%.3f ms=%.1f fps=%.1f",
                    name, counts["total"], rows[name]["gmacs"], ms, rows[name]["fps"])
        del model

    df = pd.DataFrame(rows).T
    out = Path("results/metrics/complexity_all.csv")
    df.to_csv(out)
    logger.info("Wrote %s", out)
    print(df.to_string())


if __name__ == "__main__":
    main()
