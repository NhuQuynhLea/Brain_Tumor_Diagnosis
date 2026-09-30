import json
import sys
import time
from pathlib import Path

import yaml


def _plain(obj):
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def use_run(cfg, run_dir) -> Path:
    run_dir = Path(run_dir)
    cfg.paths.checkpoints = run_dir / "models"
    cfg.paths.metrics = run_dir / "metrics"
    cfg.paths.figures = run_dir / "figures"
    cfg.paths.logs = run_dir / "logs"
    cfg.tracking.log_dir = run_dir / "tensorboard"
    return run_dir


def create_run(cfg, kind: str, name: str | None = None) -> Path:
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{kind}" + (f"_{name}" if name else "")
    run_dir = Path(cfg.paths.runs) / run_id
    i = 2
    while run_dir.exists():
        run_dir = run_dir.with_name(f"{run_id}_{i}")
        i += 1
    run_dir.mkdir(parents=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(_plain(cfg), f, sort_keys=False)
    finish_run(run_dir, "running", run_id=run_dir.name, kind=kind, name=name,
               argv=sys.argv, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    use_run(cfg, run_dir)
    return run_dir


def finish_run(run_dir, status: str = "completed", **extra) -> None:
    p = Path(run_dir) / "run_metadata.json"
    meta = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    meta.update(extra)
    meta["status"] = status
    if status != "running":
        meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def resolve_run(cfg, ref: str | None = None) -> Path:
    root = Path(cfg.paths.runs)
    if ref:
        p = Path(ref)
        if p.is_dir():
            return p
        for cand in sorted(root.iterdir()) if root.exists() else []:
            if cand.name == ref or cand.name.endswith(f"_{ref}"):
                return cand
        sys.exit(f"run '{ref}' not found under {root}")
    runs = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    if not runs:
        sys.exit(f"No runs under {root} - run training first.")
    return runs[-1]
