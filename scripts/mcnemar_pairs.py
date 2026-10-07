"""McNemar's test between two models' predictions (tasks 9.11, 9.19).

Reads <run>/metrics/<model>_predictions.csv for each model. Use --run-b when the
two checkpoints live in different runs.

    python scripts/mcnemar_pairs.py --a mpac_resnet --b resnet50 ^
        --run 20261005_163732_run_all --run-b 20261001_150620_run_all_ft
    python scripts/mcnemar_pairs.py --a mpac_resnet --b resnet18 --run 20261005_163732_run_all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scripts.compare_models import mcnemar_test
from src.utils.config import load_config
from src.utils.runs import resolve_run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="Model A name")
    p.add_argument("--b", required=True, help="Model B name")
    p.add_argument("--run", default=None, help="Run ref for A (default: latest run)")
    p.add_argument("--run-b", default=None, help="Run ref for B (default: same as A)")
    p.add_argument("--tag", default="test",
                   help="Prediction tag: 'test' -> <model>_predictions.csv, or e.g. 'figshare_all'")
    args = p.parse_args()

    cfg = load_config()
    run_a = resolve_run(cfg, args.run)
    run_b = resolve_run(cfg, args.run_b) if args.run_b else run_a

    ext = "" if args.tag == "test" else f"_{args.tag}"
    path_a = run_a / "metrics" / f"{args.a}{ext}_predictions.csv"
    path_b = run_b / "metrics" / f"{args.b}{ext}_predictions.csv"
    for path in (path_a, path_b):
        if not path.exists():
            sys.exit(f"missing predictions: {path} (run evaluate.py first)")

    def key_col(df: pd.DataFrame) -> pd.Series:
        # filepath roots differ across machines; join on <class>/<filename>
        return df["filepath"].apply(lambda p: str(Path(p).parent.name) + "/" + Path(p).name)

    pred_a = pd.read_csv(path_a)
    pred_a.index = key_col(pred_a)
    pred_b = pd.read_csv(path_b)
    pred_b.index = key_col(pred_b)
    common = pred_a.join(pred_b[["correct"]], rsuffix="_b", how="inner")
    res = mcnemar_test(common["correct"].to_numpy(), common["correct_b"].to_numpy())

    acc_a = common["correct"].mean()
    acc_b = common["correct_b"].mean()
    print(f"{args.a} ({run_a.name}) acc={acc_a:.4f}")
    print(f"{args.b} ({run_b.name}) acc={acc_b:.4f}")
    print(f"McNemar: b={res['b']} c={res['c']} chi2={res['chi2']} "
          f"p={res['p_value']} significant={res['significant']}")


if __name__ == "__main__":
    main()
