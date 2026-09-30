"""Evaluation metrics: accuracy, precision, recall/sensitivity, specificity, F1."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    """Return per-class and macro metrics. Specificity is computed one-vs-rest."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = len(classes)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    # Per-class specificity: TN / (TN + FP)
    specificity = []
    for i in range(n_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp) if (tn + fp) else 0.0)

    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "specificity_macro": float(np.mean(specificity)),
        "f1_macro": float(f1.mean()),
    }
    for i, cls in enumerate(classes):
        metrics[f"{cls}_precision"] = float(precision[i])
        metrics[f"{cls}_recall"] = float(recall[i])
        metrics[f"{cls}_specificity"] = float(specificity[i])
        metrics[f"{cls}_f1"] = float(f1[i])
        metrics[f"{cls}_support"] = int(support[i])
    return metrics


def flatten_metrics_table(all_metrics: dict[str, dict], classes: list[str]):
    """Turn {model: metrics_dict} into a tidy DataFrame for comparison tables."""
    import pandas as pd

    cols = (
        ["accuracy", "precision_macro", "recall_macro", "specificity_macro", "f1_macro"]
        + [f"{c}_{m}" for m in ("precision", "recall", "specificity", "f1") for c in classes]
    )
    rows = {name: {c: m.get(c) for c in cols} for name, m in all_metrics.items()}
    return pd.DataFrame(rows).T.round(4)
