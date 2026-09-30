"""Plot helpers: confusion matrix, ROC curves, training curves."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(y_true, y_pred, classes, out_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mat, fmt, ttl in [
        (axes[0], cm, "d", title),
        (axes[1], cm_norm, ".2f", title + " (normalized)"),
    ]:
        sns.heatmap(mat, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(ttl)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return cm


def plot_roc_curves(y_true, y_prob, classes, out_path, title="ROC Curves (one-vs-rest)"):
    """y_prob: (N, C) softmax probabilities. Returns dict of per-class AUC."""
    n_classes = len(classes)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if y_bin.shape[1] == 1:  # safety for 2-class edge case
        y_bin = np.hstack([1 - y_bin, y_bin])

    aucs = {}
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        aucs[cls] = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{cls} (AUC={aucs[cls]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    aucs["macro"] = float(np.mean(list(aucs.values())))
    return aucs


def plot_training_curves(history: pd.DataFrame, out_path, title="Training Curves"):
    """history: DataFrame with columns epoch, train_loss, val_loss, val_accuracy, val_f1_macro."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["epoch"], history["val_accuracy"], color="green")
    axes[1].set_title("Val Accuracy")
    axes[2].plot(history["epoch"], history["val_f1_macro"], color="purple")
    axes[2].set_title("Val F1 (macro)")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def overlay_cam(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a [0,1] heatmap on an RGB image ([0,1] float)."""
    import cv2
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    if heat.shape[:2] != image_rgb.shape[:2]:
        heat = cv2.resize(heat, (image_rgb.shape[1], image_rgb.shape[0]))
    return np.clip(image_rgb + alpha * heat, 0, 1)
