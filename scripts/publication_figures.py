"""Render the manuscript's architecture and selective-inference plates."""
import argparse
import json
from io import BytesIO
from pathlib import Path, PureWindowsPath
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageOps
import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PAPER = ROOT / "LaTeX2e+Proceedings+Template+ZIP"
SAMPLE = ROOT / "data/raw/Testing/glioma/Te-gl_399.jpg"
QUAL_SAMPLE = ROOT / "data/raw/Training/glioma/Tr-gl_700.jpg"
RUN = ROOT / "results/runs/20261007_103027_run_all_lsnet_s42_p15"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
INK = "#192B40"
MUTED = "#506477"
TEAL = "#007F86"
CYAN = "#D9F2F0"
BLUE = "#366DAD"
PALE = "#E5EFFA"
ORANGE = "#B15828"
CREAM = "#FAECDE"
EDGE = "#CFD9DF"
BG = "#FFFFFF"
FONT_SCALE = 0.80
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                     "svg.fonttype": "none", "figure.facecolor": BG,
                     "savefig.facecolor": BG})


def canvas(height):
    fig, ax = plt.subplots(figsize=(8, height))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def text(ax, x, y, s, size=11, color=INK, weight="normal", ha="left", va="center", **kw):
    return ax.text(x, y, s, fontsize=size * FONT_SCALE, color=color, weight=weight, ha=ha,
                   va=va, zorder=20, **kw)


def panel(ax, x, y, w, h, face=BG, edge=EDGE, radius=.10, lw=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=face, edgecolor=edge, linewidth=lw, zorder=1)
    ax.add_patch(p)
    return p


def line(ax, a, b, color=MUTED, width=1.6, arrow=False):
    if arrow:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                     lw=width, color=color, shrinkA=0, shrinkB=0, zorder=8))
    else:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=width, zorder=7)


def mri_tile(ax, image, x, y, side, color=TEAL):
    ax.imshow(image, cmap="gray", extent=(x, x + side, y, y + side), zorder=4,
              interpolation="bilinear", vmin=0, vmax=255)
    ax.add_patch(Rectangle((x, y), side, side, fill=False, ec=color, lw=1.7, zorder=5))


def feature_stack(ax, x, y, w, h, color, layers=3):
    for i in range(layers - 1, -1, -1):
        shift = .035 * i
        ax.add_patch(Rectangle((x + shift, y + shift), w, h, facecolor=BG,
                               edgecolor=color, linewidth=1.2, zorder=4 + layers - i))
        ax.add_patch(Rectangle((x + shift + .06, y + shift + .06), w - .12, h - .12,
                               facecolor=color, alpha=.12 + .10 * (layers - i),
                               edgecolor="none", zorder=4 + layers - i))
        for j in range(3):
            ax.plot([x + shift + w * (.25 + .25 * j)],
                    [y + shift + h * (.25 + .20 * (j % 2))],
                    marker="o", ms=2.5, color=color, zorder=9)


def heading(ax, number, title, descriptor, height):
    text(ax, .26, height - .30, number, 11, TEAL, "bold")
    text(ax, .66, height - .30, title, 16, INK, "bold")
    text(ax, 7.75, height - .31, descriptor, 9.5, MUTED, ha="right")
    line(ax, (.26, height - .57), (7.74, height - .57), EDGE, .9)


def architecture(image):
    fig, ax = canvas(4.85)
    heading(ax, "01", "Compact multi-scale backbone", "224 x 224  |  0.91M parameters", 4.85)
    panel(ax, .25, 2.24, 7.5, 1.92, "#F7FAFC")
    text(ax, .48, 3.92, "INPUT", 9, MUTED, "bold")
    mri_tile(ax, image, .48, 2.73, 1.02)
    text(ax, .99, 2.56, "MRI slice", 10, INK, ha="center")
    line(ax, (1.56, 3.26), (1.85, 3.26), arrow=True)
    feature_stack(ax, 1.91, 2.89, .40, .57, BLUE, 2)
    text(ax, 2.13, 2.56, "Stem", 10, INK, "bold", ha="center")
    text(ax, 2.13, 2.41, "24 ch · 112²", 10, MUTED, ha="center")
    specs = [(2.72, "48", "2", "56", BLUE), (3.69, "96", "4", "28", TEAL),
             (4.67, "192", "4", "14", TEAL), (5.66, "384", "2", "7", BLUE)]
    line(ax, (2.45, 3.26), (2.67, 3.26), arrow=True)
    for idx, (x, width, depth, side, col) in enumerate(specs):
        fw, fh = [( .72, .77), (.60, .66), (.48, .55), (.36, .44)][idx]
        feature_stack(ax, x + (.72 - fw) / 2, 3.22 - fh / 2, fw, fh, col, 3)
        text(ax, x + .35, 2.59, f"{width} ch", 11.2, INK, "bold", ha="center")
        text(ax, x + .35, 2.40, f"×{depth} · {side}²", 10.2, MUTED, ha="center")
        if idx < len(specs) - 1:
            line(ax, (x + (.72 + fw) / 2 + .07, 3.26),
                 (specs[idx + 1][0] - .06, 3.26), arrow=True)
    line(ax, (6.28, 3.26), (6.63, 3.26), arrow=True)
    panel(ax, 6.66, 3.10, .83, .51, PALE, BLUE, .07)
    text(ax, 7.075, 3.36, "1 x 1", 9.5, BLUE, "bold", ha="center")
    text(ax, 7.075, 3.18, "+ BN", 8.3, MUTED, ha="center")
    line(ax, (7.075, 3.08), (7.075, 2.97), BLUE, 1, True)
    panel(ax, 6.66, 2.65, .83, .30, CYAN, TEAL, .04)
    text(ax, 7.075, 2.80, "GAP", 9.4, TEAL, "bold", ha="center")
    text(ax, 7.075, 2.43, "2 / 4 logits", 10.5, MUTED, ha="center")
    panel(ax, .25, .18, 7.5, 1.98, BG, EDGE)
    text(ax, .47, 1.96, "INSIDE ONE MPAC RESIDUAL BLOCK", 10, INK, "bold")
    text(ax, 7.53, 1.96, "STAGES   2 / 4 / 4 / 2", 10.5, MUTED, ha="right")
    text(ax, .56, 1.28, "x", 16, INK, "bold", ha="center")
    ax.add_patch(Circle((.56, 1.28), .27, facecolor=PALE, edgecolor=BLUE, lw=1.2, zorder=2))
    line(ax, (.86, 1.28), (1.20, 1.28))
    line(ax, (1.20, .76), (1.20, 1.43), color=BLUE)
    line(ax, (1.20, 1.43), (1.48, 1.43), color=BLUE, arrow=True)
    line(ax, (1.20, .76), (1.48, .76), color=TEAL, arrow=True)
    for yy, d, col, face in [(1.10, 1, BLUE, PALE), (.43, 2, TEAL, CYAN)]:
        panel(ax, 1.53, yy, 1.50, .65, face, col, .05)
        for ii in range(-2, 3):
            for jj in range(-2, 3):
                ax.add_patch(Circle((1.86 + jj * .085, yy + .33 + ii * .085), .011,
                                    color="#ACB9C2", zorder=9))
        for ii in range(-1, 2):
            for jj in range(-1, 2):
                ax.add_patch(Circle((1.86 + jj * (.085 if d == 1 else .17),
                                     yy + .33 + ii * (.085 if d == 1 else .17)), .025,
                                    color=col, zorder=10))
        radius = .11 if d == 1 else .195
        ax.add_patch(Rectangle((1.86 - radius, yy + .33 - radius),
                               2 * radius, 2 * radius, fill=False, edgecolor=col,
                               linewidth=.85, zorder=9))
        text(ax, 2.20, yy + .41, f"DW d={d}", 10.2, col, "bold")
        text(ax, 2.20, yy + .16, f"RF {2 * d + 1} x {2 * d + 1}", 9.5, MUTED)
        line(ax, (3.04, yy + .33), (3.37, yy + .33), col, arrow=True)
    panel(ax, 3.42, .61, .96, 1.15, "#F1F5F7", EDGE, .06)
    text(ax, 3.90, 1.36, "concat", 10, INK, "bold", ha="center")
    text(ax, 3.90, 1.11, "BN + LReLU", 8.0, MUTED, ha="center")
    line(ax, (4.41, 1.28), (4.69, 1.28), arrow=True)
    panel(ax, 4.75, .84, .99, .89, PALE, BLUE, .06)
    text(ax, 5.245, 1.38, "1 x 1", 11, BLUE, "bold", ha="center")
    text(ax, 5.245, 1.12, "mix + BN", 8.6, MUTED, ha="center")
    line(ax, (5.77, 1.28), (6.34, 1.28), arrow=True)
    ax.add_patch(Circle((6.48, 1.28), .15, facecolor=BG, edgecolor=INK, lw=1.3, zorder=10))
    text(ax, 6.48, 1.28, "+", 13, INK, ha="center")
    line(ax, (6.64, 1.28), (7.26, 1.28), arrow=True)
    text(ax, 6.96, 1.55, "LReLU", 10, MUTED, ha="center")
    text(ax, 7.34, 1.28, "y", 16, INK, "bold")
    line(ax, (.56, .95), (.56, .37), BLUE, 1.1)
    line(ax, (.56, .37), (3.58, .37), BLUE, 1.1)
    projection = panel(ax, 3.58, .23, .82, .30, BG, BLUE, .04)
    projection.set_linestyle((0, (3, 2)))
    text(ax, 3.99, .38, "1 x 1", 8.3, BLUE, "bold", ha="center")
    line(ax, (4.40, .37), (6.48, .37), BLUE, 1.1)
    line(ax, (6.48, .37), (6.48, 1.10), BLUE, 1.1, True)
    text(ax, .99, .60, "skip", 8.6, BLUE, "bold", ha="center")
    return fig


def scaled_view(image, scale):
    original = Image.fromarray(image).convert("L")
    side = original.width
    if scale > 1:
        new = int(round(side * scale))
        return np.asarray(ImageOps.fit(original.resize((new, new), Image.Resampling.BICUBIC),
                                       (side, side), method=Image.Resampling.BICUBIC))
    inner = int(round(side * scale))
    reduced = original.resize((inner, inner), Image.Resampling.BICUBIC)
    result = Image.new("L", (side, side), 0)
    result.paste(reduced, ((side - inner) // 2, (side - inner) // 2))
    return np.asarray(result)


def inference(image):
    fig, ax = canvas(6.25)
    ax.set_ylim(-1.75, 4.50)
    heading(ax, "04", "Selective inference at test time", "one shared LS-Net  |  no retraining", 4.50)
    panel(ax, .25, 2.62, 7.5, 1.13, "#F7FAFC")
    mri_tile(ax, image, .44, 2.84, .68)
    text(ax, 1.27, 3.36, "Identity", 11, INK, "bold")
    text(ax, 1.27, 3.08, "p₀ = softmax(LS-Net(x))", 10, MUTED)
    line(ax, (3.05, 3.19), (3.37, 3.19), arrow=True)
    panel(ax, 3.44, 2.87, 1.72, .66, PALE, BLUE, .07)
    text(ax, 4.30, 3.31, "confidence gate", 10.6, BLUE, "bold", ha="center")
    text(ax, 4.30, 3.08, "max p₀ < τ ?", 11, INK, ha="center")
    line(ax, (5.18, 3.19), (5.49, 3.19), arrow=True)
    text(ax, 5.335, 3.42, "No", 9, TEAL, "bold", ha="center")
    panel(ax, 5.57, 2.87, 1.96, .66, CYAN, TEAL, .07)
    text(ax, 6.55, 3.31, "retain p₀", 11, TEAL, "bold", ha="center")
    text(ax, 6.55, 3.07, "one forward", 9.4, MUTED, ha="center")
    line(ax, (4.30, 2.86), (4.30, 2.36), BLUE, 1.6, True)
    text(ax, 4.51, 2.51, "Yes", 9, BLUE, "bold")
    panel(ax, .25, .17, 4.78, 2.19, BG, EDGE)
    text(ax, .47, 2.16, "SCALE-CONSISTENCY PROBE", 10, INK, "bold")
    text(ax, 4.76, 2.16, "2 extra forwards", 10.2, TEAL, "bold", ha="right")
    mri_tile(ax, image, .52, 1.00, .68, INK)
    text(ax, .86, .78, "identity", 9.7, INK, "bold", ha="center")
    mri_tile(ax, scaled_view(image, .9), 2.12, 1.36, .59, TEAL)
    line(ax, (2.17, 1.83), (2.36, 1.83), "#F3CE77", 1.5, True)
    line(ax, (2.66, 1.83), (2.47, 1.83), "#F3CE77", 1.5, True)
    text(ax, 2.415, 1.22, "0.9× (−10%)", 10.5, TEAL, "bold", ha="center")
    mri_tile(ax, scaled_view(image, 1.1), 2.12, .55, .59, TEAL)
    line(ax, (2.40, 1.03), (2.17, 1.03), "#F3CE77", 1.5, True)
    line(ax, (2.44, 1.03), (2.67, 1.03), "#F3CE77", 1.5, True)
    text(ax, 2.415, .39, "1.1× (+10%)", 10.5, TEAL, "bold", ha="center")
    line(ax, (1.21, 1.34), (1.52, 1.34), TEAL, 1.2)
    line(ax, (1.52, .85), (1.52, 1.66), TEAL, 1.2)
    line(ax, (1.52, 1.66), (2.07, 1.66), TEAL, 1.2, True)
    line(ax, (1.52, .85), (2.07, .85), TEAL, 1.2, True)
    line(ax, (1.53, 1.32), (3.12, 1.32), MUTED, 1.0, True)
    line(ax, (2.75, 1.66), (3.12, 1.60), TEAL, 1.1, True)
    line(ax, (2.75, .85), (3.12, 1.02), TEAL, 1.1, True)
    panel(ax, 3.18, .85, 1.60, .99, PALE, BLUE, .06)
    text(ax, 3.98, 1.66, "shared LS-Net", 10.4, BLUE, "bold", ha="center")
    text(ax, 3.98, 1.39, "argmax p₋, p₊", 10.4, INK, ha="center")
    text(ax, 3.98, 1.10, "both match ŷ₀ ?", 10.4, INK, ha="center")
    text(ax, .50, .47, "one shared LS-Net", 10.2, MUTED)
    line(ax, (4.82, 1.55), (5.35, 1.84), TEAL, 1.3, True)
    text(ax, 5.04, 1.91, "yes", 8.4, TEAL, "bold")
    line(ax, (4.82, 1.05), (5.35, .74), ORANGE, 1.3, True)
    text(ax, 5.08, .97, "no", 8.4, ORANGE, "bold")
    panel(ax, 5.38, 1.37, 2.36, .96, CYAN, TEAL, .08)
    text(ax, 5.58, 2.10, "AGREE", 9.3, TEAL, "bold")
    text(ax, 5.58, 1.81, "mean of 3 views", 11, INK, "bold")
    text(ax, 5.58, 1.55, "identity + scale pair", 8.7, MUTED)
    panel(ax, 5.38, .27, 2.36, .96, CREAM, ORANGE, .08)
    text(ax, 5.58, .99, "DISAGREE", 9.3, ORANGE, "bold")
    text(ax, 5.58, .71, "acquire 3 more views", 10.5, INK, "bold")
    text(ax, 5.58, .45, "complete the 6-view set", 8.7, MUTED)
    panel(ax, .25, -.90, 7.5, 1.02, "#F7FAFC", EDGE)
    text(ax, .47, -.12, "CONDITIONAL VIEW BANK", 11, INK, "bold")
    text(ax, .47, -.58, "mean of 6 views, +3 forwards", 10.7, MUTED)
    source = Image.fromarray(image)
    transforms = [(ImageOps.mirror(source), "flip"),
                  (source.rotate(15, resample=Image.Resampling.BICUBIC), "+15°"),
                  (source.rotate(-15, resample=Image.Resampling.BICUBIC), "−15°")]
    for i, (im, label) in enumerate(transforms):
        xx = 3.00 + i * 1.54
        mri_tile(ax, np.asarray(im), xx, -.78, .72, ORANGE)
        text(ax, xx + .83, -.21, ("↔", "↺", "↻")[i], 18, ORANGE, "bold")
        text(ax, xx + .83, -.56, label, 11, ORANGE, "bold")
    line(ax, (6.86, .26), (6.86, .12), ORANGE, 1.3, True)
    panel(ax, .25, -1.64, 7.50, .52, PALE, BLUE, .07)
    text(ax, .51, -1.38, "FINAL DECISION", 11, BLUE, "bold")
    text(ax, 4.05, -1.38, "one selected probability", 11, INK, "bold", ha="right")
    line(ax, (4.20, -1.38), (4.72, -1.38), BLUE, 1.4, True)
    text(ax, 4.90, -1.38, "argmax grade", 11, INK, "bold")
    for start in ((7.53, 3.19), (7.74, 1.84), (7.75, -.44)):
        line(ax, start, (7.93, start[1]), MUTED, 1.05, True)
        ax.add_patch(Circle((7.93, start[1]), .022, facecolor=MUTED, zorder=10))
    line(ax, (7.93, 3.19), (7.93, -1.38), MUTED, 1.05)
    line(ax, (7.93, -1.38), (7.75, -1.38), MUTED, 1.2, True)
    return fig


def raw_relative_path(saved_path):
    parts = PureWindowsPath(saved_path).parts
    if "data" not in parts:
        raise ValueError(f"Prediction has no dataset path: {saved_path}")
    data_index = parts.index("data")
    if parts[data_index:data_index + 2] != ("data", "raw"):
        raise ValueError(f"Unexpected prediction dataset: {saved_path}")
    return PureWindowsPath(*parts[data_index:])


def local_image_path(saved_path):
    path = ROOT.joinpath(*raw_relative_path(saved_path).parts)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_predictions():
    path = RUN / "metrics/lsnet_predictions.csv"
    df = pd.read_csv(path)
    manifest = pd.read_csv(ROOT / "data/processed/test.csv")
    if len(df) != 1052 or set(df["label"]) != set(CLASS_NAMES):
        raise ValueError("The selected Nickparvar test predictions are incomplete")
    expected = [(raw_relative_path(r.filepath).as_posix().lower(), r.label, int(r.label_idx))
                for r in manifest.itertuples(index=False)]
    observed = [(raw_relative_path(r.filepath).as_posix().lower(), r.label, int(r.label_idx))
                for r in df.itertuples(index=False)]
    if observed != expected:
        raise ValueError("Prediction rows differ from the held-out test manifest")
    if not np.array_equal(df["correct"].to_numpy(),
                          (df["label_idx"] == df["y_pred"]).to_numpy()):
        raise ValueError("Prediction correctness disagrees with class indices")
    df["confidence"] = df.apply(lambda r: r[f"prob_{r['pred_label']}"], axis=1)
    probabilities = df[[f"prob_{c}" for c in CLASS_NAMES]].to_numpy()
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4):
        raise ValueError("The stored probability vectors do not sum to one")
    counts = np.zeros((4, 4), dtype=int)
    np.add.at(counts, (df["label_idx"], df["y_pred"]), 1)
    return df, counts


def image_in_square(ax, image, x, y, side):
    h, w = image.shape[:2]
    scale = side / max(w, h)
    shown_w, shown_h = w * scale, h * scale
    ax.add_patch(Rectangle((x, y), side, side, facecolor="#0B1018", edgecolor="none", zorder=3))
    ax.imshow(image, extent=(x + (side - shown_w) / 2, x + (side + shown_w) / 2,
                             y + (side - shown_h) / 2, y + (side + shown_h) / 2),
              zorder=4, interpolation="bilinear")
    ax.add_patch(Rectangle((x, y), side, side, fill=False, edgecolor=EDGE, lw=1.1, zorder=5))
    assert abs(shown_w / shown_h - w / h) < 1e-12


def example_rows(predictions):
    sample_key = raw_relative_path(str(QUAL_SAMPLE)).as_posix().lower()
    matching = predictions[predictions["filepath"].map(
        lambda p: raw_relative_path(p).as_posix().lower() == sample_key)]
    if len(matching) != 1 or matching.iloc[0]["label"] != "glioma" or not matching.iloc[0]["correct"]:
        raise ValueError("The illustrative glioma is not a verified correct test prediction")
    errors = predictions[(predictions["label"] == "notumor") & ~predictions["correct"]].copy()
    if errors.empty:
        raise ValueError("No verified no-tumor false positive is available")
    errors["distance"] = (errors["confidence"] - .50).abs()
    selected = [matching.iloc[0], errors.sort_values(["distance", "filepath"]).iloc[0]]
    return [(row, local_image_path(row["filepath"])) for row in selected]


def qualitative_examples(predictions):
    from src.data.preprocessing import build_eval_transform
    from src.evaluation.gradcam import GradCAM
    from src.models.litespeed_net import build_lsnet
    from src.training.checkpoint import load_checkpoint
    from src.utils.config import Config

    examples = example_rows(predictions)
    model = build_lsnet(num_classes=len(CLASS_NAMES)).eval()
    checkpoint = load_checkpoint(RUN / "models/lsnet/best.pt", model, device="cpu")
    cfg = Config.wrap(checkpoint["config"])
    if cfg.model.name != "lsnet" or cfg.project.seed != 42 or cfg.training.early_stopping_patience != 15:
        raise ValueError("Checkpoint metadata does not match the selected LS-Net run")
    transform = build_eval_transform(cfg)
    cam = GradCAM(model, target_layer=model.stages[2][-1].mix[2])
    fig, ax = canvas(6.20)
    heading(ax, "03", "Observed MRI classifications", "LS-Net  ·  Nickparvar  ·  seed 42", 6.20)
    for i, (row, path) in enumerate(examples):
        raw = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        tensor = transform(image=raw)["image"].unsqueeze(0)
        with torch.no_grad():
            computed = torch.softmax(model(tensor), dim=1)[0].numpy()
        stored = row[[f"prob_{c}" for c in CLASS_NAMES]].to_numpy(dtype=float)
        if not np.allclose(computed, stored, atol=1e-3) or int(computed.argmax()) != int(row["y_pred"]):
            raise ValueError(f"Checkpoint predictions differ from stored test probabilities: {path}; "
                             f"recorded={stored.round(4)}, recomputed={computed.round(4)}")
        heat = cam.generate(tensor, int(row["y_pred"]))
        heat = cv2.resize(heat, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_LINEAR)
        opacity = np.clip((heat - .22) / .75, 0, .56)[..., None]
        colour = plt.get_cmap("inferno")(heat)[..., :3]
        overlay = np.uint8(np.clip(raw / 255 * (1 - opacity) + colour * opacity, 0, 1) * 255)
        x = .25
        y = 3.09 if i == 0 else .43
        accent = TEAL if row["correct"] else ORANGE
        panel(ax, x, y, 7.50, 2.43, "#F7FAFC", EDGE)
        text(ax, x + 4.78, y + 2.19, f"{i + 1:02d}  /  {'CORRECT' if row['correct'] else 'ERROR'}",
             11.5, accent, "bold")
        image_in_square(ax, raw, x + .18, y + .31, 2.06)
        image_in_square(ax, overlay, x + 2.42, y + .31, 2.06)
        text(ax, x + 1.21, y + .13, "ORIGINAL MRI", 10, MUTED, "bold", ha="center")
        text(ax, x + 3.45, y + .13, "GRAD-CAM OVERLAY", 10, MUTED, "bold", ha="center")
        text(ax, x + 4.78, y + 1.91,
             f"TRUE   {row['label'].replace('notumor', 'no tumor')}", 11, INK)
        text(ax, x + 4.78, y + 1.64,
             f"PRED   {row['pred_label'].replace('notumor', 'no tumor')}",
             11, accent, "bold")
        text(ax, x + 4.78, y + 1.35, "SOFTMAX OUTPUT", 11.5, MUTED, "bold")
        for j, cls in enumerate(CLASS_NAMES):
            yy = y + 1.08 - j * .24
            value = float(row[f"prob_{cls}"])
            text(ax, x + 4.78, yy, cls.replace("notumor", "no tumor"), 12, MUTED)
            ax.add_patch(Rectangle((x + 5.98, yy - .06), .66, .12,
                                   facecolor=EDGE, zorder=3))
            ax.add_patch(Rectangle((x + 5.98, yy - .06), .66 * value, .12,
                                   facecolor=accent if cls == row["pred_label"] else BLUE,
                                   zorder=4))
            text(ax, x + 6.75, yy, f"{value * 100:.1f}%", 12.5, INK,
                 "bold" if cls == row["pred_label"] else "normal")
    cam.close()
    text(ax, .30, .16, "Extracranial Grad-CAM signal may reflect background, not a lesion.",
         10, MUTED)
    return fig


def model_comparison():
    values = pd.read_csv(ROOT / "results/paper/per_class_f1.csv", index_col=0)
    runs = pd.read_csv(ROOT / "results/paper/per_run_results.csv")
    models = [("resnet50", "finetune", "R50", "ResNet-50", "#ADB8C2"),
              ("efficientnetb0", "finetune", "EB0", "EfficientNet-B0", "#D3D9DE"),
              ("mpac_resnet_tiny", "scratch", "MP", "MPAC-ResNet-Tiny", "#C0CAD2"),
              ("lsnet", "scratch", "LS", "LS-Net", TEAL)]
    deviations = {}
    for model, variant, _, _, _ in models:
        key = f"{model}-{variant}"
        group = runs[(runs["model"] == model) & (runs["variant"] == variant)]
        if key not in values or set(group["seed"]) != {42, 123, 456} or len(group) != 3:
            raise ValueError(f"Expected three verified runs for {key}")
        per_seed = np.array([[json.loads((ROOT / Path(path)).read_text())[f"{cls}_f1"] * 100
                              for cls in CLASS_NAMES] for path in group["metrics_json"]])
        if not np.allclose(per_seed.mean(axis=0), values[key].to_numpy(), atol=.02):
            raise ValueError(f"Figure values disagree with seed-level results for {key}")
        deviations[key] = per_seed.std(axis=0, ddof=1)
    fig, ax = canvas(3.15)
    fig.set_size_inches(6.4, 2.52)
    text(ax, .31, 2.91, "Per-class F1 on brain MRI", 13, INK, "bold")
    text(ax, 7.70, 2.91, "Nickparvar test · mean ± SD, 3 seeds", 10, MUTED, ha="right")
    line(ax, (.31, 2.70), (7.69, 2.70), EDGE, .8)
    low, span, base, plot_height = 94, 6, .94, 1.23
    text(ax, .24, base + plot_height, "100", 9.5, MUTED, ha="center")
    text(ax, .24, base, "94", 9.5, MUTED, ha="center")
    for k, cls in enumerate(CLASS_NAMES):
        x = .42 + k * 1.87
        text(ax, x + .83, 2.49, cls.replace("notumor", "No tumor").title(),
             11.5, INK, "bold", ha="center")
        for tick in [94, 97, 100]:
            yy = base + (tick - low) / span * plot_height
            line(ax, (x + .11, yy), (x + 1.66, yy), "#E4E8EB", .65)
        if k:
            line(ax, (x - .09, .80), (x - .09, 2.48), EDGE, .7)
        for j, (model, variant, badge, _, colour) in enumerate(models):
            v = float(values.loc[cls, f"{model}-{variant}"])
            if not 94 <= v <= 100:
                raise ValueError(f"Out-of-range class F1 for {model}, {cls}: {v}")
            centre = x + .24 + j * .42
            top = base + (v - low) / span * plot_height
            ax.add_patch(Rectangle((centre - .115, base), .23, top - base,
                                   facecolor=colour, edgecolor="none", zorder=4))
            err = float(deviations[f"{model}-{variant}"][k]) / span * plot_height
            line(ax, (centre, top - err), (centre, top + err),
                 INK if model == "lsnet" else MUTED, 1.0)
            line(ax, (centre - .045, top + err), (centre + .045, top + err),
                 INK if model == "lsnet" else MUTED, 1.0)
            if model in {"resnet50", "lsnet"}:
                text(ax, centre, top + err + .13, f"{v:.1f}", 10.2,
                     TEAL if model == "lsnet" else MUTED,
                     "bold" if model == "lsnet" else "normal", ha="center")
            text(ax, centre, .74, badge, 10.6,
                 TEAL if model == "lsnet" else MUTED,
                 "bold" if model == "lsnet" else "normal", ha="center")
    for x, (_, _, badge, full, colour) in zip((.45, 2.29, 4.20, 6.55), models):
        ax.add_patch(Rectangle((x, .42), .11, .11, facecolor=colour, edgecolor="none"))
        text(ax, x + .18, .475, f"{badge} {full}", 9.6,
             INK if badge == "LS" else MUTED, "bold" if badge == "LS" else "normal")
    text(ax, .31, .17, "Shared 94–100% scale (truncated origin); whiskers: ±1 SD across three seeds.",
         9.7, MUTED)
    return fig


def review_candidates(predictions):
    from src.data.preprocessing import build_eval_transform
    from src.evaluation.gradcam import GradCAM
    from src.models.litespeed_net import build_lsnet
    from src.training.checkpoint import load_checkpoint
    from src.utils.config import Config

    rows = predictions[(predictions["label"] == "glioma") & predictions["correct"]]
    rows = rows.sort_values(["confidence", "filepath"]).iloc[
        np.linspace(0, len(rows) - 1, 12, dtype=int)]
    model = build_lsnet(num_classes=len(CLASS_NAMES)).eval()
    cfg = Config.wrap(load_checkpoint(RUN / "models/lsnet/best.pt", model)["config"])
    transform = build_eval_transform(cfg)
    cam = GradCAM(model, target_layer=model.stages[2][-1].mix[2])
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, (_, row) in zip(axes.flat, rows.iterrows()):
        path = local_image_path(row["filepath"])
        raw = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        tensor = transform(image=raw)["image"].unsqueeze(0)
        heat = cam.generate(tensor, int(row["y_pred"]))
        heat = cv2.resize(heat, (raw.shape[1], raw.shape[0]))
        alpha = np.clip((heat - .22) / .75, 0, .56)[..., None]
        colour = plt.get_cmap("inferno")(heat)[..., :3]
        overlay = np.uint8(np.clip(raw / 255 * (1 - alpha) + colour * alpha, 0, 1) * 255)
        ax.imshow(np.concatenate((raw, overlay), axis=1))
        ax.set_title(f"{path.name}  ·  {row['confidence'] * 100:.1f}%", fontsize=11)
        ax.axis("off")
    cam.close()
    fig.tight_layout()
    target = ROOT / "results/paper/glioma_review_candidates.png"
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=PAPER / "figures")
    parser.add_argument("--review-candidates", action="store_true")
    args = parser.parse_args()
    if args.review_candidates:
        predictions, _ = load_predictions()
        review_candidates(predictions)
        return
    if not SAMPLE.is_file():
        raise FileNotFoundError(SAMPLE)
    args.out.mkdir(parents=True, exist_ok=True)
    image = np.asarray(Image.open(SAMPLE).convert("L").resize((224, 224)))
    predictions, _ = load_predictions()
    figures = [("lsnet_architecture", architecture(image)),
               ("mcg_tta", inference(image)),
               ("mri_classification_examples", qualitative_examples(predictions)),
               ("model_class_comparison", model_comparison())]
    for name, figure in figures:
        figure.savefig(args.out / f"{name}.pdf", dpi=300)
        render_a, render_b = BytesIO(), BytesIO()
        figure.savefig(render_a, format="png", dpi=300)
        figure.savefig(render_b, format="png", dpi=300)
        if render_a.getvalue() != render_b.getvalue():
            raise AssertionError(f"Non-deterministic raster: {name}")
        (args.out / f"{name}.png").write_bytes(render_a.getvalue())
        plt.close(figure)
        print(args.out / f"{name}.pdf")


if __name__ == "__main__":
    main()
