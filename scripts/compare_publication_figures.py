"""Render side-by-side manuscript/exemplar figure review sheets at equal widths."""
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "Kyrkou_Rethinking_Compact_1M_Vision_Models_Balancing_Accuracy_and_Speed_through_CVPRF_2026_paper.pdf"
OURS = ROOT / "LaTeX2e+Proceedings+Template+ZIP/figures"
OUTPUT = ROOT / "results/paper"


def side_by_side(document, filename, page, crop):
    own = Image.open(OURS / f"{filename}.png").convert("RGB")
    excerpt = document[page].get_pixmap(matrix=pymupdf.Matrix(3, 3), clip=pymupdf.Rect(crop), alpha=False)
    exemplar = Image.frombytes("RGB", (excerpt.width, excerpt.height), excerpt.samples)
    width = 1300
    exemplar = exemplar.resize((width, round(exemplar.height * width / exemplar.width)), Image.Resampling.LANCZOS)
    own = own.resize((width, round(own.height * width / own.width)), Image.Resampling.LANCZOS)
    top = 96
    output = Image.new("RGB", (2 * width + 64, max(own.height, exemplar.height) + top + 32), "#e9edf1")
    output.paste(own, (0, top))
    output.paste(exemplar, (width + 64, top))
    pen = ImageDraw.Draw(output)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 36)
    pen.text((22, 20), "OUR MANUSCRIPT", font=font, fill="#192b40")
    pen.text((width + 86, 20), "EXEMPLAR (Kyrkou, CVPRF 2026)", font=font, fill="#192b40")
    target = OUTPUT / f"paired_{filename}.png"
    output.save(target)
    print(target)


def main():
    with pymupdf.open(REFERENCE) as document:
        side_by_side(document, "lsnet_architecture", 3, (58, 72, 554, 196))
        side_by_side(document, "mcg_tta", 4, (58, 72, 295, 247))
    with pymupdf.open(ROOT / "LaTeX2e+Proceedings+Template+ZIP/main.pdf") as paper:
        for name, phrase in (("lsnet_architecture", "Compact multi-scale backbone"),
                             ("mcg_tta", "Selective inference at test time"),
                             ("mri_classification_examples", "Observed MRI classifications"),
                             ("model_class_comparison", "Per-class F1 on brain MRI")):
            for index, page in enumerate(paper):
                if phrase in page.get_text():
                    preview = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                    target = OUTPUT / f"page_{name}.png"
                    preview.save(target)
                    print(f"{target} (page {index + 1})")
                    break
            else:
                raise ValueError(f"Figure absent from PDF: {name}")


if __name__ == "__main__":
    main()
