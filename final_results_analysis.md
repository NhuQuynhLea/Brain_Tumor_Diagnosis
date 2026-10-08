# Final Results Analysis — Brain Tumor MRI Classification

*Status: 2026-10-08 (updated after `20261008_154220_run_all_brats_ft_s123`). Covers every run in `results/experiments.csv` (207 rows) and `results/runs/` (27 run dirs).*
*All numbers below were **recomputed from the raw per-run metrics**, not copied from `results/paper/` (that folder is stale/biased; see §1).*

---

## 0. TL;DR

| Question | Answer |
|---|---|
| Best model, Nickparvar 4-class | **ResNet50-FT 99.14 ± 0.16 % acc** (3 seeds). Statistically tied with **EfficientNetB0-FT 99.08 ± 0.28** (McNemar p ≥ 0.50 on all 3 seeds), which uses 5.9× fewer params and 10.6× fewer GFLOPs |
| Best sub-1M model, Nickparvar | **LS-Net 98.29 ± 0.25** (914K) ≈ **MPAC-ResNet-Tiny 98.22 ± 0.30** (717K). Both are within about 0.9 pp of ResNet50-FT with 26–33× fewer params |
| Best model on BraTS HGG/LGG grading | **VGG16-FT**, top on both seeds so far: slice F1 **89.31 ± 0.06**, AUC 96.25, subject-level F1 92.34, 11/12 LGG patients caught on each seed. Seeds 42 + 123; seed 456 is pending |
| Does the proposed MPAC-ResNet beat its ResNet-18 baseline? | **No on Nickparvar** (−0.60 pp, significantly worse on 1 of 3 seeds). **Not consistently on BraTS**: +3.5 F1 at s42 but −1.2 at s123 (2-seed mean +1.2, within noise). The comparison is also confounded (§4.1) |
| Is the dataset saturated? | **Yes.** All 13 architectures land between 95.5 and 99.1 %, and 12 of them between 97.4 and 99.1 %. BraTS separates the models by about 20 F1 points (FT protocol, 2-seed means 68.9–89.3) |
| Are the results paper-ready? | **Not yet.** Issues 1 and 3 in §1 are fixed; the `paper_figures.py` grouping bug (§1 #2) still means `results/paper/` is stale — do not re-run it until that's patched. BraTS has only 2 of 3 seeds (FT protocol only), and the literature/SOTA table and the write-up (Phases 1 and 8) are still missing |

---

## 1. Data-integrity findings (fix before writing the paper)

| # | Issue | Evidence | Impact | Fix |
|---|---|---|---|---|
| 1 | ~~**20 exact-duplicate rows in `experiments.csv`**~~ **RESOLVED 2026-10-08** | Lines 45–64 repeated lines 20–39 (FT seeds 123/456 + Custom CNN seeds 123/456). Deduplicated (187 data rows) + 2 blank lines removed | `results/paper/summary_mean_std.csv` still reports **n=5** instead of 3 (stale file, do not trust); correct values in §3 | Done: CSV deduped, `load_eval_rows` now dedupes on `["run_id","model","source","split"]`. The stale `results/paper/` outputs still need regeneration after fix #2 |
| 2 | **`paper_figures.py` groups by `(model, variant, epochs)` only** | No `patience` and no dataset key in the grouping | (a) LS-Net/Tiny summaries mix patience-7 and patience-15 runs. The current file shows LS-Net 97.65 ± 1.18 (the p7 runs), but the protocol-consistent p15 runs give **98.29 ± 0.25**. The Tiny s42 p15 reruns are missing from `per_run_results.csv`. (b) If you re-run it now, **BraTS rows (`split=test`) get merged into the Nickparvar stats**, e.g. custom_cnn 30-epoch. (c) Scratch models (Tiny, LS-Net) get labelled `freeze` because the config default is `freeze_backbone: true`, which those builders ignore | Add a `dataset` column (or filter on run_id containing `brats`), group by patience, and label models without `pretrained` support as `scratch`. Then regenerate `results/paper/*` |
| 3 | ~~**BraTS-FT run is only partially synced**~~ **RESOLVED 2026-10-08** | The run dir has been re-synced: 183 files / 1.75 GB, 0 zero-byte files, all 39 figures, all 13 checkpoints + history.csv, 26 tensorboard files, `vgg16_predictions.csv` = 1,400 rows | None — predictions, errors, figures and checkpoints are all usable now | Root cause was an aborted transfer (likely disk-full — D: had ~5 GB free). Free space before the s456 sync |
| 4 | `models/<name>/best.pt` at the repo root gets overwritten by every run | Root checkpoints come from the latest run that trained each model | The root checkpoints are not "the Nickparvar models" anymore | Treat `results/runs/<run>/models/` as the canonical checkpoints |
| 5 | One broken Figshare eval row | `experiments.csv` line 43: vgg16 acc 1.0, spec 0.0, AUC nan | Superseded by line 44 | Drop it |
| 6 | 9.19 McNemar (MPAC-ResNet vs LS-Net) output not in `results/` | Only the numbers quoted in `implementation_plan.md` exist | Its s123 "significant" result used the **p7 LS-Net s123 run (96.29 %, the outlier)** | Re-run `mcnemar_pairs.py` against the p15 LS-Net runs |
| 7 | `prepare_brats.py` has uncommitted changes; `results/prepare_brats.log` comes from a fake 20-subject smoke test | `git status` | The real BraTS prep log lives only on the GPU machine | Commit the script and copy the real log |

> Note: `results/` and `models/` are git-ignored. "Pushed" results exist only locally, not in the repo.

---

## 2. Experimental setup

| | Nickparvar (primary) | BraTS 2020 (generalization) |
|---|---|---|
| Task | 4-class: glioma / meningioma / notumor / pituitary | 2-class glioma grading: HGG / LGG |
| Data | 7,200 → 7,013 after removing 187 md5 duplicates; stratified 70/15/15 (4,909 / 1,052 / 1,052) | 369 patients; **subject-level** 70/15/15 split; ≤25 seg-guided axial slices per subject; [FLAIR, T1ce, T2] stacked as RGB |
| Test set | 1,052 images, balanced | **1,400 slices / 56 subjects (44 HGG, 12 LGG)** |
| Imbalance | none | ~4:1, handled with class-weighted CE |
| Recipe | AdamW, cosine, 224², batch 32, early stop on val F1-macro | same |
| Seeds | 42 / 123 / 456 | FT protocol: **42 + 123** (456 pending). Frozen / lr 1e-3 protocol: 42 only |
| Metrics | Acc, Prec, Sens, Spec, F1 (macro + per-class), AUC, McNemar | same + subject-level (mean-prob vote) |

Training variants:

| Variant | Applies to | Settings |
|---|---|---|
| Freeze | VGG16, ResNet18/50, EfficientNetB0, MobileNetV2 | Head only (MPAC-ResNet: stem + stages 1–2 frozen), lr 1e-3, 30 epochs, patience 7 |
| Fine-tune (FT) | Same models | Full network, lr 1e-4, 30 epochs, patience 7 |
| Scratch | Custom CNN, all `*_tiny` models, LS-Net | No pretrained weights exist for these builders, so the `pretrained` and `freeze_backbone` flags in the CSV are meaningless. Nickparvar: 50 epochs, lr 1e-3, patience 15 (Custom CNN: patience 7). BraTS: 30 epochs, lr 1e-3 or 1e-4 |

---

## 3. Nickparvar results (4-class)

### 3.1 Main table (corrected, 3 seeds each, mean ± sample-std, %)

| Model | Variant | Params | GFLOPs | CPU FPS | Accuracy | F1-macro | Sensitivity | Specificity | AUC |
|---|---|---|---|---|---|---|---|---|---|
| **ResNet50** | FT | 23.52M | 8.17 | 14.7 | **99.14 ± 0.16** | **99.15 ± 0.17** | 99.15 ± 0.16 | 99.71 ± 0.06 | 99.92 ± 0.05 |
| EfficientNetB0 | FT | 4.01M | 0.77 | 41.8 | 99.08 ± 0.28 | 99.08 ± 0.27 | 99.09 ± 0.27 | 99.70 ± 0.09 | **99.97 ± 0.02** |
| VGG16 | FT | 27.56M | 30.72 | 6.7 | 98.86 ± 0.10 | 98.86 ± 0.10 | 98.86 ± 0.10 | 99.62 ± 0.03 | 99.94 ± 0.03 |
| ResNet18 | FT | 11.18M | 3.63 | 35.8 | 98.70 ± 0.47 | 98.70 ± 0.47 | 98.71 ± 0.47 | 99.57 ± 0.16 | 99.94 ± 0.02 |
| MobileNetV2 | FT | 2.23M | 0.60 | 57.6 | 98.45 ± 0.33 | 98.45 ± 0.33 | 98.46 ± 0.33 | 99.48 ± 0.11 | 99.95 ± 0.03 |
| **LS-Net** | scratch, p15 | 0.91M | 0.36 | 88.5 | **98.29 ± 0.25** | 98.29 ± 0.25 | 98.30 ± 0.25 | 99.43 ± 0.08 | 99.79 ± 0.08 |
| **MPAC-ResNet-Tiny** | scratch, p15 | 0.72M | 0.92 | 55.5 | 98.22 ± 0.30 | 98.22 ± 0.30 | 98.23 ± 0.30 | 99.41 ± 0.11 | 99.84 ± 0.03 |
| EfficientNet-Tiny | scratch, p15 | 0.75M | 0.19 | 61.9 | 98.13 ± 0.38 | 98.14 ± 0.38 | 98.15 ± 0.38 | 99.38 ± 0.13 | 99.85 ± 0.04 |
| MPAC-ResNet (proposed) | FT, lr 1e-4 | 1.86M | 2.17 | 51.3 | 98.10 ± 0.44 | 98.10 ± 0.44 | 98.10 ± 0.44 | 99.37 ± 0.14 | 99.82 ± 0.05 |
| ResNet-Tiny | scratch, p15 | 0.98M | 2.51 | 41.3 | 97.72 ± 0.17 | 97.72 ± 0.16 | 97.74 ± 0.16 | 99.24 ± 0.05 | 99.78 ± 0.03 |
| MobileNet-Tiny | scratch, p15 | 0.40M | 0.12 | 87.5 | 97.65 ± 0.14 | 97.67 ± 0.14 | 97.68 ± 0.14 | 99.22 ± 0.05 | 99.78 ± 0.04 |
| CustomCNN-Tiny | scratch, p15 | 0.31M | 1.32 | 67.7 | 97.43 ± 0.43 | 97.44 ± 0.44 | 97.45 ± 0.43 | 99.15 ± 0.14 | 99.79 ± 0.06 |
| Custom CNN | scratch, 50 epochs, p7 | 1.24M | 5.17 | 27.6 | 95.53 ± 2.96 | 95.55 ± 2.94 | 95.56 ± 2.95 | 98.51 ± 0.99 | 99.43 ± 0.51 |

Notes on the table:
- Per-seed accuracy: ResNet50-FT [99.33, 99.05, 99.05]; Custom CNN [97.24, **92.11**, 97.24]; LS-Net p7 (superseded) [98.38, **96.29**, 98.29].
- Complexity numbers come from `results/metrics/complexity_all.csv` (CPU, batch 1).
- Tiny s42 uses the `20261007_122552_run_all_tiny_s42_p15` reruns; ResNet-Tiny s42 uses `20261006_102217_run_all`.

### 3.2 Freeze vs fine-tune (seed 42)

| Model | Frozen | FT | Δ |
|---|---|---|---|
| ResNet50 | 91.44 | 99.33 | +7.89 |
| EfficientNetB0 | 91.25 | 99.24 | +7.99 |
| MobileNetV2 | 89.64 | 98.48 | +8.84 |
| VGG16 | 97.34 | 98.86 | +1.52 |

The frozen ResNet50/EfficientNetB0/MobileNetV2 runs trained only 5–8K head parameters and underfit. VGG16's frozen head has 12.8M parameters, which explains the small gap.

### 3.3 Per-class F1 (mean of 3 seeds, %)

| Model | glioma | meningioma | notumor | pituitary |
|---|---|---|---|---|
| ResNet50-FT | 99.19 | 98.63 | 99.14 | 99.62 |
| EfficientNetB0-FT | 98.94 | 98.69 | 99.27 | 99.43 |
| VGG16-FT | 98.87 | 98.39 | 98.81 | 99.37 |
| ResNet18-FT | 98.69 | 97.94 | 98.81 | 99.37 |
| MobileNetV2-FT | 98.43 | 97.76 | 98.94 | 98.69 |
| LS-Net (p15) | 98.00 | 97.32 | 98.35 | 99.50 |
| MPAC-ResNet-Tiny | 97.93 | 97.83 | 98.02 | 99.12 |
| MPAC-ResNet-FT | 97.44 | 97.46 | 98.01 | 99.50 |
| Custom CNN | 95.23 | **92.86** | 96.10 | 98.02 |

**Meningioma is the hardest class for 12 of 14 configurations** (exceptions: MPAC-ResNet-FT and LS-Net p7, where glioma is marginally lower) (the notumor/glioma confusion is secondary). **Pituitary is the easiest** in nearly every configuration.

### 3.4 Statistical tests (McNemar, test set n=1,052)

| Pair | s42 | s123 | s456 | Verdict |
|---|---|---|---|---|
| ResNet50-FT vs EfficientNetB0-FT | p=1.00 | p=0.62 | p=0.50 | **tied** |
| ResNet18-FT vs MPAC-ResNet-FT | **p=4.8e-4 (R18 better)** | p=1.00 | p=0.55 | MPAC ≤ ResNet18 |
| MPAC-ResNet-Tiny vs EfficientNet-Tiny | p=0.48 | p=0.80 | p=1.00 | **tied** |
| VGG16-frozen vs Custom CNN (legacy) | p=0.0046 | — | — | VGG16 better |

The differences at the top of the table are mostly **not statistically significant**. Present them as tiers, not as a strict ranking.

### 3.5 Accuracy–efficiency takeaways

- **EfficientNetB0-FT** is the practical pick for full-size models: same accuracy as ResNet50 (tied) at 5.9× fewer params, 10.6× fewer GFLOPs and 2.8× faster on CPU.
- **LS-Net** is the deployment pick: 98.29 % at 0.36 GFLOPs and 88.5 FPS on CPU, which is **6× faster than ResNet50 for −0.85 pp**.
- **MPAC-ResNet-Tiny** gives 98.22 % with **32.8× fewer params** than ResNet50 (−0.92 pp).
- **Transfer learning improves both accuracy and stability.** Seed std is ≤0.47 for every FT model, versus 2.96 for Custom CNN. All sub-1M models trained with patience 15 are also stable (std ≤0.43).
- **Patience matters for scratch models.** LS-Net's std went from 1.18 (p7) to 0.25 (p15). Report p15 as the protocol and p7 as a sensitivity note.

---

## 4. Proposed-method (MPAC) analysis

### 4.1 Ablation: what replacing ResNet-18 stages 3–4 with MPAC does (Nickparvar, FT lr 1e-4)

| | ResNet18-FT | MPAC-ResNet-FT | Δ |
|---|---|---|---|
| Accuracy | 98.70 ± 0.47 | 98.10 ± 0.44 | **−0.60 pp** |
| Params | 11.18M | 1.86M | **6.0× fewer** |
| GFLOPs | 3.63 | 2.17 | 1.7× fewer |
| CPU FPS | 35.8 | 51.3 | 1.4× faster |

Other MPAC-ResNet runs (seed 42, single run each, train-time test metrics only):
- lr 1e-3 → **98.38 %**, which beats its own lr 1e-4 run (97.62 %). The 3-seed study used the weaker learning rate.
- From scratch, 50 epochs → 98.29 % (task 9.7).

⚠️ **Confound.** `build_mpac_resnet` swaps in **randomly initialised** MPAC stages 3–4, while ResNet-18 keeps **ImageNet-pretrained** stages 3–4. The ablation therefore measures "MPAC + random init" against "standard conv + pretrained". The fair control is ResNet-18 with stages 3–4 re-initialised.

### 4.2 Sub-1M downscaling study (Nickparvar, 3 seeds, p15)

Ranking: LS-Net 98.29 > MPAC-ResNet-Tiny 98.22 > EfficientNet-Tiny 98.13 > ResNet-Tiny 97.72 > MobileNet-Tiny 97.65 > CustomCNN-Tiny 97.43.

- The total spread is **0.86 pp**, and the top 3 are statistically tied. On this dataset the downscaling study shows that **every compact design survives**, not which one survives best.
- The **two MPAC-based designs rank 1st and 2nd**, which is consistent with the LS-Net paper's thesis, but the margin is within noise.
- CustomCNN-Tiny (0.31M, 97.43 ± 0.43) beats the full Custom CNN (1.24M, 95.53 ± 2.96). This is confounded: the patience is 15 vs 7.
- The planned downscaling figure (9.12) has **not been produced**.

---

## 5. Cross-dataset / generalization

### 5.0 Figshare (Phase 7): invalidated, but a useful finding

Zero-shot accuracy was 99.5–99.8 %, but **about 92 % of the Figshare images (2,810 / 3,064) are near-duplicates of Nickparvar training images** (1,523 are exact duplicates). This is a **dataset-provenance contamination** finding and worth a paragraph in the paper as a limitation and warning. The decontaminated re-evaluation (7.4b) has not been run.

### 5.1 BraTS 2020 HGG/LGG (test: 1,400 slices / 56 subjects)

#### 5.1a Main table: FT protocol, seeds 42 + 123

Every model uses the same recipe: `--finetune`, lr 1e-4, 30 epochs, patience 7. Runs: `20261007_221751_run_all_brats_ft_s42` and `20261008_154220_run_all_brats_ft_s123`. Values are mean ± std over n=2; with only two seeds, the std is indicative only.

| Model | Params | Slice Acc | **Slice F1** | F1 per seed (s42 / s123) | AUC | LGG sens (slice) | HGG sens (slice) | Subject Acc | **Subject F1** | Subject LGG caught (of 12, s42 / s123) |
|---|---|---|---|---|---|---|---|---|---|---|
| **VGG16** | 27.6M | **92.57 ± 0.10** | **89.31 ± 0.06** | 89.35 / 89.27 | **96.25 ± 0.93** | **87.2** | 94.0 | **94.64** | **92.34 ± 3.33** | 11 / 11 |
| ResNet50 | 23.5M | 89.65 ± 1.21 | 84.76 ± 0.86 | 85.37 / 84.15 | 94.85 ± 0.54 | 77.0 | 93.1 | 91.96 | 87.91 ± 1.10 | 9 / 10 |
| ResNet-Tiny | 0.98M | 90.07 ± 1.92 | 84.57 ± 2.92 | 82.51 / **86.64** | 94.71 ± 1.14 | 70.8 | 95.3 | 90.18 | 84.68 ± 2.33 | 8 / 9 |
| EfficientNetB0 | 4.0M | 88.90 ± 0.15 | 84.46 ± 0.99 | 83.76 / 85.16 | 94.82 ± 1.01 | 83.2 | 90.5 | 92.86 | 90.04 ± 3.17 | 11 / 11 |
| CustomCNN-Tiny | 0.31M | 89.04 ± 0.56 | 84.44 ± 0.85 | 83.84 / 85.04 | 93.72 ± 0.55 | 81.0 | 91.2 | 91.07 | 87.46 ± 0.48 | 10 / 11 |
| MPAC-ResNet | 1.86M | 86.93 ± 3.03 | 82.26 ± 2.28 | 83.87 / 80.65 | 93.14 ± 0.23 | 82.7 | 88.1 | 88.39 | 83.90 ± 0.27 | 9 / 11 |
| LS-Net | 0.91M | 88.25 ± 0.86 | 82.02 ± 2.36 | 83.69 / 80.35 | 89.82 ± 4.29 | 69.2 | 93.5 | 92.86 | 89.33 ± 0.92 | 11 / 9 |
| ResNet18 | 11.2M | 85.75 ± 0.16 | 81.08 ± 1.05 | 80.34 / 81.82 | 93.46 ± 1.04 | 84.5 | 86.1 | 85.71 | 80.88 ± 1.29 | 9 / 11 |
| MPAC-ResNet-Tiny | 0.72M | 86.93 ± 2.93 | 80.06 ± 3.07 | 77.89 / 82.23 | 91.26 ± 3.53 | 65.3 | 92.8 | 89.29 | 83.56 ± 0.75 | 9 / 8 |
| Custom CNN | 1.24M | 87.92 ± 0.40 | 79.56 ± 0.19 | 79.42 / 79.69 | 94.46 ± 0.06 | **55.8** | 96.7 | 88.39 | 79.78 ± 2.77 | 7 / 6 |
| MobileNetV2 | 2.2M | 85.32 ± 0.76 | 78.87 ± 1.15 | 78.06 / 79.68 | 91.21 ± 0.46 | 70.2 | 89.5 | 85.71 | 79.40 ± 3.64 | 8 / 9 |
| EfficientNet-Tiny | 0.75M | 80.78 ± 0.60 | 76.00 ± 1.03 | 76.72 / 75.27 | 89.09 ± 0.49 | 84.3 | 79.8 | 83.93 | 80.38 ± 5.67 | 12 / 11 |
| MobileNet-Tiny | 0.40M | 74.68 ± 5.50 | 68.89 ± 5.36 | 65.10 / 72.68 | 82.57 ± 6.44 | 73.0 | 75.1 | 76.79 | 71.09 ± 4.69 | 9 / 9 |

**McNemar, slice-level, top-2 per seed:**
- s42: VGG16 vs ResNet50, p=0.030 (VGG16 better).
- s123: VGG16 vs ResNet-Tiny, p=0.14 (not significant).

What the two seeds show:
- **VGG16-FT is the only model that is clearly and reproducibly best.** Its slice F1 differs by 0.08 between seeds.
- **Below VGG16 there is a four-way tie at about 84.5 F1:** ResNet50, ResNet-Tiny, EfficientNetB0 and CustomCNN-Tiny. Two of those are sub-1M scratch models.
- **The most stable models:** VGG16, Custom CNN (F1 std 0.19) and CustomCNN-Tiny.
- **The least stable models:** MobileNet-Tiny, MPAC-ResNet-Tiny, ResNet-Tiny, LS-Net and MPAC-ResNet (F1 std 2.3–5.4). LS-Net's AUC swings from 92.9 to 86.8 between seeds.
- **Custom CNN is biased toward HGG.** It catches only 55.8 % of LGG slices and half of the LGG patients, even though its accuracy (87.9 %) looks reasonable. Accuracy alone would hide this.

#### 5.1b Seed 42 only, best of both regimes (val-selected)

For each model, I picked whichever of the two seed-42 runs (frozen/lr 1e-3 vs FT/lr 1e-4) had the higher **validation** F1, so no model selection used the test set. This table exists only because the frozen/lr 1e-3 run has a single seed; for rankings, use §5.1a.

| Model | Selected run | Slice Acc | Slice F1 | AUC | LGG sens (slice) | HGG sens (slice) | **Subject Acc** | **Subject F1** | Subject LGG sens |
|---|---|---|---|---|---|---|---|---|---|
| **VGG16** | FT | **92.50** | **89.35** | **96.91** | **89.0** | 93.5 | 92.86 | 89.98 | 11/12 |
| ResNet50 | FT | 90.50 | 85.37 | 95.23 | 73.0 | 95.3 | 92.86 | 88.69 | 9/12 |
| MPAC-ResNet-Tiny | lr 1e-3 | 89.43 | 84.11 | 92.40 | 73.7 | 93.7 | 89.29 | 84.09 | 9/12 |
| MPAC-ResNet | FT | 89.07 | 83.87 | 92.98 | 75.3 | 92.8 | 89.29 | 84.09 | 9/12 |
| EfficientNetB0 | FT | 88.79 | 83.76 | 94.11 | 77.3 | 91.9 | **94.64** | **92.28** | 11/12 |
| ResNet-Tiny | lr 1e-4 | 88.71 | 82.51 | 93.91 | 68.0 | 94.4 | 89.29 | 83.03 | 8/12 |
| LS-Net | lr 1e-3 | 87.64 | 82.18 | 93.77 | 75.3 | 91.0 | 91.07 | 87.13 | 10/12 |
| ResNet18 | FT | 85.64 | 80.34 | 92.73 | 78.7 | 87.6 | 85.71 | 79.96 | 9/12 |
| EfficientNet-Tiny | lr 1e-3 | 85.14 | 80.04 | 92.71 | 80.7 | 86.4 | 87.50 | 82.92 | 10/12 |
| Custom CNN | lr 1e-4 | 87.64 | 79.42 | 94.42 | 57.0 | 96.0 | 89.29 | 81.74 | 7/12 |
| CustomCNN-Tiny | lr 1e-3 | 85.64 | 78.17 | 90.86 | 63.3 | 91.7 | 87.50 | 80.85 | 8/12 |
| MobileNetV2 | FT | 84.79 | 78.06 | 90.88 | 68.7 | 89.2 | 83.93 | 76.83 | 8/12 |
| MobileNet-Tiny | lr 1e-3 | 83.86 | 76.43 | 89.41 | 64.7 | 89.1 | 85.71 | 78.79 | 8/12 |

The binary `specificity_macro` equals `recall_macro` by construction (HGG specificity = LGG sensitivity), so the table reports per-class sensitivities instead.

**Frozen → FT (pretrained models, slice F1, seed 42; no frozen s123 run exists):**

| Model | Frozen | FT | Δ |
|---|---|---|---|
| VGG16 | 69.28 | 89.35 | +20.1 |
| ResNet50 | 68.45 | 85.37 | +16.9 |
| EfficientNetB0 | 67.45 | 83.76 | +16.3 |
| MobileNetV2 | 64.61 | 78.06 | +13.5 |
| MPAC-ResNet | 74.88 | 83.87 | +9.0 |
| ResNet18 | 73.00 | 80.34 | +7.3 |

**McNemar on BraTS (slice-level, seed 42):**
- MPAC-ResNet-Tiny vs LS-Net (frozen/lr 1e-3 run): p=0.054, not significant.

### 5.2 What BraTS shows

1. **BraTS breaks the saturation.** The Nickparvar spread of about 1 pp (FT models, 98.10–99.14) becomes about 20 F1 points (§5.1a), so this is the dataset that actually discriminates between architectures.
2. **Frozen ImageNet features fail on multi-modal stacked MRI.** Every frozen pretrained model (F1 64.6–74.9) scores *below* every sub-1M scratch model in the same lr 1e-3 run (F1 76.4–84.1). The domain gap is far larger than on Nickparvar (+1.5 to +8.8 pp there, +7 to +20 here).
3. **VGG16 is the outlier, not "large models" in general.** With 2 seeds, VGG16-FT leads by about 4.5 F1. ResNet50 and EfficientNetB0 are tied with the best sub-1M models (ResNet-Tiny 0.98M, CustomCNN-Tiny 0.31M) at about 84.5. The nuance for the downscaling story: the best compact models lose about 1 pp to the best full model on Nickparvar and about 4.7 F1 on BraTS, and that BraTS gap comes from VGG16 alone. The worst compact models (EfficientNet-Tiny, MobileNet-Tiny) lose 13–20 F1.
4. **MPAC on BraTS is inconsistent:**

   | | s42 | s123 | Mean |
   |---|---|---|---|
   | MPAC-ResNet vs ResNet18 (slice F1) | +3.5 | −1.2 | +1.2, within seed noise (MPAC std 2.28) |
   | Subject F1: MPAC-ResNet / ResNet18 | 84.09 / 79.96 | 83.71 / 81.79 | 83.90 / 80.88 |

   - The subject-level F1 is the more consistent MPAC signal.
   - MPAC-ResNet-Tiny ranks only 9th under the FT protocol, with high variance.
   - The §4.1 init confound still applies. The frozen comparison is also not fair, because MPAC-ResNet "frozen" trains stages 3–4 while ResNet18 "frozen" trains only the head.
5. **Statistical power is low.** With 12 LGG test subjects, one patient moves LGG sensitivity by 8.3 pp. At subject level, many models share identical confusion matrices; at s123, CustomCNN-Tiny and EfficientNetB0 are identical (91.07 % / F1 87.80). The seed-456 run is still needed, and with n=2 the std values are indicative only.
6. **Training is unstable for pretrained FT models.** They pick their best-val epoch very early on both seeds (s42 / s123): VGG16 0 / 3, EfficientNetB0 3 / 1, ResNet18 1 / 6, MobileNetV2 6 / 3, MPAC-ResNet 9 / 2. They overfit within a few epochs at lr 1e-4. Validation F1 is also a noisy selector: LS-Net had the best frozen-run val F1 (0.942) but only 82.18 test F1.
7. **Some scratch models are under-trained by the FT recipe** (lr 1e-4, 30 epochs, p7):
   - EfficientNet-Tiny's best epoch is 29 on **both** seeds, so it was still improving when training stopped.
   - LS-Net's best epoch at s123 is 23.
   - MobileNet-Tiny stays the worst model on both seeds (65.1 / 72.7); with lr 1e-3 at s42 it reached 76.4.
   - Scratch models should be re-run with the Nickparvar scratch protocol (lr 1e-3, 50 epochs, p15) before drawing any compact-architecture conclusions on BraTS.

---

## 6. Key findings (paper-ready statements)

1. Fully fine-tuned ImageNet CNNs reach **≥98.4 % accuracy** on Nickparvar 4-class. ResNet50-FT (99.14 ± 0.16) and EfficientNetB0-FT (99.08 ± 0.28) are statistically indistinguishable.
2. Full fine-tuning is essential. Frozen backbones lose 8–9 pp on Nickparvar and 7–20 F1 on BraTS.
3. Transfer learning cuts seed variance by 6× or more compared with from-scratch Custom CNN (std 2.96 vs ≤0.47).
4. **Sub-1M models (0.3–1.0M params) reach 97.4–98.3 %.** LS-Net and MPAC-ResNet-Tiny lead, about 0.9 pp below the best full model with 26–33× fewer params and up to 6× higher CPU throughput.
5. The Nickparvar benchmark is saturated, and about 92 % of Figshare overlaps it, so it cannot test generalization.
6. On BraTS HGG/LGG grading (subject-level split), architectures separate clearly (slice F1 68.9–89.3).
   - **VGG16-FT is reproducibly best**: slice F1 89.31 ± 0.06, AUC 96.25, subject F1 92.34, 11/12 LGG patients detected on each seed.
   - The next tier, at about 84.5 F1, is ResNet50 and EfficientNetB0 tied with two sub-1M scratch models (ResNet-Tiny, CustomCNN-Tiny).
7. The MPAC block does **not** improve accuracy over a pretrained ResNet-18:
   - **Nickparvar:** −0.6 pp.
   - **BraTS slice F1:** +1.2 mean, but inconsistent (+3.5 / −1.2).
   - **What it does deliver:** a 6× parameter reduction, plus a modest, consistent subject-level F1 gain on BraTS (+3.0, n=2).

---

## 7. Contribution check

### 7.1 Against README §2–§4 (objectives and expected results)

| README item | Status | Evidence / gap |
|---|---|---|
| Study DL methods for brain-tumor MRI (survey) | ❌ **Not done in repo** | Phase 1 (1.1–1.4) has no deliverable; there is no literature/SOTA table |
| Build and train DL models for detection and classification | ✅ | 13 architectures, freeze / FT / scratch, 2 datasets. "Detection" is covered at image level by the `notumor` class |
| Evaluate with Accuracy, Sensitivity, Specificity, F1 | ✅ (exceeded) | Plus AUC, per-class metrics, McNemar, subject-level metrics, complexity / FPS, Grad-CAM |
| Propose a high-performing model for practical diagnosis | ⚠️ **Partially** | No single winner: ResNet50/EfficientNetB0-FT (Nickparvar), VGG16-FT (BraTS), LS-Net / MPAC-Tiny (edge). The recommendation must be **conditional on use-case**. Proposed MPAC-ResNet is not the top model on either dataset |
| CNNs and improved variants | ✅ | MPAC-ResNet hybrid, LS-Net, Tiny variants |
| Data processing pipeline | ✅ | Cleaning (md5 dedup), stratified / subject-grouped split, augmentation, BraTS NIfTI→2D extraction |
| Optimal model with **high generalization** | ⚠️ | Generalization was tested only by *retraining* on BraTS (a different task). There is no zero-shot or decontaminated cross-dataset test (7.4b) |
| Standardized end-to-end workflow | ✅ code / ⚠️ docs | Scripts are complete, but the README "Reproduce" section is stale: it says "tất cả 5 models" and is missing `--config configs/config_brats.yaml`, `prepare_brats.py`, `extract_figshare_mat.py`, `prepare_external.py`, `paper_figures.py`, `mcnemar_pairs.py`, `profile_models.py`, `--seed`, `--by-subject` and `--patience` |
| Scientific contribution: paper / conference report | ❌ | Phase 8 writing has not started |

### 7.2 Against the paper contributions claimed in `integration_plan.md`

| Claimed contribution | Verdict | Why |
|---|---|---|
| **C1. MPAC-ResNet, a novel hybrid that captures multi-scale features "critical" for tumors** | ⚠️ **Weak as an accuracy claim; OK as an efficiency claim** | Worse than its ResNet-18 parent on Nickparvar (−0.6 pp). On BraTS it is mixed: slice F1 +3.5 / −1.2 over 2 seeds; subject F1 +4.1 / +1.9. The init confound (§4.1) must be fixed. The safe framing is *"6× smaller than ResNet-18 at comparable accuracy on both tasks"*; a "gain on the harder task" claim needs seed 456 plus the fair-init control |
| **C2. First systematic sub-1M study for brain-tumor MRI** | ✅ **Data done** / ⚠️ claim | 6 compact architectures × 3 seeds on Nickparvar, plus BraTS FT (n=2). On Nickparvar, MPAC designs rank top-2 but the differences are not significant. On BraTS the FT ranking among compact models flips: ResNet-Tiny and CustomCNN-Tiny lead, while LS-Net and MPAC-Tiny are unstable. That ranking is confounded by the under-training noted in §5.2 #7, so "MPAC is most resilient" is **not** supported yet. "First" needs a literature check (Phase 1). The downscaling figure (9.12) is missing |
| **C3. Most comprehensive comparative study** | ⚠️ | Breadth is real (13 models, 3 regimes, multi-seed, 2 datasets, complexity, Grad-CAM, McNemar). "Most comprehensive on this dataset" cannot be claimed without the SOTA comparison table. There are no ViT/ConvNeXt models (flagged in `project_analysis.md`) |
| (Unclaimed but real) **C4. Dataset-provenance audit**: Figshare ⊂ Nickparvar (~92 %) | ✅ **Add to paper** | Strong, verifiable finding that warns against the common "cross-dataset" practice on these two sets |
| (Unclaimed but real) **C5. Saturated vs. hard benchmark**: Nickparvar spread about 1 pp vs BraTS about 20 F1; the best compact models are about 1 pp behind the leader on Nickparvar and about 4.7 F1 behind on BraTS | ✅ (n=2 on BraTS; seed 456 pending) | This turns the BraTS phase into a core result rather than an appendix |

---

## 8. Plan status (`implementation_plan.md`)

| Phase | Status | Notes / stale markers |
|---|---|---|
| 1 Literature review | ❌ 0/4 | Blocks the SOTA table and the "first/most comprehensive" claims |
| 2 Environment | ✅ 5/5 | |
| 3 Data | ✅ 8/8 | |
| 4 Models | ✅ 7/7 | |
| 5 Training | ✅ 15/16 (5.10 tuning skipped) | |
| 6 Analysis | ✅ 14/14 | ⚠️ but 6.10–6.14 outputs in `results/paper/` are **stale/biased** (§1 #1–2): regenerate |
| 7 Figshare | ⚠️ 3/6 | 7.4 invalidated; 7.4b, 7.5, 7.6 pending. Suggest: do 7.4b (~250 imgs) and write 7.6 as the contamination finding; drop 7.5 |
| 8 Documentation / paper | ❌ 0/8 | |
| 9 MPAC / LS-Net | ✅ 15/19 | **Stale markers:** 9.6, 9.7, 9.8 and 9.9 are marked ⏳ but **are done** (see §4). Pending: 9.12 downscaling figure, 9.13 Grad-CAM comparison, 9.14 cross-dataset MPAC, 9.15 paper integration. 9.19 must be re-run with p15 LS-Net |
| 10 BraTS | ⚠️ 5/9 + partials | **10.1–10.5 are done but unmarked.** 10.6 multi-seed: **2/3 seeds** for the FT protocol (s456 pending); the frozen/lr 1e-3 protocol has s42 only. 10.7 modality ablation: `configs/config_brats_flair.yaml` exists (untracked) but has **not been run**. 10.8 MPAC ablation: n=2, inconsistent, confounded. 10.9 write-up ❌ |

---

## 9. Recommended next steps (priority order)

1. **Fix the integrity issues (§1):** dedupe `experiments.csv`; patch `paper_figures.py` (dedupe, dataset filter, patience grouping, scratch labelling); regenerate `results/paper/`; re-sync the BraTS-FT run dir.
2. **Finish BraTS multi-seed (10.6):**
   - **FT, seed 456:** run `run_all.py --config configs/config_brats.yaml --epochs 30 --lr 0.0001 --seed 456 --name brats_ft_s456 --by-subject --extra --finetune`. This completes §5.1a.
   - **Scratch models (all `*_tiny`, LS-Net, Custom CNN):** run separately with the scratch protocol (lr 1e-3, 50 epochs, `--patience 15`) for seeds 42/123/456. The FT recipe under-trains them (§5.2 #7).
   - Report slice- and subject-level mean ± std.
3. **Fair MPAC ablation:** add a ResNet-18 variant with re-initialised layer3/4 (same init as MPAC-ResNet), on both datasets. This decides whether C1 survives.
4. Re-run 9.19 (MPAC-ResNet vs LS-Net McNemar) with the p15 LS-Net runs, and save the output under `results/`.
5. Produce the 9.12 downscaling figure (Nickparvar vs BraTS drop per architecture). With step 2 done, this becomes the headline figure for C2/C5.
6. Phase 1 literature + SOTA table, then Phase 8 writing. Update the README "Reproduce" section and the plan status markers.
7. 10.7 modality ablation: the config is ready (`configs/config_brats_flair.yaml`). Run at least VGG16-FT, ResNet50-FT and EfficientNetB0-FT on FLAIR-only and compare against §5.1a.
8. Optional: 7.4b decontaminated Figshare eval, 9.13 Grad-CAM comparison.
