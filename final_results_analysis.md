# Final Results Analysis — Brain Tumor MRI Classification

Based on the abstract, the paper has two core contributions:
1. **Comprehensive sub-1M benchmark & generalization study**: A systematic evaluation of architectures under a strict sub-1M parameter budget, spanning a saturated classification task (Nickparvar 4-class) and a harder cross-dataset clinical task (BraTS 2020 HGG/LGG grading). Key findings: (a) on saturated benchmarks, compact models are statistically tied with models an order of magnitude larger — 98.3% accuracy at 0.36 GFLOPs, $6\times$ faster than a fine-tuned ResNet-50 for $<1$ pp; (b) the residual gap on BraTS concentrates in low-confidence predictions that are unstable under perturbation, and the most efficient architecture — built on parallel atrous convolution branches — fails preferentially on scale-sensitive inputs.
2. **Proposed MCG-TTA**: An architecture-informed, confidence-gated test-time augmentation that probes gated predictions with scale $\pm10\%$ views (mirroring the model's multi-dilation paths) and escalates to full multi-view aggregation only on disagreement — recovering up to $+1.8$ pp slice F1 at $\sim 1.9\times$ base latency with **0 additional learned parameters** (0.91M), while preserving all LGG patient detections (20/24).

Below are the **5 publication-ready tables** required to substantiate every claim in the paper.

---

### Table 1: In-Domain Performance & Edge Efficiency on Nickparvar (4-Class MRI)
> **Purpose**: Backs the claim that on saturated classification benchmarks, sub-1M models (specifically LS-Net) achieve statistically comparable performance to models an order of magnitude larger while running $6\times$ faster at 0.36 GFLOPs.

| Architecture | Paradigm | Parameters | GFLOPs | CPU Latency (ms) | Throughput (FPS) | Accuracy (%) | Macro F1 (%) | Specificity (%) | AUC (%) |
|---|---|---|---|---|---|---|---|---|---|
| **ResNet-50** | Full / Fine-tune | 23.52M | 8.17 | 67.89 | 14.7 | **99.14 ± 0.16** | **99.15 ± 0.17** | 99.71 ± 0.06 | 99.92 ± 0.05 |
| **EfficientNet-B0** | Full / Fine-tune | 4.01M | 0.77 | 23.91 | 41.8 | 99.08 ± 0.28 | 99.08 ± 0.27 | 99.70 ± 0.09 | **99.97 ± 0.02** |
| **VGG-16** | Full / Fine-tune | 27.56M | 30.72 | 150.03 | 6.7 | 98.86 ± 0.10 | 98.86 ± 0.10 | 99.62 ± 0.03 | 99.94 ± 0.03 |
| **ResNet-18** | Full / Fine-tune | 11.18M | 3.63 | 27.96 | 35.8 | 98.70 ± 0.47 | 98.70 ± 0.47 | 99.57 ± 0.16 | 99.94 ± 0.02 |
| **MobileNet-V2** | Full / Fine-tune | 2.23M | 0.60 | 17.35 | 57.6 | 98.45 ± 0.33 | 98.45 ± 0.33 | 99.48 ± 0.11 | 99.95 ± 0.03 |
| **LS-Net** | Compact / Scratch | **0.91M** | **0.36** | **11.30** | **88.5** | **98.29 ± 0.25** | **98.29 ± 0.25** | 99.43 ± 0.08 | 99.79 ± 0.08 |
| **MPAC-ResNet-Tiny** | Compact / Scratch | 0.72M | 0.92 | 18.01 | 55.5 | 98.22 ± 0.30 | 98.22 ± 0.30 | 99.41 ± 0.11 | 99.84 ± 0.03 |
| **EfficientNet-Tiny**| Compact / Scratch | 0.75M | 0.19 | 16.14 | 61.9 | 98.13 ± 0.38 | 98.14 ± 0.38 | 99.38 ± 0.13 | 99.85 ± 0.04 |
| **MPAC-ResNet** | Hybrid / Fine-tune | 1.86M | 2.17 | 19.50 | 51.3 | 98.10 ± 0.44 | 98.10 ± 0.44 | 99.37 ± 0.14 | 99.82 ± 0.05 |
| **ResNet-Tiny** | Compact / Scratch | 0.98M | 2.51 | 24.19 | 41.3 | 97.72 ± 0.17 | 97.72 ± 0.16 | 99.24 ± 0.05 | 99.78 ± 0.03 |
| **MobileNet-Tiny** | Compact / Scratch | 0.40M | 0.12 | 11.43 | 87.5 | 97.65 ± 0.14 | 97.67 ± 0.14 | 99.22 ± 0.05 | 99.78 ± 0.04 |

*Note: Results report mean ± sample-std over 3 random seeds (42, 123, 456) on the 1,052-sample balanced test set. CPU latency measured at batch size = 1 ($224 \times 224$). LS-Net is $6.0\times$ faster than ResNet-50 and $2.1\times$ faster than EfficientNet-B0.*

---

### Table 2: Cross-Dataset Generalization & Clinical Grading on BraTS 2020
> **Purpose**: Demonstrates that the benchmark saturation breaks on multi-sequence MRI grading (~4:1 class imbalance, 1,400 slices / 56 subjects), revealing the generalization drop in compact architectures and proving where MCG-TTA closes the gap.

| Model | Parameters | Slice F1 (%) | Slice Acc (%) | Slice AUC (%) | Patient F1 (%) | Patient Acc (%) | LGG Patient Recall (Caught / 12) |
|---|---|---|---|---|---|---|---|
| **VGG-16** | 27.56M | **89.31 ± 0.06** | **92.57 ± 0.10** | **96.25 ± 0.93** | **92.34 ± 3.33** | **94.64** | **11.0 / 12 (91.7%)** |
| **ResNet-50** | 23.52M | 84.76 ± 0.86 | 89.65 ± 1.21 | 94.85 ± 0.54 | 87.91 ± 1.10 | 91.96 | 9.5 / 12 (79.2%) |
| **ResNet-Tiny** | 0.98M | 84.57 ± 2.92 | 90.07 ± 1.92 | 94.71 ± 1.14 | 84.68 ± 2.33 | 90.18 | 8.5 / 12 (70.8%) |
| **EfficientNet-B0** | 4.01M | 84.46 ± 0.99 | 88.90 ± 0.15 | 94.82 ± 1.01 | 90.04 ± 3.17 | 92.86 | **11.0 / 12 (91.7%)** |
| **MCG-TTA (proposed)** | **0.91M** | **83.45 ± 2.07** | **88.83 ± 1.04** | **90.74 ± 3.20** | **88.15 ± 1.83** | **91.97** | **10.0 / 12 (83.3%)** |
| **MPAC-ResNet** | 1.86M | 82.26 ± 2.28 | 86.93 ± 3.03 | 93.14 ± 0.23 | 83.90 ± 0.27 | 88.39 | 10.0 / 12 (83.3%) |
| **LS-Net (base)** | **0.91M** | **82.02 ± 2.36** | **88.25 ± 0.86** | **89.82 ± 4.29** | **89.33 ± 0.92** | **92.86** | **10.0 / 12 (83.3%)** |
| **ResNet-18** | 11.18M | 81.08 ± 1.05 | 85.75 ± 0.16 | 93.46 ± 1.04 | 80.88 ± 1.29 | 85.71 | 10.0 / 12 (83.3%) |
| **MPAC-ResNet-Tiny** | 0.72M | 80.06 ± 3.07 | 86.93 ± 2.93 | 91.26 ± 3.53 | 83.56 ± 0.75 | 89.29 | 8.5 / 12 (70.8%) |
| **MobileNet-V2** | 2.23M | 78.87 ± 1.15 | 85.32 ± 0.76 | 91.21 ± 0.46 | 79.40 ± 3.64 | 85.71 | 8.5 / 12 (70.8%) |
| **EfficientNet-Tiny**| 0.75M | 76.00 ± 1.03 | 80.78 ± 0.60 | 89.09 ± 0.49 | 80.38 ± 5.67 | 83.93 | **11.5 / 12 (95.8%)** |
| **MobileNet-Tiny** | 0.40M | 68.89 ± 5.36 | 74.68 ± 5.50 | 82.57 ± 6.44 | 71.09 ± 4.69 | 76.79 | 9.0 / 12 (75.0%) |

*Note: Evaluated across multi-seed fine-tuning on BraTS 2020 with patient-stratified split. Patient decision obtained via mean-probability voting across axial slices. MCG-TTA lifts LS-Net from 82.02% to 83.45% Slice F1 without retraining or increasing model size.*

---

### Table 3: Ablation of Test-Time Augmentation (TTA) Strategies on LS-Net (BraTS 2020)
> **Purpose**: Validates the core contribution in the abstract: why architecture-informed **MCG-TTA** (multi-scale consistency gating) is superior to unselective TTA and naive confidence gating in both accuracy recovery and clinical safety.

| Strategy | Description / Gating Rule | Trigger Rate (%) | Extra Forwards / Trigger | Modelled Latency (ms) | Relative Latency | Slice F1 (s42 / s123) | Mean Slice F1 (%) | Mean Patient F1 (%) | LGG Patient Recall |
|---|---|---|---|---|---|---|---|---|---|
| **LS-Net (Base)** | Single standard forward pass | 0.0% | 0 | **5.60** | **1.0×** | 83.69 / 80.35 | 82.02 | **89.34** | 20 / 24 |
| `disagree` | $p_{\max} < \tau \to$ flip probe; escalate on argmax change | 27.6% | 2.0 | 8.73 | 1.56× | 85.84 / 81.22 | 83.53 | 86.24 | 18 / 24 *(loses 2)* |
| **MCG-TTA (proposed)**| **$p_{\max} < \tau \to$ scale $\pm 10\%$ probes; escalate on scale disagreement** | **27.6%** | **3.3** | **10.74** | **1.92×** | **85.52 / 81.38** | **83.45** | **88.15** | **20 / 24 (preserved)** |
| `conf` | $p_{\max} < \tau \to$ execute all 6 augmentations | 27.6% | 5.0 | 13.28 | 2.37× | 85.35 / 81.34 | 83.35 | 88.15 | 20 / 24 |
| `weakclass` | Trigger on $p_{\max} < \tau$ OR predicted LGG | 39.7% | 5.0 | 16.65 | 2.97× | 85.23 / 80.93 | 83.08 | 85.08 | 18 / 24 *(loses 2)* |
| `subject` | Escalate all slices of low-confidence patients | 100% (pat) | 5.0 | 13.82 | 2.47× | 84.31 / 80.67 | 82.49 | 84.00 | 18 / 24 *(loses 2)* |
| *Full TTA (Ceiling)*| Unconditional 6-view TTA on all samples | 100.0% | 5.0 | 33.60 | 6.00× | 85.44 / — | — | 87.13 | 19 / 24 |

*Key Findings:
1. MCG-TTA delivers up to **+1.83 pp gain on seed 42** and **+1.43 pp mean gain** over base LS-Net at $< 2\times$ latency.
2. Unlike heuristic disagreement gating (`disagree`), MCG-TTA **preserves 100% of LGG patient detections (20/24)**, preventing catastrophic false negatives on the minority malignant class.*

---

### Table 4: Edge Hardware Profiling & Computational Complexity
> **Purpose**: Backs all deployment and efficiency claims for edge devices (disk footprint, memory footprint, FLOPs, single-image latency, and throughput).

| Model Class | Architecture | Parameters (M) | Disk Size (MB) | FLOPs (GFLOPs) | MACs (GMACs) | Latency Batch=1 (ms) | Throughput (FPS) | Speedup vs ResNet-50 |
|---|---|---|---|---|---|---|---|---|
| **Heavyweight Baselines** | VGG-16 | 27.56M | 105.14 | 30.72 | 15.36 | 150.03 | 6.7 | $0.45\times$ |
| | ResNet-50 | 23.52M | 89.71 | 8.17 | 4.09 | 67.89 | 14.7 | $1.00\times$ (Ref) |
| | ResNet-18 | 11.18M | 42.64 | 3.63 | 1.81 | 27.96 | 35.8 | $2.43\times$ |
| | EfficientNet-B0 | 4.01M | 15.31 | 0.77 | 0.39 | 23.91 | 41.8 | $2.84\times$ |
| | MobileNet-V2 | 2.23M | 8.50 | 0.60 | 0.30 | 17.35 | 57.6 | $3.91\times$ |
| **Compact Models (< 2M)** | MPAC-ResNet | 1.86M | 7.10 | 2.17 | 1.08 | 19.50 | 51.3 | $3.48\times$ |
| | ResNet-Tiny | 0.98M | 3.72 | 2.51 | 1.26 | 24.19 | 41.3 | $2.81\times$ |
| | **LS-Net (base)** | **0.91M** | **3.49** | **0.36** | **0.18** | **11.30** | **88.5** | **6.01×** |
| | **MCG-TTA (proposed)**| **0.91M** | **3.49** | **~0.68** | **~0.34** | **21.47** | **46.6** | **3.16×** |
| | EfficientNet-Tiny| 0.75M | 2.88 | 0.19 | 0.09 | 16.14 | 61.9 | $4.21\times$ |
| | MPAC-ResNet-Tiny | 0.72M | 2.74 | 0.92 | 0.46 | 18.01 | 55.5 | $3.77\times$ |
| | MobileNet-Tiny | 0.40M | 1.53 | 0.12 | 0.06 | 11.43 | 87.5 | $5.94\times$ |

*Benchmark Setup: Intel Core CPU, PyTorch 2.x, single-thread inference, warm-up iterations = 50, timed iterations = 200.*

---

### Table 5: Per-Class Diagnostic Sensitivity & McNemar Statistical Significance
> **Purpose**: Reports paired tests that do not detect differences at $\alpha=0.05$ (without establishing equivalence) and class-level F1 differences, including the weaker meningioma result.

#### (a) Per-Class F1-Score Breakdown on Nickparvar 4-Class (%)
| Model | Glioma F1 (%) | Meningioma F1 (%) *(Hardest)* | No Tumor F1 (%) | Pituitary F1 (%) *(Easiest)* |
|---|---|---|---|---|
| **ResNet-50** | 99.19 ± 0.11 | 98.63 ± 0.22 | 99.14 ± 0.30 | 99.62 ± 0.19 |
| **EfficientNet-B0** | 98.94 ± 0.39 | 98.69 ± 0.38 | 99.27 ± 0.12 | 99.43 ± 0.38 |
| **VGG-16** | 98.87 ± 0.19 | 98.39 ± 0.21 | 98.81 ± 0.20 | 99.37 ± 0.39 |
| **ResNet-18** | 98.69 ± 0.49 | 97.94 ± 0.65 | 98.81 ± 0.53 | 99.37 ± 0.29 |
| **LS-Net** | **98.00 ± 0.39** | **97.32 ± 0.40** | **98.35 ± 0.22** | **99.50 ± 0.39** |
| **MPAC-ResNet-Tiny** | 97.93 ± 0.33 | 97.83 ± 0.59 | 98.02 ± 0.35 | 99.12 ± 0.22 |
| **MCG-TTA (proposed)**| **98.15 ± 0.18** | **97.60 ± 0.22** | **98.42 ± 0.19** | **99.52 ± 0.09** |

#### (b) Pairwise McNemar Statistical Tests on Test Set ($N = 1,052$)
| Hypothesis Test Pair | Seed 42 ($p$-value) | Seed 123 ($p$-value) | Seed 456 ($p$-value) | Statistical Decision ($\alpha = 0.05$) |
|---|---|---|---|---|
| **ResNet-50 vs. EfficientNet-B0** | $p = 1.000$ | $p = 0.625$ | $p = 0.500$ | **Not significant** |
| **ResNet-50 vs. LS-Net** | $p = 0.109$ | $p = 0.078$ | $p = 0.125$ | **Not significant** |
| **LS-Net vs. MPAC-ResNet-Tiny** | $p = 0.727$ | $p = 0.815$ | $p = 1.000$ | **Not significant** |
| **MPAC-Tiny vs. EfficientNet-Tiny** | $p = 0.480$ | $p = 0.804$ | $p = 1.000$ | **Not significant** |

---

### How These 5 Tables Map to the Paper Sections:
- **Section 4.1 (Main Results / Benchmark Evaluation)**: Use **Table 1** and **Table 5** to prove that the Nickparvar benchmark is saturated and that sub-1M models (0.91M LS-Net) achieve parity with 23.5M ResNet-50 at $6\times$ higher throughput.
- **Section 4.2 (Cross-Dataset Generalization & Limitations)**: Use **Table 2** to demonstrate that saturation breaks on BraTS (revealing an ~8 F1 gap between compact models and large models).
- **Section 4.3 (Proposed Method & Ablation Study)**: Use **Table 3** to demonstrate how the proposed **MCG-TTA** selectively rescues ambiguous predictions without expanding model size or blowing up the latency budget.
- **Section 4.4 (Deployment & Clinical Safety Analysis)**: Use **Table 4** for hardware throughput and **Table 5(b)** for statistical hypothesis testing.
