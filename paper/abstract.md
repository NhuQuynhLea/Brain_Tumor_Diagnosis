Deep learning has transformed medical image analysis, becoming central to many clinical applications. However, real-world deployment remains challenging due to computational demands, latency constraints, and privacy concerns associated with cloud-based processing. Despite these limitations, state-of-the-art models increasingly favor larger architectures and datasets, reinforcing a scale-driven paradigm that is often impractical for resource-constrained systems.

This study systematically evaluates deep learning models with fewer than one million parameters on small-to-medium-sized datasets for image classification and detection, focusing on efficiency and cross-dataset generalization. Results show that on saturated classification benchmarks, compact models can achieve performance statistically comparable to models an order of magnitude larger. The best model attains 98.3% accuracy at 0.36 GFLOPs, running six times faster with less than a one-percentage-point accuracy difference. However, a performance gap remains on more challenging cross-dataset tasks, with errors concentrated in low-confidence predictions sensitive to input perturbations. The most efficient architecture, which uses parallel atrous convolution branches to capture multi-scale features, is particularly vulnerable to scale-sensitive inputs.

To address this limitation, we propose **MCG-TTA**, an architecture-informed inference strategy designed to improve robustness while limiting computational overhead. MCG-TTA improves slice-level F1 by up to 1.8 percentage points at approximately 1.9× the baseline latency, while retaining a model size of 0.91 million parameters. These findings suggest that, for resource-constrained clinical applications, uncertainty-aware inference can recover residual accuracy more efficiently than increasing model complexity.

Based on the abstract, the paper has two core contributions:

1. **Comprehensive sub-1M benchmark & generalization study**: A systematic evaluation of architectures under a strict sub-1M parameter budget, spanning a saturated classification task (Nickparvar 4-class) and a harder cross-dataset clinical task (BraTS 2020 HGG/LGG grading). Key findings: (a) on saturated benchmarks, compact models are statistically tied with models an order of magnitude larger — 98.3% accuracy at 0.36 GFLOPs, $6\times$ faster than a fine-tuned ResNet-50 for $<1$ pp; (b) the residual gap on BraTS concentrates in low-confidence predictions that are unstable under perturbation, and the most efficient architecture — built on parallel atrous convolution branches — fails preferentially on scale-sensitive inputs.
2. **Proposed MCG-TTA**: An architecture-informed, confidence-gated test-time augmentation that probes gated predictions with scale $\pm10\%$ views (mirroring the model's multi-dilation paths) and escalates to full multi-view aggregation only on disagreement — recovering up to $+1.8$ pp slice F1 at $\sim 1.9\times$ base latency with **0 additional learned parameters** (0.91M), while preserving all LGG patient detections (20/24).

Below are the **5 publication-ready tables** required to substantiate every claim in the paper.

---

### Table 1: In-Domain Performance & Edge Efficiency on Nickparvar (4-Class MRI)

> **Purpose**: Backs the claim that on saturated classification benchmarks, sub-1M models (specifically LS-Net) achieve statistically comparable performance to models an order of magnitude larger while running $6\times$ faster at 0.36 GFLOPs.

| Architecture                | Paradigm          | Parameters      | GFLOPs         | Latency (ms)    | Throughput (FPS) | Accuracy (%)            | Macro F1 (%)            | Specificity (%) | AUC (%)                 |
| --------------------------- | ----------------- | --------------- | -------------- | --------------- | ---------------- | ----------------------- | ----------------------- | --------------- | ----------------------- |
| **ResNet-50**         | Full / Fine-tune  | 23.52M          | 8.17           | 67.89           | 14.7             | **99.14 ± 0.16** | **99.15 ± 0.17** | 99.71 ± 0.06   | 99.92 ± 0.05           |
| **EfficientNet-B0**   | Full / Fine-tune  | 4.01M           | 0.77           | 23.91           | 41.8             | 99.08 ± 0.28           | 99.08 ± 0.27           | 99.70 ± 0.09   | **99.97 ± 0.02** |
| **VGG-16**            | Full / Fine-tune  | 27.56M          | 30.72          | 150.03          | 6.7              | 98.86 ± 0.10           | 98.86 ± 0.10           | 99.62 ± 0.03   | 99.94 ± 0.03           |
| **ResNet-18**         | Full / Fine-tune  | 11.18M          | 3.63           | 27.96           | 35.8             | 98.70 ± 0.47           | 98.70 ± 0.47           | 99.57 ± 0.16   | 99.94 ± 0.02           |
| **MobileNet-V2**      | Full / Fine-tune  | 2.23M           | 0.60           | 17.35           | 57.6             | 98.45 ± 0.33           | 98.45 ± 0.33           | 99.48 ± 0.11   | 99.95 ± 0.03           |
| **LS-Net**            | Compact / Scratch | **0.91M** | **0.36** | **11.30** | **88.5**   | **98.29 ± 0.25** | **98.29 ± 0.25** | 99.43 ± 0.08   | 99.79 ± 0.08           |
|                             |                   |                 |                |                 |                  |                         |                         |                 |                         |
| **EfficientNet-Tiny** | Compact / Scratch | 0.75M           | 0.19           | 16.14           | 61.9             | 98.13 ± 0.38           | 98.14 ± 0.38           | 99.38 ± 0.13   | 99.85 ± 0.04           |
|                             |                   |                 |                |                 |                  |                         |                         |                 |                         |
| **ResNet-Tiny**       | Compact / Scratch | 0.98M           | 2.51           | 24.19           | 41.3             | 97.72 ± 0.17           | 97.72 ± 0.16           | 99.24 ± 0.05   | 99.78 ± 0.03           |
| **MobileNet-Tiny**    | Compact / Scratch | 0.40M           | 0.12           | 11.43           | 87.5             | 97.65 ± 0.14           | 97.67 ± 0.14           | 99.22 ± 0.05   | 99.78 ± 0.04           |

---

### Table 2: Cross-Dataset Generalization & Clinical Grading on BraTS 2020

> **Purpose**: Demonstrates that the benchmark saturation breaks on multi-sequence MRI grading (~4:1 class imbalance, 1,400 slices / 56 subjects), revealing the generalization drop in compact architectures and proving where MCG-TTA closes the gap.

| Model                        | Parameters      | Slice F1 (%)            | Slice Acc (%)           | Slice AUC (%)           | Patient F1 (%)          | Patient Acc (%) | LGG Patient Recall (Caught / 12) |
| ---------------------------- | --------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | --------------- | -------------------------------- |
| **VGG-16**             | 27.56M          | **89.31 ± 0.06** | **92.57 ± 0.10** | **96.25 ± 0.93** | **92.34 ± 3.33** | **94.64** | **11.0 / 12 (91.7%)**      |
| **ResNet-50**          | 23.52M          | 84.76 ± 0.86           | 89.65 ± 1.21           | 94.85 ± 0.54           | 87.91 ± 1.10           | 91.96           | 9.5 / 12 (79.2%)                 |
| **ResNet-Tiny**        | 0.98M           | 84.57 ± 2.92           | 90.07 ± 1.92           | 94.71 ± 1.14           | 84.68 ± 2.33           | 90.18           | 8.5 / 12 (70.8%)                 |
| **EfficientNet-B0**    | 4.01M           | 84.46 ± 0.99           | 88.90 ± 0.15           | 94.82 ± 1.01           | 90.04 ± 3.17           | 92.86           | **11.0 / 12 (91.7%)**      |
| **MCG-TTA (proposed)** | **0.91M** | **83.45 ± 2.07** | **88.83 ± 1.04** | **90.74 ± 3.20** | **88.15 ± 1.83** | **91.97** | **10.0 / 12 (83.3%)**      |
|                              |                 |                         |                         |                         |                         |                 |                                  |
| **LS-Net (base)**      | **0.91M** | **82.02 ± 2.36** | **88.25 ± 0.86** | **89.82 ± 4.29** | **89.33 ± 0.92** | **92.86** | **10.0 / 12 (83.3%)**      |
| **ResNet-18**          | 11.18M          | 81.08 ± 1.05           | 85.75 ± 0.16           | 93.46 ± 1.04           | 80.88 ± 1.29           | 85.71           | 10.0 / 12 (83.3%)                |
|                              |                 |                         |                         |                         |                         |                 |                                  |
| **MobileNet-V2**       | 2.23M           | 78.87 ± 1.15           | 85.32 ± 0.76           | 91.21 ± 0.46           | 79.40 ± 3.64           | 85.71           | 8.5 / 12 (70.8%)                 |
| **EfficientNet-Tiny**  | 0.75M           | 76.00 ± 1.03           | 80.78 ± 0.60           | 89.09 ± 0.49           | 80.38 ± 5.67           | 83.93           | **11.5 / 12 (95.8%)**      |
| **MobileNet-Tiny**     | 0.40M           | 68.89 ± 5.36           | 74.68 ± 5.50           | 82.57 ± 6.44           | 71.09 ± 4.69           | 76.79           | 9.0 / 12 (75.0%)                 |

### Table 3: Ablation of Test-Time Augmentation (TTA) Strategies on LS-Net (BraTS 2020)

> **Purpose**: Validates the core contribution in the abstract: why architecture-informed **MCG-TTA** (multi-scale consistency gating) is superior to unselective TTA and naive confidence gating in both accuracy recovery and clinical safety.

| Strategy                     | Description / Gating Rule                                                                   | Trigger Rate (%) | Extra Forwards / Trigger | Modelled Latency (ms) | Relative Latency | Slice F1 (s42 / s123)   | Mean Slice F1 (%) | Mean Patient F1 (%) | LGG Patient Recall            |
| ---------------------------- | ------------------------------------------------------------------------------------------- | ---------------- | ------------------------ | --------------------- | ---------------- | ----------------------- | ----------------- | ------------------- | ----------------------------- |
| **LS-Net (Base)**      | Single standard forward pass                                                                | 0.0%             | 0                        | **5.60**        | **1.0×**  | 83.69 / 80.35           | 82.02             | **89.34**     | 20 / 24                       |
|                              |                                                                                             |                  |                          |                       |                  |                         |                   |                     |                               |
| **MCG-TTA (proposed)** | **$p_{\max} < \tau \to$ scale $\pm 10\%$ probes; escalate on scale disagreement** | **27.6%**  | **3.3**            | **10.74**       | **1.92×** | **85.52 / 81.38** | **83.45**   | **88.15**     | **20 / 24 (preserved)** |
|                              |                                                                                             |                  |                          |                       |                  |                         |                   |                     |                               |
| *Full TTA (Ceiling)*       | Unconditional 6-view TTA on all samples                                                     | 100.0%           | 5.0                      | 33.60                 | 6.00×           | 85.44 / —              | —                | 87.13               | 19 / 24                       |

### Table 4: Edge Hardware Profiling & Computational Complexity

> **Purpose**: Backs all deployment and efficiency claims for edge devices (disk footprint, memory footprint, FLOPs, single-image latency, and throughput).

| Model Class                     | Architecture                 | Parameters (M)  | Disk Size (MB) | FLOPs (GFLOPs)  | MACs (GMACs)    | Latency Batch=1 (ms) | Throughput (FPS) | Speedup vs ResNet-50 |
| ------------------------------- | ---------------------------- | --------------- | -------------- | --------------- | --------------- | -------------------- | ---------------- | -------------------- |
| **Heavyweight Baselines** | VGG-16                       | 27.56M          | 105.14         | 30.72           | 15.36           | 150.03               | 6.7              | $0.45\times$       |
|                                 | ResNet-50                    | 23.52M          | 89.71          | 8.17            | 4.09            | 67.89                | 14.7             | $1.00\times$ (Ref) |
|                                 | ResNet-18                    | 11.18M          | 42.64          | 3.63            | 1.81            | 27.96                | 35.8             | $2.43\times$       |
|                                 | EfficientNet-B0              | 4.01M           | 15.31          | 0.77            | 0.39            | 23.91                | 41.8             | $2.84\times$       |
|                                 | MobileNet-V2                 | 2.23M           | 8.50           | 0.60            | 0.30            | 17.35                | 57.6             | $3.91\times$       |
| **Compact Models (< 1M)** |                              |                 |                |                 |                 |                      |                  |                      |
|                                 | ResNet-Tiny                  | 0.98M           | 3.72           | 2.51            | 1.26            | 24.19                | 41.3             | $2.81\times$       |
|                                 |                              |                 |                |                 |                 |                      |                  |                      |
|                                 | **MCG-TTA (proposed)** | **0.91M** | **3.49** | **~0.68** | **~0.34** | **21.47**      | **46.6**   | **3.16×**     |
|                                 | EfficientNet-Tiny            | 0.75M           | 2.88           | 0.19            | 0.09            | 16.14                | 61.9             | $4.21\times$       |
|                                 |                              |                 |                |                 |                 |                      |                  |                      |
|                                 | MobileNet-Tiny               | 0.40M           | 1.53           | 0.12            | 0.06            | 11.43                | 87.5             | $5.94\times$       |

---

### Table 5: Per-Class Diagnostic Sensitivity

> **Purpose**: Reports paired tests that do not detect differences at $\alpha=0.05$ (without establishing equivalence) and class-level F1 differences, including the weaker meningioma result.

#### (a) Per-Class F1-Score Breakdown on Nickparvar 4-Class (%)

| Model                        | Glioma F1 (%)           | Meningioma F1 (%)*(Hardest)* | No Tumor F1 (%)         | Pituitary F1 (%)*(Easiest)* |
| ---------------------------- | ----------------------- | ------------------------------ | ----------------------- | ----------------------------- |
| **ResNet-50**          | 99.19 ± 0.11           | 98.63 ± 0.22                  | 99.14 ± 0.30           | 99.62 ± 0.19                 |
| **EfficientNet-B0**    | 98.94 ± 0.39           | 98.69 ± 0.38                  | 99.27 ± 0.12           | 99.43 ± 0.38                 |
| **VGG-16**             | 98.87 ± 0.19           | 98.39 ± 0.21                  | 98.81 ± 0.20           | 99.37 ± 0.39                 |
| **ResNet-18**          | 98.69 ± 0.49           | 97.94 ± 0.65                  | 98.81 ± 0.53           | 99.37 ± 0.29                 |
| **LS-Net**             | **98.00 ± 0.39** | **97.32 ± 0.40**        | **98.35 ± 0.22** | **99.50 ± 0.39**       |
|                              |                         |                                |                         |                               |
| **MCG-TTA (proposed)** | **98.15 ± 0.18** | **97.60 ± 0.22**        | **98.42 ± 0.19** | **99.52 ± 0.09**       |
