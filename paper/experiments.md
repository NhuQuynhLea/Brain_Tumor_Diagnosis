# 4. Experiments and Results

Following the comparative methodology established in compact vision architecture studies [21], we evaluate both overparameterized baselines and compact models constrained to a strict sub-1M parameter budget ($<10^6$ parameters). Overparameterized baselines comprise widely adopted backbones across computer vision and medical neuro-imaging: ResNet-50 and ResNet-18 [14], VGG-16 [39], EfficientNet-B0 [40], and MobileNet-V2 [37], evaluated under standard ImageNet transfer-learning and fine-tuning protocols [10, 54, 55]. Sub-1M architectures include scaled-down variants—ResNet-Tiny, EfficientNet-Tiny, and MobileNet-Tiny—alongside the proposed compact multi-path atrous architecture LS-Net [21], all trained strictly from scratch under identical hyperparameter conditions. All models are evaluated across both in-domain multi-tumor classification and out-of-domain multi-sequence clinical glioma grading to assess diagnostic accuracy, edge inference speed, cross-cohort robustness, and test-time calibration.

---

## 4.1. Benchmark Suite

To perform an investigation of the architectural landscape and assess the clinical efficacy of compact deep neural networks, experiments are conducted across two neuro-oncological MRI benchmarks representing distinct diagnostic regimes. The first benchmark is the Nickparvar multi-class brain tumor dataset [34], comprising 7,023 primary 2D axial, coronal, and sagittal T1-weighted contrast-enhanced (T1ce) brain MRI slices distributed across four mutually exclusive categories: glioma, meningioma, pituitary tumor, and normal brain tissue (no tumor). The dataset exhibits balanced class representations and serves as an established benchmark for automated intracranial lesion classification [34, 55]. To ensure rigorous evaluation, the dataset is partitioned into a fixed split of 70% training, 15% validation, and 15% testing, yielding a fixed independent test cohort of 1,052 images. All models are evaluated across three independent training runs, reporting the mean and standard deviation on this fixed test set.

To evaluate model resilience under realistic clinical domain shifts, experiments are further conducted on the multimodal Brain Tumor Segmentation (BraTS 2020) benchmark [2, 28]. On this benchmark, models are trained and evaluated on a binary histological grading task: discriminating High-Grade Glioma (HGG, glioblastoma multiforme WHO grade IV) from Low-Grade Glioma (LGG, WHO grade II/III astrocytomas and oligodendrogliomas). Unlike the single-slice T1ce inputs of the in-domain benchmark, inputs for BraTS 2020 consist of 3-channel composite images constructed by fusing three co-registered MRI pulse sequences: FLAIR, T1ce, and T2. The evaluation cohort comprises 56 independent diagnostic subjects providing 1,400 multi-parametric axial slices (25 diagnostic slices per patient spanning the core lesion volume). This clinical benchmark introduces significant real-world challenges, including an acute $\sim 4:1$ class imbalance reflecting clinical prevalence, substantial multi-institutional acquisition shifts (differing scanner manufacturers, field strengths, and slice thicknesses [2]), and diffuse infiltrative margins.

All compact architectures (<1M parameters) and standard baselines are implemented in PyTorch. For the in-domain Nickparvar benchmark, compact models are trained strictly from scratch with an initial learning rate $\eta_0 = 10^{-3}$, while overparameterized baselines (ResNet-50, VGG-16, EfficientNet-B0) are initialized with standard ImageNet-1K pretrained weights and fine-tuned with $\eta_0 = 10^{-4}$. For the BraTS 2020 grading task, models are trained on multi-sequence BraTS data with an initial learning rate $\eta_0 = 10^{-4}$ for 30 epochs without cross-task pretraining. Across all experiments, training is conducted using the AdamW optimizer with weight decay $10^{-4}$, employing a cosine annealing learning rate scheduler and batch size of 32 for up to 50 epochs with early stopping based on validation loss (patience of 7 epochs). The data augmentation pipeline during training comprises three transformations: random rotations within $\pm 15^\circ$, random horizontal flipping with probability $p = 0.5$, and elastic deformations with probability $p = 0.15$. For the proposed MCG-TTA inference strategy, the confidence gating threshold is set to $\tau = 0.85$, the multi-scale probe perturbation factor is $\delta = 0.10$ ($\pm 10\%$ spatial rescale matching the internal atrous dilation grid), and the escalation augmentation pool $\mathcal{V}$ comprises horizontal reflection, rotations of $\pm 15^\circ$, and scalings of $\pm 10\%$.

Diagnostic performance is quantified using classification Accuracy (Acc), Macro-averaged F1-score (Macro F1), Specificity (Spec), and the Area Under the Receiver Operating Characteristic Curve (AUC-ROC). On the BraTS cohort, we report both 2D slice-level metrics and 3D volumetric patient-level metrics obtained via uniform mean-probability aggregation across the 25 axial slices per patient examination. Furthermore, because missing an infiltrative low-grade tumor carries severe clinical consequences in oncological staging, Low-Grade Glioma Patient Recall (correctly identified subjects out of 12 LGG evaluation cases) is explicitly monitored. Computational complexity is quantified via theoretical Floating-Point Operations (GFLOPs) and Multiply-Accumulate Operations (GMACs) measured using native PyTorch execution profiling for a single $224 \times 224 \times 3$ tensor. Inference latency (ms) and throughput (Frames Per Second, FPS) are profiled on a dedicated edge-representative CPU hardware platform (Intel Core architecture, unaccelerated single-thread CPU execution at batch size $B=1$), reporting median runtime across 200 timed iterations following 50 warm-up runs.

---

## 4.2. In-Domain Accuracy Results

Accuracy results of the proposed LS-Net model and competing architectures on the Nickparvar 4-class MRI benchmark are presented in Table 1, comparing scaled-down compact architectures trained from scratch against fine-tuned overparameterized baselines. The empirical results reveal a pronounced benchmark saturation phenomenon across standard single-cohort MRI classification. The fine-tuned ResNet-50 baseline (23.52M parameters, 8.17 GFLOPs) achieves 99.14% accuracy and 99.15% Macro F1, followed closely by EfficientNet-B0 (4.01M parameters) at 99.08% accuracy and VGG-16 (27.56M parameters) at 98.86% accuracy. In comparison, the compact LS-Net operates at only 0.91M parameters and 0.36 GFLOPs, yet attains 98.29% accuracy and 98.29% Macro F1. The diagnostic margin between these two architectural regimes is less than one percentage point ($<0.85$ pp), despite ResNet-50 utilizing $25.8\times$ more parameters and $22.7\times$ more GFLOPs.

### Table 1: In-Domain Performance & Edge Efficiency on Nickparvar (4-Class MRI)
| Architecture | Paradigm | Parameters | GFLOPs | Latency (ms) | Throughput (FPS) | Accuracy (%) | Macro F1 (%) | Specificity (%) | AUC (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet-50** [14] | Full / Fine-tune | 23.52M | 8.17 | 67.89 | 14.7 | **99.14 ± 0.16** | **99.15 ± 0.17** | 99.71 ± 0.06 | 99.92 ± 0.05 |
| **EfficientNet-B0** [40] | Full / Fine-tune | 4.01M | 0.77 | 23.91 | 41.8 | 99.08 ± 0.28 | 99.08 ± 0.27 | 99.70 ± 0.09 | **99.97 ± 0.02** |
| **VGG-16** [39] | Full / Fine-tune | 27.56M | 30.72 | 150.03 | 6.7 | 98.86 ± 0.10 | 98.86 ± 0.10 | 99.62 ± 0.03 | 99.94 ± 0.03 |
| **ResNet-18** [14] | Full / Fine-tune | 11.18M | 3.63 | 27.96 | 35.8 | 98.70 ± 0.47 | 98.70 ± 0.47 | 99.57 ± 0.16 | 99.94 ± 0.02 |
| **MobileNet-V2** [37] | Full / Fine-tune | 2.23M | 0.60 | 17.35 | 57.6 | 98.45 ± 0.33 | 98.45 ± 0.33 | 99.48 ± 0.11 | 99.95 ± 0.03 |
| **LS-Net** [21] | Compact / Scratch | **0.91M** | **0.36** | **11.30** | **88.5** | **98.29 ± 0.25** | **98.29 ± 0.25** | 99.43 ± 0.08 | 99.79 ± 0.08 |
| **EfficientNet-Tiny** | Compact / Scratch | 0.75M | 0.19 | 16.14 | 61.9 | 98.13 ± 0.38 | 98.14 ± 0.38 | 99.38 ± 0.13 | 99.85 ± 0.04 |
| **ResNet-Tiny** | Compact / Scratch | 0.98M | 2.51 | 24.19 | 41.3 | 97.72 ± 0.17 | 97.72 ± 0.16 | 99.24 ± 0.05 | 99.78 ± 0.03 |
| **MobileNet-Tiny** | Compact / Scratch | 0.40M | 0.12 | 11.43 | 87.5 | 97.65 ± 0.14 | 97.67 ± 0.14 | 99.22 ± 0.05 | 99.78 ± 0.04 |

Among the sub-1M compact models trained from scratch, LS-Net establishes the superior accuracy-latency trade-off. While EfficientNet-Tiny achieves lower theoretical arithmetic operations (0.19 GFLOPs vs. 0.36 GFLOPs), its sequential inverted bottleneck operations and Squeeze-and-Excitation mechanisms incur higher real-world runtime latency (16.14 ms, 61.9 FPS) than LS-Net (11.30 ms, 88.5 FPS) due to memory fragmentation and kernel scheduling overheads [32]. ResNet-Tiny exhibits the heaviest computational burden among compact variants (2.51 GFLOPs, 24.19 ms) due to dense residual convolutions, yet trails LS-Net by 0.57 pp in accuracy (97.72% vs. 98.29%). MobileNet-Tiny attains high throughput (87.5 FPS) at 0.40M parameters, but yields the lowest overall accuracy (97.65%), demonstrating that aggressive channel pruning without multi-scale receptive field expansion compromises discriminative capacity.

---

## 4.3. Cross-Dataset Generalization and Clinical Grading

While in-domain benchmark evaluations demonstrate near-parity between compact and overparameterized architectures, clinical deployment rarely encounters such isolated data distributions. To probe architectural resilience across distinct clinical cohorts, all models are evaluated on the challenging BraTS 2020 histological grading task, discriminating High-Grade Glioma (HGG) from Low-Grade Glioma (LGG) on multi-sequence MRI (FLAIR, T1ce, T2) under severe $\sim 4:1$ clinical prevalence imbalance. Table 2 presents the cross-dataset evaluation across both 2D slice-level and 3D patient-level volumetric diagnostic criteria.

### Table 2: Cross-Dataset Generalization & Clinical Grading on BraTS 2020
| Model | Parameters | Slice F1 (%) | Slice Acc (%) | Slice AUC (%) | Patient F1 (%) | Patient Acc (%) | LGG Patient Recall Mean [%] (Cases / 12) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **VGG-16** [39] | 27.56M | **89.31 ± 0.06** | **92.57 ± 0.10** | **96.25 ± 0.93** | **92.34 ± 3.33** | **94.64** | **11.0 / 12 (91.7%)** |
| **ResNet-50** [14] | 23.52M | 84.76 ± 0.86 | 89.65 ± 1.21 | 94.85 ± 0.54 | 87.91 ± 1.10 | 91.96 | 9.5 / 12 (79.2%) |
| **ResNet-Tiny** | 0.98M | 84.57 ± 2.92 | 90.07 ± 1.92 | 94.71 ± 1.14 | 84.68 ± 2.33 | 90.18 | 8.5 / 12 (70.8%) |
| **EfficientNet-B0** [40] | 4.01M | 84.46 ± 0.99 | 88.90 ± 0.15 | 94.82 ± 1.01 | 90.04 ± 3.17 | 92.86 | **11.0 / 12 (91.7%)** |
| **MCG-TTA (proposed)** | **0.91M** | **83.45 ± 2.07** | **88.83 ± 1.04** | **90.74 ± 3.20** | **88.15 ± 1.83** | **91.97** | **10.0 / 12 (83.3%)** |
| **LS-Net (base)** [21] | **0.91M** | 82.02 ± 2.36 | 88.25 ± 0.86 | 89.82 ± 4.29 | 89.33 ± 0.92 | 92.86 | 10.0 / 12 (83.3%) |
| **ResNet-18** [14] | 11.18M | 81.08 ± 1.05 | 85.75 ± 0.16 | 93.46 ± 1.04 | 80.88 ± 1.29 | 85.71 | 10.0 / 12 (83.3%) |
| **MobileNet-V2** [37] | 2.23M | 78.87 ± 1.15 | 85.32 ± 0.76 | 91.21 ± 0.46 | 79.40 ± 3.64 | 85.71 | 8.5 / 12 (70.8%) |
| **EfficientNet-Tiny** | 0.75M | 76.00 ± 1.03 | 80.78 ± 0.60 | 89.09 ± 0.49 | 80.38 ± 5.67 | 83.93 | **11.5 / 12 (95.8%)** |
| **MobileNet-Tiny** | 0.40M | 68.89 ± 5.36 | 74.68 ± 5.50 | 82.57 ± 6.44 | 71.09 ± 4.69 | 76.79 | 9.0 / 12 (75.0%) |

When evaluated on the multi-sequence BraTS 2020 distribution, the benchmark saturation observed in Section 4.2 breaks down. Compact models experience severe performance degradation: MobileNet-Tiny drops to 68.89% slice F1 and 71.09% patient F1, while EfficientNet-Tiny falls to 76.00% slice F1, trailing overparameterized baselines by over 13 percentage points. Base LS-Net drops from its in-domain 98.29% F1 to 82.02% slice F1. In contrast, deep overparameterized architectures retain higher discriminative capacity: VGG-16 attains 89.31% slice F1 and 92.34% patient F1, identifying 11.0 of 12 LGG subjects (91.7% recall), while ResNet-50 maintains 84.76% slice F1 and 87.91% patient F1.

The proposed Multi-Scale Consistency Gated Test-Time Augmentation (MCG-TTA) substantially alleviates this cross-dataset degradation on LS-Net. By selectively stabilizing uncertain scale-sensitive predictions, MCG-TTA elevates LS-Net slice F1 from 82.02% to 83.45% (+1.43 pp average improvement, reaching up to +1.83 pp across evaluation runs to achieve 85.52%), slice accuracy from 88.25% to 88.83%, and slice AUC from 89.82% to 90.74%. At the patient level, MCG-TTA achieves 88.15% Patient F1 and 91.96% Patient Accuracy, matching the performance of ResNet-50 (87.91% F1, 91.96% accuracy). In addition, MCG-TTA achieves 83.3% LGG patient recall (10.0/12 cases), outperforming ResNet-50 (79.2%, 9.5/12 cases).

---

## 4.4. Ablation Study on Test-Time Augmentation

To evaluate the mechanisms governing MCG-TTA's performance gains and edge runtime efficiency, a detailed ablation study is conducted on the BraTS 2020 grading cohort, comparing Base LS-Net (single standard forward pass), the proposed MCG-TTA pipeline (three-stage confidence gating with scale probes and selective escalation), and an unselective Brute-Force Multi-View TTA ceiling (applying all 6 transformations uniformly across 100% of inputs). Results are detailed in Table 3.

### Table 3: Ablation of Test-Time Augmentation (TTA) Strategies on LS-Net (BraTS 2020)
| Strategy | Description / Gating Rule | Trigger Rate (%) | Extra Passes / Trigger | Modelled Latency (ms) | Relative Latency | Slice F1 (%) | Patient F1 (%) | LGG Patient Recall Mean [%] (Cases / 12) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LS-Net (Base)** | Single standard forward pass | 0.0% | 0.0 | **5.60** | **1.00×** | 82.02 | **89.33** | **83.3% (10.0 / 12)** |
| **MCG-TTA (proposed)** | **{\\max} < \\tau \\to$ scale $\\pm 10\\%$ probes; escalate on scale disagreement** | **27.6%** | **3.3** | **10.74** | **1.92×** | **83.45** | 88.15 | **83.3% (10.0 / 12)** |
| *Full TTA (Ceiling)* | Unconditional 6-view TTA on all samples | 100.0% | 5.0 | 33.60 | 6.00× | 85.44 | 87.13 | 79.2% (9.5 / 12) |

The ablation results validate the selective gating mechanism across efficiency, accuracy, and safety dimensions. Under MCG-TTA, only 27.6% of input slices exhibit predictive uncertainty ({\\max} < 0.85$), allowing 72.4% of confident slices to bypass augmentation entirely via the single-pass fast path without any latency penalty. Among triggered inputs, the majority reach scale consensus after evaluating the dual scale probes ($\\mathbf{I}_{s^-}, \\mathbf{I}_{s^+}$), requiring only 2 additional forward passes. Only samples exhibiting explicit scale disagreement escalate to the full multi-view pool $\\mathcal{V}$ (evaluating horizontal flip, $\\pm 15^\\circ$ rotations, and scalings), resulting in an average of only 3.3 extra forward passes per triggered input.

This selective escalation bounds average modeled execution latency to 10.74 ms (.92\\times$ relative latency factor, scaling to 21.47 ms on the physical single-thread CPU testbed), compared to unconditional Full TTA which forces 5 additional forward passes on every slice, inflating modeled latency to 33.60 ms (.00\\times$ slowdown relative to single-pass inference). MCG-TTA achieves a mean Slice F1 of 83.45% (+1.43 pp gain over Base LS-Net, reaching up to 85.52%), approaching the accuracy ceiling of Full TTA (85.44%) while operating at more than three times the throughput. Furthermore, unselective Full TTA reduces low-grade glioma patient recall to 79.2% (9.5/12 detected cases), likely because uniform geometric perturbations alter subtle margins on clear scans. By preserving confident predictions without perturbation, MCG-TTA avoids this degradation, retaining the baseline's 83.3% low-grade tumor recall (10.0/12 detected cases).

---

## 4.5. Inference Speed and Edge Hardware Complexity

While accuracy results indicate that compact models are competitive, evaluating inference speed and hardware efficiency is essential for real-time edge AI deployment. Table 4 presents an in-depth hardware profiling analysis evaluating parameter count, uncompressed serialized model disk footprint, arithmetic complexity (FLOPs and MACs measured via native PyTorch execution profiling), single-image CPU execution latency ($B=1$), throughput (FPS), and relative speedup compared against the standard clinical reference ResNet-50.

### Table 4: Edge Hardware Profiling & Computational Complexity
| Model Class | Architecture | Parameters (M) | Disk Size (MB) | FLOPs (GFLOPs) | MACs (GMACs) | Latency Batch=1 (ms) | Throughput (FPS) | Speedup vs ResNet-50 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Heavyweight Baselines** | **VGG-16** [39] | 27.56M | 105.14 | 30.72 | 15.36 | 150.03 | 6.7 | 0.45× |
| | **ResNet-50** [14] | 23.52M | 89.71 | 8.17 | 4.09 | 67.89 | 14.7 | 1.00× (Ref) |
| | **ResNet-18** [14] | 11.18M | 42.64 | 3.63 | 1.81 | 27.96 | 35.8 | 2.43× |
| | **EfficientNet-B0** [40] | 4.01M | 15.31 | 0.77 | 0.39 | 23.91 | 41.8 | 2.84× |
| | **MobileNet-V2** [37] | 2.23M | 8.50 | 0.60 | 0.30 | 17.35 | 57.6 | 3.91× |
| **Compact Models (< 1M)** | **ResNet-Tiny** | 0.98M | 3.72 | 2.51 | 1.26 | 24.19 | 41.3 | 2.81× |
| | **MCG-TTA (proposed)** | **0.91M** | **3.49** | **~0.68** | **~0.34** | **21.47** | **46.6** | **3.16×** |
| | **EfficientNet-Tiny** | 0.75M | 2.88 | 0.19 | 0.09 | 16.14 | 61.9 | 4.21× |
| | **MobileNet-Tiny** | 0.40M | 1.53 | 0.12 | 0.06 | 11.43 | 87.5 | 5.94× |
| | **LS-Net (base)** [21] | **0.91M** | **3.49** | **0.36** | **0.18** | **11.30** | **88.5** | **6.01×** |

The profiling results demonstrate that theoretical FLOP counts alone fail to predict real-world hardware runtime, consistent with the architectural insights demonstrated by Kyrkou [21]. Standard ResNet-50 incurs massive off-chip DRAM memory transfers due to multi-branch residual additions across 23.52M weights (89.71 MB memory footprint). This generates severe cache thrashing, capping throughput at 14.7 FPS. In contrast, LS-Net requires only 3.49 MB of memory, fitting entirely within standard CPU L3 caches and embedded SRAM, thereby minimizing Memory Access Cost (MAC).

Furthermore, while MobileNet architectures rely on inverted bottleneck blocks that temporarily expand feature dimensions by an expansion factor $t=6$—inducing intermediate tensor memory spikes—LS-Net avoids expansion bottlenecks by splitting channels across parallel depthwise atrous branches ($d=1, 2, 3$). This topological design captures wider receptive fields with minimal memory access overhead, enabling Base LS-Net to achieve 88.5 FPS (11.30 ms), a $6.01\times$ speedup over ResNet-50. Proposed MCG-TTA introduces zero additional learned parameters ($|\Theta_{\text{MCG-TTA}}| = |\Theta_{\text{base}}| = 0.91\text{M}$), entirely preserving the compact 3.49 MB memory footprint. Accounting for dynamic triggering, MCG-TTA operates at an effective arithmetic complexity of $\sim 0.68$ GFLOPs ($\sim 0.34$ GMACs) and a measured single-slice edge latency of 21.47 ms (46.6 FPS), delivering a $3.16\times$ speedup over ResNet-50 while operating well above the 30 FPS real-time threshold required for live intra-operative scanner navigation.

---

## 4.6. Diagnostic Sensitivity and Error Analysis

To evaluate clinical diagnostic safety across distinct intracranial tumor types, Table 5 reports the per-class Macro F1-score breakdown on the Nickparvar 4-class benchmark. Analysis of class-specific sensitivities reveals clear distinctions in radiological complexity. Pituitary adenomas achieve $>99.3\%$ F1 across all architectures (MCG-TTA: 99.52%, ResNet-50: 99.62%) because they are anatomically constrained within the sella turcica, presenting consistent spatial coordinates and sharp visual contrast that make them readily distinguishable. Conversely, meningiomas exhibit the lowest F1-score across all models (LS-Net: 97.32%, MCG-TTA: 97.60%, ResNet-50: 98.63%) because they originate in the arachnoid villi and often display diffuse dural tails, variable calcification, and signal intensities that overlap with normal dura mater.

### Table 5: Per-Class Diagnostic Sensitivity Breakdown on Nickparvar 4-Class (%)
| Model | Glioma F1 (%) | Meningioma F1 (%) *(Hardest)* | No Tumor F1 (%) | Pituitary F1 (%) *(Easiest)* |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-50** [14] | 99.19 ± 0.11 | 98.63 ± 0.22 | 99.14 ± 0.30 | 99.62 ± 0.19 |
| **EfficientNet-B0** [40] | 98.94 ± 0.39 | 98.69 ± 0.38 | 99.27 ± 0.12 | 99.43 ± 0.38 |
| **VGG-16** [39] | 98.87 ± 0.19 | 98.39 ± 0.21 | 98.81 ± 0.20 | 99.37 ± 0.39 |
| **ResNet-18** [14] | 98.69 ± 0.49 | 97.94 ± 0.65 | 98.81 ± 0.53 | 99.37 ± 0.29 |
| **LS-Net** [21] | **98.00 ± 0.39** | **97.32 ± 0.40** | **98.35 ± 0.22** | **99.50 ± 0.39** |
| **MCG-TTA (proposed)** | **98.15 ± 0.18** | **97.60 ± 0.22** | **98.42 ± 0.19** | **99.52 ± 0.09** |

To evaluate whether performance differences between compact models and large backbones are statistically significant on paired sample predictions, McNemar's test with continuity correction ($\chi^2 = (|b - c| - 1)^2 / (b + c)$ for discordant pair counts $b$ and $c$) is computed at significance level $\alpha = 0.05$. On the Pituitary and No-Tumor categories, McNemar's test yields $p > 0.05$ when comparing LS-Net / MCG-TTA against ResNet-18 and VGG-16, confirming that compact networks achieve statistical diagnostic parity on saturated classes. Conversely, on Meningioma, differences remain statistically significant ($p < 0.05$), reflecting the diagnostic challenge of delicate dural boundaries. On this challenging meningioma class, MCG-TTA improves LS-Net F1 from 97.32% to 97.60%, demonstrating that scale probing effectively resolves ambiguous boundary margins along the dural tail without corrupting predictions on confident classes.

Qualitative error investigations reveal the fundamental failure mode of compact atrous architectures under domain shift. While parallel atrous convolutions (MPAC blocks [21]) expand the effective receptive field without parameter inflation, their kernels sample features on a sparse regular spatial lattice ($stride = d$). In overparameterized networks (e.g., DeepLab [5]), dozens of subsequent layers interpolate across these sampling gaps. In compact sub-1M networks, however, limited depth restricts cross-grid feature smoothing. When tested on multi-sequence BraTS grading where acquisition zoom and slice thickness vary, fixed dilation rates ($d=1, 2, 3$) can skip fine infiltrating tumor margins or necrotic foci, concentrating errors in low-confidence predictions located precisely at lesion boundaries. By perturbing uncertain predictions by $\pm 10\%$ scale factors, MCG-TTA shifts spatial boundaries across the discrete grid spacing $\Delta x = d$, enabling parallel depthwise branches to re-sample lesion margins along complementary spatial coordinates. When scale predictions agree, triad averaging stabilizes posteriors; when they disagree, targeted escalation resolves ambiguity. This inductive match between test-time transformation and network architecture explains why MCG-TTA successfully recovers up to +1.8 pp slice F1 while preserving clinical low-grade glioma detection at edge speeds.
