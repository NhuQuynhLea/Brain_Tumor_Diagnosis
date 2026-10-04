# 🧠 Integration Plan: LS-Net (CVPR 2026) × Brain Tumor Diagnosis Project

## Paper Synopsis — What LiteSpeed-Net Actually Does

**Paper**: *"Rethinking Compact 1M Vision Models: Balancing Accuracy and Speed through CVPR"* — Kyrkou, CVPR 2026

**Core idea**: In the **sub-1M parameter regime**, existing architectures (ResNet, MobileNet, EfficientNet, ViT, etc.) were all designed for multi-million parameters and then naively scaled *down* via channel-width multipliers. This causes them to lose discriminative power unevenly. LS-Net proposes a **bottom-up architecture** with a novel **Multi-Path Atrous Convolution (MPAC)** block that:

1. Uses **parallel depthwise atrous (dilated) convolutions** at different dilation rates (d=1, d=2, ...) to capture multi-scale spatial context cheaply
2. **Concatenates** the outputs (free channel expansion — no extra compute vs standard conv)
3. Uses **only depthwise convolutions** (not full convolutions) for spatial ops → extremely parameter-efficient
4. Employs **1×1 pointwise convolutions** for channel mixing (highly parallelizable on modern hardware)
5. Uses **LeakyReLU** + **BatchNorm** (no expensive operations like SE blocks, H-Swish)
6. Applies **early downsampling** + fewer filters in initial layers

**Key results**: LS-Net (∼919K params, 295M MACs) beats all scaled-down sub-1M models on ImageNet-16, CIFAR-10, Intel Image Classification, and object detection — while achieving **220 FPS on Jetson Orin** (vs 128 FPS for ResNet, 70 FPS for MobileNetV2).

---

## 🔗 What Aligns With Our Project

Our project already has:
- ✅ 5-model comparative study (Custom CNN, VGG16, ResNet50, EfficientNetB0, MobileNetV2)
- ✅ Freeze vs fine-tune analysis
- ✅ Multi-seed stability analysis (3 seeds × 5 models)
- ✅ ResNet50-FT as best model (99.14% ± 0.16%)
- ✅ Parameter counting + FLOPs + inference latency measurements
- ✅ Accuracy-efficiency trade-off scatter plot
- ✅ Full training/evaluation infrastructure

> [!IMPORTANT]
> The paper's paradigm of **downscaling established architectures to sub-1M and comparing** is **exactly** what we can apply to brain tumor MRI classification. The novelty contribution becomes: *"We investigate whether the MPAC architectural principle — multi-path atrous depthwise convolutions with concatenation-based channel expansion — can enhance compact models for medical image classification, using brain tumor MRI as the target domain."*

---

## 🏗️ Architecture Design: What to Build

### Contribution 1: MPAC-Enhanced ResNet (our proposed model)

**Idea**: Replace standard residual blocks in a compact ResNet variant with MPAC-augmented blocks. This is **not** just copying LS-Net — it's a novel hybrid that combines:
- ResNet's proven residual learning framework (critical for medical imaging where pretrained weights exist)
- MPAC's multi-scale spatial diversity (critical for brain tumors that vary in size/shape)

```
┌─────────────────────────────────────────────────────────┐
│                  MPAC-ResNet Architecture                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input (224×224×3)                                        │
│      ↓                                                   │
│  [Stem] Conv2d 7×7/2 → BN → LeakyReLU → MaxPool         │
│      ↓                                                   │
│  [Stage 1] ResBlock × 2  (64 ch, 56×56)                  │
│      ↓                                                   │
│  [Stage 2] ResBlock × 2  (128 ch, 28×28)                 │
│      ↓                                                   │
│  ┌─────────────────────────────────────────┐             │
│  │ [Stage 3] MPAC-ResBlock × 2 (256 ch)    │ ← NOVEL     │
│  │   Each block:                            │             │
│  │   - 1×1 reduce → DW Atrous d=1 ─┐       │             │
│  │                   DW Atrous d=2 ─┤ concat │             │
│  │   - BN + LeakyReLU               │       │             │
│  │   - 1×1 project + residual       │       │             │
│  └─────────────────────────────────────────┘             │
│      ↓                                                   │
│  ┌─────────────────────────────────────────┐             │
│  │ [Stage 4] MPAC-ResBlock × 2 (512 ch)    │ ← NOVEL     │
│  └─────────────────────────────────────────┘             │
│      ↓                                                   │
│  [GAP] → Dropout → FC(4)                                 │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Why Stages 3-4 only?** Early stages capture low-level features (edges, textures) — standard convolutions are sufficient. Stages 3-4 handle higher-level semantic features where multi-scale context matters most for distinguishing glioma vs meningioma vs pituitary tumors (they differ in shape, location, and boundary patterns).

### Contribution 2: Downscaling Study — Sub-1M Variants

Following the paper's paradigm, we also create sub-1M parameter versions of our models:

| Model Variant | Target Params | How to Scale Down |
|---|---|---|
| ResNet-Tiny | ~500K–1M | Reduce layers to [1,1,1,1], narrow channels |
| MPAC-ResNet-Tiny | ~500K–1M | Same + MPAC in later stages |
| EfficientNet-Tiny | ~500K–1M | Width multiplier 0.35 |
| MobileNetV2-Tiny | ~500K–1M | Width multiplier 0.35 |
| Custom CNN-Tiny | ~500K | Halve channel dims |
| LS-Net (original) | ~919K | Implement from paper/GitHub |

### Contribution 3: Standalone LS-Net Implementation

Implement the **original LS-Net** from the paper as a 7th model in our comparison. This serves as:
- A reference implementation to validate our understanding of MPAC
- A direct comparison point for our MPAC-ResNet hybrid

---

## 📐 MPAC Block — Detailed Implementation Design

```python
class MPACBlock(nn.Module):
    """Multi-Path Atrous Convolution block from LS-Net (Kyrkou, CVPR 2026).
    
    Applies M parallel depthwise atrous convolutions with different dilation
    rates and concatenates outputs for channel expansion without extra compute.
    """
    def __init__(self, in_channels, dilation_rates=[1, 2], 
                 kernel_size=3, use_residual=True, alpha=1.0):
        # Optional 1×1 channel expansion/reduction (controlled by alpha)
        # M parallel: DepthwiseConv2d(in_ch, in_ch, k=3, dilation=d_i)
        # Concat → out_channels = in_channels * M
        # BN + LeakyReLU
        # 1×1 pointwise projection back to target channels
        # Residual connection (if dimensions match)
```

**Key design decisions for brain tumor MRI domain**:
- **Dilation rates [1, 2]** (not [1, 2, 4]) — MRI images at 224×224 don't benefit from very large receptive fields (unlike satellite/aerial images)
- **Use residual connections** — essential for gradient flow in medical imaging where fine details matter
- **alpha=1.0** (no expansion) — keeps parameter count controlled

---

## 🔄 What We Can Reuse From Existing Project

| Component | File | Reusable? | Notes |
|---|---|---|---|
| Data pipeline | [`src/data/dataset.py`](file:///d:/Project/Brain_Tumor_Diagnosis/src/data/dataset.py), [`augmentation.py`](file:///d:/Project/Brain_Tumor_Diagnosis/src/data/augmentation.py), [`preprocessing.py`](file:///d:/Project/Brain_Tumor_Diagnosis/src/data/preprocessing.py) | ✅ 100% | Same dataset, same splits, same augmentation |
| Model factory | [`src/models/model_factory.py`](file:///d:/Project/Brain_Tumor_Diagnosis/src/models/model_factory.py) | ✅ Extend | Just register new models in `MODEL_REGISTRY` |
| Training loop | [`src/training/trainer.py`](file:///d:/Project/Brain_Tumor_Diagnosis/src/training/trainer.py) | ✅ 100% | Identical training procedure |
| Loss/optimizer | [`src/training/losses.py`](file:///d:/Project/Brain_Tumor_Diagnosis/src/training/losses.py) | ✅ 100% | Same CrossEntropy + AdamW + Cosine |
| Checkpointing | [`src/training/checkpoint.py`](file:///d:/Project/Brain_Tumor_Diagnosis/src/training/checkpoint.py) | ✅ 100% | Same save/load logic |
| Evaluation metrics | [`src/evaluation/metrics.py`](file:///d:/Project/Brain_Tumor_Diagnosis/src/evaluation/metrics.py) | ✅ 100% | Same Acc/F1/AUC/specificity |
| Grad-CAM | [`src/evaluation/gradcam.py`](file:///d:/Project/Brain_Tumor_Diagnosis/src/evaluation/gradcam.py) | ✅ 100% | Auto-detects last Conv2d |
| Complexity analysis | [`src/evaluation/complexity.py`](file:///d:/Project/Brain_Tumor_Diagnosis/src/evaluation/complexity.py) | ✅ 100% | Params + FLOPs + latency |
| Train scripts | [`scripts/train.py`](file:///d:/Project/Brain_Tumor_Diagnosis/scripts/train.py), [`train_all.py`](file:///d:/Project/Brain_Tumor_Diagnosis/scripts/train_all.py) | ✅ 100% | Just add `--model mpac_resnet` etc. |
| Evaluate/compare | [`scripts/evaluate.py`](file:///d:/Project/Brain_Tumor_Diagnosis/scripts/evaluate.py), [`scripts/compare_models.py`](file:///d:/Project/Brain_Tumor_Diagnosis/scripts/compare_models.py) | ✅ 100% | Handles any registered model |
| Paper figures | [`scripts/paper_figures.py`](file:///d:/Project/Brain_Tumor_Diagnosis/scripts/paper_figures.py) | ✅ Extend | Add new models to plots |
| Config system | [`configs/config.yaml`](file:///d:/Project/Brain_Tumor_Diagnosis/configs/config.yaml) | ✅ 100% | Works with any model name |
| Multi-seed runner | [`scripts/run_all.py`](file:///d:/Project/Brain_Tumor_Diagnosis/scripts/run_all.py) | ✅ 100% | Supports `--seed` override |
| Existing results | [`results/`](file:///d:/Project/Brain_Tumor_Diagnosis/results) | ✅ Keep | Baseline comparison numbers already done |

> [!TIP]
> **The beauty of your existing infrastructure**: We need to write **only 3 new Python files** (MPAC block, MPAC-ResNet model, LS-Net model) + update the model registry. Everything else — training, evaluation, visualization, statistical testing — is already built.

---

## 📝 Detailed Step-by-Step Plan

### Phase A: Implement Novel Architectures (3–4 days)

| Step | Task | Files to Create/Modify | Deliverable |
|---|---|---|---|
| A.1 | **Implement MPAC block** | `src/models/mpac_block.py` (NEW) | Standalone, tested module |
| A.2 | **Implement MPAC-ResNet** — replace Stages 3-4 BasicBlocks with MPAC-ResBlocks | `src/models/mpac_resnet.py` (NEW) | Full-size and Tiny variants |
| A.3 | **Implement original LS-Net** — from paper arch (stem → stacked MPAC → GAP → FC) | `src/models/litespeed_net.py` (NEW) | ~919K param model |
| A.4 | **Register all new models** in model factory | `src/models/model_factory.py` (EDIT) | `mpac_resnet`, `mpac_resnet_tiny`, `lsnet` available |
| A.5 | **Smoke test** — verify forward/backward on real batch | `scripts/smoke_test_models.py` (EXTEND) | All new models pass |

### Phase B: Training Experiments (5–7 days compute)

| Step | Task | Command | Expected Outcome |
|---|---|---|---|
| B.1 | **Train MPAC-ResNet (full, pretrained backbone)** — freeze Stages 1-2 of ResNet, train MPAC Stages 3-4 + head | `python scripts/train.py --model mpac_resnet --finetune --epochs 30` | Target: ≥99% acc |
| B.2 | **Train MPAC-ResNet (full, from scratch)** | `python scripts/train.py --model mpac_resnet --no-pretrained --epochs 50` | Compare with/without pretrain |
| B.3 | **Train LS-Net (from scratch)** — no pretrained weights available | `python scripts/train.py --model lsnet --no-pretrained --epochs 50` | Establish LS-Net baseline on our dataset |
| B.4 | **Train all sub-1M Tiny variants** from scratch | `python scripts/train_all.py --models mpac_resnet_tiny resnet_tiny efficientnet_tiny mobilenet_tiny custom_cnn_tiny lsnet --epochs 50` | Downscaling comparison |
| B.5 | **Multi-seed experiments** (seeds 42, 123, 456) for all new models | 3× `python scripts/train_all.py ... --seed <S>` | Stability analysis |

### Phase C: Evaluation & Analysis (2–3 days)

| Step | Task | Details |
|---|---|---|
| C.1 | **Full evaluation suite** for all new models | Confusion matrices, ROC/AUC, per-class F1 |
| C.2 | **Statistical testing** — McNemar between MPAC-ResNet vs ResNet50-FT, MPAC-ResNet vs LS-Net | p-values for significance |
| C.3 | **Parameter/FLOPs/FPS comparison table** — replicate Table 2 from the paper but for our models | Acc vs efficiency scatter |
| C.4 | **Grad-CAM comparison** — do MPAC blocks attend to more clinically-relevant regions? | Visual comparison figure |
| C.5 | **Downscaling analysis** — how gracefully does each architecture degrade at sub-1M? | Accuracy-drop curves (Fig 4 style from paper) |
| C.6 | **Cross-dataset eval** on Figshare (decontaminated subset) | Generalization of MPAC-ResNet |

### Phase D: Paper Integration (2–3 days)

| Step | Task |
|---|---|
| D.1 | Update **SOTA comparison table** — add LS-Net paper results + our new models |
| D.2 | Write **"Proposed Method" section** — MPAC-ResNet architecture + motivation |
| D.3 | Write **"Downscaling Analysis" section** — following the paper's sub-1M paradigm |
| D.4 | Update **accuracy-efficiency figures** with all new models |
| D.5 | Revise **Contributions list** in Introduction |

---

## 🎯 Resulting Paper Contributions

After integration, the paper will claim these **novel contributions**:

### Contribution 1: MPAC-Enhanced ResNet for Brain Tumor Classification
> *"We propose MPAC-ResNet, a novel architecture that integrates Multi-Path Atrous Convolutions (inspired by LiteSpeed-Net) into the ResNet framework for brain tumor MRI classification. By replacing standard convolutional blocks in the deeper stages with MPAC blocks, we capture multi-scale spatial features critical for distinguishing tumor types that vary in size, shape, and boundary patterns."*

### Contribution 2: Compact Model Design Space Exploration for Medical Imaging
> *"We conduct the first systematic study of sub-1M parameter architectures for brain tumor classification, demonstrating which architectural principles (multi-path vs. depthwise separable vs. inverted residual vs. channel shuffling) are most resilient to extreme parameter reduction in the medical imaging domain."*

### Contribution 3: Most Comprehensive Comparative Study
> *"We compare 8+ architectures (Custom CNN, VGG16, ResNet50, EfficientNetB0, MobileNetV2, MPAC-ResNet, LS-Net, + tiny variants) across freeze/fine-tune/from-scratch training, multi-seed stability, accuracy-efficiency trade-offs, and cross-dataset generalization — the most comprehensive such study on this dataset."*

---

## ⚠️ Novelty Justification & Differentiation from LS-Net

| Aspect | LS-Net Paper | Our Work |
|---|---|---|
| **Domain** | General vision (CIFAR-10, Intel, ImageNet-16) | **Medical imaging** (brain tumor MRI) — different data distribution |
| **Architecture** | Standalone from-scratch model | **Hybrid**: MPAC blocks integrated into ResNet residual framework |
| **Training** | From scratch only | From scratch + **transfer learning** (ImageNet pretrained backbone) |
| **Analysis** | Sub-1M regime only | Full-size pretrained models + sub-1M regime |
| **Evaluation** | General metrics (accuracy, FPS) | Clinical metrics + **Grad-CAM interpretability** + **cross-dataset generalization** |
| **Contribution** | Propose MPAC block for general vision | Adapt & validate MPAC for **medical image classification** |

> [!IMPORTANT]
> **This is NOT plagiarism or replication.** We are:
> 1. **Borrowing an architectural building block** (MPAC) — like how hundreds of papers use SE blocks, CBAM, or depthwise separable convolutions
> 2. **Designing a novel hybrid architecture** (MPAC-ResNet) that combines it with residual learning
> 3. **Applying it to a new domain** (medical imaging) with domain-specific analysis (Grad-CAM clinical relevance, cross-dataset generalization)
> 4. **Extending the paper's comparative methodology** (downscaling study) to medical imaging
> 5. We **cite the LS-Net paper** prominently as inspiration

---

## 📊 Expected Results Table (After Full Integration)

| Model | Params | From Scratch? | Acc (%) | F1 (%) | FPS | Contribution Role |
|---|---|---|---|---|---|---|
| ResNet50-FT | 23.5M | No (pretrained) | 99.14 ± 0.16 | 99.15 ± 0.17 | baseline | Existing best |
| **MPAC-ResNet-FT** | **~20M** | **No (pretrained)** | **≥99.2?** | **≥99.2?** | **faster** | **Proposed model** |
| EfficientNetB0-FT | 4.0M | No | 99.08 ± 0.28 | 99.08 ± 0.27 | — | Existing comparison |
| LS-Net | 0.92M | Yes | TBD | TBD | fastest | Paper baseline |
| MPAC-ResNet-Tiny | ~0.5-1M | Yes | TBD | TBD | — | Downscaling study |
| ResNet-Tiny | ~0.5-1M | Yes | TBD | TBD | — | Downscaling study |
| Custom CNN | 1.24M | Yes | 94.68 ± 3.63 | 94.71 ± 3.60 | — | Existing baseline |

---

## 🗓️ Timeline Summary

| Phase | Duration | Dependencies |
|---|---|---|
| **A. Implementation** | 3–4 days | None |
| **B. Training** | 5–7 days | Phase A (mostly GPU wait time) |
| **C. Evaluation** | 2–3 days | Phase B |
| **D. Paper** | 2–3 days | Phase C |
| **Total** | **12–17 days** | |

> [!TIP]
> Phases B and D can partially overlap — you can start writing the methodology section while models are training.
