# 📋 Implementation Plan: Deep Learning for Brain Tumor Diagnosis from MRI

## Paper Summary

**Title:** Nghiên cứu và xây dựng mô hình học sâu cho bài toán chuẩn đoán u não từ ảnh MRI  
*(Research and Development of Deep Learning Models for Brain Tumor Diagnosis from MRI Images)*

**Core Idea:** Build, train, and compare multiple CNN-based deep learning architectures to classify brain tumors from MRI images. Evaluate using Accuracy, Sensitivity/Recall, Specificity, and F1-score. Propose the best-performing model for clinical support.

---

## Detailed Task Breakdown

| # | Phase | Task | Description | Key Deliverable | Est. Effort |
|---|-------|------|-------------|-----------------|-------------|
| **Phase 1** | **Literature Review & Background** | | | | |
| 1.1 | Literature | Brain tumor pathology review | Study types of brain tumors (glioma, meningioma, pituitary, no tumor), clinical characteristics, and why early detection matters | Summary document | 2 days |
| 1.2 | Literature | Medical imaging techniques survey | Review MRI imaging modalities (T1, T2, FLAIR, contrast-enhanced), their characteristics and role in brain tumor diagnosis | Summary document | 1 day |
| 1.3 | Literature | Deep learning in medical imaging survey | Survey SOTA methods: CNN, ResNet, VGG, EfficientNet, Vision Transformers applied to brain tumor classification; review 15-20 recent papers | Literature review table | 3 days |
| 1.4 | Literature | Identify baseline & candidate architectures | Select 4-5 architectures to implement (e.g., Custom CNN, VGG16, ResNet50, EfficientNetB0, MobileNetV2) with justification | Architecture selection document | 1 day |
| **Phase 2** | **Environment Setup** | | | | |
| 2.1 ✅ | Setup | Set up development environment | Install Python, PyTorch/TensorFlow, CUDA, Jupyter, create virtual environment, define `requirements.txt` | Working dev environment | 0.5 day |
| 2.2 ✅ | Setup | Define project folder structure | Create standardized directory structure: `data/`, `notebooks/`, `src/`, `models/`, `results/`, `configs/` | Project skeleton | 0.5 day |
| 2.3 ✅ | Setup | Set up experiment tracking | Configure logging, Weights & Biases / TensorBoard for tracking metrics, hyperparameters, and model checkpoints | Tracking dashboard | 0.5 day |
| 2.4 ✅ | Setup | Set up configuration management | Create config files (YAML/JSON) for hyperparameters, paths, model selection — ensure reproducibility | Config system | 0.5 day |
| **Phase 3** | **Data Collection & Preprocessing** | | | | |
| 3.1 ✅ | Data | Identify & download dataset | Select public dataset (e.g., [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — 4 classes: glioma, meningioma, pituitary, no tumor) | Raw dataset downloaded | 0.5 day |
| 3.2 ✅ | Data | Exploratory Data Analysis (EDA) | Analyze class distribution, image dimensions, intensity distributions; visualize sample images from each class | EDA notebook with charts | 1 day |
| 3.3 ✅ | Data | Data cleaning & quality check | Remove corrupted/duplicate images, verify labels, handle any mislabeled samples | Clean dataset | 0.5 day |
| 3.4 ✅ | Data | Image preprocessing pipeline | Implement: resize to uniform size (e.g., 224×224), normalize pixel values (0-1 or ImageNet stats), convert to tensors | `preprocessing.py` | 1 day |
| 3.5 ✅ | Data | Data splitting | Split into Train/Validation/Test sets (70/15/15 or 80/10/10) with stratified sampling to preserve class ratios | Split datasets | 0.5 day |
| 3.6 ✅ | Data | Data augmentation strategy | Implement augmentations: random rotation, horizontal flip, zoom, brightness/contrast adjustment, elastic deformation | `augmentation.py` | 1 day |
| 3.7 ✅ | Data | Create DataLoader/Dataset classes | Build custom PyTorch `Dataset` and `DataLoader` classes with on-the-fly augmentation for training | `dataset.py` | 1 day |
| 3.8 ✅ | Data | Verify data pipeline end-to-end | Load a batch, visualize augmented samples, check shapes and labels are correct | Verification notebook | 0.5 day |
| **Phase 4** | **Model Development** | | | | |
| 4.1 ✅ | Model | Build custom CNN baseline | Design a simple CNN from scratch (3-5 conv blocks, batch norm, dropout, FC layers) as baseline | `models/custom_cnn.py` | 1 day |
| 4.2 ✅ | Model | Implement VGG16 (transfer learning) | Load pretrained VGG16, freeze early layers, replace classifier head for 4-class output | `models/vgg16.py` | 1 day |
| 4.3 ✅ | Model | Implement ResNet50 (transfer learning) | Load pretrained ResNet50, freeze backbone, add custom classification head | `models/resnet50.py` | 1 day |
| 4.4 ✅ | Model | Implement EfficientNetB0 (transfer learning) | Load pretrained EfficientNetB0, fine-tune with custom head | `models/efficientnet.py` | 1 day |
| 4.5 ✅ | Model | Implement MobileNetV2 (transfer learning) | Load pretrained MobileNetV2, adapt for brain tumor classification | `models/mobilenet.py` | 1 day |
| 4.6 ✅ | Model | Create model factory/registry | Build a unified interface to instantiate any model by name from config | `models/model_factory.py` | 0.5 day |
| 4.7 ✅ | Model | Define loss function & optimizer | Implement CrossEntropyLoss (with optional class weights for imbalance), Adam/AdamW optimizer, learning rate scheduler | `training/losses.py` | 0.5 day |
| **Phase 5** | **Training & Evaluation** | | | | |
| 5.1 ✅ | Training | Build training loop | Implement training loop with: forward pass, loss computation, backprop, gradient clipping, logging per epoch | `training/trainer.py` | 1.5 days |
| 5.2 ✅ | Training | Build validation loop | Implement validation loop: compute loss & metrics on val set after each epoch, early stopping logic | `training/trainer.py` | 0.5 day |
| 5.3 ✅ | Training | Implement metrics computation | Code Accuracy, Precision, Recall/Sensitivity, Specificity, F1-score (macro & per-class) | `evaluation/metrics.py` | 1 day |
| 5.4 ✅ | Training | Implement model checkpointing | Save best model (by val F1 or val loss), save last model, save training state for resume | Checkpoint logic | 0.5 day |
| 5.5 ✅ | Training | Train Custom CNN | Full training from scratch — **done: 95.63% test acc, F1 95.62%** (but not converged, see 5.14) | Trained model + logs | 1 day |
| 5.6 ✅ | Training | Train VGG16 | Transfer learning, frozen backbone + large head — **done: 97.34% test acc, F1 97.35%, AUC 99.83% — best model** | Trained model + logs | 1 day |
| 5.7 ✅ | Training | Train ResNet50 (freeze) | Frozen backbone baseline — **done: 91.44% test acc** (underfit: only 8.2K trainable params, see 5.11) | Trained model + logs | 1 day |
| 5.8 ✅ | Training | Train EfficientNetB0 (freeze) | Frozen backbone baseline — **done: 91.25% test acc** (underfit: only 5.1K trainable params, see 5.12) | Trained model + logs | 1 day |
| 5.9 ✅ | Training | Train MobileNetV2 (freeze) | Frozen backbone baseline — **done: 89.64% test acc** (underfit: only 5.1K trainable params, see 5.13) | Trained model + logs | 1 day |
| 5.10 ⏭️ | Training | Hyperparameter tuning | `scripts/tune.py` ready — **SKIPPED: no headroom left** (top models at 99%+ acc; tuning can't meaningfully improve) | — | — |
| 5.11 ✅ | Training | **Fine-tune ResNet50** | **done: 99.33% acc, F1 99.34%, AUC 99.94%** (vs 91.44% frozen, +7.9pp) — run `20261001_150620_run_all_ft` | Fine-tuned model + logs | 1 day |
| 5.12 ✅ | Training | **Fine-tune EfficientNetB0** | **done: 99.24% acc, AUC 99.98%** (vs 91.25% frozen, +8.0pp) — same run | Fine-tuned model + logs | 1 day |
| 5.13 ✅ | Training | **Fine-tune MobileNetV2** | **done: 98.48% acc, AUC 99.92%** (vs 89.64% frozen, +8.8pp) — same run | Fine-tuned model + logs | 1 day |
| 5.13b ✅ | Training | **Fine-tune VGG16** | **done: 98.86% acc, AUC 99.94%** (vs 97.34% frozen, +1.5pp) — run `20261002_104212_train_ft`; added for complete freeze-vs-FT coverage | Fine-tuned model + logs | 1 day |
| 5.14 ✅ | Training | Re-train Custom CNN longer | **done: 97.24% acc, AUC 99.69%** at 50 epochs (vs 95.63% at 30) — run `20261001_164924_train_ep50` | Improved model + logs | 1 day |
| 5.15 ✅ | Training | Multi-seed experiments | **done: all 5 models × seeds 42/123/456** — transfer models stable (±0.1–0.3pp); Custom CNN unstable (seed 123 → 92.11%, ±2.96pp) — key finding for paper | Mean ± std results | 1.5 days |
| **Phase 6** | **Analysis & Comparison** | | | | |
| 6.1 ✅ | Evaluation | Generate confusion matrices | Plot confusion matrix for each model on test set (per-class performance visualization) | Confusion matrix plots | 0.5 day |
| 6.2 ✅ | Evaluation | Plot training curves | Visualize train/val loss and accuracy curves for all models | Training curve plots | 0.5 day |
| 6.3 ✅ | Evaluation | Compute full evaluation metrics table | Create comparison table: Accuracy, Precision, Recall, Specificity, F1-score (macro + per-class) for all models | Metrics comparison table | 0.5 day |
| 6.4 ✅ | Evaluation | ROC curves & AUC | Plot ROC curves (one-vs-rest) and compute AUC for each model | ROC/AUC plots | 1 day |
| 6.5 ✅ | Evaluation | Statistical significance testing | Perform McNemar's test or paired t-test between top models to verify significant differences | Statistical test results | 0.5 day |
| 6.6 ✅ | Evaluation | Grad-CAM / interpretability | Implement Grad-CAM to visualize which regions the model focuses on for predictions | Grad-CAM heatmaps | 1 day |
| 6.7 ✅ | Evaluation | Error analysis | Analyze misclassified samples: identify common failure patterns, per-class error rates | Error analysis report | 1 day |
| 6.8 ✅ | Evaluation | Model complexity comparison | Compare parameter count, FLOPs, inference time, and model size across all architectures | Complexity table | 0.5 day |
| 6.9 ✅ | Evaluation | Select best model & justify | `compare_models.py` run — **VGG16 best (97.34%); McNemar vs Custom CNN: χ²=8.03, p=0.0046 (significant)** | Final recommendation | 0.5 day |
| 6.10 ✅ | Evaluation | Re-evaluate & merged multi-seed table | `scripts/paper_figures.py` merges all runs → `results/paper/per_run_results.csv` + `summary_mean_std.csv` (freeze + finetune + seeds) | Merged results tables | 0.5 day |
| 6.11 ✅ | Evaluation | Accuracy–efficiency scatter plot | `results/paper/accuracy_efficiency.png` — F1 ± std vs GFLOPs, best variant per model | Trade-off figure | 0.5 day |
| 6.12 ✅ | Evaluation | Per-class F1 grouped bar chart | `results/paper/per_class_f1.png` + `.csv` — mean per-class F1 over seeds, best variant per model | Per-class F1 figure | 0.5 day |
| 6.13 ✅ | Evaluation | Normalized confusion matrices | `results/paper/<model>_confusion_norm.png` — row-normalized, best run per model | Normalized CM plots | 0.5 day |
| 6.14 ✅ | Evaluation | Freeze vs fine-tune comparison figure | `results/paper/freeze_vs_finetune.png` + `.csv` — all 4 transfer models, mean ± std | Freeze/FT comparison figure | 0.5 day |
| **Phase 7** | **Cross-dataset Generalization Study** | | | | |
| 7.1 ✅ | Generalization | Select external dataset(s) | Figshare/Cheng chosen (3,064 T1-contrast MRI, 3 classes — **no `notumor`**); BraTS = 3D segmentation → stretch goal. ⚠️ Kaggle PNG mirror `denizkavi1` unusable (viridis-colormapped); used `.mat` original `ashkhagan/figshare-brain-tumor-dataset` + `scripts/extract_figshare_mat.py` → `data/external_raw/figshare/` | Dataset decision | 0.5 day |
| 7.2 ✅ | Generalization | Preprocess external dataset | `scripts/prepare_external.py` → `data/external/figshare/{all,train,val,test}.csv` — 3,064 imgs: glioma 1,426 / pituitary 930 / meningioma 708 | `data/external/<name>/*.csv` | 1 day |
| 7.3 ✅ | Generalization | Add external-eval support to code | `evaluate.py --csv <manifest> [--restrict-classes]` + `train.py --data-dir <dir>` — restriction masks absent-class logits & remaps labels before metrics | Code change | 0.5 day |
| 7.4 ⚠️ | Generalization | Zero-shot cross-dataset evaluation | **done, but INVALIDATED**: all 4 FT models scored 99.5–99.8% — because ~92% of Figshare images (2,810/3,064 hash near-dupes, 1,523 exact) **already exist in the Nickparvar training data** (verified: pixel corr 0.96–0.9997). Not a true external test | Cross-dataset results + overlap audit | 0.5 day |
| 7.4b ⏳ | Generalization | Decontaminated external eval | Re-evaluate on Figshare images **not** present in our dataset (hash dist > 5, ~250 imgs) — only this subset yields honest out-of-domain metrics | Decontaminated metrics | 0.5 day |
| 7.5 ⏳ | Generalization | Fine-tune on external dataset (optional) | `train.py --model resnet50 --finetune --data-dir data/external/figshare --epochs 30 --lr 0.0001` | Adapted results | 1 day |
| 7.6 ⏳ | Generalization | Cross-dataset analysis & write-up | Report contamination finding (dataset provenance overlap) + decontaminated-subset metrics + domain-shift discussion — the overlap itself is a paper-worthy limitation/analysis point | Paper section content | 0.5 day |
| **Phase 8** | **Documentation & Reporting** | | | | |
| 8.1 | Report | Write Introduction chapter | Motivation, problem statement, objectives, scope | Chapter draft | 1 day |
| 8.2 | Report | Write Literature Review chapter | Background on brain tumors, MRI, deep learning methods, related works | Chapter draft | 2 days |
| 8.3 | Report | Write Methodology chapter | Data pipeline, model architectures, training procedures, evaluation metrics | Chapter draft | 2 days |
| 8.4 | Report | Write Experiments & Results chapter | Dataset description, experimental setup, results tables, comparison charts | Chapter draft | 2 days |
| 8.5 | Report | Write Discussion & Conclusion | Interpret results, limitations, future work, propose best model | Chapter draft | 1 day |
| 8.6 | Report | Prepare figures and tables | Create publication-quality figures, format all tables consistently | Final figures/tables | 1 day |
| 8.7 | Report | Code cleanup & documentation | Add docstrings, type hints, clean notebooks, write `README.md` for reproducibility | Clean codebase | 1 day |
| 8.8 | Report | Create reproducibility package | Final `requirements.txt`, training scripts, inference demo, pretrained model weights | Release package | 1 day |
| **Phase 9** | **LS-Net / MPAC Integration** (see `integration_plan.md`) | | | | |
| 9.1 ⏳ | Model | Implement MPAC block | Multi-Path Atrous Convolution block (Kyrkou, CVPR 2026): parallel depthwise atrous convs (d=1,2) → concat → BN+LeakyReLU → 1×1 projection + residual | `src/models/mpac_block.py` | 1 day |
| 9.2 ⏳ | Model | Implement MPAC-ResNet | Hybrid: ResNet stem + Stages 1–2 standard ResBlocks, Stages 3–4 MPAC-augmented ResBlocks; full-size + Tiny (~0.5–1M) variants | `src/models/mpac_resnet.py` | 1.5 days |
| 9.3 ⏳ | Model | Implement original LS-Net | Standalone ~919K-param LS-Net from paper as reference/comparison model | `src/models/litespeed_net.py` | 1 day |
| 9.4 ⏳ | Model | Register new models in factory | Add `mpac_resnet`, `mpac_resnet_tiny`, `lsnet`, `resnet_tiny`, `efficientnet_tiny`, `mobilenet_tiny`, `custom_cnn_tiny` to `MODEL_REGISTRY` | `src/models/model_factory.py` edit | 0.5 day |
| 9.5 ⏳ | Model | Smoke test new models | Extend `scripts/smoke_test_models.py` — forward/backward/param-count check for every new model | Passing smoke test | 0.5 day |
| 9.6 ⏳ | Training | Train MPAC-ResNet (pretrained FT) | `train.py --model mpac_resnet --finetune` — ImageNet-pretrained backbone, freeze Stages 1–2, train MPAC Stages 3–4 + head | Trained model + logs (target ≥99%) | 1 day |
| 9.7 ⏳ | Training | Train MPAC-ResNet (from scratch) | `train.py --model mpac_resnet --no-pretrained --epochs 50` — with/without pretrain comparison | Trained model + logs | 1 day |
| 9.8 ⏳ | Training | Train LS-Net (from scratch) | `train.py --model lsnet --no-pretrained --epochs 50` — no pretrained weights exist; establishes LS-Net baseline on our dataset | Trained model + logs | 1 day |
| 9.9 ⏳ | Training | Train sub-1M Tiny variants | `train_all.py --models mpac_resnet_tiny resnet_tiny efficientnet_tiny mobilenet_tiny custom_cnn_tiny lsnet` — downscaling comparison per paper's paradigm | Tiny-variant results | 1.5 days |
| 9.10 ⏳ | Training | Multi-seed runs for new models | Seeds 42/123/456 × all new models — stability analysis consistent with 5.15 | Mean ± std results | 1.5 days |
| 9.11 ⏳ | Evaluation | Statistical + complexity analysis | McNemar: MPAC-ResNet vs ResNet50-FT and vs LS-Net; params/FLOPs/FPS table (paper Table-2 style); acc–efficiency scatter update | Updated comparison tables/figures | 1 day |
| 9.12 ⏳ | Evaluation | Downscaling analysis | Accuracy-drop curves across sub-1M variants (paper Fig-4 style) — which architecture degrades most gracefully in medical imaging | Downscaling figure | 0.5 day |
| 9.13 ⏳ | Evaluation | Grad-CAM comparison | Do MPAC blocks attend to more clinically relevant regions than standard convs? MPAC-ResNet vs ResNet50-FT heatmaps | Grad-CAM comparison figure | 0.5 day |
| 9.14 ⏳ | Evaluation | Cross-dataset eval of MPAC-ResNet | Evaluate on decontaminated Figshare subset (task 7.4b subset, ~250 imgs) | Generalization metrics | 0.5 day |
| 9.15 ⏳ | Report | Paper integration | Update SOTA table (add LS-Net + new models), write "Proposed Method" (MPAC-ResNet) + "Downscaling Analysis" sections, update acc-efficiency figures, revise Introduction contributions | Paper sections | 2 days |

---

## Summary

| Phase | Tasks | Est. Total Effort |
|-------|-------|-------------------|
| 1. Literature Review | 4 tasks | 7 days |
| 2. Environment Setup | 4 tasks | 2 days |
| 3. Data Processing | 8 tasks | 6 days |
| 4. Model Development | 7 tasks | 6 days |
| 5. Training & Evaluation | 16 tasks (1 skipped) | 16 days |
| 6. Analysis & Comparison | 14 tasks | 8.5 days |
| 7. Cross-dataset Generalization | 6 tasks | 4 days |
| 8. Documentation | 8 tasks | 11 days |
| 9. LS-Net / MPAC Integration | 15 tasks | ~14 days |
| **Total** | **82 tasks** | **~74 days** |

> [!TIP]
> Phases 1-2 can be partially parallelized. Phases 4-5 are the heaviest compute workload — ensure GPU access (Google Colab Pro, Kaggle, or local GPU).

> [!IMPORTANT]
> **Candidate Architectures:** Custom CNN (baseline), VGG16, ResNet50, EfficientNetB0, MobileNetV2 — **extended in Phase 9** with MPAC-ResNet (proposed hybrid), LS-Net (paper baseline), and sub-1M Tiny variants of all models for the downscaling study. See `integration_plan.md`.

> [!NOTE]
> **Recommended Dataset:** [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — 7,023 images, 4 classes (glioma, meningioma, pituitary, no tumor). Well-established and widely cited.

---

## Implementation Log — Phases 2 & 3 (completed 2026-09-30)

**Phase 2 — Environment Setup**
- `.venv/` virtualenv (Python 3.12) + `requirements.txt` — torch 2.5.1 **CPU-only** (no GPU detected on this machine).
- Folder skeleton: `data/{raw,processed}`, `src/{data,models,training,evaluation,utils}`, `models/`, `results/{figures,logs,metrics}`, `configs/`, `scripts/`, `notebooks/`.
- Tracking: `src/utils/tracking.py` — TensorBoard `SummaryWriter` + per-run `metrics.csv` (TensorBoard chosen over W&B: no API key needed).
- Config: `configs/config.yaml` + `src/utils/config.py` (paths, splits, augmentation, training hyperparams, seed).

**Phase 3 — Data Pipeline**
- Dataset v2 downloaded via `kagglehub` → `data/raw/` (**7,200 images, perfectly balanced: 1,800/class** — newer version than the 7,023-image v1).
- Cleaning (`scripts/run_cleaning.py`): 0 corrupted, **187 exact-duplicate images removed** (md5) → `data/processed/clean.csv` (7,013 images).
- Split (`scripts/run_split.py`): merged original `Training/`+`Testing/` folders and re-split **stratified 70/15/15** → `train.csv` (4,909) / `val.csv` (1,052) / `test.csv` (1,052).
- Preprocessing/augmentation: `src/data/preprocessing.py`, `src/data/augmentation.py` (albumentations: rotate, h-flip, zoom, brightness/contrast, elastic, resize 224×224, ImageNet normalize).
- `src/data/dataset.py`: `BrainTumorDataset` + `build_dataloaders()`; verified end-to-end by `scripts/verify_pipeline.py` — batch `(32, 3, 224, 224)`, labels 0–3, augmentation active.
- EDA: `scripts/run_eda.py` + executed `notebooks/01_eda.ipynb` → `results/figures/*.png`, `results/metrics/eda_summary.csv`.

**Notes / new findings for later phases**
- ⚠️ No GPU on this machine — Phase 5 training will be slow on CPU. Consider Google Colab / Kaggle GPU, or reduce epochs.
- `training.num_workers=0` in config (Windows DataLoader limitation).
- New task added below (2.5): convenience runner for the data pipeline.

| 2.5 ✅ | Setup | Pipeline runner scripts | `scripts/` entrypoints for download → EDA → clean → split → verify so the whole data pipeline is reproducible with 5 commands | `scripts/*.py` | — |

**Phase 4 — Model Development** (completed 2026-09-30)
- Model code lives in `src/models/` (the `models/` dir is reserved for checkpoints). Verified by `scripts/smoke_test_models.py` — every model ran forward + backward + optimizer step on a real training batch.

| Model | Total params | Trainable | Frozen |
|---|---|---|---|
| custom_cnn | 1,240,036 | 1,240,036 | 0 |
| vgg16 | 27,562,308 | 12,847,620 | 14,714,688 |
| resnet50 | 23,516,228 | 8,196 | 23,508,032 |
| efficientnetb0 | 4,012,672 | 5,124 | 4,007,548 |
| mobilenetv2 | 2,228,996 | 5,124 | 2,223,872 |

- Transfer models default to `freeze_backbone: true` (classifier head only); set `model.freeze_backbone: false` in `configs/config.yaml` for full fine-tuning.
- `src/training/losses.py`: `compute_class_weights` (inverse-frequency), `build_criterion` (CrossEntropy ± class weights), `build_optimizer` (AdamW/Adam/SGD), `build_scheduler` (Cosine/Step/Plateau).

**Phases 5 & 6 — code implemented 2026-09-30** (training runs pending GPU machine)
- `src/training/trainer.py`: `Trainer` — per-epoch train/val loops, grad clipping, early stopping on `val_f1_macro`, scheduler step, TensorBoard + log + `history.csv`.
- `src/training/checkpoint.py`: `save_checkpoint`/`load_checkpoint` → `models/<model>/{best.pt,last.pt}` with optimizer/scheduler state for `--resume`.
- `src/evaluation/metrics.py`: accuracy, precision/recall/**specificity**/F1 (macro + per-class via one-vs-rest confusion matrix).
- `src/evaluation/gradcam.py`: auto-detects last Conv2d; works with all 5 architectures.
- `src/evaluation/complexity.py`: params, GFLOPs (native `FlopCounterMode`), size MB, inference latency.
- Scripts: `scripts/train.py` (`--model --epochs --lr --resume --finetune --debug-batches`), `scripts/train_all.py`, `scripts/evaluate.py` (`--model all` evaluates every checkpoint in one run; also accepts comma lists), `scripts/run_all.py` (one background-friendly command: train all → eval all → compare), `scripts/tune.py` (random search), `scripts/compare_models.py` (comparison table, complexity, McNemar top-2, final recommendation).
- Verified on CPU via `--debug-batches` + `evaluate.py --max-samples` + `compare_models.py` — full pipeline runs end-to-end.
- 🐛 **Bugfix:** `Config.__setattr__` added — CLI overrides (`--epochs`, `--lr`, `--debug-batches`) previously wrote attributes instead of dict items and were silently ignored.
- Reproduce instructions added to `README.md` (venv → install → data → train → eval → compare).

**Post-training analysis — new tasks added 2026-10-01** (details: `analysis_and_next_steps.md`)
- All 5 models trained + evaluated on test set (1,052 imgs). VGG16 best (97.34%). ⚠️ Anomaly: Custom CNN (95.63%) beat frozen ResNet50/EfficientNetB0/MobileNetV2 because they trained only 5-8K head params → underfitting, hard to defend in paper.
- → Added 5.11-5.13 (🔴 **mandatory**: fine-tune the 3 frozen transfer models), 5.14 (re-train Custom CNN — not converged), 5.15 (multi-seed runs), 6.10 (re-evaluate), 6.11-6.14 (new paper figures: acc-efficiency scatter, per-class F1 bars, normalized CMs, freeze-vs-FT).
- 5.10 (hyperparameter tuning) reopened: `tune.py` exists but never actually ran; schedule it after fine-tuning identifies the top models.

**Fine-tuning + multi-seed results — completed 2026-10-02**

Freeze vs fine-tune (test acc, seed 42): ResNet50 91.44→**99.33**, EfficientNetB0 91.25→**99.24**, MobileNetV2 89.64→**98.48**, VGG16 97.34→**98.86**. Custom CNN 95.63→97.24 at 50 epochs. Confirmed the frozen-head underfitting hypothesis.

Multi-seed (acc, mean ± std over seeds 42/123/456): **ResNet50-FT 99.14 ± 0.16** > EfficientNetB0-FT 99.08 ± 0.28 > VGG16-FT 98.86 ± 0.09 > MobileNetV2-FT 98.45 ± 0.33 > Custom CNN 95.53 ± 2.96. ⚠️ Custom CNN seed 123 dropped to 92.11% — from-scratch training is unstable; strong Discussion point (transfer learning = higher acc AND lower variance). McNemar ResNet50 vs EfficientNetB0 (FT): p=1.0 — statistically tied; EfficientNetB0 is the efficiency pick (4.0M vs 23.5M params).

- New pipeline bits: `--seed` flag propagated `run_all.py → train_all.py → train.py` (recorded in each run's `config.yaml` + `experiments.csv`); `--config` forwarded the same way; `--extra` fixed to `argparse.REMAINDER` (was dropping `--finetune`); all script logs now also write into `<run_dir>/logs/`.
- `scripts/paper_figures.py` → `results/paper/`: merged per-run + mean±std tables, freeze-vs-FT table/figure, accuracy-efficiency scatter, per-class F1 bars, normalized CMs.
- 5.10 hyperparameter tuning **skipped** — no headroom at 99%+.
- Remaining before paper: sync `20261002_191342_run_all_cnn_s456` run dir (its metrics are missing locally — CNN s456 excluded from mean until synced), then Phase 8 writing.

**Phase 7 added 2026-10-02** — Cross-dataset generalization study. Note: "Figshare" and "Cheng dataset" are the same dataset (3 classes, no `notumor`). BraTS is segmentation 3D volumes → optional stretch.

**Phase 9 added 2026-10-04** — LS-Net / MPAC integration per `integration_plan.md` (Kyrkou, CVPR 2026 — see `Kyrkou_Rethinking_Compact_1M_Vision_Models_..._paper.pdf`). Three contributions: (1) **MPAC-ResNet** — proposed hybrid integrating Multi-Path Atrous Convolution blocks into ResNet Stages 3–4; (2) **sub-1M downscaling study** — Tiny variants of all architectures to test which design degrades most gracefully in medical imaging; (3) **standalone LS-Net** (~919K params) as a 7th comparison model. Only ~3 new files needed (`mpac_block.py`, `mpac_resnet.py`, `litespeed_net.py`) + model-factory registration — all training/eval/figure infrastructure is reused unchanged. Phase 9 should run **before Phase 8 writing is finalized**; tasks 9.15 paper edits overlap with Phase 8. Ordering within Phase 9: 9.1–9.5 (implementation) → 9.6–9.10 (training, mostly GPU wait) → 9.11–9.14 (analysis) → 9.15 (paper).
