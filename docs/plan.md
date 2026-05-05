# Land Use and Land Cover Classification from Satellite Imagery Using CNN and Transfer Learning on the EuroSAT Dataset
---

### Abstract

This document outlines a systematic investigation into land-use classification from Sentinel-2 satellite imagery using the EuroSAT dataset. The study employs transfer learning with a suite of state-of-the-art convolutional neural network architectures — spanning VGG, ResNet, DenseNet, EfficientNet, and MobileNet families — to establish a rigorous performance benchmark. All models are initialized with ImageNet-pretrained weights and fine-tuned on the target domain through a frozen-backbone paradigm with a custom classification head. Comprehensive evaluation encompasses per-class and aggregate accuracy metrics, confusion matrices, receiver operating characteristic (ROC) curves with area under the curve (AUC), and inference latency. The objective is to identify the optimal architecture balancing classification performance and computational efficiency for the land-use classification task.

---

## 1. Introduction

Land-use and land-cover classification represents a fundamental challenge in remote sensing, with direct applications in urban planning, agricultural monitoring, environmental conservation, and disaster response. The increasing availability of high-resolution satellite imagery — exemplified by the European Space Agency's Sentinel-2 mission — has created both opportunities and imperatives for automated classification systems capable of processing large-scale geospatial data with high accuracy.

Traditional approaches relying on hand-crafted features and classical machine learning algorithms have been substantially outperformed by deep convolutional neural networks. However, training deep architectures from scratch requires prohibitively large labeled datasets. Transfer learning — wherein models pretrained on large-scale natural image datasets such as ImageNet are adapted to domain-specific tasks — has emerged as the de facto paradigm for scenarios with limited labeled data.

This project undertakes a comprehensive benchmarking study comparing twelve distinct CNN architectures across five architectural families, all employing identical transfer learning protocols, to determine the optimal model for the EuroSAT land-use classification task.

---

## 2. Dataset Description

### 2.1 The EuroSAT Dataset

The EuroSAT dataset comprises 27,000 RGB satellite images derived from the Sentinel-2 satellite constellation. Each image is 64 × 64 pixels with a ground sampling distance (GSD) of 10 meters. Images were collected across European landscapes and manually labeled into one of ten land-use categories.

### 2.2 Class Distribution

The dataset exhibits moderate class imbalance, with class frequencies ranging from 2,000 to 3,000 samples per category.

| Class | Sample Count |
|---|---|
| AnnualCrop | 3,000 |
| Forest | 3,000 |
| HerbaceousVegetation | 3,000 |
| Highway | 2,500 |
| Industrial | 2,500 |
| Pasture | 2,000 |
| PermanentCrop | 2,500 |
| Residential | 3,000 |
| River | 2,500 |
| SeaLake | 3,000 |
| **Total** | **27,000** |

### 2.3 Label Encoding

Classes are mapped to integer labels via a predefined encoding (`dataset/label_map.json`), mapping the ten class names to indices 0 through 9 inclusive.

---

## 3. Methodology

### 3.1 Data Preprocessing and Splitting

**3.1.1 Stratified Partitioning.** The dataset is partitioned into training, validation, and test subsets using stratified sampling to preserve the class distribution across all splits. The chosen split ratio is **70:15:15** (18,900 training, 4,050 validation, 4,050 test samples). Stratification ensures that each subset reflects the original class imbalance, preventing evaluation bias.

**3.1.2 CSV Manifest Generation.** A preprocessing script (`scripts/01_generate_csv.py`) traverses the `dataset/` directory hierarchy, assigning each image a unique index and recording its relative file path, integer label, and human-readable class name. Three CSV manifests — `splits/train.csv`, `splits/val.csv`, and `splits/test.csv` — are produced, each conforming to the schema: `[index, filename, label, className]`.

### 3.2 Exploratory Data Analysis

A dedicated analysis script (`scripts/02_eda.py`) produces a comprehensive characterization of the dataset, including:

- Class frequency distributions and imbalance quantification
- Per-class sample image grids for qualitative inspection
- Image dimension verification and corruption detection
- Per-channel mean and standard deviation computed across the full dataset for input normalization
- Per-class pixel intensity histograms
- Split summary statistics verifying stratification fidelity

All visualizations and summary statistics are persisted to `outputs/eda/`.

### 3.3 Transfer Learning Protocol

**3.3.1 Backbone Initialization.** All models are initialized with weights pretrained on the ImageNet-1K dataset (ILSVRC 2012). The pretrained weights provide rich hierarchical feature representations learned from approximately 1.2 million natural images across 1,000 categories.

**3.3.2 Input Preprocessing.** The EuroSAT images (64 × 64 pixels) are resized to 224 × 224 pixels via bilinear interpolation to match the input resolution expected by the pretrained backbones. Training images undergo data augmentation — random horizontal flips, random rotations (±15°), and color jitter (brightness, contrast, saturation perturbations) — while validation and test images receive only the resize and normalization transformations.

**3.3.3 Classification Head.** The final fully connected layer of each backbone is replaced with a custom multi-layer classification head:

```
Input (backbone features) → Dropout(p=0.3) → Linear(512) → ReLU → Dropout(p=0.3) → Linear(10)
```

The intermediate layer (dimension 512) provides additional representational capacity for domain adaptation, while dropout layers mitigate overfitting.

**3.3.4 Feature Extraction and Fine-Tuning.** All backbone parameters are initially frozen. Only the custom classification head is trained. This approach prevents catastrophic forgetting of pretrained representations and enables rapid convergence on the target task.

### 3.4 Model Architectures

Twelve distinct architectures spanning five architectural families are evaluated:

| Family | Architectures | Approx. Parameters |
|---|---|---|
| VGG | VGG16-BN, VGG19-BN | 138M, 143M |
| ResNet | ResNet18, ResNet34, ResNet50, ResNet101 | 11.7M, 21.8M, 25.6M, 44.5M |
| DenseNet | DenseNet121, DenseNet169 | 8.0M, 14.1M |
| EfficientNet | EfficientNet-B0, EfficientNet-B3 | 5.3M, 12.2M |
| MobileNet | MobileNetV2, MobileNetV3-Large | 3.5M, 5.5M |

The selection spans architectures with widely varying depth, parameter counts, and computational footprints, enabling a thorough exploration of the accuracy-efficiency Pareto frontier.

### 3.5 Training Protocol

**3.5.1 Loss Function.** Cross-entropy loss is employed with class weights inversely proportional to class frequency in the training set, compensating for the moderate class imbalance.

**3.5.2 Optimizer and Scheduling.** The AdamW optimizer is used with an initial learning rate of 1 × 10⁻³ and weight decay of 1 × 10⁻⁴. A ReduceLROnPlateau scheduler reduces the learning rate by a factor of 0.5 when validation loss plateaus for 3 epochs.

**3.5.3 Training Regime.** Each model is trained for a maximum of 30 epochs with a batch size of 64. Early stopping with a patience of 5 epochs on validation loss prevents overfitting. The checkpoint achieving the lowest validation loss is retained for final evaluation.

**3.5.4 Hardware and Reproducibility.** All experiments are conducted with fixed random seeds (Python `random`, NumPy, PyTorch) to ensure reproducibility.

### 3.6 Evaluation Metrics

The following metrics are computed on the held-out test set for each trained model:

- **Top-1 Accuracy**: Overall classification accuracy
- **Precision, Recall, F1-Score**: Per-class and macro-averaged
- **Confusion Matrix**: Both raw counts and row-normalized, visualized as heatmaps
- **ROC Curves and AUC**: One-vs-all curves for each of the 10 classes, with micro- and macro-averaged AUC
- **Training Curves**: Loss and accuracy trajectories across training epochs
- **Inference Latency**: Mean inference time per image (milliseconds), measured on CPU

---

## 4. Project Architecture

### 4.1 Directory Structure

```
eurosat-cnn/
│
├── config/
│   └── config.yaml                # Central configuration (paths, hyperparameters, split ratios)
│
├── src/
│   ├── data/
│   │   ├── transforms.py          # Data augmentation and preprocessing pipelines
│   │   └── dataset.py             # PyTorch Dataset and DataModule definitions
│   ├── models/
│   │   └── factory.py             # Model factory: instantiation of all architectures
│   ├── training/
│   │   ├── config.py              # Training configuration dataclass
│   │   └── trainer.py             # Generic training and validation loop
│   └── evaluation/
│       └── metrics.py             # Evaluation utilities (metrics, confusion matrix, ROC, plots)
│
├── scripts/
│   ├── 01_generate_csv.py         # Stratified train/val/test split → CSV manifests
│   ├── 02_eda.py                  # Exploratory data analysis and visualization
│   ├── 03_train.py                # Training entry point (model selection via CLI)
│   └── 04_evaluate.py             # Comprehensive evaluation across trained models
│
├── dataset/                        # EuroSAT image data (excluded from version control)
│   ├── AnnualCrop/                 # 3,000 images
│   ├── Forest/                     # 3,000 images
│   ├── HerbaceousVegetation/       # 3,000 images
│   ├── Highway/                    # 2,500 images
│   ├── Industrial/                 # 2,500 images
│   ├── Pasture/                    # 2,000 images
│   ├── PermanentCrop/              # 2,500 images
│   ├── Residential/                # 3,000 images
│   ├── River/                      # 2,500 images
│   ├── SeaLake/                    # 3,000 images
│   ├── label_map.json              # Class-to-index mapping
│   └── dataset.md                  # Dataset documentation
│
├── splits/                         # Generated CSV manifests
│   ├── train.csv                   # 18,900 records (70%)
│   ├── val.csv                     # 4,050 records (15%)
│   └── test.csv                    # 4,050 records (15%)
│
├── outputs/
│   ├── eda/                        # EDA visualizations and statistics
│   ├── checkpoints/                # Best model weights ({model}_best.pt)
│   ├── metrics/                    # Per-model metrics in JSON ({model}.json)
│   ├── plots/                      # Confusion matrices, ROC curves, training curves
│   └── report/                     # Comparative summary tables and visualizations
│
├── logs/                           # Training logs and TensorBoard event files
│
├── docs/
│   └── plan.md                     # This document
│
├── pyproject.toml                  # Project metadata and dependencies
├── README.md                       # Project overview
└── .gitignore                      # Excludes dataset/, outputs/, logs/, .venv/
```

### 4.2 Execution Pipeline

The project is executed as a sequential pipeline of four scripts:

```
Step 1: python scripts/01_generate_csv.py    # Data splitting
Step 2: python scripts/02_eda.py             # Exploratory analysis
Step 3: python scripts/03_train.py --all     # Train all 12 architectures
Step 4: python scripts/04_evaluate.py        # Evaluate and compare all models
```

Each script is idempotent and writes its outputs to designated subdirectories, enabling incremental execution and inspection of intermediate results.

---

## 5. Expected Outcomes

### 5.1 Primary Deliverables

1. **CSV Manifests**: Three stratified CSV files mapping dataset samples to train, validation, and test splits.
2. **Exploratory Analysis Report**: Visualizations and statistics characterizing the dataset across all ten land-use classes.
3. **Trained Model Checkpoints**: Serialized weights for each of the twelve architectures, representing the best-performing state on validation data.
4. **Comprehensive Metrics Report**: Per-model JSON files containing all computed evaluation metrics.
5. **Comparative Analysis**: A ranked comparison of all architectures across accuracy, F1-score, AUC, and inference latency.

### 5.2 Research Questions Addressed

- Which pretrained CNN architecture yields the highest classification accuracy on the EuroSAT dataset?
- How does architectural depth and parameter count correlate with performance on satellite imagery?
- What is the trade-off between model size and classification performance?
- Can lightweight mobile-oriented architectures (MobileNet, EfficientNet) approach the accuracy of larger models while offering substantial latency improvements?

### 5.3 Final Recommendation

The study will conclude with a clear recommendation identifying the superior architecture for the EuroSAT land-use classification task, balancing classification performance, model size, and inference efficiency. This recommendation will be supported by quantitative evidence across all evaluated metrics.

---

## 6. Dependencies

The project relies on the following Python packages:

- **torch** (≥2.0) — Deep learning framework
- **torchvision** (≥0.15) — Pretrained model architectures and image transforms
- **scikit-learn** — Stratified splitting and classification metrics
- **pandas** — Data manipulation and CSV I/O
- **matplotlib** and **seaborn** — Visualization
- **tqdm** — Progress bars
- **PyYAML** — Configuration file parsing
- **Pillow** — Image loading
- **NumPy** — Numerical operations

---

*Document version: 1.0 · Last updated: May 2026*
