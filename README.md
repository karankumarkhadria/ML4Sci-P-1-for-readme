# ⚡ Linear Attention Vision Transformer for Particle Collision Analysis

## 1. 🌌 Title & Overview

### Project Title
**Linear Attention ViT for Joint Jet Mass Regression and Quark/Gluon Classification**

### Problem Statement
This project builds an efficient Vision Transformer pipeline for high-energy physics detector images to solve two tasks together:
- **Regression**: predict jet mass
- **Classification**: identify quark vs gluon jets

### Why This Project Matters
In collider physics, models must process large, sparse detector images efficiently and accurately. Standard self-attention becomes expensive as token count increases. This work explores how **linear attention** can preserve transformer quality while improving scalability.

### Brief on Linear Attention ViT
Instead of forming a full attention matrix, linear attention uses a kernelized formulation (ReLU feature map) to reduce attention complexity and memory pressure. This enables practical transformer training while still supporting rich global feature interactions.

---

## 2. 🚀 Project Journey (IMPORTANT)

The repository notebooks document a clear research-to-engineering progression:

### Timeline-style progression

- **v1 — `linear_attention_vit.ipynb` (prototype baseline)**
  - Built first full end-to-end pipeline (pretrain + finetune + scratch comparison).
  - Implemented XCiT-style linear attention encoder.
  - Observed core issue: regression quality was effectively collapsed (**R² ~ 0**) while classification looked reasonable.
  - Key lesson: pipeline worked mechanically, but target scaling/loss setup needed redesign.

- **v2 — `linear_attention_vit-2.ipynb` (multi-architecture benchmark)**
  - Added Standard ViT, Linear Attention ViT, L2ViT, XCiT.
  - Added SSL pretraining modules (SimMIM/MAE/MAEv2 scaffold) and richer experiment utilities.
  - Showed strong classification for Linear Attention ViT (~0.874 acc), but instability/imbalance across architectures remained.
  - Key lesson: framework matured, but training dynamics were still brittle for some models.

- **v3 — `linear_attention_vit-3.ipynb` (evaluation + reproducibility expansion)**
  - Added stronger evaluation suite: ROC-AUC, PR-AUC, ECE, balanced accuracy.
  - Added multi-seed runs and mini hyperparameter sweeps.
  - Exposed variance and confirmed model behavior under broader settings.
  - Key lesson: deeper metrics showed where models were truly robust vs superficially good.

- **v4 — `linear_attention_vit-4.ipynb` (pipeline alignment + multi-SSL fine-tuning)**
  - Shifted to requirement-aligned linear-attention-first pretraining/fine-tuning workflow.
  - Introduced compatibility patches and cleaner training utilities.
  - Multi-SSL comparison became clear (SimMIM/MAE/MAEv2 for linear attention).
  - Key lesson: architecture and training code became stable enough for targeted optimization.

- **v5 (final available notebook) — `linear_attention_vit-5_chnges.ipynb` (major improvements & optimization)**
  - Added improved preprocessing (energy centroid alignment, safer augmentation, log-mass handling).
  - Applied stronger multitask balancing and checkpointing logic (Best-F1 and Best-MAE views).
  - Achieved best overall Linear Attention ViT results (MAE-pretrained variant):
    - **Accuracy: 0.8845**
    - **R²: 0.8529**
    - **MSE: 429.7220 (RMSE ≈ 20.73)**
  - Major breakthrough: final setup significantly improved regression while maintaining high classification quality.

### Major breakthroughs
1. Transition from baseline implementation to architecture benchmarking.
2. Expansion from basic metrics to reliability metrics and seed-based validation.
3. Shift to requirement-aligned, modular SSL + fine-tune workflow.
4. Final-stage optimization that produced balanced high performance across both tasks.

---

## 3. 🧠 Model Architecture

### Simple explanation
The model converts detector images into patch tokens, processes them with linear-attention transformer blocks, aggregates global features, and predicts both mass and class in parallel.

### Components
- **Input pipeline**
  - HDF5 detector images
  - physics-aware preprocessing (normalization, augmentation, centroid alignment)
  - train/val split with class-distribution checks
- **Linear Attention mechanism**
  - Q/K/V projections + ReLU-kernelized linear attention
  - avoids full quadratic token-token attention matrix
- **Transformer blocks**
  - stacked blocks with normalization + MLP + residual pathways
- **Multi-task heads**
  - **Regression head** for jet mass
  - **Classification head** for quark/gluon prediction

### ASCII architecture diagram

```text
Input Detector Image
   │
   ├─► Preprocessing (normalization, augmentations, centroid alignment)
   │
   ├─► Patch Embedding
   │
   ├─► Positional/Token Encoding
   │
   ├─► Linear Attention Transformer Blocks (stacked)
   │       ├─ ReLU-kernelized attention
   │       ├─ MLP feed-forward
   │       └─ Residual + normalization
   │
   ├─► Feature Aggregation (pooling)
   │
   └─► Dual Heads
           ├─ Regression Head  → Jet Mass
           └─ Classification Head → Quark/Gluon
```

---

## 4. 🌳 Repository Workflow (Tree Diagram)

```text
jupyter notebook/
 ├── linear_attention_vit.ipynb
 ├── linear_attention_vit-2.ipynb
 ├── linear_attention_vit-3.ipynb
 ├── linear_attention_vit-4.ipynb
 └── linear_attention_vit-5_chnges.ipynb   (final available notebook)
```

> Note: the repository’s final notebook filename is **`linear_attention_vit-5_chnges.ipynb`** (original repo naming).

---

## 5. ⚙️ Key Features

- ✅ Linear Attention ViT implementation with scalable attention computation
- ✅ Efficient transformer experimentation across multiple architectures
- ✅ Joint **regression + classification** multitask setup
- ✅ Custom loss balancing (including MSE/Huber-style experimentation and weighted multitask terms)
- ✅ SSL pretraining support (SimMIM, MAE, MAEv2) with downstream fine-tuning
- ✅ Engineering-focused training utilities (schedulers, early stopping/checkpoints, reproducibility controls)

---

## 6. 📊 Experiments & Improvements

### What changed and why
- **From v1 to v2**: moved from single-model prototyping to architecture-level benchmarking.
- **v2 to v3**: added deeper metrics and repeatability checks to avoid misleading conclusions.
- **v3 to v4**: improved pipeline consistency and requirement alignment for linear-attention-first workflow.
- **v4 to v5**: focused on data preprocessing quality, target handling, and multitask optimization to improve both tasks together.

### Observed insights
- **Loss behavior**: later notebooks show more stable and meaningful convergence patterns, especially for final MAE-pretrained linear attention runs.
- **Stability**: multi-seed + broader metric tracking exposed brittle settings and guided robust choices.
- **GPU utilization**: benchmark outputs consistently logged inference time, training time, and peak GPU memory, enabling practical model trade-off decisions.

---

## 7. 📈 Metrics Comparison Table

| Version | Loss (MSE) | Accuracy | RMSE | Notes |
|---|---:|---:|---:|---|
| v1 (`linear_attention_vit.ipynb`) | ~0.0000 (normalized-scale artifact) | 0.8160 (scratch baseline) | N/A | Early pipeline; regression signal unreliable (R²≈0) |
| v2 (`linear_attention_vit-2.ipynb`) | 1021.0089 | 0.8740 | 31.95 | Linear Attention benchmark became classification-strong |
| v3 (`linear_attention_vit-3.ipynb`) | 1532.8457 | 0.8250 | 39.15 | Rich diagnostics revealed instability under expanded evaluation |
| v4 (`linear_attention_vit-4.ipynb`) | 1112.3878 | 0.8750 | 33.35 | SimMIM-pretrained linear attention improved balance |
| v5 final (`linear_attention_vit-5_chnges.ipynb`, MAE-pretrained) | **429.7220** | **0.8845** | **20.73** | Best overall balance; strongest final linear-attention result |

---

## 8. 📉 Charts & Visualizations (IMPORTANT)

### 8.1 Loss vs Epoch (conceptual trend across versions)

```text
Loss
│\
│ \        v3 (higher/unstable)
│  \__
│     \__        v4 (more stable)
│        \___
│            \____ v5-final (lowest + smoothest)
└────────────────────────────── Epoch
```

### 8.2 Accuracy vs Epoch (conceptual trend)

```text
Accuracy
│                         ───── v5-final plateau (~0.88)
│                  ───────
│            ──────  v4 (~0.87)
│      ──────
│  ────  v2 (~0.87)   v3 lower/stable band (~0.82-0.83)
└────────────────────────────── Epoch
```

### 8.3 Prediction vs Ground Truth (regression quality)

```text
Predicted Mass
│                         .  .  . .
│                    . . . . . .
│                . . . . .
│            . . .
│        . .
│    . .
│ .
└────────────────────────────── True Mass
            (v5-final points lie closer to diagonal)
```

---

## 9. 🔬 Final Model Insights

- The final optimized stage works best because it combines:
  1. stronger preprocessing,
  2. mature SSL-to-finetune workflow,
  3. multitask training balance,
  4. tighter evaluation/checkpoint logic.
- **Key optimization that worked**: MAE-pretrained Linear Attention ViT in final notebook gave the best combined classification-regression balance.
- **Trade-offs**:
  - Better accuracy/regression usually required longer training than the fastest baseline.
  - Some alternative architectures could optimize one metric but underperform on multitask balance.

---

## 10. 🛠️ Tech Stack

- **Python**
- **PyTorch**
- **NumPy**
- **h5py**
- **scikit-learn**
- **matplotlib**
- **Jupyter Notebook**
- **CUDA GPU environment**

---

## 11. 🚀 Future Work

- Add uncertainty-aware calibration for both heads in production inference.
- Run larger seed ensembles for confidence intervals on final metrics.
- Extend from binary quark/gluon tagging to richer particle taxonomy.
- Explore hybrid local-global token mixing for faster convergence.
- Evaluate higher-resolution inputs with memory-aware token strategies.

---

## 12. 📌 How to Run

1. **Clone repository** and move into project root.
2. **Install dependencies** used in notebooks (PyTorch, NumPy, h5py, scikit-learn, matplotlib, Jupyter).
3. **Place datasets** in the expected data path referenced by notebook configuration cells.
4. **Open notebooks in order** to understand evolution:
   - `linear_attention_vit.ipynb`
   - `linear_attention_vit-2.ipynb`
   - `linear_attention_vit-3.ipynb`
   - `linear_attention_vit-4.ipynb`
   - `linear_attention_vit-5_chnges.ipynb` (final practical notebook in this repo)
5. **Run all cells** in the final notebook to reproduce strongest final Linear Attention ViT metrics.
6. Optionally run earlier notebooks for ablation and progression comparison.

---

### ✅ Project Summary
This repository demonstrates a full engineering journey from prototype to optimized linear-attention transformer for physics data, with measurable improvements in both regression and classification, documented through iterative notebook-driven experimentation.
