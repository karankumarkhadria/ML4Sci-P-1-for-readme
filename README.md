# Linear Attention ViT for Particle Collision Analysis

This folder contains implementations for efficient Vision Transformer-based analysis of high-energy physics detector images, combining **linear attention** mechanisms with multitask learning for joint jet mass regression and quark/gluon classification.

## Task Overview

### Task 1: Dataset Preparation and Preprocessing
- **Data Loading**: Reading HDF5 detector images of quark/gluon jets
- **Physics-Aware Preprocessing**: Energy centroid alignment, normalization, and safe augmentation strategies

### Task 2: Linear Attention ViT — Multitask Learning
- Building and training a Linear Attention Vision Transformer for simultaneous:
  - **Regression**: predicting jet mass
  - **Classification**: identifying quark vs gluon jets

### Task 3: SSL Pretraining and Fine-tuning
- Using self-supervised learning methods (SimMIM, MAE, MAEv2) to pretrain the Linear Attention ViT before supervised fine-tuning on labeled data

### Bonus Task
- Multi-architecture benchmarking (Standard ViT, Linear Attention ViT, L2ViT, XCiT) and building a physics-informed model using the learned representations

## Implementation Details

### Dataset Preparation
The input data consists of HDF5 detector images representing quark and gluon jets from particle collisions. Each image encodes particle energy deposits across detector cells. Preprocessing involved:

- **Energy centroid alignment**: Shifting each jet image so the energy-weighted centroid is centered, ensuring spatial consistency across samples
- **Normalization**: Scaling pixel (energy deposit) values to a fixed range to stabilize training
- **Log-mass handling**: Applying a log transform to the raw jet mass targets to reduce skewness and improve regression convergence
- **Train/val split**: Performed with class-distribution checks to ensure balanced training across quark/gluon labels

This structured preprocessing pipeline was progressively improved across notebook versions, with the most stable version in `linear_attention_vit-5_chnges.ipynb`.

![Complete Pipeline Workflow](images/Complete%20Pipeline%20Workflow.png)

### Linear Attention ViT Architecture
Several architectures were explored across notebook iterations:

1. **Standard ViT (baseline)**: Full quadratic self-attention — strong baseline but expensive at higher resolutions.

2. **Linear Attention ViT (primary model)**: Replaces full attention with a ReLU-kernelized formulation that avoids forming the full token-token attention matrix, reducing memory and computational cost while retaining global feature mixing.

3. **L2ViT**: A variant using L2-normalized attention scores for additional numerical stability.

4. **XCiT-style Encoder**: Cross-covariance image transformer approach — attention across feature dimensions rather than spatial tokens.

![Linear Attention ViT Architecture](images/Linear%20Attention%20ViT%20For%20end%20to%20end%20mass%20regression%20and%20classification.png)

The key trade-off observed: standard ViT provided strong classification signals but was memory-intensive; linear attention achieved competitive accuracy at significantly lower compute, making it practical for large physics datasets.

![Attention Mechanism Comparison](images/Attention%20Mechanism%20Comparison.png)

### Physics-Aware Design
The model incorporates physics-motivated design choices beyond standard ViT:

- **Centroid alignment** as a preprocessing step mirrors how physicists define jet coordinate frames
- **Log-mass regression target** accounts for the approximately log-normal distribution of jet masses in QCD
- **Class-balanced sampling** reflects that quark/gluon ratios vary by process and must be controlled during training

![Physics-Aware Design](images/Physics-Aware%20Design.png)

### Two-Phase Training: SSL Pretraining + Fine-tuning
I implemented a two-phase training strategy to improve the model's generalization:

1. **SSL Pretraining Phase**: The Linear Attention ViT encoder is pretrained in a self-supervised manner using one of three strategies:
   - **SimMIM**: Predicts masked image patches in pixel space
   - **MAE (Masked Autoencoder)**: Reconstructs masked tokens from a small visible subset; produces rich general representations
   - **MAEv2**: An improved MAE variant with enhanced decoder and target handling

2. **Fine-tuning Phase**: The pretrained encoder is attached to dual heads (regression + classification) and fine-tuned end-to-end on labeled jet data, using combined MSE and cross-entropy loss with task-balancing weights.

![Two-Phase Training](images/Two-Phase%20Training.png)

This two-phase approach enabled the model to learn general jet image features before specializing to the supervised task. The MAE-pretrained variant achieved the best combined regression and classification performance.

![SSL Pretraining Methods](images/SSL%20Pretraining%20Methods.png)

### Multi-Architecture Benchmark
Across notebook versions, a systematic multi-architecture comparison was performed:

| Architecture | Classification Strength | Regression Quality | Compute Cost |
|---|---|---|---|
| Standard ViT | Strong | Moderate | High |
| Linear Attention ViT | Strong | Strong (with MAE pretrain) | Low |
| L2ViT | Moderate | Moderate | Low |
| XCiT | Moderate | Lower | Medium |

The progression across five notebook versions refined this benchmark:
- **v1**: Single-model prototype; regression signal collapsed (R² ≈ 0)
- **v2**: Multi-architecture benchmark; Linear Attention ViT showed ~0.874 accuracy
- **v3**: Expanded evaluation suite (ROC-AUC, PR-AUC, ECE, balanced accuracy); exposed instability
- **v4**: Improved pipeline alignment; SimMIM-pretrained linear attention improved balance
- **v5 (final)**: Best overall result — MAE-pretrained Linear Attention ViT; regression substantially fixed

![img-3](images/img-3.png)
![img-4](images/img-4%20(2).png)

## Results

### Performance Comparison Across Versions

| Version | Loss (MSE) | Accuracy | RMSE | Notes |
|---|---:|---:|---:|---|
| v1 | ~0.0000 (artifact) | 0.8160 | N/A | Regression signal unreliable (R²≈0) |
| v2 | 1021.0089 | 0.8740 | 31.95 | Classification-strong benchmark |
| v3 | 1532.8457 | 0.8250 | 39.15 | Instability exposed under richer metrics |
| v4 | 1112.3878 | 0.8750 | 33.35 | SimMIM-pretrained improved balance |
| **v5 (MAE-pretrained)** | **429.7220** | **0.8845** | **20.73** | **Best combined result** |

### Final Model Performance (v5)
- **Accuracy**: 0.8845
- **R²**: 0.8529
- **MSE**: 429.7220 (RMSE ≈ 20.73)

The MAE-pretrained Linear Attention ViT in the final notebook achieved the best combined classification-regression balance, with regression quality improving dramatically (R² from ≈0 to 0.85) through improved preprocessing, loss design, and SSL pretraining.

![Performance Comparison](images/Performance%20Comparison.png)

![img-7](images/img-7.png)
![img-8](images/img-8.png)

## References
- [Oracle-Preserving Latent Flows](https://arxiv.org/abs/2302.00806)
- [Masked Autoencoders Are Scalable Vision Learners (MAE)](https://arxiv.org/abs/2111.06377)
- [XCiT: Cross-Covariance Image Transformers](https://arxiv.org/abs/2106.09681)
- [SimMIM: A Simple Framework for Masked Image Modeling](https://arxiv.org/abs/2111.09886)
