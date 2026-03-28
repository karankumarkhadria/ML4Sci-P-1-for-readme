# Linear Attention ViT — GSoC Proposal Implementation Journey

This README documents the notebook journey for the GSoC implementation using only content and results shown in the notebooks.

## Notebook Order (Implementation Timeline)
1. `linear_attention_vit.ipynb`
2. `linear_attention_vit-2.ipynb`
3. `linear_attention_vit-3.ipynb`
4. `linear_attention_vit-4.ipynb`
5. `linear_attention_vit-5_chnges.ipynb` *(final notebook in this repository for step 5)*

---

## Project Scope Covered Across Notebooks
From the notebook markdown and outputs, the end-to-end scope is:
- Dataset loading from HDF5 jet-image data
- Physics-aware preprocessing and augmentation
- ViT-family model building for joint regression + classification
- SSL pretraining (SimMIM, MAE, MAEv2)
- Supervised fine-tuning with multitask objectives
- Multi-architecture benchmarking and richer evaluation metrics

Tasks repeatedly targeted in the notebooks:
- **Classification**: quark vs gluon prediction
- **Regression**: jet mass prediction

---

## Notebook-by-Notebook Journey

### 1) `linear_attention_vit.ipynb` — Initial end-to-end prototype
What is implemented in this notebook:
- Full pipeline notebook titled *Linear Attention Vision Transformers for Particle Collision Data*
- Includes paper summary context (XCiT and L2ViT), data loading, preprocessing, model, pretraining, finetuning, scratch baseline, and evaluation
- Multitask setup with classification + mass regression

What was observed in outputs:
- Finetuned model reports `Final Val Acc: 81.05%`
- Scratch model reports `Final Val Acc: 81.60%`
- Validation MSE is shown as `0.0000` in this notebook’s logging outputs, and comparison block shows pretraining not helping classification in this run

Notebook takeaway:
- Pipeline works end-to-end, but regression behavior in this version is not yet reliable and later notebooks refine it.

---

### 2) `linear_attention_vit-2.ipynb` — Multi-architecture benchmark framework
What is added in this notebook:
- Notebook reorganized as a benchmark framework with sections for configuration, preprocessing, architectures, SSL pretraining modules, training utilities, metrics, and visualization
- Four core architectures benchmarked:
  - Standard ViT
  - Linear Attention ViT
  - L2ViT
  - XCiT ViT
- SimMIM pretraining pipeline integrated for encoder pretraining/fine-tuning comparison

Final benchmark outputs in this notebook:
- **Linear Attention ViT**: Accuracy **0.8740**, F1 **0.8739**, MSE **1021.0089**, MAE **21.2226**, R² **0.6506**
- **Standard ViT**: Accuracy **0.8595**, F1 **0.8590**
- **L2ViT**: best regression R² in this notebook (**0.7236**), but poor classification in this run
- Notebook summary marks **Linear Attention ViT** as best classification model for this version

Notebook takeaway:
- Classification became strong with Linear Attention ViT in this benchmark-style setup.

---

### 3) `linear_attention_vit-3.ipynb` — Richer evaluation and stress-testing
What is expanded in this notebook:
- Adds richer classification reliability metrics beyond accuracy/F1:
  - Balanced Accuracy
  - ROC-AUC
  - PR-AUC
  - ECE (calibration)
- Keeps multi-architecture benchmarking and multitask setup

Final benchmark outputs in this notebook:
- **Standard ViT** performs best classification in this version:
  - Accuracy **0.8720**, F1 **0.8718**, ROC-AUC **0.9407**, PR-AUC **0.9269**
- **Linear Attention ViT** drops in this run:
  - Accuracy **0.8250**, F1 **0.8249**, MSE **1532.8457**

Notebook takeaway:
- This version exposes instability/variance across configurations and motivates stricter alignment in the next notebook.

---

### 4) `linear_attention_vit-4.ipynb` — Requirement-aligned Linear-Attention pipeline
What changes in this notebook:
- Notebook is explicitly aligned to required flow:
  1. Pretrain **Linear Attention encoder** on unlabeled data
  2. Fine-tune pretrained variants for supervised multitask learning
  3. Compare against scratch and other architectures
- Runs all three SSL methods for Linear Attention encoder:
  - SimMIM
  - MAE
  - MAEv2

Final outputs for main variants in this notebook:
- **Linear Attention ViT (SimMIM-pretrained)**:
  - Accuracy **0.8750**, F1 **0.8750**, MSE **1112.3878**, MAE **22.0543**, R² **0.6193**
- **Linear Attention ViT (MAE-pretrained)**:
  - Accuracy **0.8720**, F1 **0.8718**, MSE **1122.2405**
- **Linear Attention ViT (MAEv2-pretrained)**:
  - Accuracy **0.8475**, F1 **0.8473**, MSE **1240.1278**

Notebook takeaway:
- Requirement-aligned SSL→fine-tuning flow is stable, and SimMIM-pretrained linear attention is best in this notebook’s classification metrics.

---

### 5) `linear_attention_vit-5_chnges.ipynb` — Final improved version
What is refined in this final notebook:
- Keeps the same requirement-aligned pipeline but improves checkpointing/selection
- Adds explicit checkpoint/early-stop handling by MAE (with F1 tie-breaks), in addition to F1 tracking
- Re-runs SSL pretraining + supervised fine-tuning and full benchmark table

Key final results from this notebook:
- **Linear Attention ViT (MAE-pretrained)** *(best overall linear-attention variant in this notebook)*:
  - Accuracy **0.8845**
  - Balanced Accuracy **0.8849**
  - F1 **0.8845**
  - ROC-AUC **0.9502**
  - PR-AUC **0.9376**
  - ECE **0.0321**
  - MSE **429.7220**
  - MAE **14.8667**
  - R² **0.8529**
- **Linear Attention ViT (SimMIM-pretrained)**:
  - Accuracy **0.8740**, MSE **538.0564**
- **Linear Attention ViT (MAEv2-pretrained)**:
  - Accuracy **0.8245**, MSE **736.2875**

Notebook takeaway:
- Final notebook shows major regression improvement with MAE-pretrained Linear Attention ViT while keeping strong classification.

---

## Overall Journey Summary (v1 → v5)
- **v1**: first full pipeline works, but regression reporting/behavior needs improvement.
- **v2**: benchmark framework established; Linear Attention ViT reaches strong classification (0.8740).
- **v3**: richer metrics added (ROC-AUC, PR-AUC, ECE, balanced accuracy); instability becomes visible.
- **v4**: pipeline aligned tightly to requirement (Linear-Attention SSL pretraining variants + fine-tuning).
- **v5**: final improved checkpointing/selection gives strongest combined performance for MAE-pretrained Linear Attention ViT.

---

## Reproducing the Journey
Run notebooks in this exact order:
1. `jupyter notebook/linear_attention_vit.ipynb`
2. `jupyter notebook/linear_attention_vit-2.ipynb`
3. `jupyter notebook/linear_attention_vit-3.ipynb`
4. `jupyter notebook/linear_attention_vit-4.ipynb`
5. `jupyter notebook/linear_attention_vit-5_chnges.ipynb`

