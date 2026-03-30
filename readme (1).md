# Event Classification with Masked Transformer Autoencoders

## TL;DR
- **Problem:** Multi-class jet event classification with a focus on both final quality and run-to-run stability on a 100,000-event dataset.
- **Approach:** Hybrid Lorentz-aware + transformer modeling with MAE pretraining, followed by notebook-driven iterative refinement and ablation.
- **Best final metrics:** Test Accuracy `0.7020`, Macro AUC (OvR) `0.9536`, Macro AUC (OvO) `0.9536`.
- **Key MAE impact:** Accuracy `+0.0282`, AUC `+0.0070`, and approximately `4.5x` lower variance with pretraining.

## Overview
This repository documents a notebook-first research journey for jet event classification in the ML4Sci/CMS context.

The project combines:
- Lorentz-aware modeling for physics-informed representation learning,
- ParT-style transformer components for particle sequence understanding,
- MAE pretraining to improve downstream classification quality and stability.

## Problem setup
- **Dataset scale:** 100,000 events
- **Split:** 80% train / 10% validation / 10% test
- **Task type:** Multi-class event classification (JetClass-style setup)
- **Main targets:** improve quality (accuracy + AUC), improve consistency across seeds, and quantify MAE gains with controlled ablations.

## Quickstart (Notebook Reproduction)
### Prerequisites
- Python environment with the dependencies used by the notebooks.
- PyTorch installed and configured for your system.
- CUDA-capable setup is recommended for practical training time, but notebook logic remains the same without CUDA.

### Notebook location
- All experiments are under `notebook/`.

### Run order
1. `notebook/1-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb`
2. `notebook/2-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026 .ipynb`
3. `notebook/3-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb`
4. `notebook/3_v6-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb`
5. `notebook/4-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb`
6. `notebook/6-Hybrid_LorentzParT_MAE_GSoC2026_FINAL -.ipynb`

### What to check after each notebook
- Data split and preprocessing summary blocks.
- Pretraining/fine-tuning configuration cells.
- Test/validation metric outputs (accuracy and AUC).
- Ablation and multi-seed sections where present.
- Final summary cells for stage-specific conclusions.

## Key Innovations
- **Hybrid dual-branch modeling:** combines Lorentz-aware and transformer-style sequence components.
- **MAE pretraining before fine-tuning:** strengthens representations before supervised classification.
- **Stability via multi-seed protocol:** compares settings with repeated seeded runs and variance reporting.
- **Optimization and engineering improvements:** includes practical training-speed and pipeline refinements in final notebooks.

## Final headline results
From `notebook/6-Hybrid_LorentzParT_MAE_GSoC2026_FINAL -.ipynb`:
- **Test Accuracy:** `0.7020` (70.2%)
- **Macro AUC (OvR):** `0.9536`
- **Macro AUC (OvO):** `0.9536`

### MAE pretraining effect (final summary)
- **Accuracy gain:** `+0.0282` (~+2.8%)
- **AUC gain:** `+0.0070`
- **Variance reduction:** approximately `4.5x` lower variance with pretraining

| Setting | Accuracy | Macro AUC (OvR) | Notes |
|---|---:|---:|---|
| Final model (`NB6`) | `0.7020` | `0.9536` | Final polished benchmark pipeline |
| With MAE pretrain (`NB4` multi-seed) | `0.6993 ± 0.0013` | `0.9524 ± 0.0006` | Higher mean and tighter spread |
| Without MAE pretrain (`NB4` multi-seed) | `0.6866 ± 0.0059` | `0.9490 ± 0.0014` | Lower mean and larger variance |

## Notebook Journey (research progression)
The notebook sequence captures the model evolution from baseline to final benchmark.

| Notebook | What changed in this stage | Reported test accuracy | Reported macro AUC (OvR) |
|---|---|---:|---:|
| `1-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb` | Baseline full hybrid MAE pipeline + first ablation set | `0.6467` | `0.9383` |
| `2-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026 .ipynb` | Scheduler/logging updates + expanded multi-seed evaluation structure | `0.6093` | `0.9267` |
| `3-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb` | Mass-target normalization + training behavior tuning | `0.6464` | `0.9393` |
| `3_v6-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb` | Strong leap from improved pretraining + longer fine-tuning | `0.7018` | `0.9524` |
| `4-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb` | Consolidation and robust seeded comparison | `0.6968` | `0.9521` |
| `6-Hybrid_LorentzParT_MAE_GSoC2026_FINAL -.ipynb` | Final polished benchmark pipeline | **`0.7020`** | **`0.9536`** |

The progression is intentionally iterative rather than strictly monotonic. The largest quality jump appears after stronger pretraining/fine-tuning strategy updates (`3_v6` onward), and the final stages preserve these gains with better seeded consistency.

## Stability & reproducibility evidence
From `notebook/4-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb` multi-seed outputs:

- **With pretraining:** `test_acc = 0.6993 ± 0.0013`, `test_auc_ovr = 0.9524 ± 0.0006`
- **Without pretraining:** `test_acc = 0.6866 ± 0.0059`, `test_auc_ovr = 0.9490 ± 0.0014`

These results show that pretraining improves both central performance and run-to-run reliability.

## Ablation insights
From final ablation outputs:
- `with_mae_pretrain`: `val_acc = 0.5961`, `val_auc = 0.9195`
- `no_mae_pretrain`: `val_acc = 0.5726`, `val_auc = 0.9115`

These ablations align with final test trends and support the claim that MAE pretraining improves downstream event classification in this setup.

## Architecture and training design (high level)
- richer feature handling and normalization,
- hybrid fusion checks with attention gating,
- MAE pretraining followed by supervised fine-tuning,
- CLS-style pooling/class-attention design choices,
- engineering-oriented speed options (`torch.compile`, cuDNN benchmark settings).

## Visual journey
### Baseline architecture and design exploration
![alt text](image.png)
*Caption: Early architecture view used to establish the baseline modeling direction.*

![alt text](image-1.png)
*Caption: Companion baseline visualization confirming initial design behavior.*

### Intermediate architecture and representation analyses
![alt text](image-3.png)
*Caption: Shows deeper model behavior during architecture expansion.*

![alt text](image-5.png)
*Caption: Highlights reconstruction/representation changes under deeper settings.*

![9 digits](image-6.png)
*Caption: Visual check of class-level representation behavior across digits.*

![alt text](image-11.png)
*Caption: Additional diagnostic view used to compare latent/reconstruction quality.*

![alt text](image-4.png)
*Caption: Rotation-equivariant modeling view supporting symmetry-aware design choices.*

### Supervised and symmetry-focused diagnostics
![alt text](image-7.png)
*Caption: Demonstrates supervised latent transformation behavior across rotation steps.*

![alt text](image-8.png)
*Caption: Shows unsupervised symmetry flow behavior in latent space.*

![alt text](image-9.png)
*Caption: Confirms decoded transformation outputs remain label-consistent.*

![alt text](image-10.png)
*Caption: Further validates discovered symmetry behavior through visualization.*

### Invariance quality checks
![alt text](rotation_correlation.png)
*Caption: Indicates correlation structure expected from stronger rotation invariance.*

![alt text](prediction_consistency.png)
*Caption: Shows prediction consistency improvements under transformed inputs.*

## Limitations
- **Dataset scope:** conclusions are based on this task setup and may not transfer directly to other event distributions.
- **Compute dependence:** full reproduction quality depends on sufficient compute, especially for pretraining and multi-seed runs.
- **Generalization uncertainty:** external-domain and cross-detector generalization are not established in this notebook sequence.
- **Calibration/error analysis pending:** reliability calibration and detailed confusion-cluster error analysis are not yet complete.

## Repository structure
- `notebook/` — complete experiment lifecycle and final benchmark notebooks
- `images/` — architecture, curves, and analysis visuals
- `Research paper/` — foundational reading material
- `readme (1).md` — narrative project summary in this repository

## References
- [Oracle-Preserving Latent Flows](https://arxiv.org/abs/2302.00806)
- [VAE blog](https://www.ibm.com/think/topics/variational-autoencoder#:~:text=Variational%20autoencoders%20(VAEs)%20are%20generative,data%20they're%20trained%20on.)
