# Event Classification with Masked Transformer Autoencoders

## Project Context
This project implements a hybrid jet-event classifier for ML4Sci/CMS, focused on:
- improving the **L-GATr + ParT** hybrid architecture,
- using **track/particle-level masked autoencoder (MAE) pretraining**,
- quantifying gains in both **numerical performance** and **stability**.

The implementation and all reported numbers are taken from the notebooks in this repository.

---

## Specific Task (Implemented)
Built on prior GSoC work and evaluated on the JetClass setup with:
- **100,000 events**
- **80/10/10 train/validation/test split**

Main goals addressed in this notebook journey:
1. Improve hybrid architecture quality (accuracy/AUC)
2. Improve training stability across seeds
3. Validate MAE pretraining impact through ablations
4. Track practical speed-oriented engineering choices in final iteration

---

## Final Performance (from `6-Hybrid_LorentzParT_MAE_GSoC2026_FINAL -.ipynb`)
- **Test Accuracy:** `0.7020` (70.2%)
- **Macro AUC (OvR):** `0.9536`
- **Macro AUC (OvO):** `0.9536`

### MAE Pretraining Impact (final notebook)
- **Accuracy gain:** `+0.0282` (reported as +2.8%)
- **AUC gain:** `+0.0070`
- **Variance reduction:** reported as **4.5x lower variance** with pretraining

---

## Notebook Journey (strictly from notebook outputs)

| Notebook | Key change in that iteration | Reported test accuracy | Reported macro AUC (OvR) |
|---|---|---:|---:|
| `1-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb` | Initial full hybrid MAE pipeline + first ablation suite | `0.6467` | `0.938316` |
| `2-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026 .ipynb` | Scheduler/logging refinements + multi-seed evaluation block | `0.6093` | `0.9267` |
| `3-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb` | Mass-target normalization and tuned training behavior | `0.6464` | `0.9393` |
| `3_v6-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb` | Major jump with improved pretraining + longer fine-tuning | `0.7018` | `0.9524` |
| `4-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb` | Consolidation run + robust multi-seed comparison | `0.6968` | `0.9521` |
| `6-Hybrid_LorentzParT_MAE_GSoC2026_FINAL -.ipynb` | Final polished pipeline and benchmark summary | **`0.7020`** | **`0.9536`** |

### Multi-seed evidence (from notebook outputs)
From `4-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb`:
- **Pretrained:** `test_acc = 0.6993 ± 0.0013`, `test_auc_ovr = 0.9524 ± 0.0006`
- **No pretrain:** `test_acc = 0.6866 ± 0.0059`, `test_auc_ovr = 0.9490 ± 0.0014`

This supports both better central performance and improved stability under pretraining.

---

## Ablation Highlights (notebook-reported)
From final notebook ablation outputs:
- `with_mae_pretrain`: `val_acc = 0.5961`, `val_auc = 0.919528`
- `no_mae_pretrain`: `val_acc = 0.5726`, `val_auc = 0.911468`

Observed direction is consistent with final summary: MAE pretraining improves downstream classification quality.

---

## Model/Training Elements Tracked in the Final Notebook
The final notebook explicitly documents the following integrated improvements:
- richer input feature handling and normalization
- attention-gated hybrid fusion behavior checks
- MAE pretraining + supervised fine-tuning pipeline
- CLS-style pooling/class-attention style design choices
- engineering-focused speed switches (`torch.compile`, cuDNN benchmark settings)

---

## Visuals (from `/images`)

### Architecture and workflow
![Architecture](images/architecture.png)
![Feature engineering and architecture](images/feature%20engineering%20and%20model%20archi.png)
![Training flowchart](images/training%20flowchart.png)
![Evaluation metrics flowchart](images/evaluation%20metrics%20flowchart.png)

### Training behavior and results
![Pretraining curves](images/pretraining_curves.png)
![Finetuning curves](images/finetuning_curves.png)
![Per-class metrics](images/per_class_metrics.png)
![Multi-seed comparison](images/multiseed_comparison.png)
![Gate analysis](images/gate_analysis.png)

### Project framing and progress artifacts
![Proposal architecture](images/proposal_architecture.png)
![Proposal timeline](images/proposal_timeline.png)
![Proposal notebook progress](images/proposal_notebook_progress.png)
![Six key improvements](images/Six%20Key%20Improvements.png)
![Data distribution and reproducibility](images/Data%20Distribution%20%26%20Reproducibility%20.png)

---

## Research Foundations Used
Local papers included in this repository:
- [`Research paper/Research paper 1.pdf`](Research%20paper/Research%20paper%201.pdf) — **Lorentz-Equivariant Geometric Algebra Transformers (L-GATr)**
- [`Research paper/Reseach paper 2.pdf`](Research%20paper/Reseach%20paper%202.pdf) — **Particle Transformer (ParT) for Jet Tagging / JetClass context**

External references explicitly aligned with notebook narrative:
- GSoC 2025 write-up: https://medium.com/@thanhnguyen14401/gsoc-2025-with-ml4sci-event-classification-with-masked-transformer-autoencoders-6da369d42140
- Prior codebase context: https://github.com/ML4SCI/CMS/tree/main/MAEs/Hybrid_Transformer_Thanh_Nguyen
- Dataset source script reference: https://github.com/jet-universe/particle_transformer/blob/main/get_datasets.py

---

## Repository Layout
- `notebook/` — full experiment journey and final model notebooks
- `images/` — architecture, curves, ablation/analysis visuals
- `Research paper/` — reference papers used in study
- `readme (1).md` — previous README draft/context

---

## Reproduction Path (Notebook Order)
For the exact journey used in this project, run/read notebooks in this order:
1. `notebook/1-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb`
2. `notebook/2-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026 .ipynb`
3. `notebook/3-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb`
4. `notebook/3_v6-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb`
5. `notebook/4-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb`
6. `notebook/6-Hybrid_LorentzParT_MAE_GSoC2026_FINAL -.ipynb`

