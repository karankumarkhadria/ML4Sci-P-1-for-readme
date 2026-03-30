# Event Classification with Masked Transformer Autoencoders

## Overview
This repository documents a full notebook-driven research journey for **jet event classification** in the ML4Sci/CMS context using a hybrid deep learning approach.

The core idea is to combine:
- **L-GATr-style Lorentz-aware modeling** for physics-aware representation learning,
- **ParT-style transformer components** for particle sequence understanding,
- **Masked Autoencoder (MAE) pretraining** to improve downstream classification quality and stability.

This README is intentionally designed as a **narrative project report** (not just image dumps), so readers can understand the motivation, progression, decisions, and outcomes notebook by notebook.

---

## Why this project matters
In jet physics classification tasks, model quality is judged not only by final accuracy/AUC but also by:
1. **Training stability** across random seeds,
2. **Data efficiency and representation quality**,
3. **Robustness of performance gains** under ablations.

This work explores whether MAE pretraining and iterative architecture refinements produce measurable gains in these dimensions.

---

## Problem setup
- **Dataset scale:** 100,000 events
- **Split:** 80% train / 10% validation / 10% test
- **Task type:** Multi-class event classification (JetClass-style setup)
- **Main targets:**
  - improve hybrid architecture quality (accuracy + AUC),
  - improve consistency across runs,
  - quantify MAE pretraining gains via controlled ablations,
  - track practical training-speed improvements in the final version.

---

## Implementation approach
This project was implemented as an iterative notebook pipeline, where each stage solved one concrete limitation seen in the previous stage.  
Instead of trying to optimize everything at once, the approach was: stabilize training first, then improve representation quality, then improve evaluation reliability.

Implementation strategy used in this repo:
- Start from a working hybrid baseline (Lorentz-aware + transformer blocks).
- Add MAE pretraining before classification fine-tuning.
- Track both **performance** (accuracy/AUC) and **stability** (seed variance).
- Use ablation runs to verify whether each change actually helps.
- Keep only changes that improve quality consistently across multiple runs.

This step-by-step approach made the final model stronger and also more reproducible.

---

## Final headline results
From `notebook/6-Hybrid_LorentzParT_MAE_GSoC2026_FINAL -.ipynb`:
- **Test Accuracy:** `0.7020` (70.2%)
- **Macro AUC (OvR):** `0.9536`
- **Macro AUC (OvO):** `0.9536`

### MAE pretraining effect (final summary)
- **Accuracy gain:** `+0.0282` (~+2.8%)
- **AUC gain:** `+0.0070`
- **Variance reduction:** reported as **~4.5x lower variance** with pretraining

---

## Notebook Journey (research progression)
The heart of this repository is the notebook sequence. Each notebook captures a meaningful step in the model evolution.

| Notebook | What changed in this stage | Reported test accuracy | Reported macro AUC (OvR) |
|---|---|---:|---:|
| `1-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb` | Baseline full hybrid MAE pipeline + first ablation set | `0.6467` | `0.938316` |
| `2-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026 .ipynb` | Scheduler/logging updates + expanded multi-seed evaluation structure | `0.6093` | `0.9267` |
| `3-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb` | Mass-target normalization + training behavior tuning | `0.6464` | `0.9393` |
| `3_v6-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb` | Strong leap from improved pretraining + longer fine-tuning | `0.7018` | `0.9524` |
| `4-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb` | Consolidation and robust seeded comparison | `0.6968` | `0.9521` |
| `6-Hybrid_LorentzParT_MAE_GSoC2026_FINAL -.ipynb` | Final polished benchmark pipeline | **`0.7020`** | **`0.9536`** |

### Interpreting the journey
- The path is **not strictly monotonic** in every intermediate step, which is expected in experimental ML.
- The key milestone is the transition to stronger pretraining/fine-tuning strategy (`3_v6` onward), where both accuracy and AUC jump significantly.
- Final runs preserve this gain and improve consistency.

### What was changed in each next notebook to improve implementation
Below is the practical transition logic from one notebook to the next:

- **NB1 → NB2**
  - Added better scheduler/logging structure and multi-seed evaluation pattern.
  - Goal: improve monitoring and make comparisons more reliable.

- **NB2 → NB3**
  - Added mass-target normalization and tuned training behavior.
  - Goal: recover performance and improve feature scaling behavior.

- **NB3 → NB3_v6**
  - Strengthened pretraining and extended fine-tuning strategy.
  - Goal: improve representation quality before supervised learning.

- **NB3_v6 → NB4**
  - Consolidated improvements and tested seeded robustness more strictly.
  - Goal: ensure gains are stable, not just single-run effects.

- **NB4 → NB6 (Final)**
  - Polished full benchmark pipeline and optimization settings.
  - Goal: maximize final score while preserving reproducibility.

---

## Stability & reproducibility evidence
From `notebook/4-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb` multi-seed outputs:

- **With pretraining:**
  - `test_acc = 0.6993 ± 0.0013`
  - `test_auc_ovr = 0.9524 ± 0.0006`

- **Without pretraining:**
  - `test_acc = 0.6866 ± 0.0059`
  - `test_auc_ovr = 0.9490 ± 0.0014`

### Key takeaway
Pretraining improves **both**:
- the average metric values (better central performance), and
- the spread across seeds (better run-to-run reliability).

---

## Ablation insights
From final ablation outputs:
- `with_mae_pretrain`: `val_acc = 0.5961`, `val_auc = 0.919528`
- `no_mae_pretrain`: `val_acc = 0.5726`, `val_auc = 0.911468`

These ablations are directionally consistent with final test-set trends and support the central project claim: **MAE pretraining is beneficial for downstream event classification** in this setup.

---

## Architecture and training design (high level)
The final iteration tracks these integrated decisions:
- richer feature handling and normalization,
- hybrid fusion checks with attention gating,
- MAE pretraining followed by supervised fine-tuning,
- CLS-style pooling/class-attention design choices,
- engineering-oriented speed options (`torch.compile`, cuDNN benchmark settings).

---

## Visual journey (curated)
To keep this README content-first, only the most useful images are kept below.

### Core architecture view
![Architecture](images/architecture.png)

### Training behavior and evaluation quality
![Pretraining curves](images/pretraining_curves.png)
![Finetuning curves](images/finetuning_curves.png)
![Per-class metrics](images/per_class_metrics.png)
![Multi-seed comparison](images/multiseed_comparison.png)

These figures are enough to understand model design, learning dynamics, and reproducibility without overwhelming the page.

---

## Repository structure
- `notebook/` — complete experiment lifecycle and final benchmark notebooks
- `images/` — architecture, curves, and analysis visuals
- `Research paper/` — foundational reading material
- `README.md` — this narrative summary
- `readme (1).md` — older/alternate draft context

---

## How to read/reproduce this work
Follow notebooks in this order to match the evolution path:
1. `notebook/1-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb`
2. `notebook/2-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026 .ipynb`
3. `notebook/3-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb`
4. `notebook/3_v6-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb`
5. `notebook/4-Hybrid_Lorentz_ParT_MAE_JetClass_GSoC2026.ipynb`
6. `notebook/6-Hybrid_LorentzParT_MAE_GSoC2026_FINAL -.ipynb`

Recommended reading pattern per notebook:
1. configuration and preprocessing cells,
2. pretraining/fine-tuning setup,
3. metric tables and confusion/per-class outputs,
4. ablation and seeded comparison sections,
5. final summary blocks.

---

## Research foundations and external context
Local references in this repo:
- [`Research paper/Research paper 1.pdf`](Research%20paper/Research%20paper%201.pdf) — Lorentz-Equivariant Geometric Algebra Transformers (L-GATr)
- [`Research paper/Reseach paper 2.pdf`](Research%20paper/Reseach%20paper%202.pdf) — Particle Transformer / JetClass context

External context used in project framing:
- GSoC 2025 write-up: https://medium.com/@thanhnguyen14401/gsoc-2025-with-ml4sci-event-classification-with-masked-transformer-autoencoders-6da369d42140
- Prior ML4SCI/CMS code context: https://github.com/ML4SCI/CMS/tree/main/MAEs/Hybrid_Transformer_Thanh_Nguyen
- Dataset source utility reference: https://github.com/jet-universe/particle_transformer/blob/main/get_datasets.py

---

## Conclusion
This repository demonstrates an end-to-end experimental journey from baseline hybrid modeling to a stronger final benchmark, with evidence that MAE pretraining improves both **performance** and **stability**.

The final notebook result (`0.7020` accuracy, `0.9536` macro AUC) is best understood not as a one-off number, but as the endpoint of iterative design, ablation validation, and multi-seed reliability analysis.

## Next steps for the next notebook
The next notebook can focus on fewer but higher-impact experiments. A good direction is to improve class imbalance handling and calibration while preserving the current stability gains.

Suggested next-step points:
- add class-balanced or focal-style loss comparison,
- test stronger augmentation for hard classes only,
- run confidence calibration checks (ECE/reliability plots),
- include error analysis by confusion clusters,
- benchmark inference speed/memory with batch-size sweeps,
- keep the same multi-seed protocol so improvements stay comparable.
