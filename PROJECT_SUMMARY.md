# Project Summary

## Paper

- **Title:** Application and Comparative Study of Machine Learning in Credit Scoring Models
- **Venue:** AIDF 2025
- **DOI:** https://doi.org/10.1145/3764727.3764746

---

## Research Goal

Address the limited interpretability of traditional algorithms and the weak generalization of deep learning models by proposing a **Dynamic Weight Integration Algorithm (DWIA)** for credit scoring, benchmarked against six mainstream algorithms.

---

## Overview

### Datasets

| Dataset | UCI Source | Samples | Features | Label |
|---------|-----------|---------|----------|-------|
| Dataset A | German Credit | 1000 | 20 (6 numeric + 14 categorical) | Positive/negative ~3:2 |
| Dataset B | Australian Credit | 690 | 14 | Balanced |

### Preprocessing (`preprocessing.py`)

1. **Missing value imputation:** numeric features via IterativeImputer; categorical features via mode imputation.
2. **LOF outlier detection:** samples with a Local Outlier Factor significantly greater than 1 are removed.
3. **Adaptive standardization:** local mean and std computed from the quantile-trimmed region (controlled by τ).

### Baseline Models (`models.py`)

| Model | Type | Key Details |
|-------|------|-------------|
| LR | Traditional | L2 regularization, lbfgs optimizer |
| DT | Traditional | max_depth=8, Gini criterion |
| SVM | Traditional | RBF kernel, probability=True |
| NN | Deep | 3-layer MLP (128-64-32), EarlyStopping |
| ICNN | Deep | Features reshaped to 2D matrix; parallel 1×1 + 3×3 convolutions |
| RNN | Deep | Features serialized as time steps; two-layer LSTM |

### DWIA (`dwia.py`)

**Three-step process:**

**① Subset partitioning and local performance evaluation**
Partition the training set into M=5 non-overlapping subsets. Evaluate three base models (LR/DT/SVM) on each subset using a composite score:

$$P_{i,m} = \alpha \cdot \text{Accuracy} + \beta \cdot \text{Recall} + \gamma \cdot \text{F1} \quad (\alpha=0.4,\ \beta=0.3,\ \gamma=0.3)$$

**② Dynamic weight computation**
Normalize scores within each subset via softmax, then aggregate by subset size:

$$w_i = \frac{\sum_{m=1}^{M} w_{i,m} \cdot |D_m|}{\sum_{m=1}^{M} |D_m|}$$

**③ Ensemble prediction**
Weighted voting across the three base models:

$$y = \text{sign}\left(\sum_{i=1}^{3} w_i \cdot y_i\right)$$

### Experiment Design (`experiment.py`)

- **5-fold cross-validation:** evaluates all algorithms on both datasets across Accuracy / Recall / F1 / AUC.
- **Dynamic performance curves:** fixes the test set and incrementally increases training set size to observe performance trends.
  - Dataset A: 200→800 samples, step 100 (paper Figures 1 & 2)
  - Dataset B: 150→550 samples, step 80 (paper Figure 3)

### Visualization (`visualize.py`)

| Output File | Corresponds To |
|-------------|---------------|
| `table1_datasetA.csv` | Table 1 |
| `table2_datasetB.csv` | Table 2 |
| `figure1_accuracy_datasetA.png` | Figure 1 |
| `figure2_recall_datasetA.png` | Figure 2 |
| `figure3_auc_datasetB.png` | Figure 3 |
| `figure4_bar_comparison.png` | Additional comparison chart |

---

## Key Results (Dataset A)

| Algorithm | Accuracy | Recall | F1 | AUC |
|-----------|----------|--------|----|-----|
| LR | 78.2% | 65.5% | 70.4% | 0.785 |
| DT | 80.5% | 71.2% | 73.1% | 0.801 |
| SVM | 81.3% | 73.4% | 75.8% | 0.810 |
| NN | 82.1% | 74.6% | 77.1% | 0.821 |
| ICNN | 83.4% | 76.3% | 78.8% | 0.834 |
| RNN | 82.8% | 75.1% | 77.6% | 0.828 |
| **DWIA** | **86.7%** | **85.5%** | **86.1%** | **0.912** |

*These are the paper's reported values. Reproduced results may differ slightly due to randomness.*

---

## Implementation Notes

- **Data source:** UCI German Credit and Australian Credit datasets (auto-downloaded).
- **Reproducibility:** small deviations from paper values are expected due to random seeds and deep model training stochasticity.

---

## Module Dependency

```
main.py
  ├── preprocessing.py   (download, load, preprocess)
  ├── experiment.py      (CV experiment, dynamic curve experiment)
  │     ├── models.py    (baseline models)
  │     └── dwia.py      (DWIA algorithm)
  └── visualize.py       (figure generation)
```
