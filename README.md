# Credit Scoring with DWIA

> **Paper:** Application and Comparative Study of Machine Learning in Credit Scoring Models
> Published at AIDF 2025
> DOI: https://doi.org/10.1145/3764727.3764746

## Project Structure

```
credit_scoring/
├── data/                   # Datasets (auto-downloaded on first run)
│   ├── german.data         # Dataset A: German Credit (1000 samples, 20 features)
│   └── australian.dat      # Dataset B: Australian Credit (690 samples, 14 features)
├── preprocessing.py        # Missing value imputation, LOF outlier detection, adaptive standardization
├── models.py               # Baselines: LR, DT, SVM, NN, ICNN, RNN
├── dwia.py                 # Dynamic Weight Integration Algorithm (DWIA)
├── experiment.py           # 5-fold cross-validation + dynamic performance curves
├── visualize.py            # Figure generation: Table 1/2 & Figures 1-4
├── main.py                 # Entry point
└── output/                 # Generated tables and figures
```

## Requirements

Python 3.8+

```bash
pip install numpy pandas scikit-learn scipy tensorflow matplotlib
```

## Usage

```bash
cd credit_scoring
python main.py
```

## Datasets

| Dataset | Source | Samples | Features | Label Distribution |
|---------|--------|---------|----------|--------------------|
| Dataset A | German Credit (UCI) | 1000 | 20 | Positive/negative ~3:2 |
| Dataset B | Australian Credit (UCI) | 690 | 14 | Balanced |

Datasets are downloaded automatically from the UCI repository on the first run.

## Algorithms

| Abbreviation | Algorithm | Key Parameters |
|---|---|---|
| LR | Logistic Regression | L2 regularization, lbfgs solver |
| DT | Decision Tree | max_depth=8, Gini criterion |
| SVM | Support Vector Machine | RBF kernel |
| NN | Multi-layer Perceptron | 3 hidden layers (128-64-32) |
| ICNN | Improved CNN | Multi-scale adaptive convolution (1×1 + 3×3) |
| RNN | Recurrent Neural Network | Two-layer LSTM |
| DWIA | Dynamic Weight Integration | M=5 subsets, α=0.4, β=0.3, γ=0.3 |

## DWIA Overview

DWIA partitions the training set into M subsets and evaluates each base model (LR, DT, SVM) on each subset using a composite performance score:

$$P_{i,m} = \alpha \cdot \text{Accuracy} + \beta \cdot \text{Recall} + \gamma \cdot \text{F1}$$

Scores are normalized via softmax within each subset, then aggregated by subset size to produce final dynamic weights:

$$w_i = \frac{\sum_{m=1}^{M} w_{i,m} \cdot |D_m|}{\sum_{m=1}^{M} |D_m|}$$

Ensemble prediction:

$$y = \text{sign}\left(\sum_{i=1}^{3} w_i \cdot y_i\right)$$

## Outputs

After running `main.py`, the `output/` directory contains:

| File | Description |
|------|-------------|
| `table1_datasetA.csv` | Per-algorithm metrics on Dataset A (paper Table 1) |
| `table2_datasetB.csv` | Per-algorithm metrics on Dataset B (paper Table 2) |
| `figure1_accuracy_datasetA.png` | Accuracy vs. training set size, Dataset A (paper Figure 1) |
| `figure2_recall_datasetA.png` | Recall vs. training set size, Dataset A (paper Figure 2) |
| `figure3_auc_datasetB.png` | AUC vs. training set size, Dataset B (paper Figure 3) |
| `figure4_bar_comparison.png` | Bar chart comparing all metrics on both datasets |

## Notes

- Deep learning models (ICNN, RNN) reshape tabular features into a 2D matrix and a time series respectively to fit convolutional and recurrent architectures.
- Due to randomness in deep model training, reproduced results may differ slightly from the paper's reported values; overall trends remain consistent.
- To speed up experiments, reduce the `sample_sizes` range or lower `epochs` in `main.py`.
