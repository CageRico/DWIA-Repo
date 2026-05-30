"""
Data preprocessing module
Corresponds to Section 3.1 of the paper:
missing value imputation, LOF outlier detection, adaptive standardization.
"""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
import urllib.request
import os


# ── Dataset download ──────────────────────────────────────────────────────────

DATASET_A_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
DATASET_B_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat"

DATASET_A_PATH = os.path.join(os.path.dirname(__file__), "data", "german.data")
DATASET_B_PATH = os.path.join(os.path.dirname(__file__), "data", "australian.dat")


def download_datasets():
    """Download the German Credit (A) and Australian Credit (B) datasets."""
    os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
    for url, path in [(DATASET_A_URL, DATASET_A_PATH), (DATASET_B_URL, DATASET_B_PATH)]:
        if not os.path.exists(path):
            print(f"Downloading {os.path.basename(path)} ...")
            urllib.request.urlretrieve(url, path)
            print(f"  Saved to {path}")
        else:
            print(f"  {os.path.basename(path)} already exists, skipping download.")


# ── Dataset loading ───────────────────────────────────────────────────────────

# Column names for the German Credit dataset (20 features + 1 label)
GERMAN_COLS = [
    "status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment", "installment_rate", "personal_status", "other_debtors",
    "residence_since", "property", "age", "other_installment", "housing",
    "existing_credits", "job", "liable_people", "telephone", "foreign_worker",
    "label"
]

# Categorical feature columns in the German Credit dataset
GERMAN_CAT_COLS = [
    "status", "credit_history", "purpose", "savings", "employment",
    "personal_status", "other_debtors", "property", "other_installment",
    "housing", "job", "telephone", "foreign_worker"
]

# Australian Credit has no header; categorical columns identified by index
AUSTRALIAN_CAT_COLS = [0, 3, 4, 5, 7, 8, 10, 11]


def load_dataset_a():
    """
    Load the German Credit Dataset (Dataset A).
    1000 samples, 20 features (6 numeric + 14 categorical).
    Label: 1=good credit -> 0, 2=bad credit -> 1. Positive/negative ratio ~3:2.
    """
    df = pd.read_csv(DATASET_A_PATH, sep=" ", header=None, names=GERMAN_COLS)
    df["label"] = (df["label"] == 2).astype(int)

    X = df.drop("label", axis=1)
    y = df["label"].values

    X = _encode_categoricals(X, GERMAN_CAT_COLS)
    return X.values.astype(float), y


def load_dataset_b():
    """
    Load the Australian Credit Dataset (Dataset B).
    690 samples, 14 features, binary label 0/1.
    """
    df = pd.read_csv(DATASET_B_PATH, sep=" ", header=None)
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1]

    X = _encode_categoricals(X, AUSTRALIAN_CAT_COLS)
    return X.values.astype(float), y


def _encode_categoricals(X, cat_cols):
    X = X.copy()
    for col in cat_cols:
        le = LabelEncoder()
        col_vals = X[col].astype(str)
        X[col] = le.fit_transform(col_vals)
    return X


# ── Missing value imputation ──────────────────────────────────────────────────

def handle_missing_values(X, cat_col_indices=None):
    """
    Section 3.1:
    - Numeric features: multiple imputation via IterativeImputer.
    - Categorical features: mode imputation.
    """
    if not np.isnan(X).any():
        return X

    X = X.copy()
    n_cols = X.shape[1]
    num_cols = [i for i in range(n_cols) if i not in (cat_col_indices or [])]

    if num_cols:
        imp_num = IterativeImputer(max_iter=10, random_state=42)
        X[:, num_cols] = imp_num.fit_transform(X[:, num_cols])

    if cat_col_indices:
        imp_cat = SimpleImputer(strategy="most_frequent")
        X[:, cat_col_indices] = imp_cat.fit_transform(X[:, cat_col_indices])

    return X


# ── LOF outlier detection ─────────────────────────────────────────────────────

def remove_outliers_lof(X, y, n_neighbors=20, contamination=0.05):
    """
    Section 3.1: Local Outlier Factor (LOF) based outlier removal.
    Samples with LOF significantly greater than 1 are treated as outliers and removed.
    """
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    pred = lof.fit_predict(X)   # 1=inlier, -1=outlier
    mask = pred == 1
    return X[mask], y[mask]


# ── Adaptive standardization ──────────────────────────────────────────────────

class AdaptiveStandardizer:
    """
    Section 3.1: Adaptive standardization.
    x_new = (x - mu_tau) / sigma_tau
    mu_tau and sigma_tau are the local mean and std computed from the
    quantile-trimmed region defined by tau.
    """

    def __init__(self, tau=0.25):
        self.tau = tau
        self.mu_ = None
        self.sigma_ = None

    def fit(self, X):
        lower = np.quantile(X, self.tau, axis=0)
        upper = np.quantile(X, 1 - self.tau, axis=0)
        mask = (X >= lower) & (X <= upper)

        self.mu_ = np.zeros(X.shape[1])
        self.sigma_ = np.ones(X.shape[1])

        for j in range(X.shape[1]):
            col_mask = mask[:, j]
            if col_mask.sum() > 1:
                self.mu_[j] = X[col_mask, j].mean()
                self.sigma_[j] = X[col_mask, j].std()
                if self.sigma_[j] == 0:
                    self.sigma_[j] = 1.0
        return self

    def transform(self, X):
        return (X - self.mu_) / self.sigma_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ── Full preprocessing pipeline ───────────────────────────────────────────────

def preprocess(X, y, cat_col_indices=None, remove_outliers=True, tau=0.25):
    """
    Full pipeline: missing value imputation -> LOF outlier removal -> adaptive standardization.

    Returns
    -------
    (X_processed, y_processed, scaler)
    """
    X = handle_missing_values(X, cat_col_indices)

    if remove_outliers:
        X, y = remove_outliers_lof(X, y)

    scaler = AdaptiveStandardizer(tau=tau)
    X = scaler.fit_transform(X)

    return X, y, scaler
