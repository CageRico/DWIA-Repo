"""
Experiment pipeline
Corresponds to Section 4 of the paper: 5-fold cross-validation,
evaluating Accuracy / Recall / F1 / AUC.
Also supports dynamic performance curves as a function of training set size.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

from models import get_all_models
from dwia import DynamicWeightIntegrationAlgorithm


# ── Single model evaluation ───────────────────────────────────────────────────

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Train and evaluate a single model; return the four metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    return {
        "Accuracy": accuracy_score(y_test, y_pred) * 100,
        "Recall":   recall_score(y_test, y_pred, zero_division=0) * 100,
        "F1":       f1_score(y_test, y_pred, zero_division=0) * 100,
        "AUC":      roc_auc_score(y_test, y_prob),
    }


# ── 5-fold cross-validation ───────────────────────────────────────────────────

def run_cv_experiment(X, y, n_splits=5, random_state=42, verbose=True):
    """
    Section 4.3: 5-fold stratified cross-validation.
    Evaluates all baseline models and DWIA.

    Returns
    -------
    dict: {model_name: {metric: mean_value}}
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    model_names = list(get_all_models().keys()) + ["DWIA"]
    results = {name: {"Accuracy": [], "Recall": [], "F1": [], "AUC": []} for name in model_names}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        if verbose:
            print(f"  Fold {fold + 1}/{n_splits} ...", flush=True)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        models = get_all_models()
        for name, model in models.items():
            metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
            for k, v in metrics.items():
                results[name][k].append(v)

        dwia = DynamicWeightIntegrationAlgorithm(M=5, alpha=0.4, beta=0.3, gamma=0.3)
        metrics = evaluate_model(dwia, X_train, y_train, X_test, y_test)
        for k, v in metrics.items():
            results["DWIA"][k].append(v)

    mean_results = {}
    for name in model_names:
        mean_results[name] = {k: np.mean(v) for k, v in results[name].items()}

    return mean_results


# ── Dynamic performance curve experiment ─────────────────────────────────────

def run_sample_size_experiment(X, y, sample_sizes, metric="Accuracy",
                                test_ratio=0.2, random_state=42, verbose=True):
    """
    Section 4.4: performance curves as a function of training set size.
    Fixes the test set and incrementally increases the training set size.

    Parameters
    ----------
    sample_sizes : list[int]  list of training set sizes to evaluate
    metric : str              metric to record
    test_ratio : float        fraction of data held out as the test set

    Returns
    -------
    dict: {model_name: [metric_value_at_each_sample_size]}
    """
    from sklearn.model_selection import train_test_split

    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=random_state
    )

    model_names = list(get_all_models().keys()) + ["DWIA"]
    curves = {name: [] for name in model_names}

    for n in sample_sizes:
        n = min(n, len(X_pool))
        if verbose:
            print(f"  Training samples = {n} ...", flush=True)

        idx = _stratified_sample(y_pool, n, random_state)
        X_train, y_train = X_pool[idx], y_pool[idx]

        models = get_all_models()
        for name, model in models.items():
            metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
            curves[name].append(metrics[metric])

        dwia = DynamicWeightIntegrationAlgorithm(M=5, alpha=0.4, beta=0.3, gamma=0.3)
        metrics = evaluate_model(dwia, X_train, y_train, X_test, y_test)
        curves["DWIA"].append(metrics[metric])

    return curves


def _stratified_sample(y, n, random_state=42):
    """Sample n indices from y with class-proportional stratification."""
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    indices = []
    for cls, cnt in zip(classes, counts):
        cls_idx = np.where(y == cls)[0]
        k = max(1, round(n * cnt / total))
        k = min(k, len(cls_idx))
        chosen = rng.choice(cls_idx, size=k, replace=False)
        indices.extend(chosen.tolist())
    indices = list(set(indices))
    if len(indices) < n:
        remaining = list(set(range(total)) - set(indices))
        extra = rng.choice(remaining, size=min(n - len(indices), len(remaining)), replace=False)
        indices.extend(extra.tolist())
    return np.array(indices[:n])


# ── Print results table ───────────────────────────────────────────────────────

def print_results_table(results, dataset_name=""):
    """Pretty-print the comparison table (corresponds to Table 1 / Table 2 in the paper)."""
    header = f"\n{'='*60}\n  {dataset_name} Results\n{'='*60}"
    print(header)
    print(f"{'Algorithm':<10} {'Accuracy(%)':>12} {'Recall(%)':>10} {'F1(%)':>8} {'AUC':>8}")
    print("-" * 52)
    for name, metrics in results.items():
        print(
            f"{name:<10} "
            f"{metrics['Accuracy']:>12.1f} "
            f"{metrics['Recall']:>10.1f} "
            f"{metrics['F1']:>8.1f} "
            f"{metrics['AUC']:>8.3f}"
        )
    print("=" * 60)
