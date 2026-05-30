"""
Main entry point
Runs the full experiment pipeline to reproduce all results from Section 4 of the paper.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import download_datasets, load_dataset_a, load_dataset_b, preprocess
from experiment import run_cv_experiment, run_sample_size_experiment, print_results_table
from visualize import (
    save_results_table,
    plot_accuracy_vs_samples_A,
    plot_recall_vs_samples_A,
    plot_auc_vs_samples_B,
    plot_bar_comparison,
)


def main():
    print("=" * 60)
    print("  Credit Scoring Model Reproduction Experiment")
    print("  Paper: Application and Comparative Study of Machine")
    print("         Learning in Credit Scoring Models")
    print("=" * 60)

    # ── Step 1: Download datasets ─────────────────────────────────────────────
    print("\n[1/5] Downloading datasets ...")
    download_datasets()

    # ── Step 2: Load and preprocess ───────────────────────────────────────────
    print("\n[2/5] Loading and preprocessing data ...")

    X_a_raw, y_a_raw = load_dataset_a()
    X_b_raw, y_b_raw = load_dataset_b()
    print(f"  Dataset A: {X_a_raw.shape[0]} samples, {X_a_raw.shape[1]} features")
    print(f"  Dataset B: {X_b_raw.shape[0]} samples, {X_b_raw.shape[1]} features")

    X_a, y_a, _ = preprocess(X_a_raw, y_a_raw, remove_outliers=True)
    X_b, y_b, _ = preprocess(X_b_raw, y_b_raw, remove_outliers=True)
    print(f"  After preprocessing — A: {X_a.shape[0]} samples  B: {X_b.shape[0]} samples")

    # ── Step 3: 5-fold cross-validation (Table 1 / Table 2) ──────────────────
    print("\n[3/5] 5-fold cross-validation ...")

    print("\n  >> Dataset A")
    results_a = run_cv_experiment(X_a, y_a, n_splits=5, verbose=True)
    print_results_table(results_a, "Dataset A (German Credit)")
    save_results_table(results_a, "Dataset A", "table1_datasetA.csv")

    print("\n  >> Dataset B")
    results_b = run_cv_experiment(X_b, y_b, n_splits=5, verbose=True)
    print_results_table(results_b, "Dataset B (Australian Credit)")
    save_results_table(results_b, "Dataset B", "table2_datasetB.csv")

    # ── Step 4: Dynamic performance curve experiment (Figures 1-3) ───────────
    print("\n[4/5] Dynamic performance curve experiment ...")

    # Dataset A: sample sizes 200->800, step 100 (paper Figures 1 & 2)
    sample_sizes_a = list(range(200, 801, 100))
    print(f"\n  >> Dataset A accuracy curve (sample sizes: {sample_sizes_a})")
    curves_acc_a = run_sample_size_experiment(
        X_a, y_a, sample_sizes_a, metric="Accuracy", verbose=True
    )
    print(f"\n  >> Dataset A recall curve (sample sizes: {sample_sizes_a})")
    curves_rec_a = run_sample_size_experiment(
        X_a, y_a, sample_sizes_a, metric="Recall", verbose=True
    )

    # Dataset B: sample sizes 150->550, step 80 (paper Figure 3)
    sample_sizes_b = list(range(150, 551, 80))
    print(f"\n  >> Dataset B AUC curve (sample sizes: {sample_sizes_b})")
    curves_auc_b = run_sample_size_experiment(
        X_b, y_b, sample_sizes_b, metric="AUC", verbose=True
    )

    # ── Step 5: Generate figures ──────────────────────────────────────────────
    print("\n[5/5] Generating figures ...")
    plot_accuracy_vs_samples_A(sample_sizes_a, curves_acc_a)
    plot_recall_vs_samples_A(sample_sizes_a, curves_rec_a)
    plot_auc_vs_samples_B(sample_sizes_b, curves_auc_b)
    plot_bar_comparison(results_a, results_b)

    print("\n" + "=" * 60)
    print("  Experiment complete! All results saved to output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
