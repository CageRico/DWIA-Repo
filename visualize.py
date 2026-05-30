"""
Visualization module
Reproduces Table 1, Table 2, and Figures 1-3 from the paper.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless environment compatibility
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Display order of algorithms as in the paper
MODEL_ORDER = ["LR", "DT", "SVM", "NN", "ICNN", "RNN", "DWIA"]

# Color and line style mapping (matching the paper's figure style)
STYLE_MAP = {
    "LR":   {"color": "#1f77b4", "linestyle": "-",  "marker": "o"},
    "DT":   {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},
    "SVM":  {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},
    "NN":   {"color": "#d62728", "linestyle": ":",  "marker": "D"},
    "ICNN": {"color": "#9467bd", "linestyle": "-",  "marker": "v"},
    "RNN":  {"color": "#8c564b", "linestyle": "--", "marker": "P"},
    "DWIA": {"color": "#e377c2", "linestyle": "-",  "marker": "*"},
}


# ── Table 1 / Table 2: results table (CSV + console output) ──────────────────

def save_results_table(results, dataset_name, filename):
    """Save experiment results as a CSV file (corresponds to Table 1 / Table 2 in the paper)."""
    import csv
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Accuracy(%)", "Recall(%)", "F1(%)", "AUC"])
        for name in MODEL_ORDER:
            if name not in results:
                continue
            m = results[name]
            writer.writerow([
                name,
                f"{m['Accuracy']:.1f}",
                f"{m['Recall']:.1f}",
                f"{m['F1']:.1f}",
                f"{m['AUC']:.3f}",
            ])
    print(f"  [Table] {dataset_name} results saved to {path}")
    return path


# ── Figure 1: Dataset A accuracy vs. number of training samples ───────────────

def plot_accuracy_vs_samples_A(sample_sizes, curves, save=True):
    """Reproduce Figure 1: Dataset A accuracy as training set size increases (200->800)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for name in MODEL_ORDER:
        if name not in curves:
            continue
        s = STYLE_MAP[name]
        ax.plot(
            sample_sizes, curves[name],
            color=s["color"], linestyle=s["linestyle"],
            marker=s["marker"], markersize=6, linewidth=1.8,
            label=name
        )

    ax.set_xlabel("Number of Training Samples", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("The Accuracy of credit dataset A changes\nwith the number of training samples", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "figure1_accuracy_datasetA.png")
        fig.savefig(path, dpi=150)
        print(f"  [Figure 1] Saved to {path}")
        plt.close(fig)
    return fig


# ── Figure 2: Dataset A recall vs. number of training samples ────────────────

def plot_recall_vs_samples_A(sample_sizes, curves, save=True):
    """Reproduce Figure 2: Dataset A recall as training set size increases."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for name in MODEL_ORDER:
        if name not in curves:
            continue
        s = STYLE_MAP[name]
        ax.plot(
            sample_sizes, curves[name],
            color=s["color"], linestyle=s["linestyle"],
            marker=s["marker"], markersize=6, linewidth=1.8,
            label=name
        )

    ax.set_xlabel("Number of Training Samples", fontsize=12)
    ax.set_ylabel("Recall (%)", fontsize=12)
    ax.set_title("Recall rate of credit dataset A changes\nwith the number of training samples", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "figure2_recall_datasetA.png")
        fig.savefig(path, dpi=150)
        print(f"  [Figure 2] Saved to {path}")
        plt.close(fig)
    return fig


# ── Figure 3: Dataset B AUC vs. number of training samples ───────────────────

def plot_auc_vs_samples_B(sample_sizes, curves, save=True):
    """Reproduce Figure 3: Dataset B AUC as training set size increases (150->550)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for name in MODEL_ORDER:
        if name not in curves:
            continue
        s = STYLE_MAP[name]
        ax.plot(
            sample_sizes, curves[name],
            color=s["color"], linestyle=s["linestyle"],
            marker=s["marker"], markersize=6, linewidth=1.8,
            label=name
        )

    ax.set_xlabel("Number of Training Samples", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("AUC value of the B credit data set changes\nwith the number of training samples", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "figure3_auc_datasetB.png")
        fig.savefig(path, dpi=150)
        print(f"  [Figure 3] Saved to {path}")
        plt.close(fig)
    return fig


# ── Figure 4: Bar chart comparison across both datasets ──────────────────────

def plot_bar_comparison(results_A, results_B, save=True):
    """Bar chart comparing all algorithms on four metrics for Dataset A and B."""
    metrics = ["Accuracy", "Recall", "F1", "AUC"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    names = [n for n in MODEL_ORDER if n in results_A]
    x = np.arange(len(names))
    width = 0.35

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        vals_A = [results_A[n][metric] for n in names]
        vals_B = [results_B[n][metric] for n in names]

        bars_A = ax.bar(x - width / 2, vals_A, width, label="Dataset A", color="#4C72B0", alpha=0.85)
        bars_B = ax.bar(x + width / 2, vals_B, width, label="Dataset B", color="#DD8452", alpha=0.85)

        ax.set_title(metric, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for bar in bars_A:
            h = bar.get_height()
            fmt = f"{h:.3f}" if metric == "AUC" else f"{h:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, fmt,
                    ha="center", va="bottom", fontsize=7)
        for bar in bars_B:
            h = bar.get_height()
            fmt = f"{h:.3f}" if metric == "AUC" else f"{h:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, fmt,
                    ha="center", va="bottom", fontsize=7)

    plt.suptitle("Algorithm Comparison on Dataset A and B", fontsize=14, y=1.01)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "figure4_bar_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [Figure 4] Saved to {path}")
        plt.close(fig)
    return fig
