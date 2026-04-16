"""
Publication-Ready Visualisation
================================
All figures follow the style conventions of the paper:
  • Font: Computer Modern (LaTeX-compatible)
  • Colour palette: colour-blind-friendly (IBM palette)
  • Sizes: single-column (3.5 in) and double-column (7 in)
  • DPI: 300 for raster, vector PDF for final output
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless environments

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from typing import Any

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":  "cm",
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        300,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Colour-blind-friendly IBM palette
COLORS = {
    "icgvni":   "#0F62FE",   # IBM Blue
    "eml":      "#FF6B35",   # Orange
    "ds":       "#4CBB17",   # Green
    "gcn":      "#9B59B6",   # Purple
    "train":    "#0F62FE",
    "val":      "#FF6B35",
}

MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]


# ─────────────────────────────────────────────────────────────────────────────
# Training Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    log_rows: list[dict],
    save_path: str = "outputs/figures/training_curves.pdf",
) -> None:
    """
    Plot training loss and Kendall's τ over epochs.

    Parameters
    ----------
    log_rows  : list of dicts, each with keys 'epoch', 'loss', 'kendall_tau'
    save_path : output PDF path
    """
    epochs = [r["epoch"] for r in log_rows]
    losses = [r["loss"] for r in log_rows]
    taus   = [r["kendall_tau"] for r in log_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    # ── Loss ────────────────────────────────────────────────────────────────
    ax1.plot(epochs, losses, color=COLORS["train"], linewidth=1.5, label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("ListNet Loss")
    ax1.set_title("Training Loss")
    ax1.legend()

    # ── Kendall's τ ─────────────────────────────────────────────────────────
    ax2.plot(epochs, taus, color=COLORS["val"], linewidth=1.5, label=r"Kendall's $\tau$")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(r"Kendall's $\tau$")
    ax2.set_title(r"Validation Kendall's $\tau$")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Results Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_results_bar(
    results: dict[str, dict[str, float]],
    metric: str = "kendall_tau",
    save_path: str = "outputs/figures/main_results_bar.pdf",
) -> None:
    """
    Grouped bar chart comparing ICG-VNI against baselines per test graph.

    Parameters
    ----------
    results   : {graph_name: {metric_name: value}}
    metric    : metric to plot on y-axis
    save_path : output PDF path
    """
    graph_names = list(results.keys())
    values = [results[n].get(metric, 0.0) for n in graph_names]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    x = np.arange(len(graph_names))
    bars = ax.bar(x, values, color=COLORS["icgvni"], alpha=0.85, width=0.6,
                  label="ICG-VNI (ours)")

    # Annotate bar tops
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=7.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [n.replace("_", "\n") for n in graph_names], fontsize=8
    )
    ax.set_ylabel(r"Kendall's $\tau$")
    ax.set_title(r"ICG-VNI — Kendall's $\tau$ on Test Graphs")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Scalability Plot (log-log)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scalability(
    graph_sizes: list[int],
    inference_times: dict[str, list[float]],
    save_path: str = "outputs/figures/scalability.pdf",
) -> None:
    """
    Log-log inference time vs graph size.

    Parameters
    ----------
    graph_sizes     : list of |V| values (x-axis)
    inference_times : {method_name: [time_in_seconds]}
    save_path       : output PDF path
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    for i, (method, times) in enumerate(inference_times.items()):
        color = list(COLORS.values())[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        linestyle = "-" if method == "ICG-VNI" else "--"
        ax.loglog(
            graph_sizes, times,
            color=color, marker=marker, linestyle=linestyle,
            linewidth=1.5, markersize=5, label=method,
        )

    # Reference O(n) line
    n_arr = np.array(graph_sizes, dtype=float)
    ref = n_arr / n_arr[0] * inference_times["ICG-VNI"][0]
    ax.loglog(n_arr, ref, "k:", linewidth=1.0, label=r"$\mathcal{O}(n)$ ref.")

    ax.set_xlabel(r"Graph size $|\mathcal{V}|$")
    ax.set_ylabel("Inference time (s)")
    ax.set_title("Scalability (log-log)")
    ax.legend(fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX Table Generator
# ─────────────────────────────────────────────────────────────────────────────

def results_to_latex_table(
    agg_results: dict[str, dict[str, str]],
    caption: str = "ICG-VNI evaluation results (mean ± std over 5 seeds).",
    label: str = "tab:eval",
) -> str:
    """
    Convert aggregated results dict to a LaTeX tabular string.

    Parameters
    ----------
    agg_results : {graph_name: {metric: "mean ± std"}}
    caption     : table caption
    label       : table label

    Returns
    -------
    str  — complete LaTeX table source
    """
    graph_names  = list(agg_results.keys())
    metric_names = list(next(iter(agg_results.values())).keys())

    col_fmt = "l" + "c" * len(metric_names)
    header  = " & ".join(
        ["\\textbf{Graph}"] +
        [f"\\textbf{{{m.replace('_', ' ')}}}" for m in metric_names]
    ) + " \\\\"

    rows = []
    for name in graph_names:
        row_vals = [agg_results[name].get(m, "—") for m in metric_names]
        rows.append(f"  {name} & " + " & ".join(row_vals) + " \\\\")

    body = "\n".join(rows)

    return (
        f"\\begin{{table}}[H]\n"
        f"\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{col_fmt}}}\n"
        f"\\toprule\n"
        f"{header}\n"
        f"\\midrule\n"
        f"{body}\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}\n"
        f"\\end{{table}}\n"
    )
