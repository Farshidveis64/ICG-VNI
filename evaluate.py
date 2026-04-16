"""
ICG-VNI — Evaluation Script
============================
Evaluates a trained ICG-VNI checkpoint on all test graphs and produces
publication-ready tables and figures.

Usage
-----
  # Evaluate one checkpoint
  python evaluate.py --checkpoint outputs/results/best_model_seed42.pt

  # Evaluate 5 seeds and report mean ± std
  python evaluate.py --checkpoint outputs/results/best_model_seed{}.pt \\
                     --seeds 0 1 2 3 4

Output
------
  outputs/results/eval_results.csv
  outputs/figures/main_results_bar.pdf
  outputs/figures/scalability.pdf
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

from data.datasets import VNIDataset, TEST_DATASETS
from models.icg_vni import ICGVNI
from utils.metrics import evaluate
from utils.visualization import (
    plot_results_bar,
    plot_scalability,
    results_to_latex_table,
)


# ─────────────────────────────────────────────────────────────────────────────
# Load Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> tuple[ICGVNI, dict]:
    """Load model weights and config from a .pt checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    model_cfg = config["model"]
    data_cfg  = config["data"]

    # Infer in_dim from a test graph (load one to get feature dim)
    sample_ds = VNIDataset(
        names=[TEST_DATASETS[0]],
        cache_dir=data_cfg.get("cache_dir", "./data/cache"),
        rwpe_dim=model_cfg["rwpe_dim"],
        n_simulations=100,  # fast approximation for dim inference
        beta_factor=data_cfg["beta_factor"],
        seed=0,
    )
    in_dim = next(iter(sample_ds.data_objects.values())).x.shape[1]

    model = ICGVNI(
        in_dim=in_dim,
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_encoder_layers"],
        num_heads=model_cfg["num_heads"],
        support_size=model_cfg["support_size"],
        support_strategy=model_cfg["support_sampling"],
        dropout=0.0,  # disable at eval
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(τ={ckpt['kendall_tau']:.4f})")
    return model, config


# ─────────────────────────────────────────────────────────────────────────────
# Per-Graph Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_all(
    model: ICGVNI,
    test_ds: VNIDataset,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """
    Evaluate the model on every test graph.

    Returns
    -------
    dict  {graph_name: {metric_name: value}}
    """
    results = {}
    for name, data in test_ds.items():
        data = data.to(device)
        scores = model.predict(data)
        true_scores = data.y_raw.cpu().numpy()
        m = evaluate(scores, true_scores)
        results[name] = m
        print(f"  {name:20s}  τ={m['kendall_tau']:.4f}  "
              f"JS@10={m['js_at_10']:.4f}  M(R)={m['monotonicity']:.4f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Seed Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_seeds(
    seed_results: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, str]]:
    """
    Aggregate per-seed results into mean ± std strings.

    Parameters
    ----------
    seed_results : list of per-seed result dicts

    Returns
    -------
    dict  {graph_name: {metric: "mean ± std"}}
    """
    graph_names = list(seed_results[0].keys())
    metric_names = list(seed_results[0][graph_names[0]].keys())

    aggregated = {}
    for name in graph_names:
        aggregated[name] = {}
        for metric in metric_names:
            vals = [r[name][metric] for r in seed_results]
            mean, std = np.mean(vals), np.std(vals)
            aggregated[name][metric] = f"{mean:.4f} ± {std:.4f}"

    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ICG-VNI Evaluation")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to .pt checkpoint (use {} as seed placeholder for multi-seed)"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42],
        help="Random seed(s) to evaluate"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("outputs/results")
    fig_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    seed_results = []

    for seed in args.seed if hasattr(args, "seed") else args.seeds:
        ckpt_path = args.checkpoint.replace("{}", str(seed))
        print(f"\n{'─'*50}")
        print(f"Evaluating seed={seed}: {ckpt_path}")
        print(f"{'─'*50}")

        model, config = load_model(ckpt_path, device)
        data_cfg = config["data"]
        model_cfg = config["model"]

        test_ds = VNIDataset(
            names=data_cfg.get("test_datasets", TEST_DATASETS),
            cache_dir=data_cfg.get("cache_dir", "./data/cache"),
            rwpe_dim=model_cfg["rwpe_dim"],
            n_simulations=data_cfg["sir_simulations"],
            beta_factor=data_cfg["beta_factor"],
            seed=seed,
        )

        per_graph = evaluate_all(model, test_ds, device)
        seed_results.append(per_graph)

    # ── Aggregate across seeds ─────────────────────────────────────────────
    agg = aggregate_seeds(seed_results)

    # ── Print summary table ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY (mean ± std over seeds)")
    print(f"{'='*60}")
    for name, metrics in agg.items():
        print(f"  {name:20s}  " +
              "  ".join(f"{k}: {v}" for k, v in metrics.items()))

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_path = out_dir / "eval_results.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["graph"] + list(next(iter(agg.values())).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, metrics in agg.items():
            writer.writerow({"graph": name, **metrics})
    print(f"\n  Results saved → {csv_path}")

    # ── LaTeX table ────────────────────────────────────────────────────────
    latex = results_to_latex_table(agg)
    latex_path = out_dir / "eval_results_table.tex"
    latex_path.write_text(latex)
    print(f"  LaTeX table  → {latex_path}")

    # ── Figures ────────────────────────────────────────────────────────────
    if seed_results:
        plot_results_bar(
            seed_results[0],  # use first seed for bar plot
            save_path=str(fig_dir / "main_results_bar.pdf"),
        )


if __name__ == "__main__":
    main()
