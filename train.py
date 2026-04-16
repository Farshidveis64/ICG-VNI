"""
ICG-VNI — Training Script
==========================
Episode-based training for cross-graph inductive vital node identification.

Usage
-----
  python train.py --config configs/default.yaml --seed 42
  python train.py --config configs/default.yaml --seed 0 1 2 3 4  # 5 runs

Output
------
  outputs/results/best_model_seed<N>.pt      — saved checkpoint
  outputs/results/train_log_seed<N>.csv      — loss + metric per epoch
  outputs/figures/training_curves_seed<N>.pdf
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml

from data.datasets import VNIDataset, TRAIN_DATASETS, TEST_DATASETS
from models.icg_vni import ICGVNI
from utils.losses import listnet_loss
from utils.metrics import evaluate
from utils.visualization import plot_training_curves


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Config Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Load YAML config and return as nested dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# One Training Epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: ICGVNI,
    dataset: VNIDataset,
    optimizer: optim.Optimizer,
    device: torch.device,
    episodes_per_epoch: int,
    rng: np.random.Generator,
) -> float:
    """
    Run one training epoch of episode-based learning.

    Each episode:
      1. Sample one training graph at random.
      2. Forward pass → predicted scores.
      3. Compute ListNet loss on all nodes.
      4. Backward + optimiser step.

    Parameters
    ----------
    model              : ICG-VNI model
    dataset            : training dataset
    optimizer          : Adam optimiser
    device             : cuda or cpu
    episodes_per_epoch : number of episodes per epoch
    rng                : numpy random generator (seeded)

    Returns
    -------
    float  — mean loss over all episodes
    """
    model.train()
    total_loss = 0.0
    graph_names = list(dataset.data_objects.keys())

    for _ in range(episodes_per_epoch):
        # Sample a random training graph
        name = graph_names[rng.integers(len(graph_names))]
        data = dataset[name].to(device)

        optimizer.zero_grad()
        scores, _ = model(data, rng=rng)
        loss = listnet_loss(scores, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / episodes_per_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation on All Test Graphs
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_epoch(
    model: ICGVNI,
    dataset: VNIDataset,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate the model on all graphs in the dataset.

    Returns mean metrics across graphs.
    """
    model.eval()
    all_metrics: dict[str, list[float]] = {}

    for name, data in dataset.items():
        data = data.to(device)
        scores = model.predict(data)
        true_scores = data.y_raw.cpu().numpy()
        m = evaluate(scores, true_scores)
        for k, v in m.items():
            all_metrics.setdefault(k, []).append(v)

    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict, seed: int) -> None:
    """
    Full training pipeline for one random seed.

    Parameters
    ----------
    config : loaded YAML config dict
    seed   : random seed for this run
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  ICG-VNI Training  |  seed={seed}  |  device={device}")
    print(f"{'='*60}")

    # ── Load datasets ──────────────────────────────────────────────────────
    print("\n[1/4] Loading training graphs...")
    train_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg_names = train_cfg.get("train_datasets", TRAIN_DATASETS)
    test_cfg_names  = train_cfg.get("test_datasets",  TEST_DATASETS)

    train_ds = VNIDataset(
        names=train_cfg_names,
        cache_dir=train_cfg.get("cache_dir", "./data/cache"),
        rwpe_dim=model_cfg["rwpe_dim"],
        n_simulations=train_cfg["sir_simulations"],
        beta_factor=train_cfg["beta_factor"],
        seed=seed,
    )
    print("\n[2/4] Loading test graphs...")
    test_ds = VNIDataset(
        names=test_cfg_names,
        cache_dir=train_cfg.get("cache_dir", "./data/cache"),
        rwpe_dim=model_cfg["rwpe_dim"],
        n_simulations=train_cfg["sir_simulations"],
        beta_factor=train_cfg["beta_factor"],
        seed=seed,
    )

    # ── Build model ────────────────────────────────────────────────────────
    sample_data = next(iter(train_ds.data_objects.values()))
    in_dim = sample_data.x.shape[1]

    print(f"\n[3/4] Building ICG-VNI  (in_dim={in_dim}, "
          f"hidden={model_cfg['hidden_dim']}, "
          f"L={model_cfg['num_encoder_layers']})...")

    model = ICGVNI(
        in_dim=in_dim,
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_encoder_layers"],
        num_heads=model_cfg["num_heads"],
        vitality_dim=32,
        support_size=model_cfg["support_size"],
        support_strategy=model_cfg["support_sampling"],
        dropout=model_cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Optimiser & scheduler ──────────────────────────────────────────────
    train_params = config["training"]
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_params["lr"],
        weight_decay=train_params["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_params["epochs"]
    )

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\n[4/4] Training for {train_params['epochs']} epochs "
          f"({train_params['episodes_per_epoch']} episodes/epoch)...\n")

    out_dir = Path("outputs/results")
    fig_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / f"train_log_seed{seed}.csv"
    best_tau = -1.0
    best_epoch = 0
    patience_count = 0
    log_rows = []

    for epoch in range(1, train_params["epochs"] + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_ds, optimizer, device,
            train_params["episodes_per_epoch"], rng
        )
        val_metrics = evaluate_epoch(model, test_ds, device)
        scheduler.step()

        elapsed = time.time() - t0
        tau = val_metrics["kendall_tau"]

        # ── Logging ────────────────────────────────────────────────────────
        row = {"epoch": epoch, "loss": train_loss, **val_metrics}
        log_rows.append(row)

        print(
            f"  Epoch {epoch:3d}/{train_params['epochs']} | "
            f"Loss: {train_loss:.4f} | "
            f"τ: {tau:.4f} | "
            f"JS@10: {val_metrics.get('js_at_10', 0):.4f} | "
            f"M(R): {val_metrics.get('monotonicity', 0):.4f} | "
            f"{elapsed:.1f}s"
        )

        # ── Early stopping & checkpointing ─────────────────────────────────
        if tau > best_tau:
            best_tau = tau
            best_epoch = epoch
            patience_count = 0
            ckpt_path = out_dir / f"best_model_seed{seed}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "kendall_tau": tau,
                "config": config,
                "seed": seed,
            }, ckpt_path)
        else:
            patience_count += 1
            if patience_count >= train_params["patience"]:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best τ={best_tau:.4f} at epoch {best_epoch})")
                break

    # ── Save log CSV ───────────────────────────────────────────────────────
    if log_rows:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)

    # ── Plot training curves ───────────────────────────────────────────────
    plot_training_curves(
        log_rows,
        save_path=str(fig_dir / f"training_curves_seed{seed}.pdf"),
    )

    print(f"\n  Best Kendall's τ = {best_tau:.4f}  (epoch {best_epoch})")
    print(f"  Checkpoint saved → {ckpt_path}")
    print(f"  Log saved        → {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ICG-VNI Training")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--seed", type=int, nargs="+", default=[42],
        help="Random seed(s); multiple seeds run sequentially"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    for seed in args.seed:
        train(config, seed)


if __name__ == "__main__":
    main()
