# ICG-VNI — Inductive Vital Node Identification via Cross-Graph In-Context Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Paper:** *"Inductive Vital Node Identification across Unseen Networks via Cross-Graph In-Context Learning"*  
> **Journal:** Neural Networks (Elsevier), 2025  
> **Authors:** Farshid Veisi, Kamal Berahmand, Sona Taheri, Lewi Stone, Mahdi Jalili — RMIT University

---

## Overview

ICG-VNI is the **first inductive framework** for vital node identification that generalises to **unseen graphs at inference time without any parameter updates**.

The model uses two jointly trained components:
1. **Graph Context Encoder** — position-aware Graph Transformer producing node and graph embeddings.
2. **Cross-Graph Attention** — calibrates node representations using a small labelled support set from the target graph.

```
Input graph G_test ──► Graph Context Encoder ──► h_v^(L), h_G
Support set S ⊂ V  ──►                        ─►
                        Cross-Graph Attention  ──► c_u
                        VNI Predictor          ──► φ̂(v)
```

---

## Repository Structure

```
icg_vni/
├── configs/
│   └── default.yaml          # all hyperparameters
├── data/
│   ├── datasets.py           # dataset loader & PyG conversion
│   └── sir_simulation.py     # Monte-Carlo SIR spreading dynamics
├── models/
│   ├── graph_encoder.py      # Graph Context Encoder (Graph Transformer)
│   ├── cross_graph_attn.py   # Cross-Graph Attention + VitalityEncoder
│   └── icg_vni.py            # Full ICG-VNI model
├── utils/
│   ├── features.py           # structural features + RWPE
│   ├── losses.py             # ListNet + ApproxNDCG ranking losses
│   ├── metrics.py            # Kendall's τ, JS@k, M(R)
│   └── visualization.py     # publication-ready figures & LaTeX tables
├── tests/
│   ├── test_metrics.py
│   ├── test_model.py
│   └── test_sir.py
├── train.py                  # training entry point
├── evaluate.py               # evaluation & result export
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/icg-vni.git
cd icg-vni

# 2. Create conda environment
conda create -n icgvni python=3.10 -y
conda activate icgvni

# 3. Install PyTorch (adjust CUDA version as needed)
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# 5. Install remaining dependencies
pip install -r requirements.txt
```

---

## Data Preparation

Graph datasets should be placed at `data/cache/raw/<name>.pkl` as pickled
NetworkX `Graph` objects (largest connected component, integer node labels).

**Automatically handled datasets** (downloaded via PyG):
- `cora`, `citeseer`, `pubmed`

**Synthetic datasets** (generated at runtime):
- `ba_m2`, `ba_m5`, `er_p001`

**Manual download required:**
| Dataset | Source |
|---------|--------|
| Facebook | [SNAP](https://snap.stanford.edu/data/ego-Facebook.html) |
| Twitter-Ego | [SNAP](https://snap.stanford.edu/data/ego-Twitter.html) |
| GitHub | [PyG Social](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.GitHubDataset.html) |
| PowerGrid | [Watts & Strogatz 1998](https://www.nature.com/articles/30918) |
| PPI (Human/Yeast) | [BioGRID](https://thebiogrid.org/) |

SIR vitality scores are computed once and cached automatically in
`data/cache/vitality/`.

---

## Running Experiments

### 1. Sanity check (unit tests)
```bash
pytest tests/ -v
```

### 2. Train (single seed)
```bash
python train.py --config configs/default.yaml --seed 42
```

### 3. Train (5 seeds for mean ± std reporting)
```bash
python train.py --config configs/default.yaml --seed 0 1 2 3 4
```

### 4. Evaluate
```bash
# Single checkpoint
python evaluate.py --checkpoint outputs/results/best_model_seed42.pt --seeds 42

# All 5 seeds → mean ± std table
python evaluate.py --checkpoint "outputs/results/best_model_seed{}.pt" --seeds 0 1 2 3 4
```

Output:
- `outputs/results/eval_results.csv` — raw metrics
- `outputs/results/eval_results_table.tex` — LaTeX table (paste into paper)
- `outputs/figures/main_results_bar.pdf` — bar chart

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 128 | Node embedding dimension D |
| `num_encoder_layers` | 3 | Graph Transformer depth L |
| `num_heads` | 4 | Attention heads |
| `rwpe_dim` | 16 | RWPE diffusion steps k |
| `support_size` | 20 | Support set size \|S\| |
| `support_sampling` | kmeans | `kmeans` or `random` |
| `lr` | 1e-3 | Adam learning rate |
| `epochs` | 100 | Training epochs |
| `episodes_per_epoch` | 50 | Episodes per epoch |

All parameters are in `configs/default.yaml`.

---

## Reproducibility

- All random seeds are fixed via `set_seed(seed)` in `train.py`.
- SIR scores are cached after first computation.
- Results reported as **mean ± std over 5 seeds** (seeds 0–4).
- Exact library versions in `requirements.txt`.

---


---

## License

MIT License — see [LICENSE](LICENSE) for details.
