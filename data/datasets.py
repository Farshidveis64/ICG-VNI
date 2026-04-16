"""
Dataset Loader
==============
Loads benchmark complex-network graphs as NetworkX objects and converts them
to PyTorch Geometric Data objects with node features and SIR vitality labels.

Supported datasets (20 benchmark networks from the paper):
  Social     : Facebook, Twitter-Ego, GitHub, LastFM
  Citation   : Cora, Citeseer, PubMed, DBLP
  Biological : PPI (Yeast), PPI (Human), Drug-Drug, C. elegans
  Infrastructure: PowerGrid, Euroroad, Delaunay, PGP
  Synthetic  : BA (m=2), BA (m=5), ER (p=0.01), ER (p=0.001)
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from data.sir_simulation import compute_vitality_scores, vitality_to_rank
from utils.features import compute_node_features


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Registry
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_DATASETS = [
    "facebook_ego", "twitter_ego", "github", "lastfm",
    "cora", "citeseer", "dblp",
    "ppi_yeast", "drug_drug", "celegans",
    "euroroad", "delaunay", "pgp",
    "ba_m2", "ba_m5",
]

TEST_DATASETS = [
    "facebook", "pubmed", "ppi_human", "power_grid", "er_p001",
]

ALL_DATASETS = TRAIN_DATASETS + TEST_DATASETS


# ─────────────────────────────────────────────────────────────────────────────
# Graph Builders (synthetic + downloadable)
# ─────────────────────────────────────────────────────────────────────────────

def _build_synthetic(name: str, seed: int = 42) -> nx.Graph:
    """Build synthetic BA / ER graphs deterministically."""
    rng = np.random.default_rng(seed)
    nx_seed = int(rng.integers(0, 2**31))

    if name == "ba_m2":
        return nx.barabasi_albert_graph(5000, m=2, seed=nx_seed)
    if name == "ba_m5":
        return nx.barabasi_albert_graph(5000, m=5, seed=nx_seed)
    if name == "er_p001":
        return nx.erdos_renyi_graph(5000, p=0.001, seed=nx_seed)
    if name == "er_p0001":
        return nx.erdos_renyi_graph(5000, p=0.0001, seed=nx_seed)
    raise ValueError(f"Unknown synthetic dataset: {name}")


def _load_snap_or_torch_geometric(name: str, root: str) -> nx.Graph:
    """
    Load a real-world graph.

    Priority:
      1. Cached pickle in <root>/raw/<name>.pkl
      2. PyTorch Geometric built-in (Cora, Citeseer, PubMed)
      3. NetworkX / SNAP download helpers

    For reproducibility in the paper, we assume pre-cached .pkl files.
    Place your graph files at data/cache/raw/<dataset_name>.pkl as a
    pickled NetworkX Graph object.
    """
    cache_path = Path(root) / "raw" / f"{name}.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # ── Fallback: PyG built-ins ───────────────────────────────────────────
    try:
        from torch_geometric.datasets import Planetoid, SNAPDataset
        pyg_map = {
            "cora": ("Cora", Planetoid),
            "citeseer": ("CiteSeer", Planetoid),
            "pubmed": ("PubMed", Planetoid),
        }
        if name in pyg_map:
            ds_name, ds_cls = pyg_map[name]
            dataset = ds_cls(root=str(Path(root) / "pyg"), name=ds_name)
            data = dataset[0]
            G = nx.Graph()
            G.add_nodes_from(range(data.num_nodes))
            edges = data.edge_index.numpy().T.tolist()
            G.add_edges_from(edges)
            G = G.to_undirected()
            G.remove_edges_from(nx.selfloop_edges(G))
            return nx.convert_node_labels_to_integers(
                max(nx.connected_components(G), key=len).__class__(G)
            )
    except Exception as exc:
        print(f"[WARN] Could not load {name} via PyG: {exc}")

    raise FileNotFoundError(
        f"Dataset '{name}' not found at {cache_path}. "
        "Please download and cache the graph as a NetworkX pickle. "
        "See README.md § Data Preparation."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Dataset Class
# ─────────────────────────────────────────────────────────────────────────────

class VNIDataset:
    """
    Vital-Node-Identification dataset wrapper.

    Loads or computes (and caches) SIR vitality scores for each graph,
    then converts to PyG Data objects with node features.

    Parameters
    ----------
    names         : list of dataset names to include
    cache_dir     : directory for raw graphs + SIR score cache
    rwpe_dim      : random-walk positional encoding dimension
    n_simulations : Monte-Carlo SIR runs per node
    beta_factor   : β = beta_factor × β_threshold
    seed          : reproducibility seed
    """

    def __init__(
        self,
        names: List[str],
        cache_dir: str = "./data/cache",
        rwpe_dim: int = 16,
        n_simulations: int = 1000,
        beta_factor: float = 1.5,
        seed: int = 42,
    ) -> None:
        self.names = names
        self.cache_dir = Path(cache_dir)
        self.rwpe_dim = rwpe_dim
        self.n_simulations = n_simulations
        self.beta_factor = beta_factor
        self.seed = seed

        self.graphs: Dict[str, nx.Graph] = {}
        self.data_objects: Dict[str, Data] = {}
        self._load_all()

    # ── Public API ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, name: str) -> Data:
        return self.data_objects[name]

    def items(self):
        return self.data_objects.items()

    # ── Internal Loading Pipeline ─────────────────────────────────────────

    def _load_all(self) -> None:
        """Load all datasets, using caches where available."""
        for name in self.names:
            print(f"  Loading {name} ...", end=" ", flush=True)
            G = self._get_graph(name)
            G = self._preprocess(G)
            self.graphs[name] = G
            vitality = self._get_vitality(name, G)
            self.data_objects[name] = self._to_pyg(G, vitality, name)
            print(f"[{G.number_of_nodes()} nodes, {G.number_of_edges()} edges]")

    def _get_graph(self, name: str) -> nx.Graph:
        """Load graph from cache or build/download it."""
        if name in ("ba_m2", "ba_m5", "er_p001", "er_p0001"):
            return _build_synthetic(name, seed=self.seed)
        return _load_snap_or_torch_geometric(name, str(self.cache_dir))

    @staticmethod
    def _preprocess(G: nx.Graph) -> nx.Graph:
        """Retain largest connected component, remove self-loops."""
        G = G.to_undirected()
        G.remove_edges_from(nx.selfloop_edges(G))
        lcc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(lcc_nodes).copy()
        return nx.convert_node_labels_to_integers(G)

    def _get_vitality(self, name: str, G: nx.Graph) -> Dict[int, float]:
        """Load cached vitality or compute + save it."""
        vit_path = self.cache_dir / "vitality" / f"{name}_b{self.beta_factor}_n{self.n_simulations}.pkl"
        vit_path.parent.mkdir(parents=True, exist_ok=True)

        if vit_path.exists():
            with open(vit_path, "rb") as f:
                return pickle.load(f)

        print("(computing SIR...)", end=" ", flush=True)
        vitality = compute_vitality_scores(
            G,
            beta_factor=self.beta_factor,
            n_simulations=self.n_simulations,
            seed=self.seed,
        )
        with open(vit_path, "wb") as f:
            pickle.dump(vitality, f)
        return vitality

    def _to_pyg(
        self,
        G: nx.Graph,
        vitality: Dict[int, float],
        name: str,
    ) -> Data:
        """
        Convert a NetworkX graph + vitality scores to a PyG Data object.

        Node features x  : structural + RWPE features  [N × F]
        y                : normalised vitality score    [N]
        y_rank           : ordinal rank (1 = best)      [N]
        graph_name       : string tag for identification
        """
        # Node feature matrix
        x = compute_node_features(G, rwpe_dim=self.rwpe_dim)  # [N × F]

        # Vitality labels (normalised to [0, 1])
        nodes = sorted(G.nodes())
        vit_arr = np.array([vitality[n] for n in nodes], dtype=np.float32)
        vit_norm = (vit_arr - vit_arr.min()) / (vit_arr.max() - vit_arr.min() + 1e-8)

        # Ordinal ranks
        ranks = np.argsort(np.argsort(-vit_arr)) + 1  # rank 1 = highest

        # Edge index
        edge_index = torch.tensor(
            [[u, v] for u, v in G.edges()] + [[v, u] for u, v in G.edges()],
            dtype=torch.long,
        ).t().contiguous()

        data = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(vit_norm, dtype=torch.float),
            y_raw=torch.tensor(vit_arr, dtype=torch.float),
            y_rank=torch.tensor(ranks, dtype=torch.long),
            num_nodes=G.number_of_nodes(),
            graph_name=name,
        )
        return data
