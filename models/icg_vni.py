"""
ICG-VNI — Full Model
=====================
In-Context Graph learning for Vital Node Identification.

Pipeline (inference):
  1. Encode target graph:  h_v^(L), h_G = GraphContextEncoder(G_test, x)
  2. Sample support set S ⊂ V with known vitality y_S
  3. Cross-Graph Attention: c_u = CGA(h_query, h_S, y_S)
  4. Predict scores: φ̂(u) = FFN_pred( h_u ∥ c_u ∥ h_G )

No parameter updates at inference time — all adaptation is through
the support set in step 3.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from torch_geometric.data import Data

from models.graph_encoder import GraphContextEncoder
from models.cross_graph_attn import CrossGraphAttention


# ─────────────────────────────────────────────────────────────────────────────
# VNI Predictor Head
# ─────────────────────────────────────────────────────────────────────────────

class VNIPredictor(nn.Module):
    """
    FFN that fuses node embedding + context vector + graph embedding
    into a scalar vitality score.

    Input:  [h_u ∥ c_u ∥ h_G]  ∈ ℝ^{3D}
    Output: φ̂(u)              ∈ ℝ
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_node: torch.Tensor,     # [N, D]
        c_node: torch.Tensor,     # [N, D]
        h_G: torch.Tensor,        # [D]  (single graph)
    ) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor  [N]  — predicted vitality scores (unnormalised)
        """
        h_G_expanded = h_G.unsqueeze(0).expand(h_node.shape[0], -1)  # [N, D]
        fused = torch.cat([h_node, c_node, h_G_expanded], dim=-1)    # [N, 3D]
        return self.net(fused).squeeze(-1)                             # [N]


# ─────────────────────────────────────────────────────────────────────────────
# Support Set Sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_support_set(
    h_node: torch.Tensor,           # [N, D]  detached embeddings
    y: torch.Tensor,                # [N]     vitality labels
    support_size: int = 20,
    strategy: str = "kmeans",
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select |S| support nodes from the graph.

    Strategies
    ----------
    kmeans : K-means cluster centroids in embedding space (default, paper)
    random : uniform random sampling (ablation baseline)

    Parameters
    ----------
    h_node       : node embeddings   [N, D]
    y            : vitality labels   [N]
    support_size : |S|
    strategy     : 'kmeans' | 'random'
    rng          : optional numpy random generator

    Returns
    -------
    support_idx  : LongTensor  [|S|]  — indices of selected nodes
    y_support    : FloatTensor [|S|]  — vitality of support nodes
    """
    N = h_node.shape[0]
    support_size = min(support_size, N)

    if rng is None:
        rng = np.random.default_rng(0)

    if strategy == "kmeans":
        embeddings = h_node.detach().cpu().numpy()   # [N, D]
        km = KMeans(n_clusters=support_size, n_init=5, random_state=42)
        km.fit(embeddings)
        # For each cluster, pick the node closest to its centroid
        labels = km.labels_                          # [N]
        centroids = km.cluster_centers_              # [K, D]
        selected = []
        for k in range(support_size):
            mask = labels == k
            if mask.sum() == 0:
                continue
            cluster_nodes = np.where(mask)[0]
            dists = np.linalg.norm(embeddings[cluster_nodes] - centroids[k], axis=1)
            selected.append(cluster_nodes[dists.argmin()])
        support_idx = torch.tensor(selected, dtype=torch.long)

    elif strategy == "random":
        perm = rng.permutation(N)[:support_size]
        support_idx = torch.tensor(perm, dtype=torch.long)

    else:
        raise ValueError(f"Unknown support sampling strategy: {strategy}")

    y_support = y[support_idx]
    return support_idx, y_support


# ─────────────────────────────────────────────────────────────────────────────
# Full ICG-VNI Model
# ─────────────────────────────────────────────────────────────────────────────

class ICGVNI(nn.Module):
    """
    ICG-VNI: In-Context Graph learning for Vital Node Identification.

    Parameters
    ----------
    in_dim         : input node feature dimension
    hidden_dim     : internal embedding dimension D
    num_layers     : Graph Transformer layers L
    num_heads      : attention heads
    vitality_dim   : vitality encoding dimension D_v
    support_size   : |S| support set size
    support_strategy : 'kmeans' | 'random'
    dropout        : dropout rate
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        vitality_dim: int = 32,
        support_size: int = 20,
        support_strategy: str = "kmeans",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.support_size = support_size
        self.support_strategy = support_strategy

        # ── Sub-modules ────────────────────────────────────────────────────
        self.encoder = GraphContextEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.cga = CrossGraphAttention(
            hidden_dim=hidden_dim,
            vitality_dim=vitality_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.predictor = VNIPredictor(hidden_dim=hidden_dim, dropout=dropout)

    def forward(
        self,
        data: Data,
        support_idx: torch.Tensor | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for one graph episode.

        Parameters
        ----------
        data        : PyG Data object with x, edge_index, y, batch
        support_idx : pre-selected support indices (None = auto-sample)
        rng         : random generator for support sampling

        Returns
        -------
        scores      : predicted vitality scores  [N]
        support_idx : indices of support nodes   [|S|]
        """
        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
            else torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)

        # ── Step 1: Encode graph ───────────────────────────────────────────
        h_node, h_G = self.encoder(data.x, data.edge_index, batch)
        # h_node: [N, D],  h_G: [1, D]

        # ── Step 2: Sample support set ─────────────────────────────────────
        if support_idx is None:
            support_idx, y_support = sample_support_set(
                h_node=h_node,
                y=data.y,
                support_size=self.support_size,
                strategy=self.support_strategy,
                rng=rng,
            )
        else:
            y_support = data.y[support_idx]

        support_idx = support_idx.to(data.x.device)
        y_support = y_support.to(data.x.device)

        h_support = h_node[support_idx]   # [|S|, D]

        # ── Step 3: Cross-Graph Attention ──────────────────────────────────
        c_node = self.cga(h_node, h_support, y_support)   # [N, D]

        # ── Step 4: Predict vitality scores ───────────────────────────────
        scores = self.predictor(h_node, c_node, h_G[0])   # [N]

        return scores, support_idx

    @torch.no_grad()
    def predict(self, data: Data, support_idx: torch.Tensor | None = None) -> np.ndarray:
        """
        Convenience method for inference — returns numpy scores.

        Parameters
        ----------
        data        : PyG Data object
        support_idx : optional pre-selected support (for reproducibility)

        Returns
        -------
        np.ndarray  [N]  — predicted vitality scores
        """
        self.eval()
        scores, _ = self.forward(data, support_idx=support_idx)
        return scores.cpu().numpy()
