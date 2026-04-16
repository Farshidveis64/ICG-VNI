"""
Graph Context Encoder
=====================
A Graph Transformer encoder that produces position-aware, graph-agnostic
node embeddings h_v^(L) and a global graph embedding h_G.

Architecture (per layer):
  1. Multi-head self-attention on node features
  2. Feed-forward network (FFN)
  3. LayerNorm + residual connections
  4. Global readout for graph-level embedding

Reference:
    Dwivedi & Bresson, "A Generalization of Transformer Networks to Graphs",
    AAAI-W 2021.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax as pyg_softmax


# ─────────────────────────────────────────────────────────────────────────────
# Graph Transformer Layer
# ─────────────────────────────────────────────────────────────────────────────

class GraphTransformerLayer(MessagePassing):
    """
    Single Graph Transformer layer with multi-head attention.

    For each node v:
        α_{uv} = softmax_u( (W_Q h_v)^T (W_K h_u) / √d_k )
        h_v'   = FFN( h_v + Σ_u α_{uv} W_V h_u )

    Parameters
    ----------
    hidden_dim : embedding dimension
    num_heads  : number of attention heads
    dropout    : dropout probability
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projection matrices
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Feed-forward sub-layer
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : node features  [N, D]
        edge_index : edge indices   [2, E]

        Returns
        -------
        torch.Tensor  [N, D]
        """
        # ── Multi-head attention (with residual) ──────────────────────────
        q = self.W_Q(x)   # [N, D]
        k = self.W_K(x)   # [N, D]
        v = self.W_V(x)   # [N, D]

        attn_out = self.propagate(edge_index, q=q, k=k, v=v, size=None)
        attn_out = self.W_O(attn_out)
        x = self.norm1(x + self.dropout(attn_out))

        # ── Feed-forward (with residual) ───────────────────────────────────
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

    def message(
        self,
        q_i: torch.Tensor,   # query from destination node  [E, D]
        k_j: torch.Tensor,   # key from source node          [E, D]
        v_j: torch.Tensor,   # value from source node        [E, D]
        index: torch.Tensor, # destination node indices       [E]
    ) -> torch.Tensor:
        """Compute scaled dot-product attention messages."""
        # Reshape for multi-head: [E, H, d_k]
        E = q_i.shape[0]
        q_i = q_i.view(E, self.num_heads, self.head_dim)
        k_j = k_j.view(E, self.num_heads, self.head_dim)
        v_j = v_j.view(E, self.num_heads, self.head_dim)

        # Attention score per head  [E, H]
        attn = (q_i * k_j).sum(dim=-1) * self.scale

        # Softmax over incoming edges per node
        attn = pyg_softmax(attn, index=index)   # [E, H]

        # Weighted sum  [E, H, d_k] → [E, D]
        out = (attn.unsqueeze(-1) * v_j).reshape(E, self.hidden_dim)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Full Graph Context Encoder
# ─────────────────────────────────────────────────────────────────────────────

class GraphContextEncoder(nn.Module):
    """
    Stack of L Graph Transformer layers with a global graph-level readout.

    Input projection: x ∈ ℝ^{F}  →  h^(0) ∈ ℝ^{D}
    L attention layers produce h^(L) ∈ ℝ^{D} per node.
    Graph embedding h_G = MLP( [mean(h^(L)) ∥ max(h^(L))] )

    Parameters
    ----------
    in_dim     : input feature dimension F
    hidden_dim : embedding dimension D
    num_layers : number of transformer layers L
    num_heads  : attention heads per layer
    dropout    : dropout rate
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Graph-level readout MLP
        self.graph_readout = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x          : node features   [N, F]
        edge_index : edge list       [2, E]
        batch      : batch vector    [N]  (node → graph index)

        Returns
        -------
        h_node : node embeddings   [N, D]
        h_G    : graph embedding   [B, D]  (B = batch size)
        """
        # Project input features
        h = self.input_proj(x)            # [N, D]

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, edge_index)      # [N, D]

        h_node = h

        # Graph-level embedding via mean+max pooling
        h_mean = global_mean_pool(h, batch)   # [B, D]
        h_max  = global_max_pool(h, batch)    # [B, D]
        h_G = self.graph_readout(torch.cat([h_mean, h_max], dim=-1))  # [B, D]

        return h_node, h_G
