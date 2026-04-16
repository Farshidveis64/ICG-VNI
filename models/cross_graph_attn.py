"""
Cross-Graph Attention Module
=============================
The core ICG-VNI mechanism.  Given:
  • query node embeddings  H_q  ∈ ℝ^{N_q × D}  (all nodes of test graph)
  • support node embeddings H_s  ∈ ℝ^{|S| × D}  (labelled support nodes)
  • support vitality labels  y_s  ∈ ℝ^{|S|}

Cross-Graph Attention computes a context vector c_u for each query node u:

    e_{us} = ⟨W_Q h_u, W_K h_s⟩ / √D
    α_{us} = softmax_s( e_{us} )
    c_u    = Σ_s  α_{us} · ( W_V h_s ∥ ψ(y_s) )

where ψ is a learnable vitality encoder (MLP).

This allows the model to "read in context" the vitality information of the
support set and calibrate its predictions for unseen graphs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VitalityEncoder(nn.Module):
    """
    Scalar vitality value → dense embedding ψ(y) ∈ ℝ^{D_v}.

    A small MLP that converts a scalar SIR vitality score into a
    dense vector suitable for key-value attention.

    Parameters
    ----------
    out_dim : embedding dimension of the vitality encoding
    """

    def __init__(self, out_dim: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y : vitality scalars  [|S|]

        Returns
        -------
        torch.Tensor  [|S|, D_v]
        """
        return self.mlp(y.unsqueeze(-1))   # [|S|, D_v]


class CrossGraphAttention(nn.Module):
    """
    Cross-Graph Attention: transfer vitality knowledge from support set
    to all query nodes of an unseen graph.

    Parameters
    ----------
    hidden_dim    : node embedding dimension D
    vitality_dim  : vitality encoding dimension D_v
    num_heads     : attention heads
    dropout       : attention dropout
    """

    def __init__(
        self,
        hidden_dim: int,
        vitality_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vitality_dim = vitality_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query: from query node embeddings
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Key: from support node embeddings
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Value: concatenation of support embedding + vitality encoding
        self.vitality_enc = VitalityEncoder(out_dim=vitality_dim)
        self.W_V = nn.Linear(hidden_dim + vitality_dim, hidden_dim, bias=False)

        # Output projection
        self.W_O = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_query: torch.Tensor,    # [N_q, D]    all nodes of target graph
        h_support: torch.Tensor,  # [|S|, D]   support node embeddings
        y_support: torch.Tensor,  # [|S|]       support vitality labels
    ) -> torch.Tensor:
        """
        Compute context vectors c_u for every query node.

        Parameters
        ----------
        h_query   : query node embeddings    [N_q, D]
        h_support : support node embeddings  [|S|, D]
        y_support : support vitality labels  [|S|]

        Returns
        -------
        torch.Tensor
            Context-enriched node embeddings  [N_q, D]

        Pseudocode
        ----------
        1. Q  = W_Q( h_query  )     [N_q, D]
        2. K  = W_K( h_support)     [|S|, D]
        3. E  = ψ( y_support  )     [|S|, D_v]
        4. V  = W_V( [h_support ∥ E] )  [|S|, D]
        5. α  = softmax( Q K^T / √d )   [N_q, |S|]
        6. c  = α V                  [N_q, D]
        7. output = norm( h_query + W_O(c) )
        """
        N_q = h_query.shape[0]
        S   = h_support.shape[0]

        # ── Queries, Keys ─────────────────────────────────────────────────
        Q = self.W_Q(h_query)     # [N_q, D]
        K = self.W_K(h_support)   # [S,   D]

        # ── Values = support embedding + vitality encoding ────────────────
        E = self.vitality_enc(y_support)              # [S, D_v]
        V_raw = torch.cat([h_support, E], dim=-1)     # [S, D + D_v]
        V = self.W_V(V_raw)                           # [S, D]

        # ── Multi-head attention ──────────────────────────────────────────
        # Reshape to [*, H, d_k]
        Q = Q.view(N_q, self.num_heads, self.head_dim)   # [N_q, H, d_k]
        K = K.view(S,   self.num_heads, self.head_dim)   # [S,   H, d_k]
        V = V.view(S,   self.num_heads, self.head_dim)   # [S,   H, d_k]

        # Scaled dot-product attention: [N_q, H, S]
        attn = torch.einsum("qhd,shd->qhs", Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)                   # [N_q, H, S]
        attn = self.dropout(attn)

        # Weighted sum over support: [N_q, H, d_k] → [N_q, D]
        context = torch.einsum("qhs,shd->qhd", attn, V)
        context = context.reshape(N_q, self.hidden_dim)  # [N_q, D]

        # ── Output projection + residual + norm ───────────────────────────
        out = self.norm(h_query + self.W_O(context))     # [N_q, D]
        return out
