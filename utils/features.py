"""
Node Feature Extraction
=======================
Computes a feature matrix for each node combining:
  1. Structural features  (degree, k-shell, clustering, …)
  2. Random-Walk Positional Encoding (RWPE)

RWPE Reference:
    Dwivedi et al., "Graph Neural Networks with Learnable Structural
    and Positional Representations", ICLR 2022.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import scipy.sparse as sp


# ─────────────────────────────────────────────────────────────────────────────
# Structural Features
# ─────────────────────────────────────────────────────────────────────────────

def structural_features(G: nx.Graph) -> np.ndarray:
    """
    Compute per-node structural centrality features.

    Features (all log-transformed for scale invariance):
      0: log(degree + 1)
      1: log(k-shell coreness + 1)
      2: log(clustering coefficient × 10 + 1)
      3: log(local_reaching_centrality + 1)   ← approximates closeness
      4: log(triangle count + 1)
      5: log(average neighbour degree + 1)

    Parameters
    ----------
    G : undirected NetworkX graph (nodes must be int, 0-indexed)

    Returns
    -------
    np.ndarray  shape [N, 6]
    """
    N = G.number_of_nodes()
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}

    # ── degree ─────────────────────────────────────────────────────────────
    degree = np.array([G.degree(n) for n in nodes], dtype=np.float32)

    # ── k-shell coreness ───────────────────────────────────────────────────
    core = nx.core_number(G)
    coreness = np.array([core[n] for n in nodes], dtype=np.float32)

    # ── clustering coefficient ─────────────────────────────────────────────
    clust = nx.clustering(G)
    clustering = np.array([clust[n] for n in nodes], dtype=np.float32)

    # ── triangle count ─────────────────────────────────────────────────────
    triangles_dict = nx.triangles(G)
    triangles = np.array([triangles_dict[n] for n in nodes], dtype=np.float32)

    # ── average neighbour degree ───────────────────────────────────────────
    avg_nb = nx.average_neighbor_degree(G)
    avg_nb_arr = np.array([avg_nb[n] for n in nodes], dtype=np.float32)

    # ── local reaching centrality (cheap approximation: inv closeness) ─────
    # Full closeness is O(N·E); we skip for very large graphs.
    if N <= 5000:
        close = nx.closeness_centrality(G)
        reaching = np.array([close[n] for n in nodes], dtype=np.float32)
    else:
        # Approximation: normalised degree
        reaching = degree / (degree.max() + 1e-8)

    # ── Log transform & stack ──────────────────────────────────────────────
    feats = np.stack([
        np.log1p(degree),
        np.log1p(coreness),
        np.log1p(clustering * 10),
        np.log1p(reaching),
        np.log1p(triangles),
        np.log1p(avg_nb_arr),
    ], axis=1)  # [N, 6]

    return feats.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Random-Walk Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

def rwpe(G: nx.Graph, k: int = 16) -> np.ndarray:
    """
    Random-Walk Positional Encoding (RWPE).

    Computes the k-step random-walk landing probabilities for each node:
        PE[v, i] = (A_rw^i)[v, v],   i = 1, …, k

    where A_rw = D^{-1} A is the row-stochastic transition matrix.

    Parameters
    ----------
    G : undirected NetworkX graph
    k : number of diffusion steps

    Returns
    -------
    np.ndarray  shape [N, k]
    """
    N = G.number_of_nodes()
    nodes = sorted(G.nodes())

    # Sparse adjacency matrix
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=np.float32)

    # Row-normalise: A_rw = D^{-1} A
    degree_vec = np.array(A.sum(axis=1)).flatten()
    degree_vec[degree_vec == 0] = 1.0          # avoid divide-by-zero (isolated)
    D_inv = sp.diags(1.0 / degree_vec)
    A_rw = D_inv @ A                           # [N, N] sparse

    # Compute diagonal of A_rw^i for i = 1 .. k
    pe = np.zeros((N, k), dtype=np.float32)
    Ak = A_rw.copy()
    for step in range(k):
        # Diagonal of Ak  →  landing probability
        pe[:, step] = np.array(Ak.diagonal()).flatten()
        if step < k - 1:
            Ak = Ak @ A_rw

    return pe  # [N, k]


# ─────────────────────────────────────────────────────────────────────────────
# Combined Feature Matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_node_features(G: nx.Graph, rwpe_dim: int = 16) -> np.ndarray:
    """
    Build the full node feature matrix X ∈ ℝ^{N × F}.

    F = 6 (structural) + rwpe_dim (positional)

    Parameters
    ----------
    G        : undirected NetworkX graph
    rwpe_dim : number of RWPE steps (k)

    Returns
    -------
    np.ndarray  shape [N, F]
    """
    struct = structural_features(G)       # [N, 6]
    positional = rwpe(G, k=rwpe_dim)      # [N, k]
    return np.concatenate([struct, positional], axis=1)  # [N, 6+k]
