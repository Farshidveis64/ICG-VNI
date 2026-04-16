"""
Evaluation Metrics
==================
Implements all metrics reported in the paper:

  • Kendall's τ   — rank correlation between predicted and true ranking
  • JS@k          — Jaccard similarity of top-k sets
  • M(R)          — Monotonicity (uniqueness of ranking)
  • GG            — Generalisation Gap (transductive τ − inductive τ)
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Kendall's τ
# ─────────────────────────────────────────────────────────────────────────────

def kendall_tau(
    pred_scores: np.ndarray,
    true_scores: np.ndarray,
) -> float:
    """
    Kendall's rank correlation coefficient τ_b.

    τ ∈ [−1, 1]; τ = 1 means perfect rank agreement.

    Parameters
    ----------
    pred_scores : predicted vitality scores  [N]
    true_scores : ground-truth SIR scores    [N]

    Returns
    -------
    float
    """
    tau, _ = stats.kendalltau(pred_scores, true_scores)
    return float(tau)


# ─────────────────────────────────────────────────────────────────────────────
# Jaccard Similarity @ k
# ─────────────────────────────────────────────────────────────────────────────

def jaccard_at_k(
    pred_scores: np.ndarray,
    true_scores: np.ndarray,
    k: int = 10,
) -> float:
    """
    Jaccard similarity of the predicted and true top-k influential nodes.

    JS@k = |top-k(pred) ∩ top-k(true)| / |top-k(pred) ∪ top-k(true)|

    Parameters
    ----------
    pred_scores : predicted scores  [N]
    true_scores : ground-truth      [N]
    k           : number of top nodes

    Returns
    -------
    float ∈ [0, 1]
    """
    k = min(k, len(pred_scores))
    pred_top = set(np.argsort(-pred_scores)[:k])
    true_top = set(np.argsort(-true_scores)[:k])
    intersection = len(pred_top & true_top)
    union = len(pred_top | true_top)
    return float(intersection / union) if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Monotonicity M(R)
# ─────────────────────────────────────────────────────────────────────────────

def monotonicity(pred_scores: np.ndarray) -> float:
    """
    Monotonicity of the predicted ranking (higher = more unique ranks).

    M(R) = 1 − Σ_r  n_r(n_r − 1) / (N(N − 1))

    where n_r is the number of nodes sharing rank r (ties count).
    M(R) = 1 iff all ranks are unique; M(R) = 0 iff all nodes tied.

    Parameters
    ----------
    pred_scores : predicted scores  [N]

    Returns
    -------
    float ∈ [0, 1]
    """
    N = len(pred_scores)
    if N <= 1:
        return 1.0
    _, counts = np.unique(pred_scores, return_counts=True)
    penalty = sum(c * (c - 1) for c in counts)
    return float(1.0 - penalty / (N * (N - 1)))


# ─────────────────────────────────────────────────────────────────────────────
# Generalisation Gap
# ─────────────────────────────────────────────────────────────────────────────

def generalisation_gap(tau_transductive: float, tau_inductive: float) -> float:
    """
    Generalisation Gap (GG) between transductive and inductive performance.

    GG = τ_transductive − τ_inductive

    A lower GG indicates better cross-graph generalisation.
    """
    return float(tau_transductive - tau_inductive)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    pred_scores: np.ndarray,
    true_scores: np.ndarray,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """
    Compute all paper metrics in one call.

    Parameters
    ----------
    pred_scores : model output scores  [N]
    true_scores : SIR vitality labels  [N]
    k_values    : list of k for JS@k   (default: [10, 50])

    Returns
    -------
    dict with keys: kendall_tau, js_at_10, js_at_50, monotonicity
    """
    if k_values is None:
        k_values = [10, 50]

    results: dict[str, float] = {
        "kendall_tau": kendall_tau(pred_scores, true_scores),
        "monotonicity": monotonicity(pred_scores),
    }
    for k in k_values:
        results[f"js_at_{k}"] = jaccard_at_k(pred_scores, true_scores, k=k)

    return results
