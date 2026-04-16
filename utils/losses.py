"""
Ranking Losses
==============
ListNet loss for learning-to-rank vital nodes.

Reference:
    Cao et al., "Learning to Rank: From Pairwise Approach to Listwise Approach",
    ICML 2007.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def listnet_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    ListNet top-1 probability ranking loss.

    Converts both predicted scores and ground-truth vitality values to
    probability distributions via softmax, then minimises their KL divergence.

    L(f, y) = − Σ_i  P_y(i) · log P_f(i)
             = CE( softmax(y), softmax(f) )

    Parameters
    ----------
    scores  : predicted vitality logits,  shape [N]
    targets : ground-truth vitality,      shape [N]

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    p_target = F.softmax(targets, dim=0)          # [N]
    log_p_pred = F.log_softmax(scores, dim=0)     # [N]
    return -(p_target * log_p_pred).sum()


def approxndcg_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    ApproxNDCG loss (differentiable NDCG surrogate).

    Used as an optional alternative to ListNet for ablation experiments.

    Parameters
    ----------
    scores      : predicted scores  [N]
    targets     : relevance labels  [N]
    temperature : softmax temperature (lower → closer to true sort)

    Returns
    -------
    torch.Tensor scalar
    """
    N = scores.shape[0]
    # Approximate rank via sorted score differences
    scores_diff = scores.unsqueeze(1) - scores.unsqueeze(0)   # [N, N]
    approx_ranks = (F.sigmoid(-scores_diff / temperature)).sum(dim=1) + 1.0

    # Ideal DCG
    sorted_targets, _ = targets.sort(descending=True)
    ideal_dcg = (sorted_targets / torch.log2(torch.arange(1, N + 1, device=targets.device).float() + 1)).sum()

    # Approx DCG
    gains = targets / torch.log2(approx_ranks + 1.0)
    approx_dcg = gains.sum()

    ndcg = approx_dcg / (ideal_dcg + 1e-8)
    return 1.0 - ndcg
