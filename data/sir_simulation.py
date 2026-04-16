"""
SIR Spreading Dynamics Simulator
=================================
Implements Monte-Carlo SIR simulation on a NetworkX graph to compute
ground-truth vitality scores for each node.

Vitality φ(v) = expected number of nodes infected when v is the seed.

Reference (equation from paper):
    β_threshold = ⟨k⟩ / ⟨k²⟩  (heterogeneous mean-field approximation)
"""

from __future__ import annotations

import random
from typing import Dict, List

import networkx as nx
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core SIR Engine
# ─────────────────────────────────────────────────────────────────────────────

def sir_single_run(
    graph: nx.Graph,
    seed_node: int,
    beta: float,
    mu: float = 1.0,
    rng: random.Random | None = None,
) -> int:
    """
    Run one stochastic SIR episode from a single seed node.

    Parameters
    ----------
    graph     : undirected NetworkX graph
    seed_node : initial infected node
    beta      : per-edge infection probability per time step
    mu        : recovery probability per time step (default = 1.0 → SI model)
    rng       : optional seeded random.Random instance for reproducibility

    Returns
    -------
    int
        Total number of nodes ever infected (including seed).

    Algorithm (discrete-time SIR)
    ------------------------------
    1. S = V \\ {seed},  I = {seed},  R = ∅
    2. While I is non-empty:
       a. For each (i, s) ∈ I × N(i) ∩ S:  infect s w.p. β
       b. Each i ∈ I recovers w.p. μ  → R
    3. Return |R ∪ I|
    """
    if rng is None:
        rng = random.Random()

    susceptible = set(graph.nodes()) - {seed_node}
    infected = {seed_node}
    recovered: set[int] = set()
    total_infected = 1

    while infected:
        newly_infected: set[int] = set()
        newly_recovered: set[int] = set()

        for node in infected:
            # Attempt to spread to each susceptible neighbour
            for nb in graph.neighbors(node):
                if nb in susceptible and rng.random() < beta:
                    newly_infected.add(nb)
            # Recovery
            if rng.random() < mu:
                newly_recovered.add(node)

        susceptible -= newly_infected
        infected |= newly_infected
        infected -= newly_recovered
        recovered |= newly_recovered
        total_infected += len(newly_infected)

    return total_infected


# ─────────────────────────────────────────────────────────────────────────────
# Epidemic Threshold
# ─────────────────────────────────────────────────────────────────────────────

def epidemic_threshold(graph: nx.Graph) -> float:
    """
    Compute the heterogeneous mean-field epidemic threshold.

    β_th = ⟨k⟩ / ⟨k²⟩

    Parameters
    ----------
    graph : NetworkX graph

    Returns
    -------
    float
        Epidemic threshold β_th.
    """
    degrees = np.array([d for _, d in graph.degree()])
    k_mean = degrees.mean()
    k2_mean = (degrees ** 2).mean()
    if k2_mean == 0:
        return float("inf")
    return float(k_mean / k2_mean)


# ─────────────────────────────────────────────────────────────────────────────
# Vitality Score Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_vitality_scores(
    graph: nx.Graph,
    beta: float | None = None,
    beta_factor: float = 1.5,
    mu: float = 1.0,
    n_simulations: int = 1000,
    seed: int = 42,
) -> Dict[int, float]:
    """
    Compute Monte-Carlo SIR vitality φ(v) for every node v in the graph.

    Parameters
    ----------
    graph         : undirected NetworkX graph
    beta          : infection probability; if None, set to beta_factor × β_th
    beta_factor   : multiplier on epidemic threshold (paper uses 1.5)
    mu            : recovery probability
    n_simulations : number of Monte-Carlo runs per node
    seed          : random seed for reproducibility

    Returns
    -------
    dict
        {node_id: mean_infected_count}  (unnormalized vitality scores)

    Notes
    -----
    Total complexity: O(|V| × n_simulations × |E|) in the worst case.
    For large graphs, consider subsampling nodes for support-set labelling.
    """
    if beta is None:
        bt = epidemic_threshold(graph)
        beta = beta_factor * bt

    nodes = list(graph.nodes())
    vitality: Dict[int, float] = {}

    rng = random.Random(seed)

    for node in nodes:
        scores: List[int] = []
        for _ in range(n_simulations):
            infected_count = sir_single_run(graph, node, beta, mu, rng)
            scores.append(infected_count)
        vitality[node] = float(np.mean(scores))

    return vitality


# ─────────────────────────────────────────────────────────────────────────────
# Vitality → Rank
# ─────────────────────────────────────────────────────────────────────────────

def vitality_to_rank(vitality: Dict[int, float]) -> Dict[int, int]:
    """
    Convert vitality scores to ordinal ranks (rank 1 = most influential).

    Parameters
    ----------
    vitality : {node: score}

    Returns
    -------
    dict
        {node: rank}  — rank 1 is the highest vitality node.
    """
    sorted_nodes = sorted(vitality, key=vitality.__getitem__, reverse=True)
    return {node: rank + 1 for rank, node in enumerate(sorted_nodes)}
