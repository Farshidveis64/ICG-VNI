"""
Unit Tests — SIR Simulation
==============================
Correctness and statistical sanity checks for the SIR module.
Run with:  pytest tests/test_sir.py -v
"""

import random
import pytest
import networkx as nx
import numpy as np

from data.sir_simulation import (
    sir_single_run,
    epidemic_threshold,
    compute_vitality_scores,
    vitality_to_rank,
)


@pytest.fixture
def star_graph():
    """Star graph: hub node (0) should always be most vital."""
    return nx.star_graph(19)   # 1 hub + 19 leaves = 20 nodes


@pytest.fixture
def path_graph():
    return nx.path_graph(10)


class TestSIRSingleRun:
    def test_at_least_one_infected(self, star_graph):
        G = star_graph
        rng = random.Random(0)
        result = sir_single_run(G, seed_node=0, beta=0.9, mu=1.0, rng=rng)
        assert result >= 1

    def test_at_most_n_infected(self, star_graph):
        G = star_graph
        rng = random.Random(0)
        result = sir_single_run(G, seed_node=0, beta=0.9, mu=1.0, rng=rng)
        assert result <= G.number_of_nodes()

    def test_zero_beta(self, path_graph):
        """β=0: only seed node ever infected."""
        G = path_graph
        rng = random.Random(0)
        result = sir_single_run(G, seed_node=0, beta=0.0, mu=1.0, rng=rng)
        assert result == 1

    def test_full_beta(self, path_graph):
        """β=1, μ=1: whole connected component infected."""
        G = path_graph
        rng = random.Random(0)
        result = sir_single_run(G, seed_node=0, beta=1.0, mu=1.0, rng=rng)
        assert result == G.number_of_nodes()


class TestEpidemicThreshold:
    def test_complete_graph(self):
        """Complete graph K_n: β_th = 1/(n-1)."""
        n = 10
        G = nx.complete_graph(n)
        bt = epidemic_threshold(G)
        expected = 1.0 / (n - 1)
        assert abs(bt - expected) < 1e-6

    def test_star_graph(self):
        """Star K_{1,n}: β_th is small (high-degree hub)."""
        G = nx.star_graph(19)
        bt = epidemic_threshold(G)
        assert 0 < bt < 0.1   # threshold should be low for star graph

    def test_regular_graph(self):
        """k-regular graph: β_th = 1/(k)."""
        k, n = 4, 20
        G = nx.random_regular_graph(k, n, seed=0)
        bt = epidemic_threshold(G)
        expected = 1.0 / k
        assert abs(bt - expected) < 0.05   # approximate for finite graph


class TestComputeVitalityScores:
    def test_hub_is_most_vital(self, star_graph):
        """In a star graph, hub (node 0) must have highest vitality."""
        G = star_graph
        vitality = compute_vitality_scores(
            G, beta_factor=2.0, n_simulations=200, seed=42
        )
        hub = max(vitality, key=vitality.__getitem__)
        assert hub == 0, f"Hub should be 0, got {hub}"

    def test_all_nodes_have_scores(self, path_graph):
        G = path_graph
        vitality = compute_vitality_scores(
            G, beta_factor=2.0, n_simulations=50, seed=0
        )
        assert set(vitality.keys()) == set(G.nodes())

    def test_scores_positive(self, path_graph):
        G = path_graph
        vitality = compute_vitality_scores(
            G, beta_factor=2.0, n_simulations=50, seed=0
        )
        assert all(v >= 1.0 for v in vitality.values())   # at least seed


class TestVitalityToRank:
    def test_rank_1_is_highest(self):
        vitality = {0: 10.0, 1: 5.0, 2: 8.0}
        ranks = vitality_to_rank(vitality)
        assert ranks[0] == 1     # highest vitality → rank 1

    def test_all_ranks_assigned(self):
        vitality = {i: float(i) for i in range(5)}
        ranks = vitality_to_rank(vitality)
        assert set(ranks.values()) == {1, 2, 3, 4, 5}
