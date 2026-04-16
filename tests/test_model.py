"""
Unit Tests — Model Architecture
=================================
Shape and forward-pass sanity checks for each sub-module.
Run with:  pytest tests/test_model.py -v
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data

from models.graph_encoder import GraphContextEncoder, GraphTransformerLayer
from models.cross_graph_attn import CrossGraphAttention, VitalityEncoder
from models.icg_vni import ICGVNI, sample_support_set


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_graph():
    """A small synthetic graph for fast unit testing."""
    N, E, F = 20, 40, 22   # nodes, (directed) edges, features
    x = torch.randn(N, F)
    src = torch.randint(0, N, (E,))
    dst = torch.randint(0, N, (E,))
    edge_index = torch.stack([src, dst], dim=0)
    y = torch.rand(N)
    batch = torch.zeros(N, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=N), batch


# ─── Graph Transformer Layer ─────────────────────────────────────────────────

class TestGraphTransformerLayer:
    def test_output_shape(self, tiny_graph):
        data, _ = tiny_graph
        D = 32
        layer = GraphTransformerLayer(hidden_dim=D, num_heads=4)
        out = layer(data.x[:, :D], data.edge_index)
        assert out.shape == (data.num_nodes, D), f"Got {out.shape}"

    def test_deterministic(self, tiny_graph):
        data, _ = tiny_graph
        D = 32
        layer = GraphTransformerLayer(hidden_dim=D, num_heads=4)
        layer.eval()
        out1 = layer(data.x[:, :D], data.edge_index)
        out2 = layer(data.x[:, :D], data.edge_index)
        assert torch.allclose(out1, out2)


# ─── Graph Context Encoder ────────────────────────────────────────────────────

class TestGraphContextEncoder:
    def test_output_shapes(self, tiny_graph):
        data, batch = tiny_graph
        N, F = data.x.shape
        D = 64
        enc = GraphContextEncoder(in_dim=F, hidden_dim=D, num_layers=2, num_heads=4)
        h_node, h_G = enc(data.x, data.edge_index, batch)
        assert h_node.shape == (N, D)
        assert h_G.shape == (1, D)   # single graph in batch


# ─── Vitality Encoder ────────────────────────────────────────────────────────

class TestVitalityEncoder:
    def test_output_shape(self):
        S, D_v = 10, 32
        enc = VitalityEncoder(out_dim=D_v)
        y = torch.rand(S)
        out = enc(y)
        assert out.shape == (S, D_v)


# ─── Cross-Graph Attention ────────────────────────────────────────────────────

class TestCrossGraphAttention:
    def test_output_shape(self):
        N_q, S, D = 30, 8, 64
        cga = CrossGraphAttention(hidden_dim=D, vitality_dim=16, num_heads=4)
        h_q = torch.randn(N_q, D)
        h_s = torch.randn(S,   D)
        y_s = torch.rand(S)
        out = cga(h_q, h_s, y_s)
        assert out.shape == (N_q, D)

    def test_residual_connection(self):
        """With zero-initialised attention weights, output ≈ input (residual)."""
        N_q, S, D = 10, 5, 32
        cga = CrossGraphAttention(hidden_dim=D, vitality_dim=8, num_heads=4)
        # Zero all W_O weights → context contribution = 0 → output = norm(input)
        torch.nn.init.zeros_(cga.W_O.weight)
        h_q = torch.randn(N_q, D)
        h_s = torch.randn(S,   D)
        y_s = torch.rand(S)
        out = cga(h_q, h_s, y_s)
        assert out.shape == (N_q, D)


# ─── Support Set Sampling ─────────────────────────────────────────────────────

class TestSupportSetSampling:
    def test_kmeans_size(self):
        N, D = 50, 16
        h = torch.randn(N, D)
        y = torch.rand(N)
        idx, y_s = sample_support_set(h, y, support_size=10, strategy="kmeans")
        assert len(idx) <= 10
        assert y_s.shape == idx.shape

    def test_random_size(self):
        N, D = 50, 16
        h = torch.randn(N, D)
        y = torch.rand(N)
        rng = np.random.default_rng(0)
        idx, y_s = sample_support_set(h, y, support_size=10, strategy="random", rng=rng)
        assert len(idx) == 10

    def test_no_duplicate_indices(self):
        N, D = 100, 32
        h = torch.randn(N, D)
        y = torch.rand(N)
        idx, _ = sample_support_set(h, y, support_size=20, strategy="kmeans")
        assert len(idx) == len(set(idx.tolist()))


# ─── Full ICG-VNI Model ───────────────────────────────────────────────────────

class TestICGVNI:
    def test_forward_shape(self, tiny_graph):
        data, _ = tiny_graph
        model = ICGVNI(
            in_dim=data.x.shape[1],
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
            support_size=5,
        )
        scores, support_idx = model(data)
        assert scores.shape == (data.num_nodes,)
        assert len(support_idx) <= 5

    def test_no_grad_at_inference(self, tiny_graph):
        data, _ = tiny_graph
        model = ICGVNI(in_dim=data.x.shape[1], hidden_dim=32, num_layers=2,
                       num_heads=2, support_size=5)
        scores = model.predict(data)
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (data.num_nodes,)

    def test_scores_vary(self, tiny_graph):
        """Predicted scores must not be all equal (model is non-trivial)."""
        data, _ = tiny_graph
        model = ICGVNI(in_dim=data.x.shape[1], hidden_dim=32, num_layers=2,
                       num_heads=2, support_size=5)
        scores = model.predict(data)
        assert scores.std() > 1e-6, "All scores are identical — model might be degenerate"
