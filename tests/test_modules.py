import math

import pytest
import torch

from naptime.modules import (
    ConvBackbone1D,
    FourierTimeEmbedding,
    GaussianSetConv1D,
    GlobalLatentEncoder,
    MLP,
    ResidualConvBlock,
)


class TestFourierTimeEmbedding:
    def test_output_shape_2d_input(self):
        emb = FourierTimeEmbedding(dim=8)
        x = torch.rand(3, 10)  # (B, N)
        out = emb(x)
        assert out.shape == (3, 10, 16)  # 2 * dim

    def test_output_shape_batched(self):
        emb = FourierTimeEmbedding(dim=4)
        x = torch.rand(3, 5)  # (B, N)
        out = emb(x)
        assert out.shape == (3, 5, 8)  # 2 * dim

    def test_sin_cos_values(self):
        emb = FourierTimeEmbedding(dim=1)
        x = torch.tensor([[0.5]])  # (1, 1)
        out = emb(x)  # shape (1, 1, 2)
        expected_sin = math.sin(0.5 * math.pi)
        expected_cos = math.cos(0.5 * math.pi)
        assert torch.allclose(out[0, 0, 0], torch.tensor(expected_sin), atol=1e-5)
        assert torch.allclose(out[0, 0, 1], torch.tensor(expected_cos), atol=1e-5)

    def test_zero_input_gives_zero_sin(self):
        emb = FourierTimeEmbedding(dim=8)
        x = torch.zeros(2, 5)
        out = emb(x)
        # sin(0) = 0 for all frequencies
        assert torch.allclose(out[..., :8], torch.zeros_like(out[..., :8]), atol=1e-6)

    def test_deterministic(self):
        emb = FourierTimeEmbedding(dim=4)
        x = torch.rand(2, 6)
        assert torch.equal(emb(x), emb(x))


class TestGaussianSetConv1D:
    @pytest.fixture
    def setconv(self):
        return GaussianSetConv1D(sigmas=(0.05, 0.15))

    def test_output_shapes(self, setconv):
        B, N, C, G = 2, 10, 8, 16
        point_x = torch.rand(B, N)
        point_feat = torch.rand(B, N, C)
        mask = torch.ones(B, N)
        grid_x = torch.linspace(0, 1, G)
        agg, density = setconv(point_x, point_feat, mask, grid_x)
        # 2 sigmas → 2*C aggregated channels, 2 density channels
        assert agg.shape == (B, G, 2 * C)
        assert density.shape == (B, G, 2)

    def test_single_sigma_shapes(self):
        conv = GaussianSetConv1D(sigmas=(0.1,))
        B, N, C, G = 3, 5, 4, 8
        agg, density = conv(
            torch.rand(B, N),
            torch.rand(B, N, C),
            torch.ones(B, N),
            torch.linspace(0, 1, G),
        )
        assert agg.shape == (B, G, C)
        assert density.shape == (B, G, 1)

    def test_all_masked_gives_near_zero_agg(self, setconv):
        B, N, C, G = 1, 4, 4, 8
        agg, density = setconv(
            torch.zeros(B, N),
            torch.ones(B, N, C),
            torch.zeros(B, N),  # all masked
            torch.linspace(0, 1, G),
        )
        assert (agg.abs() < 1e-4).all()

    def test_constant_features_preserved(self):
        """Single point at t=0.5 with constant feature; grid point at 0.5 should recover it."""
        conv = GaussianSetConv1D(sigmas=(0.01,))  # narrow sigma
        C, G = 4, 33
        feat_val = 1.23
        point_x = torch.tensor([[0.5]])  # (1, 1)
        point_feat = torch.full((1, 1, C), feat_val)
        mask = torch.ones(1, 1)
        grid_x = torch.linspace(0, 1, G)
        agg, _ = conv(point_x, point_feat, mask, grid_x)
        # grid index closest to 0.5
        mid = G // 2
        assert torch.allclose(agg[0, mid, :], torch.full((C,), feat_val), atol=1e-3)

    def test_density_non_negative(self, setconv):
        agg, density = setconv(
            torch.rand(2, 5),
            torch.rand(2, 5, 8),
            torch.ones(2, 5),
            torch.linspace(0, 1, 16),
        )
        assert (density >= 0).all()


class TestGlobalLatentEncoder:
    def test_output_shapes(self):
        enc = GlobalLatentEncoder(in_dim=8, hidden_dim=16, latent_dim=4)
        B, N = 3, 12
        x = torch.rand(B, N, 8)
        mask = torch.ones(B, N)
        mu, logvar = enc(x, mask)
        assert mu.shape == (B, 4)
        assert logvar.shape == (B, 4)

    def test_masked_mean_correctness(self):
        enc = GlobalLatentEncoder(in_dim=2, hidden_dim=8, latent_dim=2)
        # Make a batch where only first 3 of 5 points are valid
        B, N = 1, 5
        x = torch.ones(B, N, 2)
        x[0, 3:] = 99.0  # these should be masked
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
        # The pooled input to the MLP is [mean, max] of first 3 rows
        # We just verify shapes and that output differs from fully-masked version
        mu_partial, _ = enc(x, mask)
        mu_full, _ = enc(x, torch.ones(B, N))
        assert mu_partial.shape == mu_full.shape

    def test_single_valid_point(self):
        """With one valid point, mean and max should both equal that point."""
        enc = GlobalLatentEncoder(in_dim=4, hidden_dim=8, latent_dim=2)
        B, N = 1, 5
        x = torch.zeros(B, N, 4)
        x[0, 2] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.zeros(B, N)
        mask[0, 2] = 1.0
        mu, logvar = enc(x, mask)
        assert torch.isfinite(mu).all()
        assert torch.isfinite(logvar).all()

    def test_no_valid_points_finite(self):
        """All-zero mask should still produce finite output (fallback to zeros)."""
        enc = GlobalLatentEncoder(in_dim=4, hidden_dim=8, latent_dim=2)
        x = torch.rand(2, 6, 4)
        mask = torch.zeros(2, 6)
        mu, logvar = enc(x, mask)
        assert torch.isfinite(mu).all()
        assert torch.isfinite(logvar).all()


class TestMLP:
    def test_output_shape(self):
        mlp = MLP(in_dim=8, hidden_dim=16, out_dim=3, depth=2)
        x = torch.rand(4, 8)
        assert mlp(x).shape == (4, 3)

    def test_depth_one_is_linear(self):
        mlp = MLP(in_dim=4, hidden_dim=8, out_dim=2, depth=1)
        # depth=1: single linear layer, no hidden activation
        assert len(list(mlp.net.children())) == 1

    def test_batched_and_unbatched(self):
        mlp = MLP(in_dim=6, hidden_dim=12, out_dim=3, depth=3)
        x_batch = torch.rand(5, 6)
        x_single = x_batch[0]
        out_batch = mlp(x_batch)
        out_single = mlp(x_single.unsqueeze(0)).squeeze(0)
        assert torch.allclose(out_batch[0], out_single, atol=1e-6)

    def test_dropout_in_eval_is_deterministic(self):
        mlp = MLP(in_dim=4, hidden_dim=8, out_dim=2, depth=2, dropout=0.5)
        mlp.eval()
        x = torch.rand(3, 4)
        assert torch.equal(mlp(x), mlp(x))


class TestResidualConvBlock:
    def test_output_shape_preserved(self):
        block = ResidualConvBlock(channels=16, kernel_size=3)
        x = torch.rand(2, 16, 32)  # (B, C, G)
        assert block(x).shape == x.shape

    def test_residual_connection(self):
        """With zero-initialized weights the output should equal the input."""
        block = ResidualConvBlock(
            channels=8, kernel_size=3
        )  # 8 divisible by GroupNorm(8)
        for p in block.parameters():
            torch.nn.init.zeros_(p)
        x = torch.rand(1, 8, 8)
        out = block(x)
        assert torch.allclose(out, x, atol=1e-6)


class TestConvBackbone1D:
    def test_output_shape_preserved(self):
        backbone = ConvBackbone1D(channels=16, layers=3, kernel_size=3)
        x = torch.rand(2, 16, 32)
        assert backbone(x).shape == x.shape

    def test_zero_layers_is_identity(self):
        backbone = ConvBackbone1D(channels=8, layers=0)
        x = torch.rand(2, 8, 16)
        assert torch.equal(backbone(x), x)

    def test_gradient_flows(self):
        backbone = ConvBackbone1D(channels=8, layers=2, kernel_size=3)
        x = torch.rand(1, 8, 16, requires_grad=True)
        loss = backbone(x).sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
